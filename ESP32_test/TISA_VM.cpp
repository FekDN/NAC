// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

#include "TISA_VM.h"
#include <Arduino.h>
#include <algorithm>
#include <limits>
#include "CYD28_SD.h" 

// Определяем внешние переменные, которые будут использоваться
extern CYD28_SD sdcard;
extern SemaphoreHandle_t g_sd_card_mutex; 

#define TISA_VM_DEBUG 1 // Установите в 0, чтобы отключить отладочные сообщения

// --- UCD-Based Unicode Helpers (Full Implementation) ---
namespace {
    // --- UTF-8 Utilities ---
    size_t get_utf8_char_len(unsigned char b) {
        if (b < 0x80) return 1;
        if ((b & 0xE0) == 0xC0) return 2;
        if ((b & 0xF0) == 0xE0) return 3;
        if ((b & 0xF8) == 0xF0) return 4;
        return 0; // Invalid start byte
    }

    uint32_t utf8_to_codepoint(const unsigned char* s, size_t len) {
        if (len == 0) return 0;
        switch(len) {
            case 1: return s[0];
            case 2: return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
            case 3: return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
            case 4: return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
            default: return 0;
        }
    }

    std::string codepoint_to_utf8(uint32_t cp) {
        std::string result;
        if (cp < 0x80) {
            result += static_cast<char>(cp);
        } else if (cp < 0x800) {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x110000) {
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return result;
    }

    #include "TISA_UCD_TABLES.h"
    
    uint32_t codepoint_to_lower(uint32_t cp) {
        int low = 0, high = (sizeof(LOWERCASE_EXCEPTIONS) / sizeof(UnicodeException)) - 1;
        while(low <= high) {
            int mid = low + (high - low) / 2;
            uint32_t from = pgm_read_dword(&LOWERCASE_EXCEPTIONS[mid].from);
            if (from < cp) low = mid + 1;
            else if (from > cp) high = mid - 1;
            else return pgm_read_dword(&LOWERCASE_EXCEPTIONS[mid].to);
        }
        low = 0, high = (sizeof(LOWERCASE_RANGES) / sizeof(UnicodeRange)) - 1;
        while(low <= high) {
            int mid = low + (high - low) / 2;
            uint32_t start = pgm_read_dword(&LOWERCASE_RANGES[mid].start);
            uint32_t end = pgm_read_dword(&LOWERCASE_RANGES[mid].end);
            if (cp < start) high = mid - 1;
            else if (cp > end) low = mid + 1;
            else return cp + (int32_t)pgm_read_dword(&LOWERCASE_RANGES[mid].delta);
        }
        return cp;
    }
    
    bool is_in_category_ranges(uint32_t cp, const CategoryRange* ranges, size_t count) {
        int low = 0, high = count - 1;
        while(low <= high) {
            int mid = low + (high - low) / 2;
            uint32_t start = pgm_read_dword(&ranges[mid].start);
            uint32_t end = pgm_read_dword(&ranges[mid].end);
            if (cp < start) high = mid - 1;
            else if (cp > end) low = mid + 1;
            else return true;
        }
        return false;
    }
    
    bool is_category(uint32_t cp, const std::string& cat) {
        if (cat == "Mn") return is_in_category_ranges(cp, CAT_MN_RANGES, sizeof(CAT_MN_RANGES)/sizeof(CategoryRange));
        if (cat == "Cc") return is_in_category_ranges(cp, CAT_CC_RANGES, sizeof(CAT_CC_RANGES)/sizeof(CategoryRange));
        if (cat == "Cf") return is_in_category_ranges(cp, CAT_CF_RANGES, sizeof(CAT_CF_RANGES)/sizeof(CategoryRange));
        if (cat == "P") return is_in_category_ranges(cp, CAT_P_RANGES, sizeof(CAT_P_RANGES)/sizeof(CategoryRange));
        if (cat == "Z") return is_in_category_ranges(cp, CAT_Z_RANGES, sizeof(CAT_Z_RANGES)/sizeof(CategoryRange));
        return false;
    }

    bool is_whitespace(uint32_t cp) {
        if (cp == 0x0009 || cp == 0x000A || cp == 0x000B || cp == 0x000C || cp == 0x000D) return true;
        return is_category(cp, "Z");
    }

    bool is_punctuation(uint32_t cp) {
        return is_category(cp, "P");
    }
    
    void decompose(uint32_t cp, std::string& result) {
        int low = 0, high = (sizeof(DECOMP_TABLE)/sizeof(Decomp)) - 1;
        while(low <= high) {
            int mid = low + (high - low) / 2;
            uint32_t from = pgm_read_dword(&DECOMP_TABLE[mid].from);
            if (from < cp) low = mid + 1;
            else if (from > cp) high = mid - 1;
            else {
                uint32_t to1 = pgm_read_dword(&DECOMP_TABLE[mid].to1);
                uint32_t to2 = pgm_read_dword(&DECOMP_TABLE[mid].to2);
                decompose(to1, result);
                if (to2 != 0) decompose(to2, result);
                return;
            }
        }
        result += codepoint_to_utf8(cp);
    }
    
    void trim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !isspace(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !isspace(ch); }).base(), s.end());
    }
}

// --- ResourceView & Subclasses (Implementations) ---

ResourceView::ResourceView(uint64_t offset, uint32_t size) : _offset(offset), _size(size) {
    if (_size < 4) {
        _entry_count = 0;
        return;
    }
    #if TISA_VM_DEBUG
    Serial.printf("[ResourceView] Constructor for resource at offset %llu, size %u\n", _offset, _size);
    #endif
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(_offset);
    sdcard.readData((uint8_t*)&_entry_count, 4);
    xSemaphoreGive(g_sd_card_mutex);
    #if TISA_VM_DEBUG
    Serial.printf("[ResourceView] Found %u entries.\n", _entry_count);
    #endif
}

std::string ResourceView::_internal_read_string_from_current_pos() {
    uint16_t len;
    if (sdcard.readData((uint8_t*)&len, 2) != 2) return "";
    if (len > 0) {
        std::string s(len, '\0');
        if (sdcard.readData((uint8_t*)s.data(), len) != len) return "";
        return s;
    }
    return "";
}

// --- BinaryVocabView (for encoding with block caching) ---
BinaryVocabView::BinaryVocabView(uint64_t offset, uint32_t size) : ResourceView(offset, size) {
    _offset_cache.reserve(OFFSETS_PER_CACHE);
    #if TISA_VM_DEBUG
    Serial.println("[BinaryVocabView] Initialized for block-based reading.");
    #endif
}

uint32_t BinaryVocabView::_get_offset_at(uint32_t index) {
    if (index >= _entry_count) return 0;
    uint32_t block_index = index / OFFSETS_PER_CACHE;
    uint32_t index_in_block = index % OFFSETS_PER_CACHE;
    if (block_index == _cached_block_index) return _offset_cache[index_in_block];

    #if TISA_VM_DEBUG
    Serial.printf("[BinaryVocabView] Cache miss for index %u. Loading block %u...\n", index, block_index);
    #endif

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    _cached_block_index = -1;
    uint64_t seek_pos = _offset + 4 + (uint64_t)block_index * CACHE_SIZE_BYTES;
    
    if (sdcard.seek(seek_pos)) {
        uint32_t entries_in_this_block = std::min(OFFSETS_PER_CACHE, _entry_count - (block_index * OFFSETS_PER_CACHE));
        _offset_cache.resize(entries_in_this_block);
        size_t bytes_to_read = entries_in_this_block * sizeof(uint32_t);
        if (sdcard.readData((uint8_t*)_offset_cache.data(), bytes_to_read) == bytes_to_read) {
            _cached_block_index = block_index;
        } else {
             Serial.printf("[TISA_VM][ERROR] vocab.b cache read failed for block %u\n", block_index);
        }
    }
    xSemaphoreGive(g_sd_card_mutex);

    return (_cached_block_index == block_index) ? _offset_cache[index_in_block] : 0;
}

BinaryVocabView::VocabEntry BinaryVocabView::_read_entry_at_index(uint32_t index) {
    VocabEntry entry = {"", -1, 0.0f};
    uint32_t entry_relative_offset = _get_offset_at(index);
    if (entry_relative_offset == 0 && index != 0) return entry;

    uint64_t data_start_offset = _offset + 4 + (uint64_t(_entry_count) * 4);
    uint64_t entry_absolute_offset = data_start_offset + entry_relative_offset;

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    if (sdcard.seek(entry_absolute_offset)) {
        entry.key = _internal_read_string_from_current_pos();
        if (!entry.key.empty()) {
            if (sdcard.readData((uint8_t*)&entry.id, sizeof(int32_t)) != sizeof(int32_t) ||
                sdcard.readData((uint8_t*)&entry.score, sizeof(float)) != sizeof(float)) {
                entry.key = ""; // Invalidate entry on read error
            }
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
    return entry;
}

std::string BinaryVocabView::read_token_by_data_offset(uint32_t data_offset) {
    uint64_t data_start_offset = get_base_offset() + 4 + (uint64_t(get_entry_count()) * 4);
    uint64_t entry_absolute_offset = data_start_offset + data_offset;
    std::string token;
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    if (sdcard.seek(entry_absolute_offset)) token = _internal_read_string_from_current_pos();
    xSemaphoreGive(g_sd_card_mutex);
    return token;
}

bool BinaryVocabView::find(const std::string& token, int32_t& id) { float d; return find(token, id, d); }

bool BinaryVocabView::find(const std::string& token, int32_t& id, float& score) {
    if (_entry_count == 0) return false;
    int32_t low = 0, high = _entry_count - 1;
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        VocabEntry mid_entry = _read_entry_at_index(mid);
        if (mid_entry.key.empty()) {
            #if TISA_VM_DEBUG
            Serial.printf("[BinaryVocabView] Failed to read entry at mid %d\n", mid);
            #endif
            return false;
        }
        
        int cmp = token.compare(mid_entry.key);
        if (cmp == 0) {
            id = mid_entry.id;
            score = mid_entry.score;
            return true;
        } else if (cmp < 0) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return false;
}


// --- BinaryVocabIndexView (for decoding with block caching) ---
BinaryVocabIndexView::BinaryVocabIndexView(uint64_t offset, uint32_t size) : ResourceView(offset, size) {
    _id_to_offset_cache.reserve(OFFSETS_PER_CACHE);
    #if TISA_VM_DEBUG
    Serial.println("[BinaryVocabIndexView] Initialized for block-based reading.");
    #endif
}

bool BinaryVocabIndexView::get_offset_for_id(int32_t id, uint32_t& offset) {
    if (id < 0 || id >= _entry_count) return false;
    uint32_t block_index = id / OFFSETS_PER_CACHE;
    uint32_t index_in_block = id % OFFSETS_PER_CACHE;
    if (block_index == _cached_block_index) {
        offset = _id_to_offset_cache[index_in_block];
        return true;
    }

    #if TISA_VM_DEBUG
    Serial.printf("[BinaryVocabIndexView] Cache miss for ID %d. Loading block %u...\n", id, block_index);
    #endif

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    _cached_block_index = -1;
    uint64_t seek_pos = _offset + 4 + (uint64_t)block_index * CACHE_SIZE_BYTES;
    if (sdcard.seek(seek_pos)) {
        uint32_t entries_in_this_block = std::min(OFFSETS_PER_CACHE, _entry_count - (block_index * OFFSETS_PER_CACHE));
        _id_to_offset_cache.resize(entries_in_this_block);
        size_t bytes_to_read = entries_in_this_block * sizeof(uint32_t);
        if (sdcard.readData((uint8_t*)_id_to_offset_cache.data(), bytes_to_read) == bytes_to_read) {
            _cached_block_index = block_index;
        } else {
             Serial.printf("[TISA_VM][ERROR] vidx.b cache read failed for block %u\n", block_index);
        }
    }
    xSemaphoreGive(g_sd_card_mutex);

    if (_cached_block_index == block_index) {
        offset = _id_to_offset_cache[index_in_block];
        return true;
    }
    return false;
}

// --- BinaryMergesView (No changes needed) ---
bool BinaryMergesView::find(const std::pair<std::string, std::string>& token_pair, int32_t& rank) {
    uint64_t current_pos = _offset + 4;
    bool found = false;
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    for (uint32_t i = 0; i < _entry_count; ++i) {
        if (!sdcard.seek(current_pos)) break;
        std::string token1 = _internal_read_string_from_current_pos();
        std::string token2 = _internal_read_string_from_current_pos();
        if (token1 == token_pair.first && token2 == token_pair.second) {
            rank = i;
            found = true;
            break;
        }
        current_pos = sdcard.getPosition();
    }
    xSemaphoreGive(g_sd_card_mutex);
    return found;
}

// --- TISAVM (Implementation) ---
TISAVM::TISAVM(VM_Resources& res) : m_res(res) {}

std::vector<int32_t> TISAVM::run(const std::vector<uint8_t>& manifest_data, const std::string& text) {
    Serial.println("[TISA_VM] Starting run...");
    TISA_State state;
    state.text = text;
    
    if (manifest_data.size() < 5 || memcmp(manifest_data.data(), "TISA", 4) != 0) {
        Serial.println("[TISA_VM][ERROR] Invalid manifest header.");
        return {};
    }
    
    size_t offset = 5;
    while (offset < manifest_data.size()) {
        uint8_t opcode = manifest_data[offset++];
        uint32_t payload_len;
        memcpy(&payload_len, &manifest_data[offset], sizeof(uint32_t));
        offset += 4;
        
        #if TISA_VM_DEBUG
        Serial.printf("[TISA_VM] Dispatching opcode 0x%02X\n", opcode);
        #endif
        _dispatch_opcode(opcode, &manifest_data[offset], payload_len, state);
        
        offset += payload_len;
    }

    if (state.fragments.empty() && !state.text.empty()) {
        state.fragments.push_back({state.text, false});
    }

    Serial.println("[TISA_VM] Run finished.");
    return state.ids;
}

std::string TISAVM::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) {
    if (!m_res.vocab || !m_res.vocab_idx_for_decode) {
        Serial.println("[TISA_VM][ERROR] Decode called but vocab or vocab_idx is missing.");
        return "";
    }

    std::vector<std::string> tokens;
    for (int32_t id : ids) {
        uint32_t data_offset;
        if (!m_res.vocab_idx_for_decode->get_offset_for_id(id, data_offset)) continue;
        std::string token = m_res.vocab->read_token_by_data_offset(data_offset);
        if (token.empty()) continue;
        if (skip_special_tokens && ((token.rfind("<", 0) == 0 && token.find(">") != std::string::npos) || (token.rfind("[", 0) == 0 && token.find("]") != std::string::npos))) continue;
        tokens.push_back(token);
    }
    
    ModelType model_type = _detect_model_type();
    if (model_type == ModelType::UNKNOWN) {
        for(const auto& t : tokens) {
            if (t.rfind("##", 0) == 0) { model_type = ModelType::WORDPIECE; break; }
            if (t.find("\xE2\x96\x81") != std::string::npos) { model_type = ModelType::UNIGRAM; break; }
        }
    }
    
    std::string result;
    switch (model_type) {
        case ModelType::BPE: {
             std::map<unsigned char, unsigned char> byte_decoder;
            for(const auto& pair : m_res.byte_map) {
                if (!pair.second.empty()) byte_decoder[static_cast<unsigned char>(pair.second[0])] = pair.first;
            }
            std::vector<unsigned char> full_bytes;
            for (const auto& token : tokens) {
                 for (unsigned char c : token) {
                    auto it = byte_decoder.find(c);
                    if (it != byte_decoder.end()) full_bytes.push_back(it->second);
                 }
            }
            result.assign(full_bytes.begin(), full_bytes.end());
            return result;
        }
        case ModelType::WORDPIECE: {
            if (tokens.empty()) return "";
            result = tokens[0];
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i].rfind("##", 0) == 0) result += tokens[i].substr(2);
                else result += " " + tokens[i];
            }
            return result;
        }
        case ModelType::UNIGRAM: {
            for (const auto& token : tokens) result += token;
            const std::string metaspace = "\xE2\x96\x81";
            size_t pos = 0;
            while((pos = result.find(metaspace, pos)) != std::string::npos) {
                result.replace(pos, metaspace.length(), " ");
            }
            if (result.rfind(" ", 0) == 0) result = result.substr(1);
            trim(result);
            return result;
        }
        case ModelType::UNKNOWN:
        default: {
            if (tokens.empty()) return "";
            result = tokens[0];
            for (size_t i = 1; i < tokens.size(); ++i) result += " " + tokens[i];
            return result;
        }
    }
}

TISAVM::ModelType TISAVM::_detect_model_type() {
    if (m_res.merges && m_res.merges->get_entry_count() > 0) return ModelType::BPE;
    if (!m_res.unigram_scores.empty()) return ModelType::UNIGRAM;
    return ModelType::UNKNOWN;
}

// --- Primitives (Full code) ---

void TISAVM::_dispatch_opcode(uint8_t opcode, const uint8_t* payload, size_t payload_len, TISA_State& state) {
    size_t offset = 0;
    switch (opcode) {
        case 0x01: _primitive_lowercase(state); break;
        case 0x02: {
            uint8_t form_len = payload[offset++];
            std::string form((char*)&payload[offset], form_len);
            _primitive_unicode_norm(state, form);
            break;
        }
        case 0x03: {
            uint16_t pattern_len, val_len;
            memcpy(&pattern_len, &payload[offset], 2); offset += 2;
            std::string pattern((char*)&payload[offset], pattern_len); offset += pattern_len;
            memcpy(&val_len, &payload[offset], 2); offset += 2;
            std::string val((char*)&payload[offset], val_len);
            _primitive_replace(state, pattern, val);
            break;
        }
        case 0x04: {
            uint8_t num_cats = payload[offset++];
            std::vector<std::string> cats;
            for(int i=0; i<num_cats; ++i) {
                uint8_t cat_len = payload[offset++];
                cats.push_back(std::string((char*)&payload[offset], cat_len));
                offset += cat_len;
            }
            _primitive_filter_category(state, cats);
            break;
        }
        case 0x07: {
            uint16_t val_len;
            memcpy(&val_len, &payload[offset], 2); offset += 2;
            std::string val((char*)&payload[offset], val_len);
            _primitive_prepend(state, val);
            break;
        }
        case 0x10: _primitive_partition_rules(state, std::vector<uint8_t>(payload, payload + payload_len)); break;
        case 0x15: _primitive_byte_encode(state); break;
        case 0x20: _primitive_bpe_encode(state); break;
        case 0x21: {
            uint8_t marker_len = payload[offset++];
            std::string marker((char*)&payload[offset], marker_len);
            _primitive_wordpiece_encode(state, marker);
            break;
        }
        case 0x22: _primitive_unigram_encode(state); break;
        case 0x30: _primitive_compose(state, std::vector<uint8_t>(payload, payload + payload_len)); break;
        default: break;
    }
}

void TISAVM::_primitive_lowercase(TISA_State& state) {
    std::string result;
    result.reserve(state.text.length());
    const unsigned char* c_str = (const unsigned char*)state.text.c_str();
    size_t offset = 0;
    while(offset < state.text.length()) {
        size_t len = get_utf8_char_len(c_str[offset]);
        if (len == 0 || offset + len > state.text.length()) { offset++; continue; }
        uint32_t cp = utf8_to_codepoint(&c_str[offset], len);
        result += codepoint_to_utf8(codepoint_to_lower(cp));
        offset += len;
    }
    state.text = result;
}

void TISAVM::_primitive_unicode_norm(TISA_State& state, const std::string& form) {
    if (form != "NFD") return;
    std::string normalized_text;
    normalized_text.reserve(state.text.length());
    const unsigned char* c_str = (const unsigned char*)state.text.c_str();
    size_t offset = 0;
    while (offset < state.text.length()) {
        size_t len = get_utf8_char_len(c_str[offset]);
        if (len == 0 || offset + len > state.text.length()) { offset++; continue; }
        uint32_t cp = utf8_to_codepoint(&c_str[offset], len);
        decompose(cp, normalized_text);
        offset += len;
    }
    state.text = normalized_text;
}

void TISAVM::_primitive_replace(TISA_State& state, const std::string& pattern, const std::string& val) {
    size_t start_pos = 0;
    while((start_pos = state.text.find(pattern, start_pos)) != std::string::npos) {
        state.text.replace(start_pos, pattern.length(), val);
        start_pos += val.length();
    }
}

void TISAVM::_primitive_filter_category(TISA_State& state, const std::vector<std::string>& cats) {
    std::string result;
    result.reserve(state.text.length());
    const unsigned char* c_str = (const unsigned char*)state.text.c_str();
    size_t offset = 0;
    while(offset < state.text.length()) {
        size_t len = get_utf8_char_len(c_str[offset]);
        if (len == 0 || offset + len > state.text.length()) { offset++; continue; }
        uint32_t cp = utf8_to_codepoint(&c_str[offset], len);
        bool should_filter = false;
        for (const auto& cat : cats) {
            if (is_category(cp, cat)) { should_filter = true; break; }
        }
        if (!should_filter) result.append(state.text.substr(offset, len));
        offset += len;
    }
    state.text = result;
}

void TISAVM::_primitive_prepend(TISA_State& state, const std::string& val) {
    if (state.text.rfind(val, 0) != 0) state.text.insert(0, val);
}

void TISAVM::_primitive_partition_rules(TISA_State& state, const std::vector<uint8_t>& rules_payload) {
    state.fragments.clear();
    const uint8_t* p = rules_payload.data();

    enum class PatternType { LITERAL, WHITESPACE, PUNCTUATION, COMPLEX_GPT2 };
    struct Rule {
        std::string pattern_str;
        PatternType type = PatternType::LITERAL;
        bool is_protected = false;
        std::string trim_preceding_space;
        uint8_t behavior_code = 0;
    };
    
    std::vector<Rule> rules;
    uint16_t num_rules; memcpy(&num_rules, p, 2); p += 2;

    for (int i = 0; i < num_rules; ++i) {
        Rule r;
        uint8_t flags = *p++;
        uint16_t pattern_len; memcpy(&pattern_len, p, 2); p += 2;
        r.pattern_str = std::string((char*)p, pattern_len); p += pattern_len;
        r.is_protected = (flags & 1);
        if (flags & 2) r.behavior_code = *p++;
        if (flags & 4) { uint8_t trim_len = *p++; r.trim_preceding_space = std::string((char*)p, trim_len); p += trim_len; }
        
        if (r.pattern_str == "\\s+") r.type = PatternType::WHITESPACE;
        else if (r.pattern_str == "\\p{P}") r.type = PatternType::PUNCTUATION;
        else if (r.pattern_str.find("\\p{L}") != std::string::npos || r.pattern_str.find("\\p{N}") != std::string::npos) r.type = PatternType::COMPLEX_GPT2;
        else {
            const std::string prefix = "(?: ?)";
            if (r.pattern_str.rfind(prefix, 0) == 0) r.pattern_str = r.pattern_str.substr(prefix.length());
        }
        rules.push_back(r);
    }
    
    if (rules.empty()) { if (!state.text.empty()) state.fragments.push_back({state.text, false}); return; }
    
    std::string current_chunk;
    size_t offset = 0;
    const unsigned char* text_ptr = (const unsigned char*)state.text.c_str();

    auto flush_chunk = [&](const std::string& preceding_space_char = "") {
        if (current_chunk.empty()) return;
        std::string chunk_to_add = current_chunk;
        if (!preceding_space_char.empty() && chunk_to_add.length() >= preceding_space_char.length()) {
            if (chunk_to_add.substr(chunk_to_add.length() - preceding_space_char.length()) == preceding_space_char) {
                chunk_to_add.erase(chunk_to_add.length() - preceding_space_char.length());
            }
        }
        if (!chunk_to_add.empty()) state.fragments.push_back({chunk_to_add, false});
        current_chunk.clear();
    };

    while(offset < state.text.length()) {
        bool rule_matched = false;
        // 1. Проверяем правила-литералы (самый высокий приоритет)
        for (const auto& rule : rules) {
            if (rule.type == PatternType::LITERAL && state.text.compare(offset, rule.pattern_str.length(), rule.pattern_str) == 0) {
                flush_chunk(rule.trim_preceding_space);
                if (rule.behavior_code != 1) state.fragments.push_back({rule.pattern_str, rule.is_protected});
                offset += rule.pattern_str.length();
                rule_matched = true;
                break;
            }
        }
        if (rule_matched) continue;
        
        size_t char_len = get_utf8_char_len(text_ptr[offset]);
        if (char_len == 0) { offset++; continue; }
        uint32_t cp = utf8_to_codepoint(&text_ptr[offset], char_len);
        
        // 2. Проверяем правила по классам символов
        for (const auto& rule : rules) {
            bool class_match = (rule.type == PatternType::WHITESPACE && is_whitespace(cp)) || (rule.type == PatternType::PUNCTUATION && is_punctuation(cp));
            if (class_match) {
                if (rule.behavior_code == 2) { // ISOLATE
                    flush_chunk();
                    state.fragments.push_back({state.text.substr(offset, char_len), false});
                } else if (rule.behavior_code == 1) { // REMOVE
                    flush_chunk();
                }
                // Для KEEP (behavior_code=0) ничего не делаем, символ попадет в current_chunk
                offset += char_len;
                rule_matched = true;
                break;
            }
        }
        if (rule_matched) continue;

        // 3. Если ничего не совпало, добавляем символ в текущий чанк
        current_chunk += state.text.substr(offset, char_len);
        offset += char_len;
    }
    flush_chunk();

    bool has_complex_rule = false;
    for(const auto& r : rules) if(r.type == PatternType::COMPLEX_GPT2) has_complex_rule = true;
    if (has_complex_rule) {
        std::vector<Fragment> final_fragments;
        for(const auto& frag : state.fragments) {
             if (frag.is_protected) { final_fragments.push_back(frag); continue; }
             std::string sub_chunk;
             for (char c : frag.text) {
                 if (c == ' ') {
                     if (!sub_chunk.empty()) final_fragments.push_back({sub_chunk, false});
                     sub_chunk.clear();
                     final_fragments.push_back({" ", false});
                 } else { sub_chunk += c; }
             }
             if (!sub_chunk.empty()) final_fragments.push_back({sub_chunk, false});
        }
        state.fragments = final_fragments;
    }
}

void TISAVM::_primitive_byte_encode(TISA_State& state) {
    for (auto& frag : state.fragments) {
        if (frag.is_protected) continue;
        std::string encoded_frag;
        encoded_frag.reserve(frag.text.length());
        for (unsigned char c : frag.text) {
            auto it = m_res.byte_map.find(c);
            if (it != m_res.byte_map.end()) encoded_frag += it->second;
        }
        frag.text = encoded_frag;
    }
}

void TISAVM::_primitive_unigram_encode(TISA_State& state) {
    if (m_res.unigram_scores.empty()) return;
    int32_t unk_id = 2; m_res.vocab->find("<unk>", unk_id);
    state.ids.clear();
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id; state.ids.push_back(m_res.vocab->find(frag.text, id) ? id : unk_id);
            continue;
        }
        const std::string& text = frag.text;
        int n = text.length();
        if (n == 0) continue;
        std::vector<float> dp(n + 1, -std::numeric_limits<float>::infinity());
        std::vector<int> best_path(n + 1, 0);
        dp[0] = 0.0f;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j <= std::min(n, i + 50); ++j) {
                std::string sub = text.substr(i, j - i);
                auto score_it = m_res.unigram_scores.find(sub);
                if (score_it != m_res.unigram_scores.end()) {
                    float score = dp[i] + score_it->second;
                    if (score > dp[j]) { dp[j] = score; best_path[j] = i; }
                }
            }
        }
        if (dp[n] == -std::numeric_limits<float>::infinity()) {
            const unsigned char* s = (const unsigned char*)text.c_str();
            for (size_t offset = 0; offset < text.length(); ) {
                 size_t len = get_utf8_char_len(s[offset]); if(len == 0) { offset++; continue; }
                 std::string single_char = text.substr(offset, len);
                 int32_t id; state.ids.push_back(m_res.vocab->find(single_char, id) ? id : unk_id);
                 offset += len;
            }
        } else {
            std::vector<int32_t> path; int curr = n;
            while (curr > 0) {
                int prev = best_path[curr];
                std::string token = text.substr(prev, curr - prev);
                int32_t id; path.push_back(m_res.vocab->find(token, id) ? id : unk_id);
                curr = prev;
            }
            std::reverse(path.begin(), path.end());
            state.ids.insert(state.ids.end(), path.begin(), path.end());
        }
    }
}

void TISAVM::_primitive_bpe_encode(TISA_State& state) {
    if (!m_res.vocab || !m_res.merges) return;
    int32_t unk_id = 0; m_res.vocab->find("<unk>", unk_id);
    state.ids.clear();
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id = unk_id; m_res.vocab->find(frag.text, id);
            state.ids.push_back(id);
            continue;
        }
        std::vector<std::string> word;
        const unsigned char* s = (const unsigned char*)frag.text.c_str();
        for (size_t offset = 0; offset < frag.text.length(); ) {
            size_t len = get_utf8_char_len(s[offset]); if (len == 0) { offset++; continue; }
            word.push_back(frag.text.substr(offset, len)); offset += len;
        }
        if (word.empty()) continue;
        while (word.size() > 1) {
            std::pair<std::string, std::string> best_pair;
            int32_t min_rank = std::numeric_limits<int32_t>::max();
            bool found_pair = false;
            for (size_t i = 0; i < word.size() - 1; ++i) {
                std::pair<std::string, std::string> pair = {word[i], word[i+1]};
                int32_t rank;
                if (m_res.merges->find(pair, rank) && rank < min_rank) {
                    min_rank = rank; best_pair = pair; found_pair = true;
                }
            }
            if (!found_pair) break;
            std::vector<std::string> new_word;
            new_word.reserve(word.size());
            for (size_t i = 0; i < word.size(); ) {
                if (i < word.size() - 1 && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                    new_word.push_back(best_pair.first + best_pair.second); i += 2;
                } else { new_word.push_back(word[i++]); }
            }
            word = new_word;
        }
        for (const auto& token : word) {
            int32_t id; state.ids.push_back(m_res.vocab->find(token, id) ? id : unk_id);
        }
    }
}

void TISAVM::_primitive_wordpiece_encode(TISA_State& state, const std::string& marker) {
    if (!m_res.vocab) return;
    int32_t unk_id = 100; m_res.vocab->find("[UNK]", unk_id);
    state.ids.clear();
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id; state.ids.push_back(m_res.vocab->find(frag.text, id) ? id : unk_id);
            continue;
        }
        std::string text = frag.text; if (text.empty()) continue;
        size_t start_offset = 0;
        while (start_offset < text.length()) {
            size_t end_offset = text.length();
            bool found_match = false;
            while (start_offset < end_offset) {
                std::string sub_token = text.substr(start_offset, end_offset - start_offset);
                std::string token_to_lookup = (start_offset == 0) ? sub_token : (marker + sub_token);
                int32_t id;
                if (m_res.vocab->find(token_to_lookup, id)) {
                    state.ids.push_back(id); start_offset = end_offset; found_match = true; break;
                }
                end_offset--;
            }
            if (!found_match) {
                state.ids.push_back(unk_id);
                size_t char_len = get_utf8_char_len(text[start_offset]);
                start_offset += (char_len > 0 ? char_len : 1);
            }
        }
    }
}

void TISAVM::_primitive_compose(TISA_State& state, const std::vector<uint8_t>& payload) {
    std::vector<int32_t> out_ids; size_t offset = 0;
    const uint8_t* p = payload.data();
    uint8_t num_items = p[offset++];
    for (int i = 0; i < num_items; ++i) {
        uint8_t is_fixed = p[offset++];
        if (is_fixed) {
            uint8_t is_int = p[offset++];
            if (is_int) { int32_t id; memcpy(&id, &p[offset], 4); offset += 4; out_ids.push_back(id); }
            else {
                uint8_t len = p[offset++]; std::string token((char*)&p[offset], len); offset += len;
                int32_t id;
                if (m_res.vocab && m_res.vocab->find(token, id)) out_ids.push_back(id);
                else out_ids.push_back(2);
            }
        } else { out_ids.insert(out_ids.end(), state.ids.begin(), state.ids.end()); }
    }
    state.ids = out_ids;

}
