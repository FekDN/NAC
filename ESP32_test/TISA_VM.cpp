// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com)
#include "TISA_VM.h"
#include <Arduino.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "CYD28_SD.h"

extern CYD28_SD sdcard;
extern SemaphoreHandle_t g_sd_card_mutex;

// --- Utilities for working with UTF-8 and Unicode categories ---
static size_t get_utf8_char_len(unsigned char b) {
    if (b < 0x80) return 1; if ((b & 0xE0) == 0xC0) return 2; if ((b & 0xF0) == 0xE0) return 3; if ((b & 0xF8) == 0xF0) return 4; return 0;
}
static uint32_t utf8_to_codepoint(const unsigned char* s, size_t len) {
    if (len == 0) return 0;
    switch(len) {
        case 1: return s[0];
        case 2: return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
        case 3: return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        case 4: return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        default: return 0;
    }
}
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string result;
    if (cp < 0x80) { result += static_cast<char>(cp); }
    else if (cp < 0x800) { result += static_cast<char>(0xC0 | (cp >> 6)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { result += static_cast<char>(0xE0 | (cp >> 12)); result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x110000) { result += static_cast<char>(0xF0 | (cp >> 18)); result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
    return result;
}
#include "TISA_UCD_TABLES.h"
static bool is_in_category_ranges(uint32_t cp, const CategoryRange* ranges, size_t count) {
    int low = 0, high = count - 1;
    while(low <= high) {
        int mid = low + (high - low) / 2;
        uint32_t start = pgm_read_dword(&ranges[mid].start);
        uint32_t end = pgm_read_dword(&ranges[mid].end);
        if (cp < start) high = mid - 1; else if (cp > end) low = mid + 1; else return true;
    }
    return false;
}
static bool is_category(uint32_t cp, const std::string& cat) {
    if (cat == "P") return is_in_category_ranges(cp, CAT_P_RANGES, sizeof(CAT_P_RANGES)/sizeof(CategoryRange));
    if (cat == "Z") return is_in_category_ranges(cp, CAT_Z_RANGES, sizeof(CAT_Z_RANGES)/sizeof(CategoryRange));
        // Add support for Mn, Cc, Cf for BERT normalizers
    if (cat == "Mn") return is_in_category_ranges(cp, CAT_MN_RANGES, sizeof(CAT_MN_RANGES)/sizeof(CategoryRange));
    if (cat == "Cc") return is_in_category_ranges(cp, CAT_CC_RANGES, sizeof(CAT_CC_RANGES)/sizeof(CategoryRange));
    if (cat == "Cf") return is_in_category_ranges(cp, CAT_CF_RANGES, sizeof(CAT_CF_RANGES)/sizeof(CategoryRange));
    return false;
}
static bool is_whitespace(uint32_t cp) { return (cp >= 0x0009 && cp <= 0x000D) || cp == 0x0020 || cp == 0x00A0 || is_category(cp, "Z"); }
enum CharType { LETTER, NUMBER, WHITESPACE, OTHER };
static CharType get_char_type(uint32_t cp) {
    if (is_whitespace(cp)) return WHITESPACE;
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= 0x0400 && cp <= 0x04FF) || (cp >= 0x00C0 && cp <= 0x02AF)) return LETTER;
    if (cp >= '0' && cp <= '9') return NUMBER;
    return OTHER;
}
static bool is_punctuation(uint32_t cp) { return is_category(cp, "P"); }
static bool is_cjk(uint32_t cp) {
return (cp >= 0x4E00 && cp <= 0x9FFF) ||
       (cp >= 0x3040 && cp <= 0x309F) ||
       (cp >= 0x30A0 && cp <= 0x30FF) ||
       (cp >= 0xAC00 && cp <= 0xD7AF);
}

// This function is used in TISAVM::_primitive_partition_rules, so it is NOT static
// IMPORTANT: The order of the checks must exactly match the order of the alternatives in the Python regex:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
size_t find_gpt_next_token(const std::string& text, size_t start_pos, size_t& out_end) {
    if (start_pos >= text.length()) {
        return std::string::npos;
    }

    // 1. Contractions: 's|'t|'re|'ve|'m|'ll|'d
    if (text[start_pos] == '\'') {
        const std::vector<std::string> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (const auto& c : contractions) {
            if (text.compare(start_pos, c.length(), c) == 0) {
                out_end = start_pos + c.length();
                return start_pos;
            }
        }
    }

    // 2-4. " ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+" - optional space + group
    // These are THREE alternatives, but the logic is the same: optional space + a sequence of characters of the same type
    size_t cur = start_pos;
    bool has_space = (text[cur] == ' ');
    if (has_space) ++cur;

    // Check that there is a non-whitespace character after the optional space
    if (cur < text.length()) {
        size_t cl = get_utf8_char_len(text[cur]);
        if (cl > 0) {
            uint32_t cp = utf8_to_codepoint((const unsigned char*)&text[cur], cl);
            CharType type = get_char_type(cp);
            
            // If this is NOT whitespace, form a group
            if (type != WHITESPACE) {
                size_t group_end = cur;
                
                if (type == LETTER) {
                    // " ?\p{L}+" - form letters
                    while (group_end < text.length()) {
                        cl = get_utf8_char_len(text[group_end]);
                        if (cl == 0) break;
                        if (get_char_type(utf8_to_codepoint((const unsigned char*)&text[group_end], cl)) != LETTER) break;
                        group_end += cl;
                    }
                } else if (type == NUMBER) {
                    // " ?\p{N}+" - form numbers
                    while (group_end < text.length()) {
                        cl = get_utf8_char_len(text[group_end]);
                        if (cl == 0) break;
                        if (get_char_type(utf8_to_codepoint((const unsigned char*)&text[group_end], cl)) != NUMBER) break;
                        group_end += cl;
                    }
                } else { // OTHER
                    // " ?[^\s\p{L}\p{N}]+" - form non-letters, non-numbers, and non-whitespace.
                    while (group_end < text.length()) {
                        cl = get_utf8_char_len(text[group_end]);
                        if (cl == 0) break;
                        CharType t = get_char_type(utf8_to_codepoint((const unsigned char*)&text[group_end], cl));
                        if (t == WHITESPACE || t == LETTER || t == NUMBER) break;
                        group_end += cl;
                    }
                }
                
                out_end = group_end;
                return start_pos; // Return the beginning (including the optional space if there was one)
            }
        }
    }

    // 5-6. "\s+(?!\S)|\s+" - whitespace
    // If got here, it means it starts with whitespace (or there was a space but after it there is also a space/end)
    size_t cl = get_utf8_char_len(text[start_pos]);
    if (cl > 0) {
        uint32_t cp = utf8_to_codepoint((const unsigned char*)&text[start_pos], cl);
        if (is_whitespace(cp)) {
            size_t end = start_pos;
            while (end < text.length()) {
                cl = get_utf8_char_len(text[end]);
                if (cl == 0) break;
                if (!is_whitespace(utf8_to_codepoint((const unsigned char*)&text[end], cl))) break;
                end += cl;
            }
            out_end = end;
            return start_pos;
        }
    }

    // Shouldn't end up here, but just in case
    out_end = start_pos + 1;
    return start_pos;
}

// --- Classes ResourceView ---
ResourceView::ResourceView(uint64_t off, uint32_t sz) : _offset(off), _size(sz) {
    _entry_count = 0;
    // Работаем с уже открытым файлом, как в старой версии
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    if(sdcard.seek(_offset)) {
        if (sdcard.readData((uint8_t*)&_entry_count, 4) != 4) {
            _entry_count = 0;
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
}

std::string ResourceView::_internal_read_string_from_current_pos() {
    uint16_t len; if (sdcard.readData((uint8_t*)&len, 2) != 2) return "";
    std::string s(len, '\0'); if (len > 0 && sdcard.readData((uint8_t*)s.data(), len) != len) return "";
    return s;
}

BinaryVocabView::BinaryVocabView(uint64_t off, uint32_t sz) : ResourceView(off, sz) { _offset_cache.reserve(OFFSETS_PER_CACHE); }

uint32_t BinaryVocabView::_get_offset_at(uint32_t idx) {
    if (idx >= _entry_count) return 0xFFFFFFFF;
    if (idx / OFFSETS_PER_CACHE == _cached_block_index) return _offset_cache[idx % OFFSETS_PER_CACHE];

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    _cached_block_index = -1;
    // БЕЗ openFile/closeFile. ИСПОЛЬЗУЕМ СМЕЩЕНИЕ +4 для пропуска заголовка с кол-вом записей
    if (sdcard.seek(_offset + 4 + (uint64_t)(idx/OFFSETS_PER_CACHE) * CACHE_SIZE_BYTES)) {
        uint32_t entries = std::min(OFFSETS_PER_CACHE, _entry_count-(idx/OFFSETS_PER_CACHE)*OFFSETS_PER_CACHE);
        _offset_cache.resize(entries);
        if (sdcard.readData((uint8_t*)_offset_cache.data(), entries * sizeof(uint32_t)) == entries * sizeof(uint32_t)) {
            _cached_block_index = idx / OFFSETS_PER_CACHE;
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
    return (_cached_block_index == idx/OFFSETS_PER_CACHE) ? _offset_cache[idx % OFFSETS_PER_CACHE] : 0xFFFFFFFF;
}

BinaryVocabView::VocabEntry BinaryVocabView::_read_entry_at_index(uint32_t idx) {
    VocabEntry e = {"", -1, 0.0f}; uint32_t rel = _get_offset_at(idx); if (rel == 0xFFFFFFFF) return e;
    uint64_t data_section_start = _offset + (uint64_t(_entry_count) * 4) + 4;

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    // БЕЗ openFile/closeFile
    if (sdcard.seek(data_section_start + rel)) {
        e.key = _internal_read_string_from_current_pos();
        if (!e.key.empty()) {
            sdcard.readData((uint8_t*)&e.id, 4);
            sdcard.readData((uint8_t*)&e.score, 4);
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
    return e;
}

std::string BinaryVocabView::read_token_by_data_offset(uint32_t off) {
    if (off == 0xFFFFFFFF) return "";
    std::string tok;
    uint64_t data_section_start = _offset + (uint64_t(_entry_count) * 4) + 4;
    
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    // БЕЗ openFile/closeFile
    if(sdcard.seek(data_section_start + off)) {
        tok = _internal_read_string_from_current_pos();
    }
    xSemaphoreGive(g_sd_card_mutex);
    return tok;
}

bool BinaryVocabView::find(const std::string& tok, int32_t& id, float& score) {
    if (_entry_count==0) return false; int32_t l=0, h=_entry_count-1;
    while(l<=h) {
        int32_t m=l+(h-l)/2; VocabEntry e=_read_entry_at_index(m); if(e.key.empty())return false;
        int cmp=tok.compare(e.key);
        if(cmp==0){id=e.id;score=e.score;return true;} else if(cmp<0)h=m-1; else l=m+1;
    }
    return false;
}
bool BinaryVocabView::find(const std::string& tok, int32_t& id){float d; return find(tok,id,d);}

BinaryVocabIndexView::BinaryVocabIndexView(uint64_t off,uint32_t sz) : ResourceView(off,sz){_id_to_offset_cache.reserve(OFFSETS_PER_CACHE);}

bool BinaryVocabIndexView::get_offset_for_id(int32_t id,uint32_t& off){
    if(id<0||id>=_entry_count)return false;
    uint32_t blk=id/OFFSETS_PER_CACHE,idx=id%OFFSETS_PER_CACHE;
    if(blk==_cached_block_index){off=_id_to_offset_cache[idx];return off!=0xFFFFFFFF;}
    
    xSemaphoreTake(g_sd_card_mutex,portMAX_DELAY);
    _cached_block_index=-1;
    // БЕЗ openFile/closeFile. ИСПОЛЬЗУЕМ СМЕЩЕНИЕ +4
    if(sdcard.seek(_offset + 4 + (uint64_t)blk * CACHE_SIZE_BYTES)){
        uint32_t entries=std::min(OFFSETS_PER_CACHE,_entry_count-(blk*OFFSETS_PER_CACHE));
        _id_to_offset_cache.resize(entries);
        if(sdcard.readData((uint8_t*)_id_to_offset_cache.data(), entries * sizeof(uint32_t)) == entries * sizeof(uint32_t)) {
            _cached_block_index=blk;
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
    
    if(_cached_block_index==blk){off=_id_to_offset_cache[idx];return off!=0xFFFFFFFF;}
    return false;
}

BinaryMergesView::BinaryMergesView(uint64_t off,uint32_t sz):ResourceView(off,sz){_offset_cache.reserve(OFFSETS_PER_CACHE);}

uint32_t BinaryMergesView::_get_offset_at(uint32_t idx){
    if(idx>=_entry_count)return 0xFFFFFFFF;
    if(idx/OFFSETS_PER_CACHE==_cached_block_index)return _offset_cache[idx%OFFSETS_PER_CACHE];
    
    xSemaphoreTake(g_sd_card_mutex,portMAX_DELAY);
    _cached_block_index=-1;
    // БЕЗ openFile/closeFile. ИСПОЛЬЗУЕМ СМЕЩЕНИЕ +4
    if(sdcard.seek(_offset + 4 + (uint64_t)(idx/OFFSETS_PER_CACHE) * CACHE_SIZE_BYTES)){
        uint32_t entries=std::min(OFFSETS_PER_CACHE,_entry_count-(idx/OFFSETS_PER_CACHE)*OFFSETS_PER_CACHE);
        _offset_cache.resize(entries);
        if(sdcard.readData((uint8_t*)_offset_cache.data(), entries * sizeof(uint32_t)) == entries * sizeof(uint32_t)) {
            _cached_block_index=idx/OFFSETS_PER_CACHE;
        }
    }
    xSemaphoreGive(g_sd_card_mutex);
    return(_cached_block_index==idx/OFFSETS_PER_CACHE)?_offset_cache[idx%OFFSETS_PER_CACHE]:0xFFFFFFFF;
}

bool BinaryMergesView::find(const std::pair<std::string,std::string>& p,int32_t& r){
    if(_entry_count==0)return false;
    int32_t low=0,high=_entry_count-1;
    while(low<=high){
        int32_t mid=low+(high-low)/2;
        uint32_t rel=_get_offset_at(mid); if(rel==0xFFFFFFFF)return false;
        std::string t1,t2; int32_t rank; bool ok=false;
        uint64_t data_section_start = _offset + (uint64_t(_entry_count) * 4) + 4;

        xSemaphoreTake(g_sd_card_mutex,portMAX_DELAY);
        // БЕЗ openFile/closeFile
        if(sdcard.seek(data_section_start + rel)){
            t1=_internal_read_string_from_current_pos();
            t2=_internal_read_string_from_current_pos();
            if(sdcard.readData((uint8_t*)&rank,4)==4)ok=true;
        }
        xSemaphoreGive(g_sd_card_mutex);

        if(!ok)return false;
        int c1=p.first.compare(t1);
        if(c1==0){
            int c2=p.second.compare(t2);
            if(c2==0){r=rank;return true;}else if(c2<0)high=mid-1;else low=mid+1;
        }else if(c1<0)high=mid-1;else low=mid+1;
    }
    return false;
}

// --- TISAVM ---

TISAVM::TISAVM(VM_Resources& res):m_res(res){}

std::vector<int32_t> TISAVM::run(const std::vector<uint8_t>& mf, const std::string& txt) {
    TISA_State s;
    s.text = txt;
    
    // Signature verification
    if (mf.size() < 5 || memcmp(mf.data(), "TISA", 4) != 0) {
        return {};
    }
    
    // Execute all operations from the manifest
    size_t offset = 5;
    while (offset < mf.size()) {
        uint8_t op = mf[offset++];
        uint32_t len;
        memcpy(&len, &mf[offset], 4);
        offset += 4;
        
        _dispatch_opcode(op, &mf[offset], len, s);
        offset += len;
    }
    
    // Each primitive function is responsible for creating fragments when needed.
    return s.ids;
}

void TISAVM::_dispatch_opcode(uint8_t op, const uint8_t* p, size_t len, TISA_State& s){
    switch (op) {
case 0x01: { // LOWERCASE — full-fledged Unicode (Cyrillic + Latin Extended)
    std::string lower_text;
    for (size_t i = 0; i < s.text.length(); ) {
        size_t char_len = get_utf8_char_len(s.text[i]);
        if (char_len == 0) { i++; continue; }

        uint32_t cp = utf8_to_codepoint((const unsigned char*)&s.text[i], char_len);
        uint32_t lower_cp = cp;

        if (cp >= 'A' && cp <= 'Z') lower_cp = cp + 32;                    // ASCII
        else if (cp >= 0x0410 && cp <= 0x042F) lower_cp = cp + 32;         // А-Я → а-я
        else if (cp >= 0x00C0 && cp <= 0x00D6) lower_cp = cp + 0x20;       // À-Ö
        else if (cp >= 0x00D8 && cp <= 0x00DE) lower_cp = cp + 0x20;       // Ø-Þ
        else if (cp >= 0x0391 && cp <= 0x03A1) lower_cp = cp + 0x20;       // Greek (just in case)

        lower_text += codepoint_to_utf8(lower_cp);
        i += char_len;
    }
    s.text = lower_text;
    break;
}
case 0x02: { // UNICODE_NORM
    std::string form((char*)&p[1], p[0]); // form = "NFD" or "NFC"
    std::string normalized_text;
    for (size_t i = 0; i < s.text.length(); ) {
        size_t cl = get_utf8_char_len(s.text[i]);
        if (cl == 0) { i++; continue; }
        uint32_t cp = utf8_to_codepoint((const unsigned char*)&s.text[i], cl);

        if (form == "NFD" && is_category(cp, "Mn")) {
            // skip combining marks (strip accents после NFD)
            i += cl;
            continue;
        }
        // For NFC and the rest, leave it as is (or you can add decomposition, but for now it's enough)
        normalized_text += s.text.substr(i, cl);
        i += cl;
    }
    s.text = normalized_text;
    break;
}
        case 0x03: { 
            std::string pat((char*)&p[2],*(uint16_t*)p);
            std::string val((char*)&p[4+pat.length()],*(uint16_t*)&p[2+pat.length()]); 
            size_t pos=0; 
            while((pos=s.text.find(pat,pos))!=std::string::npos) {
                s.text.replace(pos,pat.length(),val);
                pos += val.length(); // Continue the search after replacement
            }
            break; 
        }
        case 0x04: { // FILTER_CATEGORY
            uint8_t num_cats = p[0];
            const uint8_t* ptr = &p[1];
            std::vector<std::string> cats;
            for(int i = 0; i < num_cats; ++i) {
                uint8_t cat_len = ptr[0]; ptr++;
                cats.push_back(std::string((char*)ptr, cat_len));
                ptr += cat_len;
            }

            std::string filtered_text;
            for (size_t i = 0; i < s.text.length(); ) {
                size_t char_len = get_utf8_char_len(s.text[i]);
                if (char_len == 0) { i++; continue; }
                uint32_t cp = utf8_to_codepoint((const unsigned char*)&s.text[i], char_len);
                bool should_filter = false;
                for (const auto& cat : cats) {
                    if (is_category(cp, cat)) {
                        should_filter = true;
                        break;
                    }
                }
                if (!should_filter) {
                    filtered_text += s.text.substr(i, char_len);
                }
                i += char_len;
            }
            s.text = filtered_text;
            break;
        }
        case 0x07: { 
            std::string val((char*)&p[2],*(uint16_t*)p); 
            if(s.text.rfind(val,0)!=0) s.text.insert(0,val); 
            break; 
        }
        case 0x10: _primitive_partition_rules(s, p, len); break;
case 0x15: { // BYTE_ENCODE
    // If there are no fragments, create one from the entire text
    if (s.fragments.empty() && !s.text.empty()) {
        s.fragments.push_back({s.text, false});
    }
    
    // Apply byte encoding to unprotected fragments
    for (auto& f : s.fragments) {
        if (f.is_protected) continue;
        
        std::string encoded;
        // IMPORTANT: convert to unsigned char for correct operation with byte_map
        for (size_t i = 0; i < f.text.length(); ++i) {
            unsigned char byte = static_cast<unsigned char>(f.text[i]);
            auto it = m_res.byte_map.find(byte);
            if (it != m_res.byte_map.end()) {
                encoded += it->second;
            }
            // The 'else' block has been removed. A complete byte_map with 256 entries is a
            // precondition for this algorithm. Appending a raw byte here corrupts
            // the UTF-8 stream and is a critical bug. If a byte is missing from the map,
            // it's better to drop it (or fix the resource generator) than to crash the entire pipeline.
        }
        f.text = encoded;
    }
    break;
}
        case 0x20: _primitive_bpe_encode(s); break;
        case 0x21: { std::string marker((char*)&p[1],p[0]); _primitive_wordpiece_encode(s, marker); break; }
        case 0x22: _primitive_unigram_encode(s); break;
        case 0x30: { 
            std::vector<int32_t> out;
            uint8_t n=p[0];
            const uint8_t* ptr=&p[1]; 
            for(int i=0;i<n;i++){
                if(ptr[0]){ // is_fixed
                    ptr++;
                    if(ptr[0]){ // is_int
                        ptr++;
                        int32_t id;memcpy(&id,ptr,4);ptr+=4;out.push_back(id);
                    }else{ // is_string
                        ptr++;
                        std::string tok((char*)&ptr[1],ptr[0]);ptr+=1+tok.length();
                        int32_t id=2; // default unk
                        m_res.vocab->find(tok,id);
                        out.push_back(id);
                    }
                }else{ // is_slot
                    ptr++;
                    out.insert(out.end(),s.ids.begin(),s.ids.end());
                }
            } 
            s.ids=out; 
            break; 
        }
    }
}

// Structure for storing the parsed rule
struct Rule {
    std::string pattern;
    bool is_protected = false;
    std::string behavior; // "REMOVE", "ISOLATE", ""
    std::string trim_preceding_space;
    // Flag to simplify, so as not to parse \p{P} every time
    bool is_punctuation_rule = false;
    bool is_whitespace_rule = false;
    bool is_gpt_style_rule = false; // For complex regex gpt2/roberta
};

// Removes escaping from protected tokens (\\| → |, \\. → . and etc.)
static std::string unescape_protected_pattern(const std::string& p) {
    std::string result;
    for (size_t i = 0; i < p.length(); ++i) {
        if (p[i] == '\\' && i + 1 < p.length()) {
            result += p[i + 1];   // skip \
            ++i;
        } else {
            result += p[i];
        }
    }
    return result;
}


// A completely rewritten function that mimics Python logic
void TISAVM::_primitive_partition_rules(TISA_State& state, const uint8_t* p, size_t len) {
    state.fragments.clear();
    const std::string& text = state.text;
    
    // Read the number of rules
    const uint8_t* ptr = p;
    uint16_t num_rules;
    memcpy(&num_rules, ptr, 2); 
    ptr += 2;

    // IMPORTANT: If there are no rules, the entire text is one fragment (like in Python line 47)
    if (num_rules == 0) {
        if (!text.empty()) {
            state.fragments.push_back({text, false});
        }
        return;
    }

    // Parsing all the rules
    std::vector<Rule> rules;
    for (uint16_t i = 0; i < num_rules; ++i) {
        Rule rule;
        uint8_t flags = *ptr++;
        rule.is_protected = (flags & 1);

        uint16_t pat_len;
        memcpy(&pat_len, ptr, 2); ptr += 2;
        rule.pattern = std::string((const char*)ptr, pat_len);
        ptr += pat_len;

        if (flags & 2) {
            uint8_t b = *ptr++;
            rule.behavior = (b == 1) ? "REMOVE" : (b == 2 ? "ISOLATE" : "");
        }
        if (flags & 4) {
            uint8_t tlen = *ptr++;
            rule.trim_preceding_space = std::string((const char*)ptr, tlen);
            ptr += tlen;
        }
        
        rules.push_back(rule);
    }

    // Modifying pattern for protected rules
    // In Python (lines 42-45): If a protected rule AND starts with < or [,
    // AND DOES NOT start with "(?: ?)", then "(?: ?)" is added to the beginning
    for (auto& rule : rules) {
        if (rule.is_protected) {
            const std::string& p = rule.pattern;
            if (!p.empty() && (p[0] == '<' || p[0] == '[')) {
                // Check that the pattern does NOT start with "(?: ?)"
                if (rule.pattern.rfind("(?: ?)", 0) != 0) {
                    rule.pattern = "(?: ?)" + rule.pattern;
                }
            }
        }
    }

    // Basic text processing loop
    size_t last_pos = 0;
    
    while (last_pos < text.length()) {
        size_t best_start = std::string::npos;
        size_t best_end   = 0;
        int    best_rule_idx = -1;

        // Looking for the closest match among all the rules
        for (size_t i = 0; i < rules.size(); ++i) {
            const Rule& rule = rules[i];
            size_t m_start = std::string::npos;
            size_t m_end   = 0;

            // ─────── Finding a match based on the pattern type ───────
            
            if (rule.pattern.rfind("(?: ?)", 0) == 0) {               
                // Special tokens with optional space: (?: ?)<token>
                std::string inner = rule.pattern.substr(6);
                std::string literal = unescape_protected_pattern(inner);
                size_t p1 = text.find(literal, last_pos);
                size_t p2 = text.find(" " + literal, last_pos);
                
                if (p1 != std::string::npos && (p2 == std::string::npos || p1 <= p2)) {
                    m_start = p1; 
                    m_end = p1 + literal.length();
                } else if (p2 != std::string::npos) {
                    m_start = p2; 
                    m_end = p2 + 1 + literal.length();
                }
            }
            else if (rule.pattern == "\\p{P}") {
                // Guaranteed isolation of punctuation for BERT/WordPiece (including ASCII)
                for (size_t k = last_pos; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]);
                    if (cl == 0) { ++k; continue; }
                    uint32_t cp = utf8_to_codepoint((const unsigned char*)&text[k], cl);

                    // COMBINED CHECK: first a quick ASCII check, then a full one using the Unicode table
                    bool is_punc = 
                        (cp == '.' || cp == ',' || cp == '!' || cp == '?' ||
                         cp == ':' || cp == ';' || cp == '"' || cp == '\'' ||
                         cp == '(' || cp == ')' || cp == '[' || cp == ']' ||
                         cp == '{' || cp == '}') 
                        || is_category(cp, "P");

                    if (is_punc) {
                        m_start = k;    // Found a match
                        m_end = k + cl;
                        break;          // Stop searching, as we only need the first character
                    }
                    
                    k += cl;
                }
            }
            else if (rule.pattern == "\\s+") {
                // Search for whitespace sequences
                for (size_t k = last_pos; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]);
                    if (cl == 0) { ++k; continue; }
                    if (is_whitespace(utf8_to_codepoint((const unsigned char*)&text[k], cl))) {
                        m_start = k;
                        size_t e = k + cl;
                        while (e < text.length()) {
                            size_t nl = get_utf8_char_len(text[e]);
                            if (nl == 0 || !is_whitespace(utf8_to_codepoint((const unsigned char*)&text[e], nl))) 
                                break;
                            e += nl;
                        }
                        m_end = e; 
                        break;
                    }
                    k += cl;
                }
            }
            else if (rule.pattern.find("[\u4E00") != std::string::npos || 
                     rule.pattern.find("\\u4E00") != std::string::npos) { 
                // Search CJK characters
                for (size_t k = last_pos; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]);
                    if (cl == 0) { ++k; continue; }
                    if (is_cjk(utf8_to_codepoint((const unsigned char*)&text[k], cl))) {
                        m_start = k; 
                        m_end = k + cl; 
                        break;
                    }
                    k += cl;
                }
            }
            else if (rule.pattern.find("'s|'t|'re") != std::string::npos) { 
                // GPT-2 / ByteLevel complex pattern
                m_start = find_gpt_next_token(text, last_pos, m_end);
            }
            else { 
                // A regular string (including protected tokens)
                std::string search_pat = rule.pattern;
                if (rule.is_protected) {
                    search_pat = unescape_protected_pattern(rule.pattern);
                }
                m_start = text.find(search_pat, last_pos);
                if (m_start != std::string::npos) {
                    m_end = m_start + search_pat.length();
                }
            }

            // Select the earliest match
            if (m_start != std::string::npos) {
                if (m_start < best_start) {
                    // This is a new, earlier match. Take it.
                    best_start = m_start;
                    best_end = m_end;
                    best_rule_idx = i;
                } else if (m_start == best_start) {
                    // A match at the same position. Prefer the longer one.
                    if ((m_end - m_start) > (best_end - best_start)) {
                        best_end = m_end;
                        best_rule_idx = i;
                    }
                }
            }
        }

        // If nothing is found, the rest of the text is presented as one fragment
        if (best_rule_idx == -1) {
            if (last_pos < text.length()) {
                state.fragments.push_back({text.substr(last_pos), false});
            }
            break;
        }

        const Rule& best_rule = rules[best_rule_idx];

        // Add preceding text (text BEFORE the match)
        if (best_start > last_pos) {
            std::string pre = text.substr(last_pos, best_start - last_pos);
            
            // Processing trim_preceding_space (Python lines 55-57)
            if (!best_rule.trim_preceding_space.empty() &&
                pre.size() >= best_rule.trim_preceding_space.size() &&
                pre.compare(pre.size() - best_rule.trim_preceding_space.size(),
                            best_rule.trim_preceding_space.size(),
                            best_rule.trim_preceding_space) == 0) {
                pre.resize(pre.size() - best_rule.trim_preceding_space.size());
            }
            
            if (!pre.empty()) {
                state.fragments.push_back({pre, false});
            }
        }

        // Add the match (Python lines 60-65)
        std::string match = text.substr(best_start, best_end - best_start);
        
        // Trim leading space for protected tokens (Python lines 61-63)
        if (best_rule.is_protected && match.size() > 1 && match[0] == ' ') {
            match = match.substr(1);
        }

        // Add fragment only if behavior != REMOVE (Python line 64)
        if (best_rule.behavior != "REMOVE") {
            state.fragments.push_back({match, best_rule.is_protected});
        }

        last_pos = best_end;
    }

    // If there are no fragments after processing,
    // but the text is not empty, add the entire text as a single fragment
    // This corresponds to Python lines 67-68
    if (state.fragments.empty() && !text.empty()) {
        state.fragments.push_back({text, false});
    }
    
    // DEBUGGING: Printing fragments
    Serial.printf("PARTITION: %d fragments from text '%s'\n", state.fragments.size(), text.c_str());
    for (size_t i = 0; i < state.fragments.size(); ++i) {
        Serial.printf("  [%d] '%s' %s\n", i, state.fragments[i].text.c_str(), 
                     state.fragments[i].is_protected ? "(protected)" : "");
    }
}


void TISAVM::_primitive_bpe_encode(TISA_State& state) {
    Serial.println("=== BPE_ENCODE START ===");
    Serial.printf("Fragments: %d\n", state.fragments.size());
    
    // CRITICAL CHECK: If vocab or merges are not loaded, cannot tokenize
    if (!m_res.vocab || !m_res.merges) {
        Serial.println("ERROR: vocab or merges not loaded!");
        // Leave ids empty - this indicates a resource problem.
        state.ids.clear();
        return;
    }
    
    Serial.printf("Vocab entries: %d\n", m_res.vocab->get_entry_count());
    Serial.printf("Merges entries: %d\n", m_res.merges->get_entry_count());
    
    // Defining the UNK token (Python lines 110-111)
    int32_t unk_id = 0;
    m_res.vocab->find("<unk>", unk_id);
    if (unk_id == 0) m_res.vocab->find("[UNK]", unk_id);

    state.ids.clear();
    
    // If there are no fragments, process the entire text
    if (state.fragments.empty() && !state.text.empty()) {
        state.fragments.push_back({state.text, false});
    }
    
    for (const auto& frag : state.fragments) {
        Serial.printf("Processing fragment: '%s' (protected=%d)\n", 
                     frag.text.substr(0, 20).c_str(), frag.is_protected);
        
        // Protected fragments (Python lines 112-115)
        if (frag.is_protected) {
            int32_t id = unk_id;
            // First, let's try to find it as it is.
            if (!m_res.vocab->find(frag.text, id)) {
                // If haven't found it and there is a leading space, try without it.
                if (frag.text.size() > 1 && frag.text[0] == ' ') {
                    std::string trimmed = frag.text.substr(1);
                    if (!m_res.vocab->find(trimmed, id)) {
                        id = unk_id; // If still haven't found it, use unk
                    }
                }
            }
            state.ids.push_back(id);
            continue;
        }
        
        if (frag.text.empty()) continue;

        // Splitting into UTF-8 characters (Python line 116: word = tuple(frag.text))
        std::vector<std::string> word;
        for (size_t off = 0; off < frag.text.size(); ) {
            size_t len = get_utf8_char_len(frag.text[off]);
            if (len == 0) { 
                // Invalid UTF-8 - skip byte
                off++; 
                continue; 
            }
            word.push_back(frag.text.substr(off, len));
            off += len;
        }
        
        if (word.empty()) continue;

        Serial.printf("  Word has %d chars: ", word.size());
        for (size_t i = 0; i < std::min(word.size(), size_t(5)); ++i) {
            Serial.printf("'%s' ", word[i].c_str());
        }
        Serial.println(word.size() > 5 ? "..." : "");

        // BPE merge loop (Python lines 117-126)
        while (word.size() > 1) {
            //Find the best pair (with minimum rank)
            int32_t min_rank = std::numeric_limits<int32_t>::max();
            std::pair<std::string, std::string> best_pair{"", ""};
            
            for (size_t i = 0; i < word.size() - 1; ++i) {
                int32_t rank;
                if (m_res.merges->find({word[i], word[i+1]}, rank)) {
                    if (rank < min_rank) {
                        min_rank = rank;
                        best_pair = {word[i], word[i+1]};
                    }
                }
            }
            
            // If haven't found a suitable pair, exit.
            if (min_rank == std::numeric_limits<int32_t>::max()) break;

            // Merge ALL occurrences of best_pair (Python lines 121-126)
            std::vector<std::string> new_word;
            size_t i = 0;
            while (i < word.size()) {
                if (i < word.size() - 1 &&
                    word[i] == best_pair.first && 
                    word[i+1] == best_pair.second) {
                    new_word.push_back(word[i] + word[i+1]);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    ++i;
                }
            }
            word = std::move(new_word);
        }

        // Convert tokens to IDs (Python line 127)
        for (const auto& token : word) {
            int32_t id = unk_id;
            m_res.vocab->find(token, id);
            state.ids.push_back(id);
        }
    }
}

void TISAVM::_primitive_wordpiece_encode(TISA_State& state, const std::string& marker) {
    if (!m_res.vocab) return;
    int32_t unk_id = 100; m_res.vocab->find("[UNK]", unk_id);
    state.ids.clear();
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) { 
            int32_t id; 
            state.ids.push_back(m_res.vocab->find(frag.text, id) ? id : unk_id); 
            continue; 
        }
        std::string text = frag.text; if (text.empty()) continue;
        size_t start = 0;
        while (start < text.length()) {
            size_t end = text.length(); bool found = false;
            while (start < end) {
                std::string sub = text.substr(start, end-start);
                std::string tok = (start==0)?sub:(marker+sub); int32_t id;
                if (m_res.vocab->find(tok, id)) { 
                    state.ids.push_back(id); start=end; found=true; break; 
                }
                // Decrement end by moving back one UTF-8 character
                if (end > start) {
                    size_t prev_end = end - 1;
                    while(prev_end > start && (text[prev_end] & 0xC0) == 0x80) {
                        prev_end--;
                    }
                    end = prev_end;
                } else {
                    break;
                }
            }
            if (!found) { 
                state.ids.push_back(unk_id); 
                size_t char_len = get_utf8_char_len(text[start]);
                start += (char_len > 0) ? char_len : 1;
            }
        }
    }
}

void TISAVM::_primitive_unigram_encode(TISA_State& state) {
    if (!m_res.vocab) return;
    int32_t unk_id = 2; m_res.vocab->find("<unk>", unk_id);

    state.ids.clear();
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id = unk_id;
            m_res.vocab->find(frag.text, id);
            state.ids.push_back(id);
            continue;
        }
        const std::string& text = frag.text;
        if (text.empty()) continue;

        std::vector<size_t> char_positions;
        for (size_t i = 0; i < text.length(); ) {
            char_positions.push_back(i);
            size_t len = get_utf8_char_len(text[i]);
            i += (len > 0) ? len : 1;
        }
        char_positions.push_back(text.length());
        int n_chars = char_positions.size() - 1;
        if (n_chars == 0) continue;

        std::vector<float> dp(n_chars + 1, -std::numeric_limits<float>::infinity());
        std::vector<int> path(n_chars + 1, 0);
        dp[0] = 0.0f;

        for (int i = 0; i < n_chars; ++i) {
            if (dp[i] <= -std::numeric_limits<float>::infinity() / 2) continue;
            size_t start_pos = char_positions[i];
            for (int j = i + 1; j <= n_chars && j <= i + 50; ++j) {  // ← limit 50 как в Python
                size_t end_pos = char_positions[j];
                std::string sub = text.substr(start_pos, end_pos - start_pos);
                int32_t id; float score;
                if (m_res.vocab->find(sub, id, score)) {
                    if (dp[i] + score > dp[j]) {
                        dp[j] = dp[i] + score;
                        path[j] = i;
                    }
                }
            }
        }

        if (dp[n_chars] <= -std::numeric_limits<float>::infinity() / 2) {
            // fallback — single chars
            for (int i = 0; i < n_chars; ++i) {
                std::string ch = text.substr(char_positions[i], char_positions[i+1] - char_positions[i]);
                int32_t id = unk_id;
                m_res.vocab->find(ch, id);
                state.ids.push_back(id);
            }
        } else {
            std::vector<int32_t> ids;
            int curr = n_chars;
            while (curr > 0) {
                int prev = path[curr];
                std::string token = text.substr(char_positions[prev], char_positions[curr] - char_positions[prev]);
                int32_t id = unk_id;
                m_res.vocab->find(token, id);          // ← safely
                ids.push_back(id);
                curr = prev;
            }
            std::reverse(ids.begin(), ids.end());
            state.ids.insert(state.ids.end(), ids.begin(), ids.end());
        }
    }
}
