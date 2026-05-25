// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
#include "TISA_VM.h"
#include "platform.h"
#include "NacFile.h"

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>
#include <numeric>
#include <cstring>

extern SemaphoreHandle_t g_sd_card_mutex;

// --- UTF-8 Utilities ---
static inline size_t get_utf8_char_len(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}
static inline uint32_t utf8_to_codepoint(const unsigned char* s, size_t len) {
    switch (len) {
        case 1: return s[0];
        case 2: return ((s[0] & 0x1F) << 6)  | (s[1] & 0x3F);
        case 3: return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6)  | (s[2] & 0x3F);
        case 4: return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        default: return 0;
    }
}
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string r;
    if      (cp < 0x80)    { r += (char)cp; }
    else if (cp < 0x800)   { r += (char)(0xC0|(cp>>6)); r += (char)(0x80|(cp&0x3F)); }
    else if (cp < 0x10000) { r += (char)(0xE0|(cp>>12)); r += (char)(0x80|((cp>>6)&0x3F)); r += (char)(0x80|(cp&0x3F)); }
    else                   { r += (char)(0xF0|(cp>>18)); r += (char)(0x80|((cp>>12)&0x3F)); r += (char)(0x80|((cp>>6)&0x3F)); r += (char)(0x80|(cp&0x3F)); }
    return r;
}

#include "TISA_UCD_TABLES.h"

static bool is_in_category_ranges(uint32_t cp, const CategoryRange* ranges, size_t count) {
    int lo = 0, hi = (int)count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint32_t s = pgm_read_dword(&ranges[mid].start);
        uint32_t e = pgm_read_dword(&ranges[mid].end);
        if (cp < s) hi = mid - 1; else if (cp > e) lo = mid + 1; else return true;
    }
    return false;
}
static bool is_category(uint32_t cp, std::string_view cat) {
    if (cat == "P")  return is_in_category_ranges(cp, CAT_P_RANGES,  sizeof(CAT_P_RANGES) /sizeof(CategoryRange));
    if (cat == "Z")  return is_in_category_ranges(cp, CAT_Z_RANGES,  sizeof(CAT_Z_RANGES) /sizeof(CategoryRange));
    if (cat == "Mn") return is_in_category_ranges(cp, CAT_MN_RANGES, sizeof(CAT_MN_RANGES)/sizeof(CategoryRange));
    if (cat == "Cc") return is_in_category_ranges(cp, CAT_CC_RANGES, sizeof(CAT_CC_RANGES)/sizeof(CategoryRange));
    if (cat == "Cf") return is_in_category_ranges(cp, CAT_CF_RANGES, sizeof(CAT_CF_RANGES)/sizeof(CategoryRange));
    return false;
}
static bool is_whitespace(uint32_t cp) {
    return (cp >= 0x0009 && cp <= 0x000D) || cp == 0x0020 || cp == 0x00A0 || is_category(cp, "Z");
}
enum CharType { LETTER, NUMBER, WHITESPACE, OTHER };
static CharType get_char_type(uint32_t cp) {
    if (is_whitespace(cp)) return WHITESPACE;
    if ((cp>='a'&&cp<='z')||(cp>='A'&&cp<='Z')||(cp>=0x0400&&cp<=0x04FF)||(cp>=0x00C0&&cp<=0x02AF)) return LETTER;
    if (cp >= '0' && cp <= '9') return NUMBER;
    return OTHER;
}
static bool is_cjk(uint32_t cp) {
    return (cp>=0x4E00&&cp<=0x9FFF)||(cp>=0x3040&&cp<=0x309F)||(cp>=0x30A0&&cp<=0x30FF)||(cp>=0xAC00&&cp<=0xD7AF);
}

static size_t find_gpt_next_token(const std::string& text, size_t start, size_t& out_end) {
    if (start >= text.length()) return std::string::npos;
    if (text[start] == '\'') {
        const char* contractions[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
        for (const char* c : contractions) {
            size_t l = strlen(c);
            if (text.compare(start, l, c) == 0) { out_end = start + l; return start; }
        }
    }
    size_t cur = start;
    if (text[cur] == ' ') ++cur;
    if (cur < text.length()) {
        size_t cl = get_utf8_char_len(text[cur]);
        if (cl > 0) {
            uint32_t cp = utf8_to_codepoint((const unsigned char*)&text[cur], cl);
            CharType type = get_char_type(cp);
            if (type != WHITESPACE) {
                size_t ge = cur;
                while (ge < text.length()) {
                    cl = get_utf8_char_len(text[ge]); if (!cl) break;
                    CharType t = get_char_type(utf8_to_codepoint((const unsigned char*)&text[ge], cl));
                    if (type == LETTER && t != LETTER) break;
                    if (type == NUMBER && t != NUMBER) break;
                    if (type == OTHER  && (t==WHITESPACE||t==LETTER||t==NUMBER)) break;
                    ge += cl;
                }
                out_end = ge; return start;
            }
        }
    }
    size_t cl = get_utf8_char_len(text[start]);
    if (cl > 0 && is_whitespace(utf8_to_codepoint((const unsigned char*)&text[start], cl))) {
        size_t e = start;
        while (e < text.length()) {
            cl = get_utf8_char_len(text[e]); if (!cl) break;
            if (!is_whitespace(utf8_to_codepoint((const unsigned char*)&text[e], cl))) break;
            e += cl;
        }
        out_end = e; return start;
    }
    out_end = start + 1; return start;
}

static std::string unescape_pattern(std::string_view p) {
    std::string r; r.reserve(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] == '\\' && i + 1 < p.size()) { r += p[++i]; }
        else                                  { r += p[i]; }
    }
    return r;
}

// =============================================================================
// Resource loading and parsing
// =============================================================================

static std::vector<uint8_t> read_resource_section(uint64_t offset, uint32_t size) {
    std::vector<uint8_t> buf(size);
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    bool ok = sdcard.seek(offset) && sdcard.readData(buf.data(), size) == size;
    xSemaphoreGive(g_sd_card_mutex);
    if (!ok) buf.clear();
    return buf;
}

static std::string read_str16_buf(const uint8_t* buf, size_t buf_size, size_t& pos) {
    if (pos + 2 > buf_size) return {};
    uint16_t len; memcpy(&len, buf + pos, 2); pos += 2;
    if (pos + len > buf_size) return {};
    std::string s((const char*)buf + pos, len); pos += len;
    return s;
}

ResourceView::ResourceView(uint64_t off, uint32_t sz) : _offset(off), _size(sz) {
    _entry_count = 0;
    if (sz < 4) return;
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    bool ok = sdcard.seek(off) && sdcard.readData((uint8_t*)&_entry_count, 4) == 4;
    xSemaphoreGive(g_sd_card_mutex);
    if (!ok) _entry_count = 0;
}
std::string ResourceView::_internal_read_string_from_current_pos() { return {}; }

// --- VOCAB.B ---
BinaryVocabView::BinaryVocabView(uint64_t off, uint32_t sz) : ResourceView(off, sz) {
    if (_entry_count == 0 || sz == 0) return;
    auto buf = read_resource_section(off, sz);
    if (buf.size() < 4 + (size_t)_entry_count * 4) return;

    const uint32_t* offsets = (const uint32_t*)(buf.data() + 4);
    size_t data_start = 4 + (size_t)_entry_count * 4;
    const uint8_t* ds = buf.data() + data_start;
    size_t ds_size    = buf.size() - data_start;

    _vocab_map.reserve(_entry_count * 2);
    _id_to_token.reserve(_entry_count);

    for (uint32_t i = 0; i < _entry_count; ++i) {
        uint32_t rel = offsets[i];
        if (rel + 2 > ds_size) continue;
        size_t pos = rel;
        std::string key = read_str16_buf(ds, ds_size, pos);
        if (key.empty() && pos == rel + 2) continue;
        if (pos + 8 > ds_size) continue;
        int32_t id; float score;
        memcpy(&id,    ds + pos, 4); pos += 4;
        memcpy(&score, ds + pos, 4);
        _vocab_map[key] = {id, score};
        if (_id_to_token.find(id) == _id_to_token.end()) _id_to_token[id] = key;
    }
#ifdef DBG
    ESP_LOGI("VocabView", "Loaded %zu vocab entries.", _vocab_map.size());
#endif
}

bool BinaryVocabView::find(const std::string& tok, int32_t& id, float& score) {
    auto it = _vocab_map.find(tok);
    if (it == _vocab_map.end()) return false;
    id = it->second.id; score = it->second.score;
    return true;
}
bool BinaryVocabView::find(const std::string& tok, int32_t& id) { float d; return find(tok, id, d); }
std::string BinaryVocabView::read_token_by_data_offset(uint32_t) { return {}; }

// --- VIDX.B ---
BinaryVocabIndexView::BinaryVocabIndexView(uint64_t off, uint32_t sz) : ResourceView(off, sz) {
    if (_entry_count == 0 || sz == 0) return;
    auto buf = read_resource_section(off, sz);
    if (buf.size() < 4 + (size_t)_entry_count * 4) return;
    _id_to_offset.resize(_entry_count);
    memcpy(_id_to_offset.data(), buf.data() + 4, (size_t)_entry_count * 4);
#ifdef DBG
    ESP_LOGI("VocabIdxView", "Loaded %u id->offset entries.", _entry_count);
#endif
}
bool BinaryVocabIndexView::get_offset_for_id(int32_t id, uint32_t& off) {
    if (id < 0 || (uint32_t)id >= _id_to_offset.size()) return false;
    off = _id_to_offset[id];
    return (off != 0xFFFFFFFF);
}

// --- MERGES.B PARSER ---
// Reads a flat structure without an offset table. Rank = loop index i.
BinaryMergesView::BinaryMergesView(uint64_t off, uint32_t sz) : ResourceView(off, sz) {
    if (_entry_count == 0 || sz == 0) return;

    auto buf = read_resource_section(off, sz);
    if (buf.size() < 4) return;

    const uint8_t* ptr = buf.data() + 4;
    const uint8_t* end = buf.data() + buf.size();

    _rank_map.reserve(_entry_count * 2);

    for (uint32_t i = 0; i < _entry_count; ++i) {
        if (ptr + 2 > end) break;
        uint16_t len1; memcpy(&len1, ptr, 2); ptr += 2;
        if (ptr + len1 > end) break;
        std::string_view t1((const char*)ptr, len1); ptr += len1;

        if (ptr + 2 > end) break;
        uint16_t len2; memcpy(&len2, ptr, 2); ptr += 2;
        if (ptr + len2 > end) break;
        std::string_view t2((const char*)ptr, len2); ptr += len2;

        if (!t1.empty()) {
            std::string key; key.reserve(t1.size() + 1 + t2.size());
            key.append(t1); key.push_back('\0'); key.append(t2);
            _rank_map[std::move(key)] = (int32_t)i; // Rank is an index!
        }
    }
    _rank_map_loaded = true;
#ifdef DBG
    ESP_LOGI("MergesView", "Loaded %zu merge pairs.", _rank_map.size());
#endif
}

bool BinaryMergesView::find_raw(const std::string& key, int32_t& rank) {
    auto it = _rank_map.find(key);
    if (it == _rank_map.end()) return false;
    rank = it->second; return true;
}

bool BinaryMergesView::find(const std::pair<std::string,std::string>& p, int32_t& r) {
    std::string key = p.first + '\0' + p.second;
    return find_raw(key, r);
}
uint32_t BinaryMergesView::_get_offset_at(uint32_t) { return 0xFFFFFFFF; }
void BinaryMergesView::_load_rank_map() {}

// =============================================================================
// TISAVM Execution Engine
// =============================================================================

TISAVM::TISAVM(VM_Resources& res) : m_res(res) {
    // Preparing a fast reverse map to decode bytes if it is empty
    if (m_res.fast_rev_map.empty() && !m_res.byte_map.empty()) {
        m_res.fast_rev_map.reserve(256);
        for (const auto& kv : m_res.byte_map) {
            uint32_t val = 0;
            memcpy(&val, kv.second.data(), std::min<size_t>(4, kv.second.size()));
            m_res.fast_rev_map[val] = kv.first;
        }
    }
}

std::vector<int32_t> TISAVM::run(const std::vector<uint8_t>& mf, const std::string& txt) {
    TISA_State s; s.text = txt;
    if (mf.size() < 5 || memcmp(mf.data(), "TISA", 4) != 0) return {};
    size_t offset = 5;
    while (offset < mf.size()) {
        if (offset + 5 > mf.size()) break;
        uint8_t op = mf[offset++];
        uint32_t len; memcpy(&len, &mf[offset], 4); offset += 4;
        if (len > mf.size() - offset) break;
        _dispatch_opcode(op, &mf[offset], len, s);
        offset += len;
    }
    return s.ids;
}

struct Rule {
    std::string_view pattern;
    bool             is_protected = false;
    std::string_view behavior;
    std::string_view trim_preceding_space;

    std::string unescaped_lit;
    bool is_opt_space_lit = false;
    bool is_punct = false;
    bool is_space = false;
    bool is_cjk = false;
    bool is_gpt = false;
};

void TISAVM::_primitive_partition_rules(TISA_State& state, const uint8_t* p, size_t /*len*/) {
    state.fragments.clear();
    const std::string& text = state.text;
    uint16_t num_rules; memcpy(&num_rules, p, 2);
    const uint8_t* ptr = p + 2;
    if (num_rules == 0) {
        if (!text.empty()) state.fragments.push_back({text, false});
        return;
    }
    
    std::vector<Rule> rules; rules.reserve(num_rules);
    for (uint16_t i = 0; i < num_rules; ++i) {
        Rule r;
        uint8_t flags = *ptr++;
        r.is_protected = (flags & 1);
        uint16_t pl; memcpy(&pl, ptr, 2); ptr += 2;
        r.pattern = std::string_view((const char*)ptr, pl); ptr += pl;
        if (flags & 2) { uint8_t b=*ptr++; r.behavior=(b==1)?"REMOVE":(b==2?"ISOLATE":""); }
        if (flags & 4) { uint8_t tl=*ptr++; r.trim_preceding_space=std::string_view((const char*)ptr,tl); ptr+=tl; }
        
        // If the token is protected but doesn't have an optional space,
        // treat it as if it does (especially important for RoBERTa <mask>)
        if (r.pattern.rfind("(?: ?)", 0) == 0) {
            r.unescaped_lit = unescape_pattern(r.pattern.substr(6));
            r.is_opt_space_lit = true;
        } else if (r.is_protected && r.pattern.find("\\") == std::string_view::npos) {
            // If it's just a special token (e.g. "<mask\>"), enable optional space logic for it.
            r.unescaped_lit = std::string(r.pattern);
            r.is_opt_space_lit = true;
        } else if (r.pattern == "\\p{P}") {
            r.is_punct = true;
        } else if (r.pattern == "\\s+") {
            r.is_space = true;
        } else if (r.pattern.find("\xe4\xb8\x80")!=std::string_view::npos || r.pattern.find("\\u4E00")!=std::string_view::npos) {
            r.is_cjk = true;
        } else if (r.pattern.find("'s|'t|'re") != std::string_view::npos) {
            r.is_gpt = true;
        } else {
            r.unescaped_lit = unescape_pattern(r.pattern);
        }
        // -------------------------

        rules.push_back(std::move(r));
    }

    size_t last = 0;
    while (last < text.length()) {
        size_t bs = std::string::npos, be = 0; int bi = -1;
        for (size_t i = 0; i < rules.size(); ++i) {
            const Rule& rule = rules[i];
            size_t ms = std::string::npos, me = 0;

            if (rule.is_opt_space_lit) {
                size_t p1 = text.find(rule.unescaped_lit, last);
                if (p1 != std::string::npos) {
                    if (p1 > last && text[p1 - 1] == ' ') { ms = p1 - 1; } 
                    else { ms = p1; }
                    me = p1 + rule.unescaped_lit.size();
                }
            } else if (rule.is_punct) {
                for (size_t k = last; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]); if(!cl){++k;continue;}
                    uint32_t cp = utf8_to_codepoint((const unsigned char*)&text[k], cl);
                    if ((cp=='.'||cp==','||cp=='!'||cp=='?'||cp==':'||cp==';'||cp=='"'||cp=='\''||
                         cp=='('||cp==')'||cp=='['||cp==']'||cp=='{'||cp=='}')||is_category(cp,"P"))
                        { ms=k; me=k+cl; break; }
                    k += cl;
                }
            } else if (rule.is_space) {
                for (size_t k = last; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]); if(!cl){++k;continue;}
                    if (is_whitespace(utf8_to_codepoint((const unsigned char*)&text[k],cl))) {
                        ms = k; size_t e = k + cl;
                        while(e < text.length()){
                            size_t nl=get_utf8_char_len(text[e]); 
                            if(!nl||!is_whitespace(utf8_to_codepoint((const unsigned char*)&text[e],nl)))break; 
                            e+=nl;
                        }
                        me = e; break;
                    }
                    k += cl;
                }
            } else if (rule.is_cjk) {
                for (size_t k = last; k < text.length(); ) {
                    size_t cl = get_utf8_char_len(text[k]); if(!cl){++k;continue;}
                    if (is_cjk(utf8_to_codepoint((const unsigned char*)&text[k],cl))) { ms=k; me=k+cl; break; }
                    k += cl;
                }
            } else if (rule.is_gpt) {
                ms = find_gpt_next_token(text, last, me);
            } else {
                if (!rule.unescaped_lit.empty()) {
                    size_t p = text.find(rule.unescaped_lit, last);
                    if (p != std::string::npos) { ms = p; me = p + rule.unescaped_lit.size(); }
                }
            }

            if (ms != std::string::npos) {
                if (ms < bs || (ms==bs && (me-ms)>(be-bs))) { bs=ms; be=me; bi=(int)i; }
            }
        }
        
        if (bi == -1) {
            if (last < text.length()) state.fragments.push_back({text.substr(last), false});
            break;
        }
        const Rule& br = rules[bi];
        if (bs > last) {
            std::string pre = text.substr(last, bs - last);
            if (!br.trim_preceding_space.empty() &&
                pre.size() >= br.trim_preceding_space.size() &&
                pre.compare(pre.size()-br.trim_preceding_space.size(), br.trim_preceding_space.size(), br.trim_preceding_space)==0)
                pre.resize(pre.size()-br.trim_preceding_space.size());
            if (!pre.empty()) state.fragments.push_back({std::move(pre), false});
        }
        std::string match = text.substr(bs, be-bs);
        if (br.is_protected && match.size()>1 && match[0]==' ') match = match.substr(1);
        if (br.behavior != "REMOVE") state.fragments.push_back({std::move(match), br.is_protected});
        last = be;
    }
    if (state.fragments.empty() && !text.empty())
        state.fragments.push_back({text, false});
}

struct BPESymbol {
    int prev, next;
    size_t start, len;
};

void TISAVM::_primitive_bpe_encode(TISA_State& state) {
    if (!m_res.vocab || !m_res.merges) {
        state.ids.clear();
        return;
    }

    int32_t unk_id = 0;
    if (!m_res.vocab->find("<unk>", unk_id))
        m_res.vocab->find("[UNK]", unk_id);

    state.ids.clear();
    if (state.fragments.empty() && !state.text.empty()) {
        state.fragments.push_back({state.text, false});
    }

    std::string lookup_key; lookup_key.reserve(256);
    std::vector<BPESymbol> syms;

    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id = unk_id;
            if (!m_res.vocab->find(frag.text, id)) {
                if (frag.text.size() > 1 && frag.text[0] == ' ') {
                    if (!m_res.vocab->find(frag.text.substr(1), id)) id = unk_id;
                }
            }
            state.ids.push_back(id);
            continue;
        }
        if (frag.text.empty()) continue;

        syms.clear();
        syms.reserve(frag.text.size());

        int prev_idx = -1;
        for (size_t off = 0; off < frag.text.size(); ) {
            size_t cl = get_utf8_char_len((unsigned char)frag.text[off]);
            if (!cl) { ++off; continue; }
            syms.push_back({prev_idx, -1, off, cl});
            int cur_idx = (int)syms.size() - 1;
            if (prev_idx != -1) syms[prev_idx].next = cur_idx;
            prev_idx = cur_idx;
            off += cl;
        }

        if (syms.empty()) continue;
        int head = 0;

        while (true) {
            int32_t min_rank = std::numeric_limits<int32_t>::max();
            int best_i = -1;

            for (int i = head; i != -1; i = syms[i].next) {
                int nxt = syms[i].next;
                if (nxt == -1) break;

                lookup_key.clear();
                lookup_key.append(frag.text, syms[i].start, syms[i].len);
                lookup_key.push_back('\0'); // Separator expected in find_raw!
                lookup_key.append(frag.text, syms[nxt].start, syms[nxt].len);

                int32_t rank;
                if (m_res.merges->find_raw(lookup_key, rank)) {
                    if (rank < min_rank) {
                        min_rank = rank;
                        best_i = i;
                    }
                }
            }

            if (best_i == -1) break;

            int nxt = syms[best_i].next;
            syms[best_i].len += syms[nxt].len;
            syms[best_i].next = syms[nxt].next;
            if (syms[nxt].next != -1) {
                syms[syms[nxt].next].prev = best_i;
            }
        }

        for (int i = head; i != -1; i = syms[i].next) {
            lookup_key.assign(frag.text, syms[i].start, syms[i].len);
            int32_t id = unk_id;
            m_res.vocab->find(lookup_key, id);
            state.ids.push_back(id);
        }
    }
}

void TISAVM::_primitive_wordpiece_encode(TISA_State& state, const std::string& marker) {
    if (!m_res.vocab) return;
    int32_t unk_id = 100; m_res.vocab->find("[UNK]", unk_id);
    state.ids.clear();
    std::string lookup_key; lookup_key.reserve(256);
    
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id; 
            state.ids.push_back(m_res.vocab->find(frag.text, id) ? id : unk_id); 
            continue; 
        }
        const std::string& text = frag.text; if(text.empty()) continue;
        size_t start = 0;
        while (start < text.length()) {
            size_t end = text.length(); bool found = false;
            while (start < end) {
                lookup_key.clear();
                if (start > 0) lookup_key.append(marker);
                lookup_key.append(text, start, end - start);
                
                int32_t id;
                if (m_res.vocab->find(lookup_key, id)) { state.ids.push_back(id); start=end; found=true; break; }
                if (end > start) { size_t pv=end-1; while(pv>start&&(text[pv]&0xC0)==0x80)--pv; end=pv; }
                else break;
            }
            if (!found) {
                state.ids.push_back(unk_id);
                size_t cl=get_utf8_char_len((unsigned char)text[start]); start+=(cl>0)?cl:1;
            }
        }
    }
}

void TISAVM::_primitive_unigram_encode(TISA_State& state) {
    if (!m_res.vocab) return;
    int32_t unk_id = 2; m_res.vocab->find("<unk>", unk_id);
    state.ids.clear();
    std::string lookup_key; lookup_key.reserve(256);
    
    for (const auto& frag : state.fragments) {
        if (frag.is_protected) {
            int32_t id;
            state.ids.push_back(m_res.vocab->find(frag.text, id) ? id : unk_id); continue;
        }
        const std::string& text = frag.text; if(text.empty()) continue;
        std::vector<size_t> cp; for(size_t i=0;i<text.length();){ cp.push_back(i); size_t l=get_utf8_char_len((unsigned char)text[i]); i+=(l>0)?l:1; } cp.push_back(text.length());
        int n=(int)(cp.size()-1); if(!n) continue;
        std::vector<float> dp(n+1,-std::numeric_limits<float>::infinity());
        std::vector<int>   path(n+1,0); dp[0]=0.f;
        for (int i=0;i<n;++i) {
            if (dp[i]<=-std::numeric_limits<float>::infinity()/2) continue;
            for (int j=i+1;j<=n&&j<=i+50;++j) {
                lookup_key.assign(text, cp[i], cp[j] - cp[i]);
                int32_t id; float score;
                if (m_res.vocab->find(lookup_key, id, score)) {
                    if (dp[i] + score > dp[j]) { dp[j] = dp[i] + score; path[j] = i; }
                }
            }
        }
        if (dp[n]<=-std::numeric_limits<float>::infinity()/2) {
            for(int i=0;i<n;++i){
                lookup_key.assign(text, cp[i], cp[i+1]-cp[i]);
                int32_t id = unk_id; m_res.vocab->find(lookup_key, id); state.ids.push_back(id);
            }
        } else {
            std::vector<int32_t> ids; int cur=n;
            while(cur>0){
                int prev=path[cur];
                lookup_key.assign(text, cp[prev], cp[cur]-cp[prev]);
                int32_t id = unk_id; m_res.vocab->find(lookup_key, id); ids.push_back(id); cur=prev;
            }
            std::reverse(ids.begin(),ids.end()); state.ids.insert(state.ids.end(),ids.begin(),ids.end());
        }
    }
}

void TISAVM::_dispatch_opcode(uint8_t op, const uint8_t* p, size_t len, TISA_State& s) {
    switch (op) {
        case 0x01: {
            const size_t exc_count = sizeof(LOWERCASE_EXCEPTIONS)/sizeof(UnicodeException);
            const size_t rng_count = sizeof(LOWERCASE_RANGES)/sizeof(UnicodeRange);
            std::string result; result.reserve(s.text.length());
            for (size_t i = 0; i < s.text.length(); ) {
                size_t cl = get_utf8_char_len(s.text[i]); if (!cl){ result += s.text[i++]; continue; }
                uint32_t cp = utf8_to_codepoint((const unsigned char*)&s.text[i], cl);
                uint32_t lo = cp;
                { int a=0,b=(int)exc_count-1;
                  while(a<=b){int m=a+(b-a)/2; uint32_t f=pgm_read_dword(&LOWERCASE_EXCEPTIONS[m].from);
                    if(cp==f){lo=pgm_read_dword(&LOWERCASE_EXCEPTIONS[m].to);break;}
                    else if(cp<f)b=m-1;else a=m+1;}}
                if (lo==cp){int a=0,b=(int)rng_count-1;
                  while(a<=b){int m=a+(b-a)/2;uint32_t ss=pgm_read_dword(&LOWERCASE_RANGES[m].start),ee=pgm_read_dword(&LOWERCASE_RANGES[m].end);
                    if(cp<ss)b=m-1;else if(cp>ee)a=m+1;else{lo=(uint32_t)((int32_t)cp+(int32_t)pgm_read_dword(&LOWERCASE_RANGES[m].delta));break;}}}
                
                if (lo == cp) result.append(s.text, i, cl);
                else result += codepoint_to_utf8(lo);
                i += cl;
            }
            s.text = std::move(result);
            break;
        }
        case 0x02: {
            std::string_view form((const char*)&p[1], p[0]);
            if (form == "NFD") {
                const size_t dc = sizeof(DECOMP_TABLE)/sizeof(Decomp);
                std::string decomposed; decomposed.reserve(s.text.length() * 2);
                for (size_t i = 0; i < s.text.length(); ) {
                    size_t cl = get_utf8_char_len(s.text[i]); if(!cl){ decomposed += s.text[i++]; continue; }
                    uint32_t cp = utf8_to_codepoint((const unsigned char*)&s.text[i], cl);
                    int a=0,b=(int)dc-1; bool found=false;
                    while(a<=b){int m=a+(b-a)/2; uint32_t f=pgm_read_dword(&DECOMP_TABLE[m].from);
                      if(cp==f){decomposed+=codepoint_to_utf8(pgm_read_dword(&DECOMP_TABLE[m].to1));
                        uint32_t t2=pgm_read_dword(&DECOMP_TABLE[m].to2); if(t2)decomposed+=codepoint_to_utf8(t2);
                        found=true;break;}else if(cp<f)b=m-1;else a=m+1;}
                    if(!found) decomposed.append(s.text, i, cl);
                    i += cl;
                }
                s.text = std::move(decomposed);
            }
            break;
        }
        case 0x03: {
            uint16_t pat_len; memcpy(&pat_len, &p[0], 2);
            std::string_view pat((const char*)&p[2], pat_len);
            uint16_t val_len; memcpy(&val_len, &p[2 + pat_len], 2);
            std::string_view val((const char*)&p[4 + pat_len], val_len);
            
            std::string res_str; res_str.reserve(s.text.size());
            size_t last = 0, pos;
            while ((pos = s.text.find(pat, last)) != std::string::npos) {
                res_str.append(s.text, last, pos - last);
                res_str.append(val);
                last = pos + pat.size();
            }
            res_str.append(s.text, last, std::string::npos);
            s.text = std::move(res_str);
            break;
        }
        case 0x04: {
            uint8_t nc = p[0]; const uint8_t* ptr = &p[1];
            std::vector<std::string_view> cats; cats.reserve(nc);
            for (int i=0;i<nc;++i){ uint8_t l=*ptr++; cats.push_back({(const char*)ptr,l}); ptr+=l; }
            std::string out; out.reserve(s.text.length());
            for (size_t i=0;i<s.text.length();) {
                size_t cl=get_utf8_char_len(s.text[i]); if(!cl){out+=s.text[i++];continue;}
                uint32_t cp=utf8_to_codepoint((const unsigned char*)&s.text[i],cl);
                bool filt=false; for(auto&c:cats) if(is_category(cp, c)){filt=true;break;}
                if(!filt) out.append(s.text, i, cl);
                i+=cl;
            }
            s.text=std::move(out);
            break;
        }
        case 0x07: {
            uint16_t val_len; memcpy(&val_len, &p[0], 2);
            std::string_view val((const char*)&p[2], val_len);
            if (s.text.compare(0, val.size(), val) != 0) s.text.insert(0, val.data(), val.size());
            break;
        }
        case 0x10: _primitive_partition_rules(s, p, len); break;
        case 0x15: {
            if (s.fragments.empty() && !s.text.empty())
                s.fragments.push_back({s.text, false});
            for (auto& f : s.fragments) {
                if (f.is_protected) continue;
                std::string enc; enc.reserve(f.text.size() * 2);
                for (unsigned char b : f.text)
                    enc += m_res.byte_map[b];
                f.text = std::move(enc);
            }
            break;
        }
        case 0x20: _primitive_bpe_encode(s); break;
        case 0x21: {
            std::string marker((char*)&p[1], p[0]);
            _primitive_wordpiece_encode(s, marker);
            break;
        }
        case 0x22: _primitive_unigram_encode(s); break;
        case 0x30: {
            std::vector<int32_t> out; out.reserve(s.ids.size() + p[0]);
            uint8_t n = p[0]; const uint8_t* ptr = &p[1];
            for (int i=0;i<n;++i) {
                if (ptr[0]) {
                    ptr++;
                    if (ptr[0]) {
                        ptr++; int32_t id; memcpy(&id,ptr,4); ptr+=4; out.push_back(id);
                    } else {
                        ptr++; std::string_view tok((const char*)&ptr[1],ptr[0]); ptr+=1+tok.size();
                        int32_t id = 2; // unk default
                        if (m_res.vocab) m_res.vocab->find(std::string(tok), id);
                        out.push_back(id);
                    }
                } else {
                    ptr++; out.insert(out.end(),s.ids.begin(),s.ids.end());
                }
            }
            s.ids = std::move(out);
            break;
        }
    }
}

// ── Decode ───────────────────────────────────────────────────────────────────

static std::string wp_postprocess(std::string_view s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0, n = s.size(); i < n;) {
        if (s[i] == ' ') {
            size_t j = i; while (j < n && s[j] == ' ') ++j;
            const char puncts[] = ".,!?\"'";
            if (j < n && strchr(puncts, s[j])) { i = j; continue; }
        }
        out += s[i++];
    }
    std::string out2; out2.reserve(out.size());
    for (size_t i = 0, n = out.size(); i < n;) {
        if (i + 2 < n && out[i] == ' ' && out[i + 1] == '\'' && out[i + 2] == ' ') {
            out2 += '\''; i += 3;
        } else {
            out2 += out[i++];
        }
    }
    return out2;
}

std::string TISAVM::decode(const std::vector<int>& ids, bool skip_special) {
    if (!m_res.vocab) return "";

    std::vector<std::string_view> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        const std::string* tok = m_res.vocab->token_by_id(id);
        if (!tok) continue;
        if (skip_special && tok->size() > 2 && ((tok->front()=='<'&&tok->back()=='>')||(tok->front()=='['&&tok->back()==']'))) continue;
        tokens.push_back(*tok);
    }

    bool has_sp=false, has_wp=false;
    static const uint8_t SP[3]={0xe2,0x96,0x81};
    
    for (const auto& t : tokens) {
        if (!has_sp && t.size()>=3 && (uint8_t)t[0]==SP[0]&&(uint8_t)t[1]==SP[1]&&(uint8_t)t[2]==SP[2]) has_sp=true;
        if (!has_wp && t.size()>=2 && t[0]=='#'&&t[1]=='#') has_wp=true;
        if (has_sp && has_wp) break;
    }
    if (!has_sp && !has_wp && m_res.vocab_idx_for_decode && m_res.vocab_idx_for_decode->get_entry_count() > 0) {
        for (uint32_t i=0; i<512 && i<m_res.vocab_idx_for_decode->get_entry_count(); ++i) {
            const std::string* t = m_res.vocab->token_by_id(i);
            if (!t) continue;
            if (!has_sp && t->size()>=3 && (uint8_t)t->at(0)==SP[0]&&(uint8_t)t->at(1)==SP[1]&&(uint8_t)t->at(2)==SP[2]) has_sp=true;
            if (!has_wp && t->size()>=2 && t->at(0)=='#'&&t->at(1)=='#') has_wp=true;
            if (has_sp && has_wp) break;
        }
    }

    ModelKind kind = ModelKind::Unknown;
    if (has_sp) kind = ModelKind::Unigram;
    else if (m_res.merges && m_res.merges->get_entry_count() > 0) kind = ModelKind::BPE;
    else if (has_wp) kind = ModelKind::WordPiece;

    switch (kind) {
    case ModelKind::BPE: {
        if (m_res.fast_rev_map.empty()) {
            std::string r; r.reserve(tokens.size() * 8);
            for (auto t:tokens){
                for (size_t i=0;i<t.size();) {
                    if(i+1<t.size()&&(uint8_t)t[i]==0xC4&&(uint8_t)t[i+1]==0xA0){r+=' ';i+=2;}
                    else r+=t[i++];
                }
            }
            size_t s=0, e=r.size();
            while(s<e && (uint8_t)r[s]<=0x20) s++;
            while(e>s && (uint8_t)r[e-1]<=0x20) e--;
            return r.substr(s, e-s);
        }
        std::string result; result.reserve(tokens.size()*6);
        for (std::string_view tok : tokens) {
            for (size_t i=0; i<tok.size(); ) {
                size_t cl = get_utf8_char_len((unsigned char)tok[i]); if(!cl){++i;continue;}
                uint32_t cval = 0; memcpy(&cval, tok.data()+i, cl);
                auto it = m_res.fast_rev_map.find(cval);
                if (it != m_res.fast_rev_map.end()) result += (char)it->second;
                i += cl;
            }
        }
        return result;
    }
    case ModelKind::WordPiece: {
        if (tokens.empty()) return {};
        std::string out; out.reserve(tokens.size() * 8);
        for (size_t i=0; i<tokens.size(); ++i) {
            std::string_view t = tokens[i];
            if (i > 0) {
                if (t.size() >= 2 && t[0] == '#' && t[1] == '#') t = t.substr(2);
                else out += ' ';
            }
            out += t;
        }
        return wp_postprocess(out);
    }
    case ModelKind::Unigram: {
        std::string s; s.reserve(tokens.size()*6);
        for (auto t:tokens) s+=t;
        std::string out; out.reserve(s.size());
        for (size_t i=0;i<s.size();) {
            if(i+2<s.size()&&(uint8_t)s[i]==SP[0]&&(uint8_t)s[i+1]==SP[1]&&(uint8_t)s[i+2]==SP[2]){out+=' ';i+=3;}
            else out+=s[i++];
        }
        size_t s_idx=0, e_idx=out.size();
        while(s_idx<e_idx && (uint8_t)out[s_idx]<=0x20) s_idx++;
        while(e_idx>s_idx && (uint8_t)out[e_idx-1]<=0x20) e_idx--;
        return out.substr(s_idx, e_idx-s_idx);
    }
    default: {
        std::string r; r.reserve(tokens.size()*8);
        for(size_t i=0;i<tokens.size();++i){if(i)r+=' ';r+=tokens[i];} return r;
    }
    }
}