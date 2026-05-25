// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
#ifndef TISA_VM_H
#define TISA_VM_H

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include "NacFile.h"

struct Fragment { std::string text; bool is_protected = false; };
struct TISA_State { std::string text; std::vector<Fragment> fragments; std::vector<int32_t> ids; };

// =============================================================================
// ResourceView — base class
// =============================================================================
class ResourceView {
public:
    ResourceView(uint64_t offset, uint32_t size);
    virtual ~ResourceView() = default;
    uint32_t get_entry_count() const { return _entry_count; }
    uint64_t get_base_offset() const { return _offset; }

protected:
    std::string _internal_read_string_from_current_pos();
    uint64_t _offset;
    uint32_t _size;
    uint32_t _entry_count;
};

// =============================================================================
// BinaryVocabView
// =============================================================================
class BinaryVocabView : public ResourceView {
public:
    BinaryVocabView(uint64_t offset, uint32_t size);
    bool find(const std::string& token, int32_t& id, float& score);
    bool find(const std::string& token, int32_t& id);
    std::string read_token_by_data_offset(uint32_t data_offset);

    const std::string* token_by_id(int32_t id) const {
        auto it = _id_to_token.find(id);
        return (it != _id_to_token.end()) ? &it->second : nullptr;
    }

private:
    struct VEntry { int32_t id; float score; };
    std::unordered_map<std::string, VEntry> _vocab_map;
    std::unordered_map<int32_t, std::string> _id_to_token;
};

// =============================================================================
// BinaryVocabIndexView
// =============================================================================
class BinaryVocabIndexView : public ResourceView {
public:
    BinaryVocabIndexView(uint64_t offset, uint32_t size);
    bool get_offset_for_id(int32_t id, uint32_t& offset);
private:
    std::vector<uint32_t> _id_to_offset;
};

// =============================================================================
// BinaryMergesView
// =============================================================================
class BinaryMergesView : public ResourceView {
public:
    BinaryMergesView(uint64_t offset, uint32_t size);
    bool find(const std::pair<std::string, std::string>& token_pair, int32_t& rank);
    
    // Quick search for optimized BPE (key: "t1\0t2")
    bool find_raw(const std::string& key, int32_t& rank);

private:
    uint32_t _get_offset_at(uint32_t index);
    void     _load_rank_map();

    bool _rank_map_loaded = false;
    std::unordered_map<std::string, int32_t> _rank_map;
};

// =============================================================================
// VM_Resources
// =============================================================================
struct VM_Resources {
    std::unique_ptr<BinaryVocabView>      vocab;
    std::unique_ptr<BinaryVocabIndexView> vocab_idx_for_decode;
    std::unique_ptr<BinaryMergesView>     merges;
    std::map<std::string, float>          unigram_scores;
    std::map<uint8_t, std::string>        byte_map;
    std::unordered_map<uint32_t, uint8_t> fast_rev_map; // To speed up ByteLevel decoding
};

// =============================================================================
// TISAVM
// =============================================================================
class TISAVM {
public:
    TISAVM(VM_Resources& res);
    std::vector<int32_t> run(const std::vector<uint8_t>& manifest_data,
                              const std::string& text);
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = true);
private:
    enum class ModelKind : uint8_t { Unknown, BPE, WordPiece, Unigram };
    VM_Resources& m_res;
    
    void _dispatch_opcode(uint8_t opcode, const uint8_t* payload,
                          size_t payload_len, TISA_State& state);
                          
    void _primitive_partition_rules(TISA_State& state, const uint8_t* p, size_t len);
    void _primitive_bpe_encode(TISA_State& state);
    void _primitive_wordpiece_encode(TISA_State& state, const std::string& marker);
    void _primitive_unigram_encode(TISA_State& state);
};

#endif