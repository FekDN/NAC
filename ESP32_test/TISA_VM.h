// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

#ifndef TISA_VM_H
#define TISA_VM_H

#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <memory>
#include "CYD28_SD.h"

struct Fragment { std::string text; bool is_protected = false; };
struct TISA_State { std::string text; std::vector<Fragment> fragments; std::vector<int32_t> ids; };

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
    uint32_t _entry_count = 0;
};

// Класс теперь использует поблочное чтение и корректную логику поиска
class BinaryVocabView : public ResourceView {
public:
    BinaryVocabView(uint64_t offset, uint32_t size);
    bool find(const std::string& token, int32_t& id, float& score);
    bool find(const std::string& token, int32_t& id);
    std::string read_token_by_data_offset(uint32_t data_offset);

private:
    struct VocabEntry {
        std::string key;
        int32_t id;
        float score;
    };

    static const uint32_t CACHE_SIZE_BYTES = 4096;
    static const uint32_t OFFSETS_PER_CACHE = CACHE_SIZE_BYTES / sizeof(uint32_t);
    
    std::vector<uint32_t> _offset_cache;
    int32_t _cached_block_index = -1;

    uint32_t _get_offset_at(uint32_t index);
    VocabEntry _read_entry_at_index(uint32_t index);
};

// Класс теперь использует поблочное чтение
class BinaryVocabIndexView : public ResourceView {
public:
    BinaryVocabIndexView(uint64_t offset, uint32_t size);
    bool get_offset_for_id(int32_t id, uint32_t& offset);

private:
    static const uint32_t CACHE_SIZE_BYTES = 4096;
    static const uint32_t OFFSETS_PER_CACHE = CACHE_SIZE_BYTES / sizeof(uint32_t);

    std::vector<uint32_t> _id_to_offset_cache;
    int32_t _cached_block_index = -1;
};

class BinaryMergesView : public ResourceView {
public:
    BinaryMergesView(uint64_t offset, uint32_t size) : ResourceView(offset, size) {}
    bool find(const std::pair<std::string, std::string>& token_pair, int32_t& rank);
};

struct VM_Resources {
    std::unique_ptr<BinaryVocabView> vocab;
    std::unique_ptr<BinaryVocabIndexView> vocab_idx_for_decode;
    std::unique_ptr<BinaryMergesView> merges;
    std::map<std::string, float> unigram_scores; 
    std::map<uint8_t, std::string> byte_map;
};

class TISAVM {
public:
    TISAVM(VM_Resources& res);
    std::vector<int32_t> run(const std::vector<uint8_t>& manifest_data, const std::string& text);
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true);

private:
    enum class ModelType { UNKNOWN, BPE, WORDPIECE, UNIGRAM };
    ModelType _detect_model_type();

    VM_Resources& m_res;
    void _dispatch_opcode(uint8_t opcode, const uint8_t* payload, size_t payload_len, TISA_State& state);
    
    // --- Primitives ---
    void _primitive_lowercase(TISA_State& state);
    void _primitive_unicode_norm(TISA_State& state, const std::string& form);
    void _primitive_replace(TISA_State& state, const std::string& pattern, const std::string& val);
    void _primitive_filter_category(TISA_State& state, const std::vector<std::string>& cats);
    void _primitive_prepend(TISA_State& state, const std::string& val);
    void _primitive_partition_rules(TISA_State& state, const std::vector<uint8_t>& rules_payload);
    void _primitive_byte_encode(TISA_State& state);
    void _primitive_bpe_encode(TISA_State& state);
    void _primitive_wordpiece_encode(TISA_State& state, const std::string& marker);
    void _primitive_unigram_encode(TISA_State& state);
    void _primitive_compose(TISA_State& state, const std::vector<uint8_t>& payload);
};


#endif // TISA_VM_H
