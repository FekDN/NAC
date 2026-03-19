#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <map>
#include <set>
#include <Arduino.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "TISA_VM.h"

// Внешние объявления функций, которые понадобятся классам
void* alloc_fast(size_t size);

// Pool of Tensor slots pre-allocated at startup.
// Each slot occupies ~180-200 bytes of SRAM regardless of tensor data.
// Without PSRAM, use 32 (=~6 KB overhead).
// With PSRAM, 64 is safe (=~12 KB overhead).
// Override by defining TENSOR_POOL_SIZE before including this header.
#ifndef TENSOR_POOL_SIZE
  #define TENSOR_POOL_SIZE 32
#endif

// Tensor data allocations >= this threshold prefer PSRAM when available.
// Small metadata / shape arrays stay in fast internal SRAM.
#ifndef PSRAM_ALLOC_THRESHOLD
  #define PSRAM_ALLOC_THRESHOLD 4096
#endif

// =========================================================================
// ОБЩИЕ СТРУКТУРЫ И КЛАССЫ
// =========================================================================

enum class MmapAction : uint8_t { SAVE_RESULT = 10, FREE = 20, FORWARD = 30, PRELOAD = 40 };
enum class DataType { FLOAT32, FLOAT64, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL };

enum class QuantExecMode : uint8_t {
    DEQUANT_ON_LOAD = 0,          // Деквантование при загрузке тензора в память
    DEQUANT_EXEC_PASS_FP = 1,     // Деквантование перед выполнением операции, результат FP32
    DEQUANT_EXEC_REQUANT = 2      // Деквантование перед выполнением операции, ре-квантование результата
};

// Структура для метаданных квантизации
struct QuantizationMetadata {
    uint8_t quant_type = 0; // 0=none, 1=FP16, 2=INT8_TENSOR, 3=INT8_CHANNEL, 4=BLOCK_FP8
    
    // Для INT8_TENSOR
    float scale = 1.0f;
    
    // Для INT8_CHANNEL
    uint8_t axis = 0;
    std::vector<float> scales;
    
    // Для BLOCK_FP8
    uint16_t block_size = 0;
    std::vector<int> original_shape;
    std::vector<float> block_scales;

    void clear() {
        quant_type = 0;
        scale = 1.0f;
        axis = 0;
        // NOTE: intentionally NOT calling shrink_to_fit() — keeps allocated capacity
        // so that pool tensors can be reused without triggering new heap allocations.
        scales.clear();
        block_size = 0;
        original_shape.clear();
        block_scales.clear();
    }
};

struct TensorLocation {
    uint64_t meta_offset;
    uint32_t meta_len;
    uint64_t file_offset;
    uint64_t data_size;
};

struct Tensor {
    void* data = nullptr;
    std::vector<int> shape;
    DataType dtype = DataType::FLOAT32;
    size_t num_elements = 0;
    size_t size = 0; // Размер в байтах
    QuantizationMetadata quant_meta; // Метаданные квантизации
    TensorLocation param_location;   // поле для streaming

    size_t get_element_byte_size() const {
        switch (dtype) {
            case DataType::FLOAT32: return 4;
            case DataType::INT32:   return 4;
            case DataType::INT8:    return 1;
            default: return 0;
        }
    }

    size_t get_byte_size() const {
        return num_elements * get_element_byte_size();
    }

    void update_from_shape() {
        if (shape.empty()) {
            num_elements = 0;
        } else {
            num_elements = 1;
            for (int dim : shape) {
                if (dim <= 0) { // Если измерение 0 или отрицательное, тензор пуст
                    num_elements = 0;
                    break;
                }
                num_elements *= dim;
            }
        }
        size = get_byte_size();
    }

    void update_from_byte_size(size_t new_byte_size) {
        this->size = new_byte_size;
        size_t element_size = get_element_byte_size();
        if (element_size > 0) {
            this->num_elements = new_byte_size / element_size;
        } else {
            this->num_elements = 0;
        }
        this->shape = { (int)this->num_elements };
    }

    // Destructor is intentionally lightweight.
    // Tensors owned by TensorPool have their data freed by TensorPool::release()
    // BEFORE the Tensor slot is returned to the pool — the pool never calls delete
    // on pool-allocated Tensors.
    //
    // The ONLY path that constructs a Tensor via `new` (and therefore calls this
    // destructor) is:
    //   • TensorPool::acquire() when the pool is exhausted  → data freed by
    //     TensorPool::release() before `delete tensor` runs, so data==nullptr here.
    //   • ctx.constants unique_ptr<Tensor> entries created in initialize_nac_context
    //     → data was allocated with heap_caps_malloc and has NOT been freed yet,
    //     so we must free it here.
    ~Tensor() {
        if (data) { heap_caps_free(data); data = nullptr; }
    }
};

class TensorPool {
private:
    std::vector<Tensor> m_pool;
    std::vector<size_t> m_free_indices;
    SemaphoreHandle_t m_mutex;
public:
    TensorPool(size_t pool_size) {
        m_mutex = xSemaphoreCreateMutex();
        m_pool.resize(pool_size);
        m_free_indices.reserve(pool_size);
        for (size_t i = 0; i < pool_size; ++i) m_free_indices.push_back(i);
    }
    ~TensorPool() {
        // Free data buffers for any slots still in-use (acquired but not released).
        // Must null the pointer BEFORE m_pool destructs, because ~Tensor() also
        // calls heap_caps_free(data).  Without the null, both this loop AND
        // ~Tensor() would free the same pointer → double-free → CORRUPT HEAP.
        for (auto& tensor : m_pool) {
            if (tensor.data) { heap_caps_free(tensor.data); tensor.data = nullptr; }
        }
        vSemaphoreDelete(m_mutex);
    }
    Tensor* acquire() {
        xSemaphoreTake(m_mutex, portMAX_DELAY);
        if (m_free_indices.empty()) {
            xSemaphoreGive(m_mutex);
            ESP_LOGW("TensorPool", "Pool exhausted — allocating heap Tensor. Consider increasing TENSOR_POOL_SIZE.");
            return new Tensor();
        }
        size_t index = m_free_indices.back();
        m_free_indices.pop_back();
        xSemaphoreGive(m_mutex);
        return &m_pool[index];
    }
void release(Tensor* tensor) {
        if (!tensor) return;

        // ── Free data buffer unconditionally ─────────────────────────────
        if (tensor->data) {
            heap_caps_free(tensor->data);
            tensor->data = nullptr;          // ← обязательно!
        }

        // ── Определяем, pool-слот или new Tensor() ───────────────────────
        const Tensor* pool_begin = m_pool.empty() ? nullptr : m_pool.data();
        const Tensor* pool_end   = pool_begin ? pool_begin + m_pool.size() : nullptr;
        bool is_pool_tensor = (pool_begin && tensor >= pool_begin && tensor < pool_end);

        if (is_pool_tensor) {
            tensor->size = 0;
            tensor->num_elements = 0;
            tensor->shape.clear();
            tensor->quant_meta.clear();
            xSemaphoreTake(m_mutex, portMAX_DELAY);
            m_free_indices.push_back(static_cast<size_t>(tensor - pool_begin));
            xSemaphoreGive(m_mutex);
        } else {
            // ── Это heap-allocated Tensor (pool был исчерпан) ────────────
            // ~Tensor() теперь увидит data == nullptr → не будет double-free
            delete tensor;
        }
    }
};

struct MmapCommand { MmapAction action; uint16_t target_id; };
struct ParsedInstruction { uint8_t A=0, B=0; std::vector<uint16_t> C; std::vector<int16_t> D; size_t bytes_consumed=0; };

struct NacRuntimeContext {
    std::vector<ParsedInstruction> decoded_ops;
    std::vector<TensorLocation> param_locations;
    std::vector<bool> param_present;
    std::map<uint16_t, std::vector<MmapCommand>> mmap_schedule;
    std::vector<std::string> permutations;
    std::vector<std::unique_ptr<Tensor>> constants;
    std::vector<Tensor*> results;
    std::vector<Tensor*> fast_memory_cache;
    SemaphoreHandle_t cache_mutex;
    std::atomic<uint32_t> current_instruction_idx;
    std::atomic<bool> stop_flag;
    TensorPool tensor_pool;
    std::unique_ptr<TISAVM> tokenizer;
    VM_Resources tokenizer_resources;
    std::vector<uint8_t> tisa_manifest;
    std::map<uint16_t, std::string> id_to_name_map;
    std::vector<Tensor*> user_input_tensors;
    uint16_t num_inputs;
    uint16_t num_outputs;
    QuantExecMode quant_mode; // Новый член для выбора режима квантизации

    NacRuntimeContext() : current_instruction_idx(0), stop_flag(false), tensor_pool(TENSOR_POOL_SIZE), quant_mode(QuantExecMode::DEQUANT_ON_LOAD) {}
    ~NacRuntimeContext() {
        if (cache_mutex) vSemaphoreDelete(cache_mutex);

        // Исправленная логика: сбор уникальных указателей и однократное освобождение
        // для предотвращения double-free.
        auto release_once = [&](Tensor* t, std::set<Tensor*>& seen) {
            if (!t || seen.count(t)) return;
            seen.insert(t);
            tensor_pool.release(t);
        };
        std::set<Tensor*> released;
        for (auto* p : results)            release_once(p, released);
        for (auto* p : fast_memory_cache)  release_once(p, released);
        for (auto* p : user_input_tensors) release_once(p, released);

        // ctx.constants - это unique_ptr, они очистятся автоматически.
    }
};

#endif
