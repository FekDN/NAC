// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <map>
#include <Arduino.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "TISA_VM.h"

// Внешние объявления функций, которые понадобятся классам
void* alloc_fast(size_t size);

// =========================================================================
// ОБЩИЕ СТРУКТУРЫ И КЛАССЫ
// =========================================================================

enum class MmapAction : uint8_t { SAVE_RESULT = 10, FREE = 20, FORWARD = 30, PRELOAD = 40 };
enum class DataType { FLOAT32, FLOAT64, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL };

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
        if (!scales.empty()) {
            scales.clear();
            scales.shrink_to_fit();
        }
        block_size = 0;
        if (!original_shape.empty()) {
            original_shape.clear();
            original_shape.shrink_to_fit();
        }
        if (!block_scales.empty()) {
            block_scales.clear();
            block_scales.shrink_to_fit();
        }
    }
};

struct Tensor {
    void* data = nullptr;
    std::vector<int> shape;
    DataType dtype = DataType::FLOAT32;
    size_t num_elements = 0;
    size_t size = 0; // Размер в байтах
    QuantizationMetadata quant_meta; // Метаданные квантизации

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
        for(auto& tensor : m_pool) { if(tensor.data) heap_caps_free(tensor.data); }
        vSemaphoreDelete(m_mutex);
    }
    Tensor* acquire() {
        xSemaphoreTake(m_mutex, portMAX_DELAY);
        if (m_free_indices.empty()) {
            xSemaphoreGive(m_mutex);
            return new Tensor();
        }
        size_t index = m_free_indices.back();
        m_free_indices.pop_back();
        xSemaphoreGive(m_mutex);
        return &m_pool[index];
    }
    void release(Tensor* tensor) {
        if (!tensor) return;
        xSemaphoreTake(m_mutex, portMAX_DELAY);
        size_t index = tensor - &m_pool[0];
        if (index < m_pool.size()) {
            if (tensor->data) heap_caps_free(tensor->data);
            tensor->data = nullptr;
            tensor->size = 0;
            tensor->num_elements = 0;
            tensor->shape.clear();
            tensor->quant_meta.clear();
            m_free_indices.push_back(index);
        } else {
            delete tensor;
        }
        xSemaphoreGive(m_mutex);
    }
};

struct MmapCommand { MmapAction action; uint16_t target_id; };
struct ParsedInstruction { uint8_t A=0, B=0; std::vector<uint16_t> C; std::vector<int16_t> D; size_t bytes_consumed=0; };
struct TensorLocation {
    uint64_t meta_offset;
    uint32_t meta_len;
    uint64_t file_offset;
    uint64_t data_size;
};

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

    NacRuntimeContext() : current_instruction_idx(0), stop_flag(false), tensor_pool(256) {}
    ~NacRuntimeContext() {
        if (cache_mutex) vSemaphoreDelete(cache_mutex);
        for (auto tensor_ptr : results) { tensor_pool.release(tensor_ptr); }
        for (auto tensor_ptr : fast_memory_cache) { tensor_pool.release(tensor_ptr); }
        for (auto tensor_ptr : user_input_tensors) { if (tensor_ptr) tensor_pool.release(tensor_ptr); }
    }
};


#endif // TYPES_H
