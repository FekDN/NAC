// =========================================================================
// 1. Библиотеки и заголовочные файлы
// =========================================================================
#include <Arduino.h>
#include "TFT_eSPI_Compat.h"
#include "CYD28_TouchscreenR.h"
#include "CYD28_SD.h"
#include "types.h"
#include "op_kernels.h" 

#include "TJpg_Decoder.h"
const char* g_input_image_path = "/hen.jpg";

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cstring>
#include <atomic>
#include <climits>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <numeric> // Для std::accumulate
#include <ArduinoJson.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include "esp_heap_caps.h"

// =========================================================================
// 2. Глобальные объекты и пины
// =========================================================================
TFT_eSPI tft = TFT_eSPI();
CYD28_TouchR touch(320, 240);

Tensor* g_target_tensor_for_decode = nullptr; // Глобальный указатель на целевой тензор
int g_decode_y_offset = 0; // Смещение по Y для записи в тензор

#define PIN_SD_SCK  18
#define PIN_SD_MISO 19
#define PIN_SD_MOSI 23
#define PIN_SD_CS   5

// =========================================================================
// 3. Глобальные переменные и определения
// =========================================================================
extern std::map<std::string, KernelFunc> g_kernel_string_map;
extern std::map<uint8_t, KernelFunc> g_op_kernels;

#define MAX_INSTRUCTION_ARITY 8

enum class SystemState { FINDING_FILES, SELECTING_FILE, ENTERING_PROMPT, EXECUTION_MODE };
std::atomic<SystemState> g_current_state = {SystemState::FINDING_FILES};

std::vector<std::string> g_nac_files;
std::string g_selected_nac_path;
std::string g_user_prompt = "";
bool g_caps_lock = false;
bool g_cursor_visible = true;
unsigned long g_last_cursor_toggle = 0;
const int CURSOR_BLINK_RATE_MS = 500;

EventGroupHandle_t g_system_events;
SemaphoreHandle_t g_sd_card_mutex;
const int EVT_START_EXECUTION = BIT0;
const int EVT_COMPUTE_TASK_DONE = BIT1;
const int EVT_MEMORY_TASK_DONE = BIT2;
TaskHandle_t g_nac_compute_task_handle = NULL;
TaskHandle_t g_nac_memory_task_handle = NULL;

const char* TAG_MAIN = "MAIN";
const char* TAG_COMPUTE = "NAC_COMPUTE";
const char* TAG_MEM = "NAC_MEM";

const int LIST_ITEM_HEIGHT = 25;
const int LIST_Y_OFFSET = 30;

const int KEY_ROWS = 5;
const int KEY_COLS = 10;
const int KEY_HEIGHT = 25; 
const int KEY_WIDTH = 29;
const int KEY_MARGIN = 3;
const int KEYBOARD_Y_OFFSET = 85; // Сдвинута выше для лучшей компоновки

// Новый 5-рядный макет клавиатуры (10 колонок для более широких клавиш)
const char *keyboard_layout[KEY_ROWS][KEY_COLS] = {
    {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
    {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
    {"a", "s", "d", "f", "g", "h", "j", "k", "l", "<-"},
    {"CAPS", "z", "x", "c", "v", "b", "n", "m", ".", ","},
    {"Back", " ", " ", " ", " ", " ", " ", "START", "START", "START"}
};

// =========================================================================
// 5. Вспомогательные функции и задачи
// =========================================================================
// Функция для нахождения индекса максимального элемента
size_t argmax(const float* data, size_t len) {
    if (len == 0) return 0;
    size_t max_idx = 0;
    for (size_t i = 1; i < len; ++i) {
        if (data[i] > data[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

// Функция для вычисления Softmax
void softmax(const float* input, float* output, size_t len) {
    if (len == 0) return;
    float max_val = input[0];
    for (size_t i = 1; i < len; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }

    for (size_t i = 0; i < len; ++i) {
        output[i] /= sum_exp;
    }
}

// НОВАЯ функция-callback для конвертации и записи в int8 тензор
bool tensor_jpeg_output(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (!g_target_tensor_for_decode || !g_target_tensor_for_decode->data) return false;

    // Получаем размеры тензора
    int tensor_height = g_target_tensor_for_decode->shape[2];
    int tensor_width = g_target_tensor_for_decode->shape[3];

    // Указатели на R, G, B каналы в тензоре (формат NCHW)
    int8_t* r_channel = static_cast<int8_t*>(g_target_tensor_for_decode->data);
    int8_t* g_channel = r_channel + tensor_height * tensor_width;
    int8_t* b_channel = g_channel + tensor_height * tensor_width;

    // Проходим по блоку пикселей, который передал декодер
    for (int j = 0; j < h; j++) {
        int current_y = y + j;
        if (current_y >= tensor_height) continue;

        for (int i = 0; i < w; i++) {
            int current_x = x + i;
            if (current_x >= tensor_width) continue;

            // Получаем пиксель в формате RGB565
            uint16_t pixel = bitmap[j * w + i];

            // Конвертируем из RGB565 (uint16) в RGB888 (3x uint8)
            uint8_t r = (pixel >> 11) & 0x1F;
            uint8_t g = (pixel >> 5) & 0x3F;
            uint8_t b = pixel & 0x1F;
            r = (r * 255) / 31;
            g = (g * 255) / 63;
            b = (b * 255) / 31;

            // Конвертируем из uint8 [0, 255] в int8 [-128, 127]
            // Это стандартная практика для квантованных моделей
            r_channel[current_y * tensor_width + current_x] = (int8_t)(r - 128);
            g_channel[current_y * tensor_width + current_x] = (int8_t)(g - 128);
            b_channel[current_y * tensor_width + current_x] = (int8_t)(b - 128);
        }
    }
    return true; // Продолжаем декодирование
}

// Функция для вывода изображения из буфера на TFT (для отладки)
// Эта функция будет вызываться TJpg_Decoder
bool tft_output(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (y >= tft.height()) return false; // Остановка, если вышли за пределы экрана
    tft.pushImage(x, y, w, h, bitmap);
    return true; // Возвращаем true для продолжения декодирования
}

//void* alloc_fast(size_t size) {
//    if (size == 0) return nullptr;
//    void* ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
//    if (!ptr) {
//        ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
//    }
//    return ptr;
//}

void* alloc_fast(size_t size) {
    if (size == 0) return nullptr;
    // Всегда выделяем память из внутреннего SRAM, так как PSRAM нет.
    // Используем MALLOC_CAP_INTERNAL.
    void* ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ptr) {
        ESP_LOGE("alloc_fast", "FATAL: Failed to allocate %d bytes from internal SRAM!", size);
    }
    return ptr;
}

bool parse_instruction_at(const uint8_t* buffer, size_t buffer_size, size_t offset, const std::vector<std::string>& permutations, uint16_t num_outputs, ParsedInstruction& ins){
    ins.C.clear(); ins.D.clear(); ins.bytes_consumed = 0;
    if (offset + 2 > buffer_size) return false;

    const uint8_t* p_start = buffer + offset;
    const uint8_t* p = p_start;
    const uint8_t* buffer_end = buffer + buffer_size;
    
    ins.A = *p++; 
    ins.B = *p++;

    if (ins.A < 10) { // Спеціальні операції
        // <INPUT> (ID=2)
        if (ins.A == 2) {
            if (ins.B == 0) {
                // Data Input - C=[], D=[]
            }
            else if (ins.B >= 1 && ins.B <= 3) {
                // Parameter/State/Const - C=[2, id], D=[]
                if (p + 4 > buffer_end) return false;
                int16_t first_val, id_val;
                memcpy(&first_val, p, 2); p += 2;
                memcpy(&id_val, p, 2); p += 2;
                
                // Зберігаємо ОБА значення згідно зі специфікацією
                ins.C.push_back(first_val);  // = 2
                ins.C.push_back(id_val);     // = param_id
            }
        }
        // <OUTPUT> (ID=3)
        else if (ins.A == 3) {
            if (ins.B == 0) {
                // Final Output - C=[num_outputs+1,...], D=[offsets...]
                size_t nC = num_outputs + 1;
                if (p + nC * 2 > buffer_end) return false;
                ins.C.reserve(nC);
                for (size_t i = 0; i < nC; ++i) {
                    int16_t val;
                    memcpy(&val, p, 2); p += 2;
                    ins.C.push_back(val);
                }
                
                size_t nD = num_outputs;
                if (p + nD * 2 > buffer_end) return false;
                ins.D.reserve(nD);
                for (size_t i = 0; i < nD; ++i) {
                    int16_t val;
                    memcpy(&val, p, 2); p += 2;
                    ins.D.push_back(val);
                }
            }
        }
        
    } else { // Обычные операции (ID >= 10)
        if (ins.B >= permutations.size() || permutations[ins.B].empty()) {
            ins.bytes_consumed = 2;
            return true;
        }
        
        const std::string& perm = permutations[ins.B];
        
        // РАЗДЕЛ 3.4.1: Правило чтения поля C
        bool signature_expects_constants = false;
        for (char p_code : perm) {
            if (strchr("SAfibsc", p_code)) {
                signature_expects_constants = true;
                break;
            }
        }

        if (signature_expects_constants) {
            if (p + 2 > buffer_end) return false;
            int16_t num_consts_from_c; 
            memcpy(&num_consts_from_c, p, 2); p += 2;
            
            ins.C.push_back(num_consts_from_c);
            
            if (num_consts_from_c > 0) {
                if (p + (size_t)num_consts_from_c * 2 > buffer_end) return false;
                ins.C.reserve(num_consts_from_c + 1);
                for(int i = 0; i < num_consts_from_c; ++i) { 
                    uint16_t id; 
                    memcpy(&id, p, 2); p += 2; 
                    ins.C.push_back(id); 
                }
            }
        }
        
        // РАЗДЕЛ 3.5.1: Правило чтения поля D
        size_t nD = perm.length();
        if (nD > 0) {
            if (p + nD * 2 > buffer_end) return false;
            ins.D.reserve(nD);
            for (size_t i = 0; i < nD; ++i) { 
                int16_t d; 
                memcpy(&d, p, 2); p += 2; 
                ins.D.push_back(d); 
            }
        }
    }
    
    ins.bytes_consumed = (p - p_start);
    return true;
}

size_t gather_arguments(NacRuntimeContext& ctx, const ParsedInstruction& ins, uint32_t idx, Tensor** args_out) {
    if (ins.D.size() > MAX_INSTRUCTION_ARITY) return 0;
    
    size_t argc = 0;
    size_t c_idx = 1; 
    bool can_use_constants = !ins.C.empty() && ins.C[0] > 0;

    for (int16_t d_val : ins.D) {
        if (argc >= MAX_INSTRUCTION_ARITY) break;

        if (d_val != 0) {
            uint32_t ancestor_idx = idx + d_val;
            args_out[argc++] = (ancestor_idx < ctx.results.size()) ? ctx.results[ancestor_idx] : nullptr;
        } else {
            if (can_use_constants && c_idx < ins.C.size()) {
                uint16_t const_id = ins.C[c_idx++];
                args_out[argc++] = (const_id < ctx.constants.size() && ctx.constants[const_id]) ? ctx.constants[const_id].get() : nullptr;
            } else {
                 args_out[argc++] = nullptr;
            }
        }
    }
    return argc;
}

void nac_memory_task(void* pvParameters) {
    auto* ctx = static_cast<NacRuntimeContext*>(pvParameters);
    ESP_LOGI(TAG_MEM, "Memory Task started.");

    while (!ctx->stop_flag.load()) {
        // Ожидаем уведомления от вычислительной задачи
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) > 0) {
            if (ctx->stop_flag.load()) break;

            // Получаем текущий шаг вычислений
            uint32_t tick = ctx->current_instruction_idx.load(std::memory_order_acquire);
            
            auto it = ctx->mmap_schedule.find(tick);
            if (it == ctx->mmap_schedule.end()) {
                continue;
            }

            const auto& commands = it->second;
            for (const auto& cmd : commands) {
                switch (cmd.action) {
                    case MmapAction::PRELOAD: {
                        if (cmd.target_id >= ctx->decoded_ops.size()) continue;
                        const auto& preload_ins = ctx->decoded_ops[cmd.target_id];
                        // Эта логика предполагает, что PRELOAD всегда для веса, что может быть неверно,
                        // но для текущей задачи это основное применение.
                        if (preload_ins.A != 2 || preload_ins.B != 1 || preload_ins.C.size() < 2) continue;
                        
                        uint16_t param_id = preload_ins.C[1]; // В <INPUT B=1> ID параметра находится в C[1]
                        if (param_id >= ctx->param_present.size() || !ctx->param_present[param_id]) continue;
                        
                        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                        bool exists = (param_id < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[param_id]);
                        xSemaphoreGive(ctx->cache_mutex);
                        if (exists) continue;

                        Tensor* tensor = ctx->tensor_pool.acquire();
                        tensor->update_from_byte_size(ctx->param_locations[param_id].data_size);
                        tensor->data = alloc_fast(tensor->size);

                        if (tensor->data) {
                            xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                            sdcard.seek(ctx->param_locations[param_id].file_offset);
                            sdcard.readData((uint8_t*)tensor->data, tensor->size);
                            xSemaphoreGive(g_sd_card_mutex);

                            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                            if (param_id >= ctx->fast_memory_cache.size()) ctx->fast_memory_cache.resize(param_id + 1, nullptr);
                            ctx->fast_memory_cache[param_id] = tensor;
                            xSemaphoreGive(ctx->cache_mutex);
                        } else {
                            ESP_LOGE(TAG_MEM, "PRELOAD failed for param %u: OOM", param_id);
                            ctx->tensor_pool.release(tensor);
                        }
                        break;
                    }
                    
                    case MmapAction::FREE: {
                        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                        // Освобождаем результат операции.
                        // target_id в команде FREE - это индекс операции, результат которой нужно освободить.
                        if (cmd.target_id < ctx->results.size() && ctx->results[cmd.target_id]) {
                            // Проверяем, не сохранен ли этот же указатель в кеше, и если да - обнуляем там.
                            for (size_t i = 0; i < ctx->fast_memory_cache.size(); ++i) {
                                if (ctx->fast_memory_cache[i] == ctx->results[cmd.target_id]) {
                                    ctx->fast_memory_cache[i] = nullptr;
                                }
                            }
                            // Освобождаем тензор.
                            ctx->tensor_pool.release(ctx->results[cmd.target_id]);
                            ctx->results[cmd.target_id] = nullptr;
                        }
                        xSemaphoreGive(ctx->cache_mutex);
                        break;
                    }

                    case MmapAction::SAVE_RESULT: {
                        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                        
                        if (tick < ctx->results.size() && ctx->results[tick]) {
                            if (cmd.target_id >= ctx->fast_memory_cache.size()) {
                                ctx->fast_memory_cache.resize(cmd.target_id + 1, nullptr);
                            }
                            
                            // --- ИСПРАВЛЕНИЕ: Копируем указатель, а не перемещаем ---
                            // Мы просто сохраняем указатель на результат в кеш.
                            // Оригинальный указатель в ctx->results[tick] остается на месте.
                            ctx->fast_memory_cache[cmd.target_id] = ctx->results[tick];
                            // ctx->results[tick] = nullptr; // <--- ЭТА СТРОКА ВЫЗЫВАЛА ОШИБКИ
                            
                            ESP_LOGD(TAG_MEM, "[Tick %u] COPIED result pointer to cache slot %u", tick, cmd.target_id);
                        } else {
                            ESP_LOGW(TAG_MEM, "[Tick %u] SAVE_RESULT failed: no result found to save.", tick);
                        }
                        
                        xSemaphoreGive(ctx->cache_mutex);
                        break;
                    }

                    case MmapAction::FORWARD: {
                        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);

                        if (tick < ctx->results.size() && ctx->results[tick] && cmd.target_id < ctx->results.size()) {
                            
                            if (ctx->results[cmd.target_id] != nullptr) {
                                ESP_LOGW(TAG_MEM, "[Tick %u] FORWARD is overwriting a non-null tensor at index %u. Releasing old tensor.", tick, cmd.target_id);
                                ctx->tensor_pool.release(ctx->results[cmd.target_id]);
                            }

                            // --- ИСПРАВЛЕНИЕ: Копируем указатель, а не перемещаем ---
                            // Мы копируем указатель на результат в будущую ячейку.
                            // Оригинальный указатель в ctx->results[tick] остается на месте.
                            ctx->results[cmd.target_id] = ctx->results[tick];
                            // ctx->results[tick] = nullptr; // <--- ЭТА СТРОКА ВЫЗЫВАЛА ОШИБКИ
                            
                            ESP_LOGD(TAG_MEM, "[Tick %u] COPIED result pointer to future result slot %u", tick, cmd.target_id);

                        } else {
                             ESP_LOGW(TAG_MEM, "[Tick %u] FORWARD to %u failed: source/dest out of bounds or source is null.", tick, cmd.target_id);
                        }

                        xSemaphoreGive(ctx->cache_mutex);
                        break;
                    }
                }
            }
        }
    }
    xEventGroupSetBits(g_system_events, EVT_MEMORY_TASK_DONE);
    ESP_LOGI(TAG_MEM, "Memory Task finished.");
    vTaskDelete(NULL);
}

// НОВАЯ ФУНКЦИЯ: Читает и парсит метаданные для тензора с SD-карты
bool read_and_parse_quant_metadata(Tensor* tensor, const TensorLocation& loc) {
    if (loc.meta_len == 0) {
        tensor->quant_meta.quant_type = 0; // 'none'
        return true;
    }

    std::vector<uint8_t> meta_buffer(loc.meta_len);
    
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(loc.meta_offset);
    size_t bytes_read = sdcard.readData(meta_buffer.data(), loc.meta_len);
    xSemaphoreGive(g_sd_card_mutex);
    
    if (bytes_read != loc.meta_len) {
        ESP_LOGE("META_PARSE", "Failed to read full metadata block.");
        return false;
    }

    const uint8_t* p = meta_buffer.data();
    const uint8_t* end = p + loc.meta_len;
    
    // Пропускаем dtype_id (1Б) и rank (1Б) + shape (rank * 4Б), т.к. они для данных, а не для квантизации.
    // Это уже должно быть в тензоре или неважно для деквантизации.
    uint8_t dtype_id, rank;
    memcpy(&dtype_id, p, 1); p += 1;
    memcpy(&rank, p, 1); p += 1;
    p += rank * 4;

    if (p >= end) return false;
    
    memcpy(&tensor->quant_meta.quant_type, p, 1); p += 1;

    switch (tensor->quant_meta.quant_type) {
        case 2: // INT8_TENSOR
            if (p + 4 > end) return false;
            memcpy(&tensor->quant_meta.scale, p, 4);
            break;
        case 3: // INT8_CHANNEL
            if (p + 5 > end) return false;
            uint32_t num_scales;
            memcpy(&tensor->quant_meta.axis, p, 1); p += 1;
            memcpy(&num_scales, p, 4); p += 4;
            if (p + num_scales * 4 > end) return false;
            tensor->quant_meta.scales.resize(num_scales);
            memcpy(tensor->quant_meta.scales.data(), p, num_scales * 4);
            break;
        case 4: // BLOCK_FP8
            if (p + 3 > end) return false;
            uint8_t original_rank;
            memcpy(&tensor->quant_meta.block_size, p, 2); p += 2;
            memcpy(&original_rank, p, 1); p += 1;
            
            if (p + original_rank * 4 > end) return false;
            tensor->quant_meta.original_shape.resize(original_rank);
            if (original_rank > 0) {
                memcpy(tensor->quant_meta.original_shape.data(), p, original_rank * 4);
            }
            p += original_rank * 4;

            if (p + 4 > end) return false;
            uint32_t num_block_scales;
            memcpy(&num_block_scales, p, 4); p += 4;

            if (p + num_block_scales * 4 > end) return false;
            tensor->quant_meta.block_scales.resize(num_block_scales);
            memcpy(tensor->quant_meta.block_scales.data(), p, num_block_scales * 4);
            break;
    }

    return true;
}

// Функция деквантизации тензора согласно метаданным
bool dequantize_tensor(Tensor* tensor) {
    if (!tensor || !tensor->data) return false;
    
    uint8_t quant_type = tensor->quant_meta.quant_type;
    
    if (quant_type == 0 || quant_type == 1) { // none or FP16
        return true;
    }
    
    size_t num_elem = tensor->num_elements;
    float* float_data = static_cast<float*>(alloc_fast(num_elem * sizeof(float)));
    if (!float_data) {
        ESP_LOGE("dequantize", "Failed to allocate memory for float32 buffer (size: %u)", num_elem * sizeof(float));
        return false;
    }

    if (quant_type == 2) { // INT8_TENSOR
        int8_t* int8_data = static_cast<int8_t*>(tensor->data);
        float scale = tensor->quant_meta.scale;
        for (size_t i = 0; i < num_elem; ++i) {
            float_data[i] = static_cast<float>(int8_data[i]) * scale;
        }
    } else if (quant_type == 3) { // INT8_CHANNEL
        int8_t* int8_data = static_cast<int8_t*>(tensor->data);
        uint8_t axis = tensor->quant_meta.axis;
        const std::vector<float>& scales = tensor->quant_meta.scales;
        
        size_t axis_size = (axis < tensor->shape.size()) ? tensor->shape[axis] : 1;
        if (axis_size == 0) axis_size = 1; // Защита от деления на ноль
        size_t stride = 1;
        for (size_t i = axis + 1; i < tensor->shape.size(); ++i) {
            stride *= tensor->shape[i];
        }
        
        for (size_t i = 0; i < num_elem; ++i) {
            size_t scale_idx = (i / stride) % axis_size;
            float scale = (scale_idx < scales.size()) ? scales[scale_idx] : 1.0f;
            float_data[i] = static_cast<float>(int8_data[i]) * scale;
        }
    } else if (quant_type == 4) { // BLOCK_FP8
        int8_t* int8_data = static_cast<int8_t*>(tensor->data);
        uint16_t block_size = tensor->quant_meta.block_size;
        const std::vector<float>& block_scales = tensor->quant_meta.block_scales;
        const std::vector<int>& original_shape = tensor->quant_meta.original_shape;

        size_t original_num_elements = 1;
        for (int dim : original_shape) {
            original_num_elements *= dim;
        }

        // Перевыделяем память, если оригинальный размер меньше
        if (original_num_elements < num_elem) {
             heap_caps_free(float_data);
             float_data = static_cast<float*>(alloc_fast(original_num_elements * sizeof(float)));
             if (!float_data) {
                 ESP_LOGE("dequantize", "Failed to re-allocate memory for BLOCK_FP8");
                 heap_caps_free(tensor->data);
                 tensor->data = nullptr;
                 return false;
             }
        }
        
        size_t num_blocks = block_scales.size();
        size_t output_idx = 0;

        for (size_t block_idx = 0; block_idx < num_blocks && output_idx < original_num_elements; ++block_idx) {
            float scale = block_scales[block_idx];
            size_t block_start = block_idx * block_size;
            for (size_t i = 0; i < block_size && output_idx < original_num_elements; ++i) {
                float_data[output_idx++] = static_cast<float>(int8_data[block_start + i]) * scale;
            }
        }
        
        // Обновляем метаданные тензора после деквантизации
        tensor->shape = original_shape;
        tensor->num_elements = original_num_elements;
    } else {
        ESP_LOGW("dequantize", "Unknown quantization type: %u", quant_type);
        heap_caps_free(float_data);
        tensor->quant_meta.clear(); // Очищаем метаданные даже в случае ошибки
        return false;
    }

    // Обновляем тензор: заменяем старые данные на новые float32 данные
    heap_caps_free(tensor->data);
    tensor->data = float_data;
    tensor->dtype = DataType::FLOAT32;
    tensor->size = tensor->num_elements * sizeof(float);
    tensor->quant_meta.clear();
    return true;
}

void nac_compute_task(void* pvParameters) {
    auto* ctx = static_cast<NacRuntimeContext*>(pvParameters);
    size_t user_input_idx = 0;
    for (uint32_t idx = 0; idx < ctx->decoded_ops.size() && !ctx->stop_flag.load(); ++idx) {
        ctx->current_instruction_idx.store(idx, std::memory_order_release);
        if (g_nac_memory_task_handle) xTaskNotifyGive(g_nac_memory_task_handle);
        
        const auto& ins = ctx->decoded_ops[idx];

        // --- Блок отладочного вывода ---
        {
            std::string op_name_str;
            if (ins.A == 2) {
                switch (ins.B) {
                    case 0: op_name_str = "INPUT (User Data)"; break;
                    case 1: op_name_str = "INPUT (Weight)"; break;
                    case 2: op_name_str = "INPUT (State)"; break;
                    case 3: op_name_str = "INPUT (Constant)"; break;
                    default: op_name_str = "INPUT (Unknown)"; break;
                }
                if (ins.C.size() > 1) { // Для B=1,2,3 ID в C[1]
                    op_name_str += " ID=" + std::to_string(ins.C[1]);
                }
            } else if (ins.A >= 10) {
                if (ctx->id_to_name_map.count(ins.A)) {
                    op_name_str = ctx->id_to_name_map.at(ins.A);
                } else {
                    op_name_str = "!!! Unknown Op !!!";
                }
            } else {
                op_name_str = "--- Special/Reserved Op ---";
            }

            Serial.printf("[COMPUTE] Step[%u / %u]: OpID=%u, B=%u, Name: %s\n", 
                idx, 
                ctx->decoded_ops.size() > 0 ? ctx->decoded_ops.size() - 1 : 0, 
                ins.A, 
                ins.B, 
                op_name_str.c_str());
            size_t free_heap = ESP.getFreeHeap();
            Serial.printf("  Mem: %u KB free\n", free_heap / 1024);
        }
        
        if (ins.A == 2) { // <INPUT>
            Tensor* source_tensor = nullptr;
            if (ins.B == 1) { // Загрузка веса
                if (ins.C.size() < 2) continue;
                uint16_t param_id = ins.C[1];

                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (param_id < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[param_id]) {
                    source_tensor = ctx->fast_memory_cache[param_id];
                    ctx->fast_memory_cache[param_id] = nullptr;
                }
                xSemaphoreGive(ctx->cache_mutex);

                if (!source_tensor && param_id < ctx->param_present.size() && ctx->param_present[param_id]) {
                    ESP_LOGD(TAG_COMPUTE, "On-demand load for param %u", param_id);
                    source_tensor = ctx->tensor_pool.acquire();
                    
                    // --- ИСПРАВЛЕНИЕ: Объявляем переменную loc здесь ---
                    const auto& loc = ctx->param_locations[param_id];
                    
                    source_tensor->update_from_byte_size(loc.data_size);
                    source_tensor->data = alloc_fast(loc.data_size);

                    if (source_tensor->data) {
                        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                        sdcard.seek(loc.file_offset);
                        sdcard.readData((uint8_t*)source_tensor->data, loc.data_size);
                        xSemaphoreGive(g_sd_card_mutex);
                        
                        if (!read_and_parse_quant_metadata(source_tensor, loc)) {
                             ESP_LOGE(TAG_COMPUTE, "Failed to read metadata for param %u", param_id);
                             ctx->tensor_pool.release(source_tensor); source_tensor = nullptr;
                        } else if (!dequantize_tensor(source_tensor)) {
                            ESP_LOGE(TAG_COMPUTE, "Failed to dequantize param %u", param_id);
                            ctx->tensor_pool.release(source_tensor); source_tensor = nullptr;
                        }

                    } else {
                        ESP_LOGE(TAG_COMPUTE, "On-demand load failed for param %u: OOM", param_id);
                        ctx->tensor_pool.release(source_tensor); source_tensor = nullptr;
                    }
                }
            } else if (ins.B == 0) { // Ввод пользователя (Data Input)
                if (user_input_idx < ctx->user_input_tensors.size()) {
                    source_tensor = ctx->user_input_tensors[user_input_idx];
                    ctx->user_input_tensors[user_input_idx] = nullptr; 
                    user_input_idx++;
                } else {
                    Serial.printf("[COMPUTE][ERROR] Requested user input, but input tensor vector is exhausted!\n");
                    ctx->stop_flag.store(true);
                }
            }
            if(idx < ctx->results.size()) ctx->results[idx] = source_tensor;

        } else if (ins.A >= 10) {
            Tensor* arguments[MAX_INSTRUCTION_ARITY] = {nullptr};
            size_t argc = gather_arguments(*ctx, ins, idx, arguments);
            Tensor* result_tensor = nullptr;

            auto it = g_op_kernels.find(ins.A);
            if (it != g_op_kernels.end()) {
                KernelFunc kernel = it->second;
                result_tensor = kernel(ctx, ins, arguments, argc);
            } else {
                const char* op_name = ctx->id_to_name_map.count(ins.A) ? ctx->id_to_name_map[ins.A].c_str() : "Unknown";
                ESP_LOGW(TAG_COMPUTE, "Unimplemented op ID=%u (%s). Passing through.", ins.A, op_name);
                result_tensor = op_nac_pass(ctx, ins, arguments, argc);
            }

            if(idx < ctx->results.size()) {
                ctx->results[idx] = result_tensor;
            } else if (result_tensor) {
                ESP_LOGE(TAG_COMPUTE, "Result for op %u at idx %u out of bounds. Releasing.", ins.A, idx);
                ctx->tensor_pool.release(result_tensor);
            }
        }
    }
    xEventGroupSetBits(g_system_events, EVT_COMPUTE_TASK_DONE);
    ESP_LOGI(TAG_COMPUTE, "Compute task finished.");
    vTaskDelete(NULL);
}

bool initialize_nac_context(NacRuntimeContext& ctx) {
    register_kernels();

    if (!sdcard.isFileOpen()) {
        ESP_LOGE(TAG_MAIN, "NAC file is not open.");
        return false;
    }
    
    uint8_t header[88];
    sdcard.seek(0);
    if (sdcard.readData(header, 88) != 88) {
        ESP_LOGE(TAG_MAIN, "Failed to read NAC header");
        return false;
    }
    if (memcmp(header, "NAC", 3) != 0) {
        ESP_LOGE(TAG_MAIN, "Invalid NAC magic bytes");
        return false;
    }
    if (header[3] != 1) {
        ESP_LOGE(TAG_MAIN, "Unsupported NAC version: %d", header[3]);
        return false;
    }
    
    // storage flag с заголовка
    uint8_t quant_byte = header[4];
    bool store_weights_internally = (quant_byte & 0x80) != 0;
    Serial.printf("[MAIN] Internal storage: %s\n", store_weights_internally ? "YES" : "NO");
    
    // io_counts с заголовка
    memcpy(&ctx.num_inputs, header + 5, 2);
    memcpy(&ctx.num_outputs, header + 7, 2);
    Serial.printf("[MAIN] Model IO: %u inputs, %u outputs\n", ctx.num_inputs, ctx.num_outputs);

    uint64_t offsets[9] = {0};
    memcpy(offsets, header + 12, sizeof(offsets));
    uint64_t mmap_off = offsets[0], ops_off = offsets[1], cmap_off = offsets[2], cnst_off = offsets[3],
             perm_off = offsets[4], data_off = offsets[5], proc_off = offsets[6], meta_off = offsets[7], rsrc_off = offsets[8];

    if (mmap_off > 0) {
        if (!sdcard.seek(mmap_off + 4)) return false;
        uint32_t num_records;
        if (sdcard.readData((uint8_t*)&num_records, 4) != 4) return false;
        for (uint32_t i = 0; i < num_records; ++i) {
            uint16_t tick_id; uint8_t num_commands;
            if (sdcard.readData((uint8_t*)&tick_id, 2) != 2 || sdcard.readData((uint8_t*)&num_commands, 1) != 1) return false;
            std::vector<MmapCommand> commands;
            commands.reserve(num_commands);
            for (uint8_t j = 0; j < num_commands; ++j) {
                MmapCommand cmd; uint8_t action_type;
                if (sdcard.readData((uint8_t*)&action_type, 1) != 1 || sdcard.readData((uint8_t*)&cmd.target_id, 2) != 2) return false;
                cmd.action = static_cast<MmapAction>(action_type);
                commands.push_back(cmd);
            }
            ctx.mmap_schedule[tick_id] = std::move(commands);
        }
    }

    if (cmap_off > 0) {
        if (!sdcard.seek(cmap_off + 4)) return false;
        uint32_t count;
        if (sdcard.readData((uint8_t*)&count, 4) != 4) return false;
        for (uint32_t i = 0; i < count; ++i) {
            uint16_t op_id; uint8_t name_len;
            if (sdcard.readData((uint8_t*)&op_id, 2) != 2 || sdcard.readData((uint8_t*)&name_len, 1) != 1) return false;
            std::string op_name(name_len, '\0');
            if (sdcard.readData((uint8_t*)op_name.data(), name_len) != name_len) return false;
            ctx.id_to_name_map[op_id] = op_name;
            auto it = g_kernel_string_map.find(op_name);
            if (it != g_kernel_string_map.end()) {
                g_op_kernels[op_id] = it->second;
            }
        }
    }

    if (perm_off > 0) {
        if (!sdcard.seek(perm_off + 4)) return false;
        uint32_t count;
        if (sdcard.readData((uint8_t*)&count, 4) != 4) return false;
        uint16_t max_perm_id = 0;
        std::map<uint16_t, std::string> temp_perms;
        for (uint32_t i = 0; i < count; ++i) {
            uint16_t id; uint8_t len;
            if (sdcard.readData((uint8_t*)&id, 2) != 2 || sdcard.readData((uint8_t*)&len, 1) != 1) return false;
            std::string s(len, '\0');
            if (sdcard.readData((uint8_t*)s.data(), len) != len) return false;
            temp_perms[id] = s;
            if (id > max_perm_id) max_perm_id = id;
        }
        ctx.permutations.resize(max_perm_id + 1);
        for(const auto& pair : temp_perms) ctx.permutations[pair.first] = std::move(pair.second);
    }

    if (ops_off > 0) {
        if (!sdcard.seek(ops_off + 4)) return false;
        uint32_t num_ops;
        if (sdcard.readData((uint8_t*)&num_ops, 4) != 4) return false;
        uint64_t ops_data_start = sdcard.getPosition();
        uint64_t next_section_start = (uint64_t)-1;
        for(int i = 0; i < 9; ++i) {
            if(offsets[i] > ops_off && (next_section_start == (uint64_t)-1 || offsets[i] < next_section_start)) {
                next_section_start = offsets[i];
            }
        }
        if (next_section_start == (uint64_t)-1) {
            next_section_start = sdcard.size(); 
        }
        long ops_section_size = next_section_start - ops_data_start;
        std::vector<uint8_t> ops_buffer(ops_section_size);
        sdcard.seek(ops_data_start);
        if (sdcard.readData(ops_buffer.data(), ops_section_size) != ops_section_size) return false;
        
        ctx.decoded_ops.reserve(num_ops);
        size_t byte_ptr = 0;
        while (byte_ptr < ops_buffer.size() && ctx.decoded_ops.size() < num_ops) {
            ParsedInstruction ins;
            if (!parse_instruction_at(ops_buffer.data(), ops_buffer.size(), byte_ptr, ctx.permutations, ctx.num_outputs, ins) || ins.bytes_consumed == 0) return false;
            ctx.decoded_ops.push_back(std::move(ins));
            byte_ptr += ins.bytes_consumed;
        }
        ctx.results.resize(ctx.decoded_ops.size(), nullptr);
    }
    
    if (data_off > 0) {
        if (!sdcard.seek(data_off + 4)) return false;
        uint32_t param_name_count, input_name_count, num_tensors;
        sdcard.readData((uint8_t*)&param_name_count, 4);
        for(uint32_t i=0; i<param_name_count; ++i) { 
            uint16_t id, len; sdcard.readData((uint8_t*)&id, 2); sdcard.readData((uint8_t*)&len, 2);
            sdcard.seek(sdcard.getPosition() + len);
        }
        sdcard.readData((uint8_t*)&input_name_count, 4);
        for(uint32_t i=0; i<input_name_count; ++i) { 
            uint16_t id, len; sdcard.readData((uint8_t*)&id, 2); sdcard.readData((uint8_t*)&len, 2);
            sdcard.seek(sdcard.getPosition() + len);
        }

        if (store_weights_internally) {
            sdcard.readData((uint8_t*)&num_tensors, 4);
            Serial.printf("[MAIN] Loading %u tensor locations...\n", num_tensors);
            for (uint32_t i = 0; i < num_tensors; ++i) {
                uint16_t p_id; uint32_t meta_len; uint64_t data_len;
                if (sdcard.readData((uint8_t*)&p_id, 2)!=2 || sdcard.readData((uint8_t*)&meta_len, 4)!=4 || sdcard.readData((uint8_t*)&data_len, 8)!=8) return false;
                
                uint64_t meta_offset = sdcard.getPosition();
                sdcard.seek(meta_offset + meta_len);
                uint64_t data_offset = sdcard.getPosition();

                if (p_id >= ctx.param_locations.size()) {
                    ctx.param_locations.resize(p_id + 1);
                    ctx.param_present.resize(p_id + 1, false);
                }
                ctx.param_locations[p_id] = { meta_offset, meta_len, data_offset, data_len };
                ctx.param_present[p_id] = true;

                sdcard.seek(data_offset + data_len);
            }
        } else {
            Serial.println("[MAIN] Weights stored externally (skipping DATA section tensors)");
        }
    }

    if (proc_off > 0) {
        if (!sdcard.seek(proc_off + 4)) return false;
        uint32_t manifest_size;
        sdcard.readData((uint8_t*)&manifest_size, 4);
        ctx.tisa_manifest.resize(manifest_size);
        sdcard.readData(ctx.tisa_manifest.data(), manifest_size);
    }
    
 if (rsrc_off > 0) {
    Serial.println("[MAIN] Found RSRC section, parsing resources...");
    if (!sdcard.seek(rsrc_off + 4)) return false;
    uint32_t num_files;
    sdcard.readData((uint8_t*)&num_files, 4);
    Serial.printf("[MAIN] RSRC contains %u files.\n", num_files);

    // Временный буфер для JSON файла
    std::vector<uint8_t> json_buffer;

    for (uint32_t i = 0; i < num_files; ++i) {
        uint16_t name_len;
        sdcard.readData((uint8_t*)&name_len, 2);
        std::string filename(name_len, '\0');
        sdcard.readData((uint8_t*)filename.data(), name_len);
        uint32_t data_len;
        sdcard.readData((uint8_t*)&data_len, 4);
        uint64_t data_offset = sdcard.getPosition();
        Serial.printf("[MAIN]  - Found resource '%s' with size %u\n", filename.c_str(), data_len);

        if (filename == "vocab.b") {
            ctx.tokenizer_resources.vocab = std::make_unique<BinaryVocabView>(data_offset, data_len);
            Serial.println("[MAIN]    -> Created BinaryVocabView for vocab.b.");
        } else if (filename == "merges.b") {
            ctx.tokenizer_resources.merges = std::make_unique<BinaryMergesView>(data_offset, data_len);
             Serial.println("[MAIN]    -> Created BinaryMergesView for merges.b.");
        } else if (filename == "vocab.json") {
            // Загружаем JSON в буфер
            json_buffer.resize(data_len);
            sdcard.seek(data_offset);
            sdcard.readData(json_buffer.data(), data_len);
            Serial.println("[MAIN]    -> Read vocab.json into buffer.");
        }
        sdcard.seek(data_offset + data_len);
    }

    // Если был загружен vocab.json, парсим его
    if (!json_buffer.empty()) {
        Serial.println("[MAIN] Parsing vocab.json...");
        DynamicJsonDocument doc(20480); // Выделим достаточно памяти
        DeserializationError error = deserializeJson(doc, json_buffer.data(), json_buffer.size());

        if (error) {
            Serial.printf("[MAIN][ERROR] deserializeJson() failed: %s\n", error.c_str());
        } else {
            // НЕ ПОДДЕРЖИВАЕТСЯ. Мы не можем создать BinaryVocabView из JSON.
            // Вместо этого, нужно было бы загрузить это в map в памяти,
            // но это съест всю память.
            Serial.println("[MAIN][WARN] vocab.json found, but runtime only supports pre-compiled vocab.b. Tokenizer may fail.");
            // Создаем пустой unique_ptr, чтобы избежать падения на nullptr
            // Это неверный подход, но он покажет, что проблема именно в этом
             ctx.tokenizer_resources.vocab = std::make_unique<BinaryVocabView>(0, 0);
        }
    }
} else {
    Serial.println("[MAIN] RSRC section not found in NAC file.");
}

    if (!ctx.tisa_manifest.empty() && ctx.tokenizer_resources.vocab) {
        ctx.tokenizer = std::make_unique<TISAVM>(ctx.tokenizer_resources);
    }

    ctx.cache_mutex = xSemaphoreCreateMutex();
    return true;
}

// =========================================================================
// 6. UI Функции
// =========================================================================
void draw_keyboard() {
    // Устанавливаем размер шрифта для всей клавиатуры
    tft.setTextSize(1);

    for (int row = 0; row < KEY_ROWS; ++row) {
        for (int col = 0; col < KEY_COLS; ++col) {
            // --- Логика для объединения кнопок ---
            if (col > 0 && strcmp(keyboard_layout[row][col], keyboard_layout[row][col - 1]) == 0) {
                continue;
            }

            const char* key = keyboard_layout[row][col];
            if (strcmp(key, "") == 0) continue;

            // --- Определяем геометрию кнопки ---
            int x = col * (KEY_WIDTH + KEY_MARGIN) + 2;
            int y = row * (KEY_HEIGHT + KEY_MARGIN) + KEYBOARD_Y_OFFSET;
            
            int button_span = 1;
            for (int i = col + 1; i < KEY_COLS; ++i) {
                if (strcmp(keyboard_layout[row][i], key) == 0) button_span++;
                else break;
            }
            int current_key_width = button_span * KEY_WIDTH + (button_span - 1) * KEY_MARGIN;

            // --- Определяем цвета ---
            bool is_start = (strcmp(key, "START") == 0);
            bool is_caps = (strcmp(key, "CAPS") == 0);
            bool is_back = (strcmp(key, "Back") == 0);
            bool is_backspace = (strcmp(key, "<-") == 0);

            uint16_t fill_color = TFT_BLACK;
            uint16_t text_color = TFT_WHITE;

            if (is_caps && g_caps_lock) fill_color = TFT_DARKGREY;
            else if (is_start) fill_color = TFT_DARKGREEN;
            else if (is_back || is_backspace) fill_color = TFT_MAROON;

            // --- Отрисовка фона и рамки ---
            tft.fillRoundRect(x, y, current_key_width, KEY_HEIGHT, 4, fill_color);
            tft.drawRoundRect(x, y, current_key_width, KEY_HEIGHT, 4, TFT_WHITE);

            // Устанавливаем цвет текста И ФОНА для корректной очистки
            tft.setTextColor(text_color, fill_color);

            // --- Отрисовка текста на кнопке ---
            // Для корректного отображения регистра нужно временно создать буфер
            char display_key[10]; // Буфер для текста на кнопке
            if (strlen(key) == 1 && std::isalpha(key[0])) {
                sprintf(display_key, "%c", g_caps_lock ? std::toupper(key[0]) : key[0]);
            } else {
                strcpy(display_key, key);
            }
            
            int16_t text_w = tft.textWidth(display_key);
            int16_t text_h = tft.fontHeight();
            
            int text_x = x + (current_key_width - text_w) / 2;
            int text_y = y + (KEY_HEIGHT - text_h) / 2 + 1;
            
            tft.setCursor(text_x, text_y);
            tft.print(display_key);
        }
    }

    tft.setTextColor(TFT_WHITE, TFT_BLACK);
}

void draw_file_list() {
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(0, 5);
    tft.setTextSize(2);
    tft.setTextColor(TFT_YELLOW);
    tft.println("Select a .nac file:");
    tft.setTextColor(TFT_WHITE);
    tft.setTextSize(1);
    for (size_t i = 0; i < g_nac_files.size(); ++i) {
        tft.setCursor(5, LIST_Y_OFFSET + i * LIST_ITEM_HEIGHT);
        tft.print(g_nac_files[i].c_str());
    }
}

void draw_prompt_screen() {
    tft.fillScreen(TFT_BLACK);
    tft.setTextSize(1);
    tft.setTextColor(TFT_GREEN);
    tft.setCursor(5, 5);
    tft.printf("File: %s", g_selected_nac_path.c_str());
    tft.setTextColor(TFT_YELLOW);
    tft.setCursor(5, 20);
    tft.println("Enter your prompt:");
    
    // Обновленный вызов
    update_prompt_and_cursor(); 

    draw_keyboard();
}

void update_prompt_and_cursor() {
    // 1. Очищаем всю область ввода (включая место для курсора)
    tft.fillRect(5, 40, 310, 30, TFT_BLACK); 
    
    // 2. Устанавливаем цвет и размер для текста промпта
    tft.setCursor(5, 45);
    tft.setTextColor(TFT_CYAN);
    tft.setTextSize(2);

    // 3. Выводим сам текст
    tft.print(g_user_prompt.c_str());

    // 4. Рисуем курсор, если он должен быть видимым
    if (g_cursor_visible) {
        // Вычисляем X-координату конца текста
        int16_t cursor_x = 5 + tft.textWidth(g_user_prompt.c_str());
        // Рисуем курсор в виде тонкого прямоугольника
        tft.fillRect(cursor_x, 45, 2, 16, TFT_CYAN); 
    }
    
    // Сбрасываем цвет текста в стандартный для других элементов UI
    tft.setTextColor(TFT_WHITE);
}

// =========================================================================
// 7. ARDUINO SETUP() и LOOP()
// =========================================================================
void setup() {
    Serial.begin(115200);
    vTaskDelay(pdMS_TO_TICKS(100)); 
    
    Serial.println("\n\n--- NAC Model Executor ---");

    // <<< ИЗМЕНЕНИЕ: Создаем мьютекс и группу событий в САМОМ НАЧАЛЕ >>>
    Serial.println("[MAIN] Creating FreeRTOS objects (early)...");
    g_system_events = xEventGroupCreate();
    g_sd_card_mutex = xSemaphoreCreateMutex(); // Создаем мьютекс здесь
    if (!g_system_events || !g_sd_card_mutex) {
        Serial.println("[MAIN][FATAL] FreeRTOS objects creation FAILED!");
        // Нет смысла продолжать, если мьютекса нет
        while (1) { delay(1000); }
    }
    Serial.println("[MAIN] FreeRTOS objects OK.");

    Serial.println("[MAIN] Initializing TFT display...");
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    Serial.println("[MAIN] TFT OK.");

    Serial.println("[MAIN] Initializing Touchscreen...");
    touch.begin();
    touch.setRotation(1);
    Serial.println("[MAIN] Touchscreen OK.");

    Serial.println("[MAIN] Initializing SD card (call to begin)...");
    sdcard.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
    Serial.println("[MAIN] sdcard.begin() called. Now checking card presence...");
    
    File root = SD.open("/");
    if (!root) {
        Serial.println("[MAIN][ERROR] SD Card mount FAILED! Could not open root directory.");
        tft.setCursor(0,0);
        tft.setTextSize(2);
        tft.setTextColor(TFT_RED);
        tft.println("SD Card FAILED!");
        tft.setTextSize(1);
        tft.println("Check formatting & connection.");
        while(1);
    }
    root.close();
    Serial.println("[MAIN] SD Card OK. Root directory is accessible.");
    
    g_current_state.store(SystemState::FINDING_FILES);
    Serial.println("[MAIN] Setup complete. Moving to FINDING_FILES state.");
}

void loop() {
    switch (g_current_state.load()) {
case SystemState::FINDING_FILES: {
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(0, 0);
    tft.setTextSize(2);
    tft.println("Scanning for .nac files...");
    Serial.println("[MAIN] State: FINDING_FILES. Scanning SD card...");

    g_nac_files.clear();
    
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    Serial.println("[MAIN] Opening root directory...");
    File root = SD.open("/");
    if(root){
        Serial.println("[MAIN] Root directory opened. Iterating files...");
        File file = root.openNextFile();
        while(file){
            std::string fileName = file.name();
            Serial.printf("[MAIN] Found file: %s\n", fileName.c_str());
            if(fileName.length() > 4 && fileName.substr(fileName.length() - 4) == ".nac"){
                if (fileName.rfind("/", 0) == 0) {
                   fileName = fileName.substr(1);
                }
                g_nac_files.push_back(fileName);
                Serial.printf("[MAIN] -> Added '%s' to list.\n", fileName.c_str());
            }
            file.close();
            file = root.openNextFile();
        }
        root.close();
        Serial.println("[MAIN] File iteration complete.");
    } else {
         Serial.println("[MAIN][ERROR] Failed to open root directory!");
    }
    xSemaphoreGive(g_sd_card_mutex);

    if (g_nac_files.empty()) {
        tft.setTextColor(TFT_RED);
        tft.println("No .nac files found!");
        Serial.println("[MAIN][ERROR] No .nac files found on SD card. Halting.");
        while(1);
    } else {
        draw_file_list();
        g_current_state.store(SystemState::SELECTING_FILE);
        Serial.printf("[MAIN] Found %d .nac files. Moving to SELECTING_FILE state.\n", g_nac_files.size());
    }
    break;
}

        case SystemState::SELECTING_FILE: {
            bool touched = false;
            CYD28_TS_Point p;
            
            xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
            if (touch.touched()) {
                p = touch.getPointScaled();
                touched = true;
            }
            xSemaphoreGive(g_sd_card_mutex);

            if (touched) {
                if (p.y > LIST_Y_OFFSET) {
                    int selected_idx = (p.y - LIST_Y_OFFSET) / LIST_ITEM_HEIGHT;
                    if (selected_idx >= 0 && selected_idx < g_nac_files.size()) {
                        g_selected_nac_path = "/" + g_nac_files[selected_idx];
                        draw_prompt_screen();
                        update_prompt_and_cursor();
                        g_current_state.store(SystemState::ENTERING_PROMPT);

                        bool still_pressed = true;
                        while(still_pressed) {
                            xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                            still_pressed = touch.touched();
                            xSemaphoreGive(g_sd_card_mutex);
                            vTaskDelay(pdMS_TO_TICKS(50));
                        }
                    }
                }
            }
            break;
        }
        
        case SystemState::ENTERING_PROMPT: {
            // --- Логика мигания курсора ---
            // Эта часть выполняется в каждом цикле, независимо от нажатия
            if (millis() - g_last_cursor_toggle > CURSOR_BLINK_RATE_MS) {
                g_cursor_visible = !g_cursor_visible; // Инвертируем видимость
                g_last_cursor_toggle = millis();
                update_prompt_and_cursor(); // Перерисовываем промпт с новым состоянием курсора
            }

            // --- Логика обработки нажатий ---
            bool touched = false;
            CYD28_TS_Point p;
            
            xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
            if (touch.touched()) {
                p = touch.getPointScaled();
                touched = true;
            }
            xSemaphoreGive(g_sd_card_mutex);

            if (touched) {
                const char* key_pressed_str = nullptr;
                bool key_found = false;

                // 1. Ищем, какая клавиша была нажата
                for (int row = 0; row < KEY_ROWS; ++row) {
                    for (int col = 0; col < KEY_COLS; ++col) {
                        int x = col * (KEY_WIDTH + KEY_MARGIN) + 2;
                        int y = row * (KEY_HEIGHT + KEY_MARGIN) + KEYBOARD_Y_OFFSET;
                         if (p.x >= x && p.x < (x + KEY_WIDTH) && p.y >= y && p.y < (y + KEY_HEIGHT)) {
                             key_pressed_str = keyboard_layout[row][col];
                             key_found = true;
                             break;
                         }
                    }
                    if (key_found) break;
                }

                // 2. Обрабатываем нажатую клавишу
                if (key_found) {
                    bool needs_prompt_update = false;

                    if (strcmp(key_pressed_str, "<-") == 0) {
                        if (!g_user_prompt.empty()) g_user_prompt.pop_back();
                        needs_prompt_update = true;
                    } else if (strcmp(key_pressed_str, "CAPS") == 0) {
                        g_caps_lock = !g_caps_lock;
                        draw_keyboard(); // Перерисовываем клавиатуру, чтобы показать состояние CAPS
                    } else if (strcmp(key_pressed_str, "Back") == 0) {
                        g_user_prompt = "";
                        g_caps_lock = false;
                        draw_file_list();
                        g_current_state.store(SystemState::SELECTING_FILE);
                    } else if (strcmp(key_pressed_str, "START") == 0) {
                        g_current_state.store(SystemState::EXECUTION_MODE);
                        xEventGroupSetBits(g_system_events, EVT_START_EXECUTION);
                    } else {
                        // ИСПРАВЛЕНИЕ: Этот блок теперь обрабатывает ВСЕ остальные символы, включая пробел
                        char char_to_add = key_pressed_str[0];
                        if (g_caps_lock && std::isalpha(char_to_add)) {
                            char_to_add = std::toupper(char_to_add);
                        }
                        g_user_prompt += char_to_add;
                        needs_prompt_update = true;
                    }
                    
                    if (needs_prompt_update) {
                        // Принудительно показываем курсор и сбрасываем таймер мигания
                        g_cursor_visible = true;
                        g_last_cursor_toggle = millis();
                        update_prompt_and_cursor();
                    }
                }
                
                // 3. Ожидание отпускания пальца
                bool still_pressed = true;
                while(still_pressed) {
                    vTaskDelay(pdMS_TO_TICKS(50));
                    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                    still_pressed = touch.touched();
                    xSemaphoreGive(g_sd_card_mutex);
                }
            }
            break;
        }

        case SystemState::EXECUTION_MODE: {
            EventBits_t bits = xEventGroupWaitBits(g_system_events, EVT_START_EXECUTION, pdTRUE, pdFALSE, portMAX_DELAY);
            if (bits & EVT_START_EXECUTION) {
                tft.fillScreen(TFT_BLACK);
                tft.setCursor(0, 0); tft.setTextSize(2); tft.println("EXECUTION MODE");
                tft.setTextSize(1);
                tft.printf("File: %s\n", g_selected_nac_path.c_str());
                Serial.printf("[MAIN] Starting execution. File: '%s'\n", g_selected_nac_path.c_str());

                NacRuntimeContext* context = new NacRuntimeContext();
                
                // <<< ИЗМЕНЕНИЕ: Открываем файл здесь ОДИН РАЗ >>>
                bool file_opened_successfully = false;
                xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                if (sdcard.openFile(g_selected_nac_path.c_str())) {
                    file_opened_successfully = true;
                }
                xSemaphoreGive(g_sd_card_mutex);

                if (!file_opened_successfully) {
                    tft.setTextColor(TFT_RED);
                    tft.println("Error: Failed to open file!");
                    Serial.println("[MAIN][ERROR] Failed to open NAC file for execution.");
                    // Перезагрузка через 10 секунд
                    vTaskDelay(pdMS_TO_TICKS(10000));
                    ESP.restart();
                    break; // Выход из кейса
                }

                // --- Теперь, когда файл точно открыт, продолжаем ---

                bool is_vision_model = false;
                // Файл уже открыт, просто читаем из него. Мьютекс не нужен, т.к. другие задачи еще не запущены.
                uint8_t header[88];
                sdcard.seek(0);
                if (sdcard.readData(header, 88) == 88) {
                    uint64_t proc_off;
                    memcpy(&proc_off, header + 12 + (6 * 8), sizeof(uint64_t));
                    if (proc_off == 0) is_vision_model = true;
                }

                uint8_t* jpg_buffer = nullptr;
                size_t jpg_size = 0;
                if (is_vision_model) {
                    tft.println("Model type: Vision. Reading image...");
                    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                    File jpgFile = SD.open(g_input_image_path, FILE_READ);
                    if (jpgFile) {
                        jpg_size = jpgFile.size();
                        if (jpg_size > 0) {
                            jpg_buffer = (uint8_t*)alloc_fast(jpg_size);
                            if (jpg_buffer) jpgFile.read(jpg_buffer, jpg_size);
                            else jpg_size = 0;
                        }
                        jpgFile.close();
                    }
                    xSemaphoreGive(g_sd_card_mutex);
                }

                // <<< ИЗМЕНЕНИЕ: Просто вызываем initialize_nac_context, мьютекс не нужен >>>
                 if (initialize_nac_context(*context)) {
                    tft.printf("Ops loaded: %u\n", context->decoded_ops.size());

                    // --- Логика создания входного тензора (vision или language) ---
                    if (is_vision_model) {
                        tft.println("Model type: Vision. Reading image...");
                        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                        File jpgFile = SD.open(g_input_image_path, FILE_READ);
                        if (jpgFile) {
                            jpg_size = jpgFile.size();
                            if (jpg_size > 0) {
                                jpg_buffer = (uint8_t*)alloc_fast(jpg_size);
                                if (jpg_buffer) jpgFile.read(jpg_buffer, jpg_size);
                                else jpg_size = 0;
                            }
                            jpgFile.close();
                        }
                        xSemaphoreGive(g_sd_card_mutex);
    
                        // Создаем тензор для изображения
                        if (jpg_buffer && jpg_size > 0) {
                            Tensor* input_tensor = context->tensor_pool.acquire();
                            input_tensor->dtype = DataType::INT8; 
                            input_tensor->shape = {1, 3, 224, 224};
                            input_tensor->update_from_shape();
                            input_tensor->data = alloc_fast(input_tensor->size);

                            if (input_tensor->data) {
                                g_target_tensor_for_decode = input_tensor;
                                TJpgDec.setJpgScale(1);
                                TJpgDec.setSwapBytes(true);
                                TJpgDec.setCallback(tensor_jpeg_output);
                                JRESULT res = TJpgDec.drawJpg(0, 0, jpg_buffer, jpg_size);
            
                                if (res == JDR_OK) {
                                    context->user_input_tensors.push_back(input_tensor);
                                    Serial.println("[MAIN] Vision input tensor created successfully.");
                                } else {
                                    Serial.printf("[MAIN][ERROR] JPEG decode failed with code: %d\n", res);
                                    context->tensor_pool.release(input_tensor);
                                }
                            } else {
                                Serial.println("[MAIN][ERROR] Failed to allocate memory for vision tensor.");
                                context->tensor_pool.release(input_tensor);
                            }
                            free(jpg_buffer);
                        }
                    } else {
                        // Для языковых моделей, токенизация
                        // Эта функция будет использовать мьютекс внутри себя
                        std::vector<int32_t> token_ids = context->tokenizer->run(context->tisa_manifest, g_user_prompt);

                        // Удаляем токенизатор
                        context->tokenizer.reset(); // Уничтожает TISAVM
    
                        // Удаляем ресурсы токенизатора (vocab, merges и т.д.)
                        context->tokenizer_resources.vocab.reset();
                        context->tokenizer_resources.vocab_idx_for_decode.reset();
                        context->tokenizer_resources.merges.reset();
                        context->tokenizer_resources.byte_map.clear();
                        context->tokenizer_resources.unigram_scores.clear();

                        // Ограничение длины последовательности, как в Python
                        const int max_len = 512;
                        if (token_ids.size() > max_len) {
                            token_ids.resize(max_len);
                        }
                        
                        Serial.printf("[MAIN] Tokenizer returned %d IDs.\n", token_ids.size());

                        if (!token_ids.empty()) {
                            int seq_len = token_ids.size();

                            // --- Создаем тензоры в том же порядке, что и в Python ---
                            
                            // 1. lifted_one (для v0)
                            Tensor* one_tensor = context->tensor_pool.acquire();
                            one_tensor->dtype = DataType::FLOAT32;
                            one_tensor->shape = {1};
                            one_tensor->update_from_shape();
                            one_tensor->data = alloc_fast(one_tensor->size);
                            if (one_tensor->data) {
                                *(static_cast<float*>(one_tensor->data)) = 1.0f;
                                context->user_input_tensors.push_back(one_tensor);
                            } else { context->tensor_pool.release(one_tensor); }

                            // 2. input_ids (для v1)
                            Tensor* ids_tensor = context->tensor_pool.acquire();
                            ids_tensor->dtype = DataType::INT32;
                            ids_tensor->shape = {1, seq_len};
                            ids_tensor->update_from_shape();
                            ids_tensor->data = alloc_fast(ids_tensor->size);
                            if (ids_tensor->data) {
                                memcpy(ids_tensor->data, token_ids.data(), ids_tensor->size);
                                context->user_input_tensors.push_back(ids_tensor);
                            } else { context->tensor_pool.release(ids_tensor); }

                            // 3. attention_mask (для v2)
                            Tensor* mask_tensor = context->tensor_pool.acquire();
                            mask_tensor->dtype = DataType::INT32;
                            mask_tensor->shape = {1, seq_len};
                            mask_tensor->update_from_shape();
                            mask_tensor->data = alloc_fast(mask_tensor->size);
                            if (mask_tensor->data) {
                                int32_t* mask_data = static_cast<int32_t*>(mask_tensor->data);
                                for (int i = 0; i < seq_len; ++i) mask_data[i] = 1;
                                context->user_input_tensors.push_back(mask_tensor);
                            } else { context->tensor_pool.release(mask_tensor); }

                            // 4. position_ids (для v6)
                            Tensor* pos_tensor = context->tensor_pool.acquire();
                            pos_tensor->dtype = DataType::INT32;
                            pos_tensor->shape = {1, seq_len};
                            pos_tensor->update_from_shape();
                            pos_tensor->data = alloc_fast(pos_tensor->size);
                            if (pos_tensor->data) {
                                int32_t* pos_data = static_cast<int32_t*>(pos_tensor->data);
                                for (int i = 0; i < seq_len; ++i) pos_data[i] = i;
                                context->user_input_tensors.push_back(pos_tensor);
                            } else { context->tensor_pool.release(pos_tensor); }

                            Serial.printf("[MAIN] Prepared %d input tensors for the model.\n", context->user_input_tensors.size());
                        }
                    }
                    
                    if (context->user_input_tensors.empty()) { // <<< ИЗМЕНЕНИЕ: Проверяем вектор
                        tft.setTextColor(TFT_RED);
                        tft.println("Error: Input tensors are empty!");
                    } else {
                        tft.println("Running model...");
                        Serial.println("[MAIN] Starting compute and memory tasks...");
                        BaseType_t mem_task_created = xTaskCreatePinnedToCore(nac_memory_task, "nac_memory_task", 6144, context, 8, &g_nac_memory_task_handle, 0); //8192
                        BaseType_t compute_task_created = xTaskCreatePinnedToCore(nac_compute_task, "nac_compute_task", 10240, context, 10, &g_nac_compute_task_handle, 1); //12288
                        
                        if (mem_task_created != pdPASS) {
                            Serial.println("[MAIN][FATAL] Failed to create nac_memory_task! Halting.");
                            tft.setTextColor(TFT_RED);
                            tft.println("FATAL: MEM TASK FAILED");
                            while(1);
                        } else {
                             Serial.println("[MAIN][DIAG] nac_memory_task created successfully.");
                        }

                        if (compute_task_created != pdPASS) {
                            Serial.println("[MAIN][FATAL] Failed to create nac_compute_task! Halting.");
                            tft.setTextColor(TFT_RED);
                            tft.println("FATAL: COMPUTE TASK FAILED");
                            while(1);
                        } else {
                            Serial.println("[MAIN][DIAG] nac_compute_task created successfully.");
                        }
                        //xTaskCreatePinnedToCore(nac_memory_task, "nac_memory_task", 8192, context, 8, &g_nac_memory_task_handle, 0);
                        //xTaskCreatePinnedToCore(nac_compute_task, "nac_compute_task", 12288, context, 10, &g_nac_compute_task_handle, 1);
                        
                        xEventGroupWaitBits(g_system_events, EVT_COMPUTE_TASK_DONE, pdFALSE, pdFALSE, portMAX_DELAY);
                        
                        context->stop_flag.store(true);
                        if (g_nac_memory_task_handle) xTaskNotifyGive(g_nac_memory_task_handle);
                        xEventGroupWaitBits(g_system_events, EVT_MEMORY_TASK_DONE, pdTRUE, pdFALSE, portMAX_DELAY);
                        
                        tft.println("Execution finished!");
                        Serial.println("[MAIN] Execution finished!");
                        
                        // <<< ИЗМЕНЕНИЕ: Блок постобработки и вывода результата >>>
                        Tensor* final_logits_tensor = nullptr;
                        // Ищем последний не-нулевой результат
                        for(int i = context->results.size() - 1; i >= 0; --i) {
                            if (context->results[i] != nullptr) {
                                final_logits_tensor = context->results[i];
                                break;
                            }
                        }

                        if (final_logits_tensor && final_logits_tensor->data && final_logits_tensor->num_elements >= 2) {
                            Serial.println("[MAIN] Found final logits tensor. Processing...");
                            float* logits_data = static_cast<float*>(final_logits_tensor->data);
                            size_t prediction_idx = argmax(logits_data, final_logits_tensor->num_elements);

                            std::map<int, std::string> label_map = {{0, "NEGATIVE"}, {1, "POSITIVE"}};
                            std::string prediction_label = label_map.count(prediction_idx) ? label_map[prediction_idx] : "UNKNOWN";

                            float probabilities[final_logits_tensor->num_elements];
                            softmax(logits_data, probabilities, final_logits_tensor->num_elements);

                            Serial.printf("\n--- Analysis Results for '%s'---\n", g_user_prompt);
                            Serial.printf("  Prediction: %s\n", prediction_label.c_str());
                            Serial.printf("  Confidence (NEGATIVE): %.2f%%\n", probabilities[0] * 100.0f);
                            Serial.printf("  Confidence (POSITIVE): %.2f%%\n", probabilities[1] * 100.0f);
                            
                            tft.println("\n--- Result ---");
                            tft.printf("Prediction: %s\n", prediction_label.c_str());
                            tft.printf("Conf: %.1f%% NEG / %.1f%% POS\n", probabilities[0] * 100.0f, probabilities[1] * 100.0f);
                        } else {
                            Serial.println("[MAIN][ERROR] Could not find a valid final result tensor.");
                            tft.setTextColor(TFT_RED);
                            tft.println("Error: No result found!");
                        }
                    }
                    
                } else {
                    tft.setTextColor(TFT_RED);
                    tft.println("Error: NAC init failed!");
                }
                
                // <<< ИЗМЕНЕНИЕ: Закрываем файл здесь, после того как все задачи завершились >>>
                sdcard.closeFile();
                
                delete context;
                g_nac_compute_task_handle = NULL;
                g_nac_memory_task_handle = NULL;

                tft.setTextColor(TFT_WHITE);
                tft.println("\nRestarting in 10s...");
                vTaskDelay(pdMS_TO_TICKS(10000));
                ESP.restart();
            }
            break;
        }
    }
    vTaskDelay(pdMS_TO_TICKS(20));
}