// =========================================================================
// 1. Библиотеки и заголовочные файлы
// =========================================================================
#include <Arduino.h>
#include "TFT_eSPI_Compat.h"
#include "CYD28_TouchscreenR.h"
#include "CYD28_SD.h"
#include "types.h" // Обновленный types.h содержит enum QuantExecMode
#include "op_kernels.h"
#include "MEP_interpreter.h"  // MEP ISA v1.0 интерпретатор (оркестрация)

#include "TJpg_Decoder.h"

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

// ── STACK SIZE OVERRIDE ─────────────────────────────────────────────────────
// StoreProhibited @EXCVADDR=0x24 root cause: loopTask default stack = 8 KB,
// but initialize_nac_context() alone uses ~10.6 KB → overflow → g_sd_card_mutex
// zeroed → xSemaphoreTake(NULL) → writes at NULL+0x24 → crash.
// Overrides __weak symbol from ESP32 Arduino core main.cpp (available ≥ v2.0).
size_t getArduinoLoopTaskStackSize(void) { return 32768; }  // 32 KB


// =========================================================================
// 2. Глобальные объекты и пины
// =========================================================================
TFT_eSPI tft = TFT_eSPI();
CYD28_TouchR touch(320, 240);

int g_decode_y_offset = 0; // Смещение по Y для записи в тензор

// Global tensor pointer used by TJpgDec callback and MEP h_res_load_dynamic (0x13).
// MEP_interpreter.cpp references this via `extern Tensor* g_target_tensor_for_decode`.
Tensor* g_target_tensor_for_decode = nullptr;

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

// Функция для вывода изображения из буфера на TFT (для отладки)
// Эта функция будет вызываться TJpg_Decoder
bool tft_output(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (y >= tft.height()) return false; // Остановка, если вышли за пределы экрана
    tft.pushImage(x, y, w, h, bitmap);
    return true; // Возвращаем true для продолжения декодирования
}

// TJpgDec callback used by MEP h_res_load_dynamic (opcode 0x13, file_type==3).
// Writes RGB565 tile from JPEG decoder into g_target_tensor_for_decode as
// INT8 CHW (1,3,224,224), normalised to [-128,127] range.
// MEP_interpreter.cpp references this via:
//   extern bool tft_jpeg_output_to_tensor(int16_t,int16_t,uint16_t,uint16_t,uint16_t*);
bool tft_jpeg_output_to_tensor(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
    if (!g_target_tensor_for_decode || !g_target_tensor_for_decode->data) return false;
    if (g_target_tensor_for_decode->shape.size() < 4) return false;

    const int tensor_h = g_target_tensor_for_decode->shape[2]; // 224
    const int tensor_w = g_target_tensor_for_decode->shape[3]; // 224
    int8_t* base = static_cast<int8_t*>(g_target_tensor_for_decode->data);
    int8_t* r_ch = base;
    int8_t* g_ch = base + tensor_h * tensor_w;
    int8_t* b_ch = base + 2 * tensor_h * tensor_w;

    for (int row = 0; row < (int)h; ++row) {
        int dst_row = (int)y + row;
        if (dst_row < 0 || dst_row >= tensor_h) continue;
        for (int col = 0; col < (int)w; ++col) {
            int dst_col = (int)x + col;
            if (dst_col < 0 || dst_col >= tensor_w) continue;
            uint16_t px = bitmap[row * w + col];
            // RGB565 → INT8 (centre at 0): subtract 128 after scaling to [0,255]
            int idx = dst_row * tensor_w + dst_col;
            r_ch[idx] = (int8_t)(((px >> 11) & 0x1F) * 255 / 31 - 128);
            g_ch[idx] = (int8_t)(((px >>  5) & 0x3F) * 255 / 63 - 128);
            b_ch[idx] = (int8_t)(((px      ) & 0x1F) * 255 / 31 - 128);
        }
    }
    return true;
}

void* alloc_fast(size_t size) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;

    // ── Стратегия распределения памяти ──────────────────────────────────────
    // alloc_fast используется ТОЛЬКО для тензорных буферов (не для DMA/стеков).
    // Приоритет зависит от размера:
    //
    //  size < PSRAM_ALLOC_THRESHOLD (малые тензоры, метаданные):
    //      1. SRAM  — быстрее (x5-x10 по сравнению с PSRAM через SPI).
    //      2. PSRAM — overflow когда SRAM исчерпан; допустимо для тензоров.
    //         Если size > доступного SRAM — PSRAM-fallback корректно обрабатывает
    //         этот случай; функция вернёт валидный указатель вне зависимости от
    //         того, хватает ли SRAM конкретно для этого буфера.
    //
    //  size >= PSRAM_ALLOC_THRESHOLD (большие тензоры весов / FP32-активаций):
    //      1. PSRAM — защищаем SRAM-резерв для FreeRTOS стеков и маленьких аллокаций.
    //      2. SRAM  — аварийный fallback, если PSRAM исчерпан.
    //
    // Согласованность с quant_mode:
    //   DEQUANT_ON_LOAD (PSRAM > 512 KB): большие FP32-тензоры (~4× крупнее INT8)
    //     идут через верхнюю ветку прямо в PSRAM — SRAM не занимают.
    //   DEQUANT_EXEC_REQUANT (нет PSRAM): psramFound()==false, обе ветки сводятся
    //     к одному вызову SRAM — идентично простому malloc.

#if CONFIG_SPIRAM_SUPPORT || CONFIG_ESP32_SPIRAM_SUPPORT || defined(BOARD_HAS_PSRAM)
    if (psramFound() && size >= PSRAM_ALLOC_THRESHOLD) {
        // Большой буфер — сначала PSRAM
        ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (ptr) return ptr;
        ESP_LOGW("alloc_fast", "PSRAM full for %u B, falling back to SRAM", (unsigned)size);
        // Аварийный fallback в SRAM — может вытеснить стеки, но лучше чем OOM
        ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!ptr) ESP_LOGE("alloc_fast", "OOM: %u B not available in PSRAM or SRAM", (unsigned)size);
        return ptr;
    }
#endif

    // Малый буфер или нет PSRAM — сначала SRAM
    ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (ptr) return ptr;

    // SRAM исчерпан — overflow в PSRAM
#if CONFIG_SPIRAM_SUPPORT || CONFIG_ESP32_SPIRAM_SUPPORT || defined(BOARD_HAS_PSRAM)
    if (psramFound()) {
        ptr = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (ptr) return ptr;
    }
#endif

    ESP_LOGE("alloc_fast", "OOM: %u B — SRAM free: %u, PSRAM free: %u",
             (unsigned)size,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    return nullptr;
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

/**
 * @brief Создает новый FP32 тензор из квантованного тензора.
 * @note Оригинальный тензор (source_tensor) не изменяется.
 * @param ctx Контекст выполнения.
 * @param source_tensor Квантованный тензор-источник.
 * @return Новый FP32 тензор или nullptr в случае ошибки.
 */
Tensor* create_fp32_copy_from_quantized(NacRuntimeContext& ctx, Tensor* source_tensor) {
    if (!source_tensor || !source_tensor->data || source_tensor->quant_meta.quant_type == 0 || source_tensor->dtype == DataType::FLOAT32) {
        return nullptr; // Уже FP32 или невалидный источник
    }

    size_t num_elem = source_tensor->num_elements;
    Tensor* result_tensor = ctx.tensor_pool.acquire();
    if (!result_tensor) return nullptr;

    result_tensor->shape = source_tensor->shape;
    result_tensor->num_elements = num_elem;
    result_tensor->dtype = DataType::FLOAT32;
    result_tensor->size = num_elem * sizeof(float);
    result_tensor->data = alloc_fast(result_tensor->size);

    if (!result_tensor->data) {
        ESP_LOGE(TAG_COMPUTE, "Failed to allocate memory for temporary FP32 buffer.");
        ctx.tensor_pool.release(result_tensor);
        return nullptr;
    }

    float* float_data = static_cast<float*>(result_tensor->data);
    uint8_t quant_type = source_tensor->quant_meta.quant_type;

    if (quant_type == 2) { // INT8_TENSOR
        int8_t* int8_data = static_cast<int8_t*>(source_tensor->data);
        float   scale     = source_tensor->quant_meta.scale;
        // If min_val was stashed in scales[0] by requantize_result_tensor, use it.
        // Encoding was: int8 = round((fp32 - min_val) / scale) - 128
        // Decoding:     fp32 = (int8 + 128) * scale + min_val
        float min_val = (!source_tensor->quant_meta.scales.empty())
                            ? source_tensor->quant_meta.scales[0]
                            : 0.0f;
        for (size_t i = 0; i < num_elem; ++i) {
            float_data[i] = (static_cast<float>(int8_data[i]) + 128.0f) * scale + min_val;
        }
    } else if (quant_type == 3) { // INT8_CHANNEL
        int8_t* int8_data = static_cast<int8_t*>(source_tensor->data);
        uint8_t axis = source_tensor->quant_meta.axis;
        const std::vector<float>& scales = source_tensor->quant_meta.scales;
        size_t axis_size = (axis < source_tensor->shape.size()) ? source_tensor->shape[axis] : 1;
        size_t stride = 1;
        for (size_t i = axis + 1; i < source_tensor->shape.size(); ++i) {
            stride *= source_tensor->shape[i];
        }
        for (size_t i = 0; i < num_elem; ++i) {
            size_t scale_idx = (i / stride) % axis_size;
            float scale = (scale_idx < scales.size()) ? scales[scale_idx] : 1.0f;
            float_data[i] = static_cast<float>(int8_data[i]) * scale;
        }
    } else if (quant_type == 4) { // BLOCK_FP8
        int8_t* int8_data = static_cast<int8_t*>(source_tensor->data);
        uint16_t block_size = source_tensor->quant_meta.block_size;
        const std::vector<float>& block_scales = source_tensor->quant_meta.block_scales;
        size_t num_blocks = block_scales.size();
        size_t output_idx = 0;
        for (size_t block_idx = 0; block_idx < num_blocks && output_idx < num_elem; ++block_idx) {
            float scale = block_scales[block_idx];
            size_t block_start = block_idx * block_size;
            for (size_t i = 0; i < block_size && output_idx < num_elem; ++i) {
                float_data[output_idx++] = static_cast<float>(int8_data[block_start + i]) * scale;
            }
        }
    } else {
        ESP_LOGW(TAG_COMPUTE, "Unknown quantization type: %u. Skipping copy-dequantization.", quant_type);
        ctx.tensor_pool.release(result_tensor);
        return nullptr;
    }

    return result_tensor;
}

/**
 * @brief Ре-квантизирует FP32 тензор в INT8.
 * @note Выполняет преобразование на месте, освобождая старый FP32 буфер.
 *       Для упрощения, использует динамический min/max для вычисления scale.
 * @param ctx Контекст выполнения.
 * @param tensor Тензор-источник (FP32).
 * @return true в случае успеха, false в случае ошибки.
 */
bool requantize_result_tensor(NacRuntimeContext& ctx, Tensor* tensor) {
    if (!tensor || !tensor->data || tensor->dtype != DataType::FLOAT32) {
        return false;
    }

    size_t num_elem = tensor->num_elements;
    float* fp32_data = static_cast<float*>(tensor->data);

    // ── NaN/Inf scan before quantization ────────────────────────────────────
    {
        size_t scan = std::min(num_elem, (size_t)512);
        size_t nans = 0, infs = 0;
        for (size_t i = 0; i < scan; ++i) {
            if (std::isnan(fp32_data[i])) nans++;
            else if (std::isinf(fp32_data[i])) infs++;
        }
        if (nans || infs)
            Serial.printf("[REQUANT] NaN=%u Inf=%u in tensor (ne=%u, scan=%u) ptr=%p\n",
                          (unsigned)nans, (unsigned)infs,
                          (unsigned)num_elem, (unsigned)scan, (void*)fp32_data);
    }

    // 1. Находим min/max для вычисления scale и zero-point
    float min_val = fp32_data[0];
    float max_val = fp32_data[0];
    for (size_t i = 1; i < num_elem; ++i) {
        if (fp32_data[i] < min_val) min_val = fp32_data[i];
        if (fp32_data[i] > max_val) max_val = fp32_data[i];
    }
    float range = max_val - min_val;
    // scale maps the full FP32 range onto [-128, 127].
    // Encoding:  int8 = round((fp32 - min_val) / scale) - 128
    // Decoding:  fp32 ≈ (int8 + 128) * scale + min_val
    // (equivalent to storing as unsigned [0,255] then re-centering)
    float scale = range > 0.0f ? range / 255.0f : 1.0f;

    // 2. Выделяем новый буфер INT8
    size_t new_byte_size = num_elem;
    int8_t* int8_data = static_cast<int8_t*>(alloc_fast(new_byte_size));
    if (!int8_data) {
        ESP_LOGE(TAG_COMPUTE, "Failed to allocate memory for INT8 requantization buffer.");
        return false;
    }

    // 3. Конвертируем FP32 в INT8 (symmetric around min_val)
    for (size_t i = 0; i < num_elem; ++i) {
        float shifted = (fp32_data[i] - min_val) / scale; // → [0, 255]
        int   q       = (int)roundf(shifted) - 128;       // → [-128, 127]
        int8_data[i]  = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }

    // 4. Обновляем метаданные тензора
    // Сохраняем min_val в поле zero_point через scale и отдельное поле:
    // scale хранит шкалу, min_val — нулевую точку (stored in scales[0] for recovery).
    heap_caps_free(tensor->data);
    tensor->data = int8_data;
    tensor->dtype = DataType::INT8;
    tensor->size = new_byte_size;
    tensor->quant_meta.clear();
    tensor->quant_meta.quant_type = 2; // INT8_TENSOR
    tensor->quant_meta.scale = scale;
    // Stash min_val in scales[0] so create_fp32_copy_from_quantized can recover it.
    tensor->quant_meta.scales.resize(1);
    tensor->quant_meta.scales[0] = min_val;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: parse quantization metadata from an already-loaded buffer.
// Called by both read_and_parse_quant_metadata() and load_tensor_from_disk().
// ─────────────────────────────────────────────────────────────────────────────
// DType encoding from NAC spec §6.1.1:
//  0=float32  1=float64  2=float16  3=bfloat16  4=int32
//  5=int64    6=int16    7=int8     8=uint8      9=bool
static DataType nac_dtype_id_to_datatype(uint8_t id) {
    switch (id) {
        case 0: return DataType::FLOAT32;
        case 4: return DataType::INT32;
        case 7: return DataType::INT8;
        default: return DataType::FLOAT32; // safe fallback
    }
}

static bool parse_quant_meta_from_buffer(Tensor* tensor, const uint8_t* buf, size_t len) {
    if (len == 0) {
        tensor->quant_meta.quant_type = 0;
        return true;
    }
    const uint8_t* p   = buf;
    const uint8_t* end = buf + len;

    uint8_t dtype_id, rank;
    if (p + 2 > end) return false;
    dtype_id = *p++;
    rank     = *p++;

    // ── Populate dtype from the NAC dtype_id ──────────────────────────────────
    // Previously discarded with (void)dtype_id.  This left tensor->dtype at its
    // pool default (FLOAT32) even for INT8 tensors, so dequant code treated them
    // as pre-decoded FP32 and skipped the int8*scale step.
    tensor->dtype = nac_dtype_id_to_datatype(dtype_id);

    // ── Populate shape ────────────────────────────────────────────────────────
    // Previously skipped with p += rank * 4, leaving tensor->shape empty.
    // Streaming tensors never had alloc_fast called, so shape was never set via
    // update_from_shape(), making every guard `shape.size() < 2` fire → nullptr.
    if (p + rank * 4 > end) return false;
    tensor->shape.resize(rank);
    for (uint8_t r = 0; r < rank; ++r) {
        uint32_t dim;
        memcpy(&dim, p, 4); p += 4;
        tensor->shape[r] = (int)dim;
    }
    tensor->update_from_shape();   // sets num_elements and size correctly

    if (p >= end) return false;
    tensor->quant_meta.quant_type = *p++;

    switch (tensor->quant_meta.quant_type) {
        case 2: // INT8_TENSOR
            if (p + 4 > end) return false;
            memcpy(&tensor->quant_meta.scale, p, 4);
            break;
        case 3: { // INT8_CHANNEL
            if (p + 5 > end) return false;
            uint32_t num_scales;
            memcpy(&tensor->quant_meta.axis, p, 1); p += 1;
            memcpy(&num_scales, p, 4);              p += 4;
            if (p + num_scales * 4 > end) return false;
            tensor->quant_meta.scales.resize(num_scales);
            memcpy(tensor->quant_meta.scales.data(), p, num_scales * 4);
            break;
        }
        case 4: { // BLOCK_FP8
            if (p + 3 > end) return false;
            uint8_t orig_rank;
            memcpy(&tensor->quant_meta.block_size, p, 2); p += 2;
            orig_rank = *p++;
            if (p + orig_rank * 4 > end) return false;
            tensor->quant_meta.original_shape.resize(orig_rank);
            if (orig_rank > 0) {
                memcpy(tensor->quant_meta.original_shape.data(), p, orig_rank * 4);
            }
            p += orig_rank * 4;
            if (p + 4 > end) return false;
            uint32_t num_block_scales;
            memcpy(&num_block_scales, p, 4); p += 4;
            if (p + num_block_scales * 4 > end) return false;
            tensor->quant_meta.block_scales.resize(num_block_scales);
            memcpy(tensor->quant_meta.block_scales.data(), p, num_block_scales * 4);
            break;
        }
        default:
            break;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// OPT: Load tensor metadata + data in a SINGLE SD mutex lock, reading
// sequentially (meta_offset → data_offset) to avoid backward seeks.
// Previously: two separate lock/seek/read cycles with a backward seek.
// ─────────────────────────────────────────────────────────────────────────────
static bool load_tensor_from_disk(Tensor* tensor, const TensorLocation& loc) {
    // ── 1. Read metadata (small, stack-friendly) ──────────────────────────
    std::vector<uint8_t> meta_buf;
    if (loc.meta_len > 0) {
        meta_buf.resize(loc.meta_len);
    }

    // ── 2. Allocate data buffer before taking the SD mutex ────────────────
    //    (alloc_fast may be slow on PSRAM; do it outside the critical section)
    tensor->update_from_byte_size(loc.data_size);
    tensor->data = alloc_fast(loc.data_size);
    if (!tensor->data) {
        ESP_LOGE(TAG_MEM, "load_tensor_from_disk: OOM for %llu bytes", (unsigned long long)loc.data_size);
        return false;
    }

    // ── 3. Single SD lock: seek to meta_offset, read meta, then read data ─
    //    meta_offset < data_offset always (file format guarantees this),
    //    so both reads are sequential forward accesses.
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    bool sd_ok = true;
    if (loc.meta_len > 0) {
        sd_ok &= sdcard.seek(loc.meta_offset);
        sd_ok &= (sdcard.readData(meta_buf.data(), loc.meta_len) == loc.meta_len);
    }
    sd_ok &= sdcard.seek(loc.file_offset);
    sd_ok &= (sdcard.readData((uint8_t*)tensor->data, loc.data_size) == (size_t)loc.data_size);
    xSemaphoreGive(g_sd_card_mutex);

    if (!sd_ok) {
        ESP_LOGE(TAG_MEM, "load_tensor_from_disk: SD read error");
        heap_caps_free(tensor->data);
        tensor->data = nullptr;
        return false;
    }

    // ── 4. Parse metadata from buffer (no SD access needed) ───────────────
    return parse_quant_meta_from_buffer(tensor, meta_buf.data(), meta_buf.size());
}

// ── process_mmap_tick ────────────────────────────────────────────────────────
// Выполняет все MMAP-команды для одного тика.
// Вынесена отдельно, чтобы nac_memory_task мог вызывать её в цикле drain,
// обрабатывая сразу несколько пропущенных тиков за одно пробуждение.
static void process_mmap_tick(NacRuntimeContext* ctx, uint32_t tick) {
    auto it = ctx->mmap_schedule.find(tick);
    if (it == ctx->mmap_schedule.end()) return;

    for (const auto& cmd : it->second) {
        switch (cmd.action) {

            // ─── PRELOAD ──────────────────────────────────────────────────────
            // Ключ кэша = target_op_idx (instruction index), а НЕ param_id.
            case MmapAction::PRELOAD: {
                uint16_t target_op_idx = cmd.target_id;
                if (target_op_idx >= ctx->decoded_ops.size()) continue;
                const auto& preload_ins = ctx->decoded_ops[target_op_idx];
                if (preload_ins.A != 2 || preload_ins.B != 1 || preload_ins.C.size() < 2) continue;
                uint16_t param_id = preload_ins.C[1];
                if (param_id >= ctx->param_present.size() || !ctx->param_present[param_id]) continue;
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                bool exists = (target_op_idx < ctx->fast_memory_cache.size() &&
                               ctx->fast_memory_cache[target_op_idx] != nullptr);
                xSemaphoreGive(ctx->cache_mutex);
                if (exists) continue;
                Tensor* tensor = ctx->tensor_pool.acquire();
                if (!load_tensor_from_disk(tensor, ctx->param_locations[param_id])) {
                    ESP_LOGE(TAG_MEM, "PRELOAD: SD read failed for param %u (op %u)", param_id, target_op_idx);
                    ctx->tensor_pool.release(tensor); continue;
                }
                if (ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
                    if (!dequantize_tensor(tensor)) {
                        ESP_LOGE(TAG_MEM, "PRELOAD: dequantize failed for param %u", param_id);
                        ctx->tensor_pool.release(tensor); continue;
                    }
                }
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (target_op_idx >= ctx->fast_memory_cache.size())
                    ctx->fast_memory_cache.resize(target_op_idx + 1, nullptr);
                ctx->fast_memory_cache[target_op_idx] = tensor;
                xSemaphoreGive(ctx->cache_mutex);
                ESP_LOGD(TAG_MEM, "[Tick %u] PRELOADED param %u -> op slot %u", tick, param_id, target_op_idx);
                break;
            }

            // ─── FREE ─────────────────────────────────────────────────────────
            // Зануляем ВСЕ алиасы перед release (FORWARD/op_nac_pass создают алиасы).
            case MmapAction::FREE: {
                uint16_t tid = cmd.target_id;
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                Tensor* to_free = (tid < ctx->results.size()) ? ctx->results[tid] : nullptr;
                if (to_free) {
                    for (auto& rp : ctx->results)           { if (rp == to_free) rp = nullptr; }
                    for (auto& cp : ctx->fast_memory_cache) { if (cp == to_free) cp = nullptr; }
                }
                xSemaphoreGive(ctx->cache_mutex);
                if (to_free) {
                    ctx->tensor_pool.release(to_free);
                    ESP_LOGD(TAG_MEM, "[Tick %u] FREED tensor (target slot %u)", tick, tid);
                }
                break;
            }

            // ─── SAVE_RESULT ──────────────────────────────────────────────────
            // results[tick] должен быть уже заполнен (notify идёт ПОСЛЕ записи).
            case MmapAction::SAVE_RESULT: {
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (tick < ctx->results.size() && ctx->results[tick]) {
                    uint16_t slot_id = cmd.target_id;
                    if (slot_id >= ctx->fast_memory_cache.size())
                        ctx->fast_memory_cache.resize(slot_id + 1, nullptr);
                    ctx->fast_memory_cache[slot_id] = ctx->results[tick];
                    ESP_LOGD(TAG_MEM, "[Tick %u] SAVE_RESULT -> cache slot %u", tick, slot_id);
                } else {
                    ESP_LOGW(TAG_MEM, "[Tick %u] SAVE_RESULT: results[%u] is null", tick, tick);
                }
                xSemaphoreGive(ctx->cache_mutex);
                break;
            }

            // ─── FORWARD ─────────────────────────────────────────────────────
            // results[src] НЕ обнуляем — gather_arguments читает его по D-offset.
            case MmapAction::FORWARD: {
                uint16_t src = (uint16_t)tick;
                uint16_t dst = cmd.target_id;
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (src < ctx->results.size() && ctx->results[src] &&
                    dst < ctx->results.size()) {
                    if (ctx->results[dst] && ctx->results[dst] != ctx->results[src])
                        ctx->tensor_pool.release(ctx->results[dst]);
                    ctx->results[dst] = ctx->results[src];
                    ESP_LOGD(TAG_MEM, "[Tick %u] FORWARD src=%u -> dst=%u", tick, src, dst);
                } else {
                    ESP_LOGW(TAG_MEM, "[Tick %u] FORWARD src=%u dst=%u: oob or null", tick, src, dst);
                }
                xSemaphoreGive(ctx->cache_mutex);
                break;
            }
        }
    }
}

void nac_memory_task(void* pvParameters) {
    auto* ctx = static_cast<NacRuntimeContext*>(pvParameters);

    // ── Защита от пропуска тиков (tick-skipping bug) ──────────────────────────
    //
    // ПРОБЛЕМА: xTaskNotifyGive инкрементирует счётчик уведомлений, а
    // current_instruction_idx хранит только ПОСЛЕДНЕЕ значение.
    // Если compute обработал тики 5, 6, 7 пока memory task спал, счётчик = 3,
    // но load() вернёт 7 — тики 5 и 6 будут пропущены:
    //   FREE   для тиков 5-6 не выполнится → тензоры остаются в памяти → OOM
    //   SAVE_RESULT для 5-6 не выполнится → кэш не заполнен → on-demand загрузки
    //   FORWARD для 5-6 не выполнится → неверные алиасы в results[]
    //
    // РЕШЕНИЕ: last_processed отслеживает последний обработанный тик.
    // При каждом пробуждении дренируем ВСЕ тики от last+1 до current.
    // UINT32_MAX = sentinel «ещё ничего не обработано» → first drain с тика 0.
    uint32_t last_processed = UINT32_MAX;

    ESP_LOGI(TAG_MEM, "Memory Task started.");

    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (ctx->stop_flag.load()) break;

        uint32_t current = ctx->current_instruction_idx.load(std::memory_order_acquire);

        // Дренируем все тики [last+1 .. current].
        uint32_t from = (last_processed == UINT32_MAX) ? 0u : last_processed + 1u;
        for (uint32_t t = from; t <= current; ++t) {
            process_mmap_tick(ctx, t);
        }
        last_processed = current;
    }

    xEventGroupSetBits(g_system_events, EVT_MEMORY_TASK_DONE);
    ESP_LOGI(TAG_MEM, "Memory Task finished.");
    vTaskDelete(NULL);
}

// Reads quantization metadata from SD and parses it.
// NOTE: For bulk tensor loading, prefer load_tensor_from_disk() which
// combines this with the data read under a single SD mutex lock.
bool read_and_parse_quant_metadata(Tensor* tensor, const TensorLocation& loc) {
    if (loc.meta_len == 0) {
        tensor->quant_meta.quant_type = 0;
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

    return parse_quant_meta_from_buffer(tensor, meta_buffer.data(), meta_buffer.size());
}

// Функция деквантизации тензора согласно метаданным (in place)
bool dequantize_tensor(Tensor* tensor) {
    if (!tensor || !tensor->data) return false;
    
    uint8_t quant_type = tensor->quant_meta.quant_type;
    
    if (quant_type == 0 || quant_type == 1) { // none or FP16
        tensor->quant_meta.clear();
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

        const auto& ins = ctx->decoded_ops[idx];

        // --- Отладочный вывод (включить #define NAC_VERBOSE_DEBUG для диагностики) ---
#ifdef NAC_VERBOSE_DEBUG
        {
            char op_name_buf[48];
            if (ins.A == 2) {
                const char* subtypes[] = {"User Data","Weight","State","Constant","Unknown"};
                uint8_t sub = (ins.B <= 3) ? ins.B : 4;
                if (ins.C.size() > 1)
                    snprintf(op_name_buf, sizeof(op_name_buf), "INPUT(%s) ID=%u", subtypes[sub], ins.C[1]);
                else
                    snprintf(op_name_buf, sizeof(op_name_buf), "INPUT(%s)", subtypes[sub]);
            } else if (ins.A >= 10 && ctx->id_to_name_map.count(ins.A)) {
                snprintf(op_name_buf, sizeof(op_name_buf), "%s", ctx->id_to_name_map.at(ins.A).c_str());
            } else {
                snprintf(op_name_buf, sizeof(op_name_buf), "OpID=%u", ins.A);
            }
            Serial.printf("[COMPUTE] Step[%u/%u] %s\n", idx, (unsigned)ctx->decoded_ops.size()-1, op_name_buf);
            Serial.printf("  Mem: %u KB free\n", ESP.getFreeHeap() / 1024);
        }
#endif
        
        Tensor* result_tensor = nullptr;

        if (ins.A == 2) { // <INPUT>
            Tensor* source_tensor = nullptr;
            if (ins.B == 1) { // Загрузка веса
                if (ins.C.size() < 2) continue;
                uint16_t param_id = ins.C[1];

                // ── Ключ кэша = instruction index (idx), а не param_id ──────
                // nac_memory_task PRELOAD хранит тензор по target_op_idx
                // (= индекс B=1 инструкции в decoded_ops), чтобы не
                // пересекаться с SAVE_RESULT, который тоже использует
                // instruction index. Оба потока обязаны использовать
                // один ключ, иначе PRELOAD не будет обнаружен.
                uint16_t cache_key = (uint16_t)idx;
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (cache_key < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[cache_key]) {
                    source_tensor = ctx->fast_memory_cache[cache_key];
                    ctx->fast_memory_cache[cache_key] = nullptr;
                }
                xSemaphoreGive(ctx->cache_mutex);

                if (!source_tensor && param_id < ctx->param_present.size() && ctx->param_present[param_id]) {
                    ESP_LOGD(TAG_COMPUTE, "On-demand load for param %u (op %u)", param_id, idx);
                    source_tensor = ctx->tensor_pool.acquire();
                    const auto& loc = ctx->param_locations[param_id];

                    // OPT: single SD lock reads meta+data sequentially
                    if (!load_tensor_from_disk(source_tensor, loc)) {
                        ESP_LOGE(TAG_COMPUTE, "On-demand load failed for param %u", param_id);
                        ctx->tensor_pool.release(source_tensor); source_tensor = nullptr;
                    } else if (ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
                        if (!dequantize_tensor(source_tensor)) {
                            ESP_LOGE(TAG_COMPUTE, "Failed to dequantize param %u", param_id);
                            ctx->tensor_pool.release(source_tensor); source_tensor = nullptr;
                        }
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
            result_tensor = source_tensor;

        } else if (ins.A >= 10) {
            Tensor* arguments[MAX_INSTRUCTION_ARITY] = {nullptr};
            size_t argc = gather_arguments(*ctx, ins, idx, arguments);
            
            // --- НАЧАЛО: Логика деквантизации/ре-квантизации ---
            Tensor* dequant_arguments[MAX_INSTRUCTION_ARITY] = {nullptr};
            std::vector<Tensor*> temp_dequantized; // Вектор для временных тензоров, которые нужно освободить

            if (ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_PASS_FP || ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT) {
                for (size_t i = 0; i < argc; ++i) {
                    if (arguments[i] && arguments[i]->dtype != DataType::FLOAT32) {
                        Tensor* fp32_copy = create_fp32_copy_from_quantized(*ctx, arguments[i]);
                        if (fp32_copy) {
                            dequant_arguments[i] = fp32_copy;
                            temp_dequantized.push_back(fp32_copy);
                        } else {
                            dequant_arguments[i] = arguments[i];
                        }
                    } else {
                        dequant_arguments[i] = arguments[i];
                    }
                }
            } else {
                for(size_t i = 0; i < argc; ++i) dequant_arguments[i] = arguments[i];
            }
            // --- КОНЕЦ: Логика деквантизации ---

            auto it = g_op_kernels.find(ins.A);
            if (it != g_op_kernels.end()) {
                KernelFunc kernel = it->second;
                result_tensor = kernel(ctx, ins, dequant_arguments, argc);
            } else {
                const char* op_name = ctx->id_to_name_map.count(ins.A) ? ctx->id_to_name_map[ins.A].c_str() : "Unknown";
                ESP_LOGW(TAG_COMPUTE, "Unimplemented op ID=%u (%s). Passing through.", ins.A, op_name);
                result_tensor = op_nac_pass(ctx, ins, dequant_arguments, argc);
            }
            
            // --- НАЧАЛО: Логика освобождения временных тензоров ---
            for(Tensor* temp_tensor : temp_dequantized) {
                ctx->tensor_pool.release(temp_tensor);
            }
            // --- КОНЕЦ: Логика освобождения ---
        }

        // --- НАЧАЛО: Логика ре-квантизации результата ---
        if (result_tensor && ins.A >= 10 && ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT) {
            if (result_tensor->dtype == DataType::FLOAT32) {
                requantize_result_tensor(*ctx, result_tensor);
            }
        }
        // --- КОНЕЦ: Логика ре-квантизации ---

        if(idx < ctx->results.size()) {
            ctx->results[idx] = result_tensor;
            // Нотифицируем nac_memory_task ПОСЛЕ записи results[idx].
            ctx->current_instruction_idx.store(idx, std::memory_order_release);
            if (g_nac_memory_task_handle) xTaskNotifyGive(g_nac_memory_task_handle);
        } else if (result_tensor) {
            ESP_LOGE(TAG_COMPUTE, "Result for op %u at idx %u out of bounds. Releasing.", ins.A, idx);
            ctx->tensor_pool.release(result_tensor);
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
             perm_off = offsets[4], data_off = offsets[5], proc_off = offsets[6], orch_off = offsets[7], rsrc_off = offsets[8];
    (void)orch_off; // ORCH section is parsed by mep_load_from_nac(), not here

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

    // ── CNST section — constant scalars / lists referenced by OPS C-fields ──
    // Format (per NAC spec §5.3):
    //   4-byte tag 'CNST' (already consumed as part of the section header)
    //   uint32  count          — number of records
    //   per record:
    //     uint16  id
    //     uint8   type         0=null 1=bool 2=int64 3=float64 4=string
    //                          5=list[int32] 6=list[float32]
    //     uint16  length       bytes for types 2-4; element count for 5-6;
    //                          ignored for 0,1
    //     <value bytes>
    // Each constant is materialised as a unique_ptr<Tensor> stored at
    // ctx.constants[id].  Kernels receive it via gather_arguments().
    if (cnst_off > 0) {
        if (!sdcard.seek(cnst_off + 4)) return false;   // skip 4-byte 'CNST' tag
        uint32_t cnst_count = 0;
        if (sdcard.readData((uint8_t*)&cnst_count, 4) != 4) return false;

        // Gather all records first so we know the max id before resizing
        struct RawConst {
            uint16_t id;
            uint8_t  type;
            uint16_t length;
            std::vector<uint8_t> data;
        };
        std::vector<RawConst> raw(cnst_count);
        uint16_t max_id = 0;

        for (uint32_t i = 0; i < cnst_count; ++i) {
            RawConst& rc = raw[i];
            if (sdcard.readData((uint8_t*)&rc.id,     2) != 2) return false;
            if (sdcard.readData((uint8_t*)&rc.type,   1) != 1) return false;
            if (sdcard.readData((uint8_t*)&rc.length, 2) != 2) return false;

            size_t byte_count = 0;
            switch (rc.type) {
                case 1: byte_count = 1;                          break; // bool
                case 2: byte_count = 8;                          break; // int64
                case 3: byte_count = 8;                          break; // float64
                case 4: byte_count = rc.length;                  break; // string
                case 5: byte_count = (size_t)rc.length * 4;     break; // list[int32]
                case 6: byte_count = (size_t)rc.length * 4;     break; // list[float32]
                default: byte_count = 0;                         break; // null / unknown
            }
            if (byte_count > 0) {
                rc.data.resize(byte_count);
                if (sdcard.readData(rc.data.data(), byte_count) != byte_count) return false;
            }
            if (rc.id > max_id) max_id = rc.id;
        }

        ctx.constants.resize((size_t)max_id + 1);   // unique_ptr default = nullptr

        for (const RawConst& rc : raw) {
            if (rc.type == 0 || rc.type == 4) {
                // null or string — not representable as a numeric Tensor; leave nullptr
                ctx.constants[rc.id] = nullptr;
                continue;
            }

            auto t_up = std::make_unique<Tensor>();
            Tensor* t  = t_up.get();

            switch (rc.type) {
                case 1: {   // bool → INT32 scalar
                    t->dtype  = DataType::INT32;
                    t->shape  = {1};
                    t->update_from_shape();
                    t->data   = heap_caps_malloc(sizeof(int32_t), MALLOC_CAP_8BIT);
                    if (!t->data) return false;
                    *static_cast<int32_t*>(t->data) = rc.data[0] ? 1 : 0;
                    break;
                }
                case 2: {   // int64 → INT32 scalar (all NAC ops use int* for shape args)
                    int64_t v64 = 0;
                    memcpy(&v64, rc.data.data(), 8);
                    t->dtype  = DataType::INT32;
                    t->shape  = {1};
                    t->update_from_shape();
                    t->data   = heap_caps_malloc(sizeof(int32_t), MALLOC_CAP_8BIT);
                    if (!t->data) return false;
                    *static_cast<int32_t*>(t->data) = (int32_t)v64;
                    break;
                }
                case 3: {   // float64 → FLOAT32 scalar
                    double v64 = 0.0;
                    memcpy(&v64, rc.data.data(), 8);
                    t->dtype  = DataType::FLOAT32;
                    t->shape  = {1};
                    t->update_from_shape();
                    t->data   = heap_caps_malloc(sizeof(float), MALLOC_CAP_8BIT);
                    if (!t->data) return false;
                    *static_cast<float*>(t->data) = (float)v64;
                    break;
                }
                case 5: {   // list[int32]
                    int n    = (int)rc.length;
                    t->dtype = DataType::INT32;
                    t->shape = {n};
                    t->update_from_shape();
                    t->data  = heap_caps_malloc((size_t)n * sizeof(int32_t), MALLOC_CAP_8BIT);
                    if (!t->data) return false;
                    memcpy(t->data, rc.data.data(), (size_t)n * sizeof(int32_t));
                    break;
                }
                case 6: {   // list[float32]
                    int n    = (int)rc.length;
                    t->dtype = DataType::FLOAT32;
                    t->shape = {n};
                    t->update_from_shape();
                    t->data  = heap_caps_malloc((size_t)n * sizeof(float), MALLOC_CAP_8BIT);
                    if (!t->data) return false;
                    memcpy(t->data, rc.data.data(), (size_t)n * sizeof(float));
                    break;
                }
                default:
                    ctx.constants[rc.id] = nullptr;
                    continue;
            }

            ctx.constants[rc.id] = std::move(t_up);
        }
        Serial.printf("[MAIN] CNST: loaded %u constants (max_id=%u).\n",
                      (unsigned)cnst_count, (unsigned)max_id);
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
        size_t ops_section_size = next_section_start - ops_data_start;
                if (ops_section_size > 512u * 1024u) {
                    ESP_LOGE(TAG_MAIN, "ops_section_size %u > 512KB", (unsigned)ops_section_size);
                    return false;
                }
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
    
    // d_model — bytes 10-11 of header (uint16, NAC spec §2.2.4).
    // Not used by the runtime; logged here for diagnostics.
    uint16_t d_model = 0;
    memcpy(&d_model, header + 10, 2);
    if (d_model > 0) Serial.printf("[MAIN] d_model: %u\n", d_model);

    if (data_off > 0) {
        if (!sdcard.seek(data_off + 4)) return false;
        uint32_t param_name_count = 0, input_name_count = 0, num_tensors = 0;

        // ── Block 1: param_id → name mapping ─────────────────────────────
        // NAC spec §6.1 Block 1: uint32 count; per record: uint16 id, uint16 name_len, name
        if (sdcard.readData((uint8_t*)&param_name_count, 4) != 4) return false;
        for (uint32_t i = 0; i < param_name_count; ++i) {
            uint16_t id = 0, len = 0;
            if (sdcard.readData((uint8_t*)&id,  2) != 2) return false;
            if (sdcard.readData((uint8_t*)&len, 2) != 2) return false;
            if (!sdcard.seek(sdcard.getPosition() + len)) return false;
        }

        // ── Block 2: input_index → name mapping ───────────────────────────
        // NAC spec §6.1 Block 2: uint32 count; per record: uint16 index, uint16 name_len, name
        if (sdcard.readData((uint8_t*)&input_name_count, 4) != 4) return false;
        for (uint32_t i = 0; i < input_name_count; ++i) {
            uint16_t id = 0, len = 0;
            if (sdcard.readData((uint8_t*)&id,  2) != 2) return false;
            if (sdcard.readData((uint8_t*)&len, 2) != 2) return false;
            if (!sdcard.seek(sdcard.getPosition() + len)) return false;
        }

        if (store_weights_internally) {
            // ── Block 3: weight tensors ───────────────────────────────────
            // NAC spec §6.1 Block 3: uint32 count; per record:
            //   uint16 p_id, uint32 meta_len, uint64 data_len, metadata, data
            if (sdcard.readData((uint8_t*)&num_tensors, 4) != 4) return false;
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
        uint32_t manifest_size = 0;
        if (sdcard.readData((uint8_t*)&manifest_size, 4) != 4) return false;
        ctx.tisa_manifest.resize(manifest_size);
        if (sdcard.readData(ctx.tisa_manifest.data(), manifest_size) != manifest_size) return false;
    }
    
 if (rsrc_off > 0) {
    Serial.println("[MAIN] Found RSRC section, parsing resources...");
    if (!sdcard.seek(rsrc_off + 4)) return false;
    uint32_t num_files = 0;
    if (sdcard.readData((uint8_t*)&num_files, 4) != 4) return false;
    Serial.printf("[MAIN] RSRC contains %u files.\n", num_files);

    for (uint32_t i = 0; i < num_files; ++i) {
        uint16_t name_len = 0;
        if (sdcard.readData((uint8_t*)&name_len, 2) != 2) return false;
        std::string filename(name_len, '\0');
        if (sdcard.readData((uint8_t*)filename.data(), name_len) != name_len) return false;
        uint32_t data_len = 0;
        if (sdcard.readData((uint8_t*)&data_len, 4) != 4) return false;
        uint64_t data_offset = sdcard.getPosition();
        Serial.printf("[MAIN]  - Found resource '%s' with size %u\n", filename.c_str(), data_len);

        if (filename == "vocab.b") {
            ctx.tokenizer_resources.vocab = std::make_unique<BinaryVocabView>(data_offset, data_len);
            Serial.println("[MAIN]    -> Created BinaryVocabView for vocab.b.");
        } else if (filename == "vidx.b") {
            // Reverse-lookup index: maps token ID → byte offset in vocab.b data section.
            // Required by TISAVM::decode() and by any opcode that looks up by ID.
            ctx.tokenizer_resources.vocab_idx_for_decode =
                std::make_unique<BinaryVocabIndexView>(data_offset, data_len);
            Serial.println("[MAIN]    -> Created BinaryVocabIndexView for vidx.b.");
        } else if (filename == "merges.b") {
            ctx.tokenizer_resources.merges = std::make_unique<BinaryMergesView>(data_offset, data_len);
            Serial.println("[MAIN]    -> Created BinaryMergesView for merges.b.");
        } else if (filename == "vocab.json") {
            // Runtime only supports pre-compiled vocab.b — skip loading this file.
            // Previously this allocated a json_buffer + DynamicJsonDocument(20480),
            // wasting ~20KB of SRAM for a file we cannot use.
            Serial.println("[MAIN]    -> Skipping vocab.json (only vocab.b is supported).");
        }
        sdcard.seek(data_offset + data_len);
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
                         // Проверяем, что нажатие попало в границы кнопки
                         if (p.x >= x && p.x < (x + KEY_WIDTH) && p.y >= y && p.y < (y + KEY_HEIGHT)) {
                             // Учитываем, что большие кнопки занимают несколько ячеек в сетке
                             int current_col = col;
                             const char* key_val = keyboard_layout[row][current_col];
                             if (strcmp(key_val, "") == 0) continue; // Пустые ячейки (объединенные кнопки)
                             
                             // Находим начало объединенной кнопки
                             while(current_col > 0 && strcmp(keyboard_layout[row][current_col], keyboard_layout[row][current_col - 1]) == 0) {
                                current_col--;
                             }
                             key_pressed_str = keyboard_layout[row][current_col];
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

                // ── 1. Открываем NAC-файл ─────────────────────────────────
                // Один дескриптор на всё время выполнения. NAC-граф, параметры
                // и ORCH-байткод читаются через него, защищённые g_sd_card_mutex.
                xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
                bool file_opened_successfully = sdcard.openFile(g_selected_nac_path.c_str());
                xSemaphoreGive(g_sd_card_mutex);

                if (!file_opened_successfully) {
                    tft.setTextColor(TFT_RED);
                    tft.println("Error: Failed to open file!");
                    Serial.println("[MAIN][ERROR] Failed to open NAC file for execution.");
                    delete context;
                    vTaskDelay(pdMS_TO_TICKS(10000)); ESP.restart(); break;
                }

                // ── 2. Инициализируем контекст NAC ───────────────────────
                // Читает MMAP, OPS, CMAP, PERM, DATA, PROC, RSRC секции.
                // Создаёт tokenizer — НЕ уничтожать до завершения MEP-плана:
                // RES_LOAD_EXTERN (0x12) берёт его из ctx->tokenizer.
                if (!initialize_nac_context(*context)) {
                    tft.setTextColor(TFT_RED); tft.println("Error: NAC init failed!");
                    Serial.println("[MAIN][ERROR] initialize_nac_context failed.");
                    sdcard.closeFile(); delete context;
                    vTaskDelay(pdMS_TO_TICKS(10000)); ESP.restart(); break;
                }

                // ── 3. Автовыбор режима квантизации ──────────────────────
                // PSRAM > 512 KB: DEQUANT_ON_LOAD — FP32 в PSRAM, максимальная
                //   скорость, больший расход памяти.
                // SRAM-only: DEQUANT_EXEC_REQUANT — веса INT8, деквантуются
                //   и ре-квантуются per-op (4× меньше промежуточных тензоров,
                //   критично при 320 KB SRAM).
                if (psramFound() && ESP.getFreePsram() > 512 * 1024) {
                    context->quant_mode = QuantExecMode::DEQUANT_ON_LOAD;
                    Serial.println("[MAIN] Quant mode: DEQUANT_ON_LOAD (PSRAM)");
                    tft.println("Quant: ON_LOAD");
                } else {
                    context->quant_mode = QuantExecMode::DEQUANT_EXEC_REQUANT;
                    Serial.println("[MAIN] Quant mode: DEQUANT_EXEC_REQUANT (SRAM-only)");
                    tft.println("Quant: REQUANT");
                }
                tft.printf("Ops: %u\n", (unsigned)context->decoded_ops.size());

                // и вызывает drawJpg(); в легаси-пути vision — тоже используется.

                // ── 4. Выбор пути выполнения ──────────────────────────────
                //
                // ┌─ MEP-ПУТЬ (файл имеет секцию ORCH) ─────────────────────
                // │  MEPInterpreter.run() — полный рецепт выполнения.
                // │  Последовательность инструкций в байткоде:
                // │    0x10 RES_LOAD_MODEL   → no-op (модель уже в ctx)
                // │    0x12 RES_LOAD_EXTERN  → токенизатор из ctx->tokenizer
                // │    0x02 SRC_USER_PROMPT  → g_user_prompt из pre_answers
                // │    0x20 PREPROC_ENCODE   → токенизация
                // │    0x30 TENSOR_CREATE    → attention_mask, position_ids…
                // │    0x80 MODEL_RUN_STATIC → run_model_sync() (inline, sync)
                // │    0x60/0x62             → softmax, argmax постобработка
                // │    0xF0 IO_WRITE         → вывод на Serial + TFT
                // │    0xFE EXEC_RETURN / 0xFF EXEC_HALT
                // │
                // └─ ЛЕГАСИ-ПУТЬ (нет секции ORCH) ─────────────────────────
                //    main.cpp строит входные тензоры вручную и запускает
                //    nac_compute_task (Core 1). Постобработка вручную.
                //
                // Вспомогательный поток nac_memory_task (Core 0) активен
                // ВСЕГДА — он обрабатывает MMAP-расписание для обоих путей.
                // Compute/MEP нотифицируют задачу после записи каждого
                // results[idx], чтобы SAVE_RESULT видел корректный тензор.

                // ── 4a. Запуск постоянного memory-потока ─────────────────
                xEventGroupClearBits(g_system_events, EVT_MEMORY_TASK_DONE);
                g_nac_memory_task_handle = NULL;
                BaseType_t mem_created = xTaskCreatePinnedToCore(
                    nac_memory_task, "NAC_MEM",
                    4096,            // stack bytes
                    context,         // pvParameters
                    5,               // priority
                    &g_nac_memory_task_handle,
                    0                // Core 0 (Arduino loop / MEP run на Core 1)
                );
                if (mem_created != pdPASS) {
                    tft.setTextColor(TFT_RED);
                    tft.println("Error: memory task failed!");
                    Serial.println("[MAIN][ERROR] xTaskCreate(NAC_MEM) failed.");
                    sdcard.closeFile(); delete context;
                    vTaskDelay(pdMS_TO_TICKS(10000)); ESP.restart(); break;
                }
                Serial.println("[MAIN] NAC_MEM task started on Core 0.");

                // Пробуем загрузить ORCH секцию из NAC-файла
                std::vector<uint8_t>       mep_bc;
                std::map<uint16_t, MepVal> mep_consts;
                bool has_orch = mep_load_from_nac(mep_bc, mep_consts, context);

                if (has_orch) {
                    // ════════════════════════════════════════════════════════
                    // MEP-ПУТЬ
                    // g_user_prompt → pre_answers → SRC_USER_PROMPT (0x02)
                    // MODEL_RUN_STATIC (0x80) вызывает run_model_sync(),
                    // который нотифицирует NAC_MEM после каждого results[idx].
                    // ════════════════════════════════════════════════════════
                    Serial.println("[MAIN] ORCH section found — MEP orchestrated path.");
                    tft.println("MEP execution...");

                    MEPInterpreter mep(mep_bc.data(), mep_bc.size(), mep_consts, context);
                    mep.set_pre_answers({ g_user_prompt });
                    mep.run();
                    // Результат выведен инструкциями IO_WRITE внутри плана.

                    Serial.println("[MAIN] MEP execution completed.");
                    tft.println("MEP done.");
                }
                // (else: легаси-путь — nac_compute_task нотифицирует NAC_MEM)

                // ── 4b. Остановка memory-потока ──────────────────────────
                context->stop_flag.store(true);
                if (g_nac_memory_task_handle)
                    xTaskNotifyGive(g_nac_memory_task_handle); // разблокируем из portMAX_DELAY
                xEventGroupWaitBits(g_system_events, EVT_MEMORY_TASK_DONE,
                                    pdTRUE, pdFALSE, pdMS_TO_TICKS(5000));
                g_nac_memory_task_handle = NULL;

                // ── 5. Очистка ────────────────────────────────────────────
                sdcard.closeFile();
                delete context;

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