// =============================================================================
// MEP_interpreter.cpp  —  MEP ISA v1.0 interpreter for ESP32
//
// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
// Licensed under the Apache License, Version 2.0
// =============================================================================
#include "MEP_interpreter.h"
#include "CYD28_SD.h"

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <cmath>
#include <algorithm>
#include <numeric>

// External SD card object declared in main.cpp
extern CYD28_SD sdcard;

// ABI note: TISAVM::decode() signature was changed from vector<int32_t> to
// vector<int> in TISA_VM.h. On xtensa-esp-elf gcc 14, int32_t resolves to
// `long` in this TU but to `int` in TISA_VM.cpp — different mangled symbols.
// `int` is always stable. See TISA_VM.h for the canonical explanation.
static_assert(sizeof(int) == 4, "int must be 32-bit on ESP32/Xtensa");

// TFT for IO_WRITE display output (optional — controlled by MEP_IO_USE_TFT)
#ifndef MEP_IO_USE_TFT
#define MEP_IO_USE_TFT 1
#endif
#if MEP_IO_USE_TFT
#include "TFT_eSPI_Compat.h"
extern TFT_eSPI tft;
#endif

static const char* TAG_MEP = "MEP";

// =============================================================================
// mep_load_from_nac  —  read ORCH section from the open SD file
// =============================================================================
bool mep_load_from_nac(std::vector<uint8_t>&       out_bytecode,
                        std::map<uint16_t, MepVal>&  out_consts,
                        NacRuntimeContext*           ctx)
{
    // Re-read the 88-byte NAC header to locate the ORCH section offset.
    uint8_t header[88];
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(0);
    size_t bytes_read = sdcard.readData(header, 88);
    xSemaphoreGive(g_sd_card_mutex);

    if (bytes_read != 88 || memcmp(header, "NAC", 3) != 0) {
        ESP_LOGE(TAG_MEP, "mep_load_from_nac: invalid NAC header");
        return false;
    }

    // Header layout (NAC spec §2.2):
    //   magic(3) + version(1) + quant_storage(1)
    //   + num_inputs(2) + num_outputs(2) + reserved(1)   ← io_counts area = 5 bytes
    //   + d_model(2)
    //   = 12 bytes total before section offset table
    // offsets[9] = 9 × uint64 = 72 bytes, starts at byte 12
    uint64_t offsets[9];
    memcpy(offsets, header + 12, sizeof(offsets));  // sizeof = 9*8 = 72
    uint64_t orch_off = offsets[7];   // slot 7 = ORCH (spec §2.2.5 table)

    if (orch_off == 0) {
        ESP_LOGI(TAG_MEP, "No ORCH section in this NAC file.");
        return false;
    }

    // ── Parse ORCH section ──────────────────────────────────────────────────
    // Layout: b'ORCH'(4) | bytecode_len(4) | constants_count(4) | bytecode | constants
    uint8_t orch_header[12];
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(orch_off);
    sdcard.readData(orch_header, 12);
    xSemaphoreGive(g_sd_card_mutex);

    if (memcmp(orch_header, "ORCH", 4) != 0) {
        ESP_LOGE(TAG_MEP, "Expected ORCH magic at offset %llu", (unsigned long long)orch_off);
        return false;
    }

    uint32_t bytecode_len, constants_count;
    memcpy(&bytecode_len,    orch_header + 4, 4);
    memcpy(&constants_count, orch_header + 8, 4);

    if (bytecode_len == 0) {
        ESP_LOGI(TAG_MEP, "Empty ORCH section.");
        return false;
    }

    // Read bytecode
    out_bytecode.resize(bytecode_len);
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.readData(out_bytecode.data(), bytecode_len);
    xSemaphoreGive(g_sd_card_mutex);

    // Read constants pool
    // Each entry: uint16 id | uint8 type | uint16 length | payload
    // type codes: 0=None 1=bool 2=int64 3=f64 4=str 5=int[] 6=f[]
    for (uint32_t i = 0; i < constants_count; ++i) {
        uint8_t entry_hdr[5];
        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
        sdcard.readData(entry_hdr, 5);

        uint16_t const_id; uint8_t type_code; uint16_t length;
        memcpy(&const_id, entry_hdr,     2);
        memcpy(&type_code, entry_hdr+2,  1);
        memcpy(&length,   entry_hdr+3,   2);

        MepVal val;
        switch (type_code) {
            case 0: // None
                val.set_none();
                break;
            case 1: { // bool (1 byte)
                uint8_t b; sdcard.readData(&b, 1);
                val.set_bool(b != 0);
                break;
            }
            case 2: { // int64
                int64_t v; sdcard.readData((uint8_t*)&v, 8);
                val.set_i64(v);
                break;
            }
            case 3: { // float64
                double v; sdcard.readData((uint8_t*)&v, 8);
                val.set_f64(v);
                break;
            }
            case 4: { // UTF-8 string
                std::string s(length, '\0');
                sdcard.readData((uint8_t*)s.data(), length);
                val.set_str(s);
                break;
            }
            case 5: { // int32[] — store as INT32 tensor
                std::vector<int32_t> arr(length);
                if (length > 0) sdcard.readData((uint8_t*)arr.data(), length * 4);
                if (ctx && length > 0) {
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = DataType::INT32;
                        t->shape = {1, (int)length};
                        t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) {
                            memcpy(t->data, arr.data(), length * sizeof(int32_t));
                            val.set_tensor(t);
                        } else {
                            ctx->tensor_pool.release(t);
                            val.set_i64(arr[0]); // fallback
                        }
                    } else {
                        val.set_i64(arr[0]); // fallback
                    }
                } else if (length > 0) {
                    val.set_i64((int64_t)arr[0]); // no ctx: store first element
                }
                break;
            }
            case 6: { // float32[] — store as FLOAT32 tensor
                std::vector<float> arr(length);
                if (length > 0) sdcard.readData((uint8_t*)arr.data(), length * 4);
                if (ctx && length > 0) {
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = DataType::FLOAT32;
                        t->shape = {1, (int)length};
                        t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) {
                            memcpy(t->data, arr.data(), length * sizeof(float));
                            val.set_tensor(t);
                        } else {
                            ctx->tensor_pool.release(t);
                            val.set_f64((double)arr[0]); // fallback
                        }
                    } else {
                        val.set_f64((double)arr[0]); // fallback
                    }
                } else if (length > 0) {
                    val.set_f64((double)arr[0]); // no ctx: store first element
                }
                break;
            }
            default:
                // Unknown type — skip payload
                sdcard.seek(sdcard.getPosition() + length);
                break;
        }
        xSemaphoreGive(g_sd_card_mutex);

        out_consts[const_id] = val;
    }

    ESP_LOGI(TAG_MEP, "ORCH loaded: %u bytes bytecode, %u constants.", bytecode_len, constants_count);
    return true;
}

// =============================================================================
// MEPInterpreter  —  constructor / destructor
// =============================================================================

MEPInterpreter::MEPInterpreter(const uint8_t*                   bytecode,
                                size_t                            bytecode_len,
                                const std::map<uint16_t, MepVal>& constants,
                                NacRuntimeContext*                primary_ctx)
    : m_plan(bytecode), m_plan_size(bytecode_len),
      m_consts(constants), m_primary_ctx(primary_ctx)
{
    ESP_LOGI(TAG_MEP, "MEPInterpreter ready: %u bytes bytecode, %u constants.",
             (unsigned)bytecode_len, (unsigned)constants.size());
}

MEPInterpreter::~MEPInterpreter() {
    // Release any tensors still sitting in context slots.
    for (int k = 0; k < 256; ++k) {
        if (m_ctx[k].type == MepValType::TENSOR && m_ctx[k].tensor) {
            m_primary_ctx->tensor_pool.release(m_ctx[k].tensor);
            m_ctx[k].tensor = nullptr;
        }
        m_ctx[k]._free_str();
    }
}

// =============================================================================
// Bytecode readers
// =============================================================================
uint8_t  MEPInterpreter::ru8()  { return m_plan[m_ip++]; }
uint16_t MEPInterpreter::ru16() {
    uint16_t v; memcpy(&v, m_plan + m_ip, 2); m_ip += 2; return v;
}
int16_t  MEPInterpreter::ri16() {
    int16_t v; memcpy(&v, m_plan + m_ip, 2); m_ip += 2; return v;
}

// =============================================================================
// Helpers
// =============================================================================

const MepVal& MEPInterpreter::get_const(uint16_t id) const {
    static const MepVal s_none;
    auto it = m_consts.find(id);
    return (it != m_consts.end()) ? it->second : s_none;
}

void MEPInterpreter::release_slot(uint8_t k) {
    MepVal& v = m_ctx[k];
    if (v.type == MepValType::TENSOR && v.tensor) {
        m_primary_ctx->tensor_pool.release(v.tensor);
        v.tensor = nullptr;
    }
    v._free_str();
    v.type = MepValType::NONE;
}

void MEPInterpreter::put_tensor(uint8_t k, Tensor* t) {
    release_slot(k);
    m_ctx[k].set_tensor(t);
}

Tensor* MEPInterpreter::new_tensor(DataType dt, const std::vector<int>& shape) {
    Tensor* t = m_primary_ctx->tensor_pool.acquire();
    if (!t) return nullptr;
    t->dtype = dt;
    t->shape = shape;
    t->update_from_shape();
    if (t->num_elements > 0) {
        t->data = alloc_fast(t->size);
        if (!t->data) {
            m_primary_ctx->tensor_pool.release(t);
            return nullptr;
        }
    }
    return t;
}

// ── Instruction length table (opcode byte included) ──────────────────────────
uint32_t MEPInterpreter::instr_len(uint32_t ip) const {
    if (ip >= m_plan_size) return 1;
    uint8_t op = m_plan[ip];
    switch (op) {
        case 0x02: return 5;   // out(1)+dtype(1)+const_id(2)
        case 0x04: return 4;   // out(1)+const_id(2)
        case 0x10: return 4;   // model_id(1)+path_const_id(2)
        case 0x11: return 5;   // out(1)+file_type(1)+path_const_id(2)
        case 0x12: return 5;   // out(1)+res_type(1)+res_id_const_id(2)
        case 0x13: return 4;   // out(1)+path_key(1)+file_type(1)
        case 0x1F: return 3;   // res_type(1)+id_or_key(1)
        case 0x20: return 4;   // proc(1)+in(1)+out(1)
        case 0x21: return 4;
        case 0x22: return 5;   // proc(1)+item_const_id(2)+out(1)
        case 0x2A: {            // out(1)+fmt_const_id(2)+count(1)+N*key(1)
            if (ip + 5 > m_plan_size) return 1;
            return 1 + 1 + 2 + 1 + m_plan[ip + 4];
        }
        case 0x30: return 5;   // out(1)+dtype(1)+ctype(1)+src_key(1)
        case 0x38: return 6;   // op(1)+out(1)+in(1)+pad_w_key(1)+pad_v_key(1)
        case 0x39: {            // op(1)+out(1)+count(1)+N*key(1)+axis_key(1)
            if (ip + 4 > m_plan_size) return 1;
            return 1 + 1 + 1 + 1 + m_plan[ip + 3] + 1;
        }
        case 0x3A: {            // op(1)+out(1)+in(1)[+dim_key(1)]
            if (ip + 2 > m_plan_size) return 1;
            return 1 + 3 + (m_plan[ip + 1] == 1 ? 1 : 0);
        }
        case 0x3B: return 4;   // out(1)+in_tensor(1)+in_idx(1)
        case 0x59: return 3;   // out(1)+in(1)
        case 0x5F: return 4;   // key(1)+msg_const_id(2)
        case 0x60: return 4;   // op(1)+out(1)+in(1)
        case 0x61: return 5;   // op(1)+out(1)+k1(1)+k2(1)
        case 0x62: return 4;   // op(1)+out(1)+in(1)
        case 0x68: return 5;   // op(1)+out(1)+k1(1)+k2(1)
        case 0x70: return 5;   // in(1)+k_key(1)+out_idx(1)+out_val(1)
        case 0x71: return 5;   // logits(1)+temp(1)+topk(1)+out(1)
        case 0x80: {
            if (ip + 3 > m_plan_size) return 1;
            uint8_t ci = m_plan[ip + 2];
            if (ip + 3 + ci >= m_plan_size) return 1;
            uint8_t co = m_plan[ip + 3 + ci];
            return 1 + 1 + 1 + ci + 1 + co;
        }
        case 0xA0: return 2;   // counter_key(1)
        case 0xA1: return 3;   // jump_offset(2)
        case 0xA8: return 4;   // cond_key(1)+jump_offset(2)
        case 0xA9: return 4;   // cond_key(1)+jump_offset(2)
        case 0xE0: return 4;   // out(1)+in(1)+format_type(1)
        case 0xF0: return 5;   // in(1)+dest_type(1)+dest_key(1)+write_mode(1)
        case 0xFE: {            // count(1)+N*key(1)
            if (ip + 2 > m_plan_size) return 1;
            return 1 + 1 + m_plan[ip + 1];
        }
        case 0xFF: return 1;
        default:   return 1;
    }
}

// =============================================================================
// Opcode name helper (for debug trace only)
// =============================================================================
static const char* mep_opcode_name(uint8_t op) {
    switch (op) {
        case 0x02: return "SRC_USER_PROMPT";
        case 0x04: return "SRC_CONSTANT";
        case 0x10: return "RES_LOAD_MODEL";
        case 0x11: return "RES_LOAD_DATAFILE";
        case 0x12: return "RES_LOAD_EXTERN";
        case 0x13: return "RES_LOAD_DYNAMIC";
        case 0x1F: return "RES_UNLOAD";
        case 0x20: return "PREPROC_ENCODE";
        case 0x21: return "PREPROC_DECODE";
        case 0x22: return "PREPROC_GET_ID";
        case 0x2A: return "STRING_FORMAT";
        case 0x30: return "TENSOR_CREATE";
        case 0x38: return "TENSOR_MANIPULATE";
        case 0x39: return "TENSOR_COMBINE";
        case 0x3A: return "TENSOR_INFO";
        case 0x3B: return "TENSOR_EXTRACT";
        case 0x59: return "SYS_COPY";
        case 0x5F: return "SYS_DEBUG_PRINT";
        case 0x60: return "MATH_UNARY";
        case 0x61: return "MATH_BINARY";
        case 0x62: return "MATH_AGGREGATE";
        case 0x68: return "LOGIC_COMPARE";
        case 0x70: return "ANALYSIS_TOP_K";
        case 0x71: return "ANALYSIS_SAMPLE";
        case 0x80: return "MODEL_RUN_STATIC";
        case 0xA0: return "FLOW_LOOP_START";
        case 0xA1: return "FLOW_LOOP_END";
        case 0xA8: return "FLOW_BRANCH_IF";
        case 0xA9: return "FLOW_BREAK_LOOP_IF";
        case 0xE0: return "SERIALIZE_OBJECT";
        case 0xF0: return "IO_WRITE";
        case 0xFE: return "EXEC_RETURN";
        case 0xFF: return "EXEC_HALT";
        default:   return "UNKNOWN";
    }
}

// =============================================================================
// Main execution loop
// =============================================================================
MepVal MEPInterpreter::run() {
    ESP_LOGI(TAG_MEP, "--- MEP execution started ---");
    m_ip      = 0;
    m_running = true;

    while (m_ip < m_plan_size && m_running) {
        uint8_t op     = ru8();
        uint32_t ip_before = m_ip - 1;  // ip before reading op
        uint32_t heap  = (uint32_t)esp_get_free_heap_size();
        uint32_t stack = (uint32_t)uxTaskGetStackHighWaterMark(NULL);
        Serial.printf("[MEP] ip=%3u op=0x%02X %-22s heap=%6u stack_hwm=%5u\n",
                      ip_before, op, mep_opcode_name(op), heap, stack);

        switch (op) {
            case 0x02: h_src_user_prompt();    break;
            case 0x04: h_src_constant();       break;
            case 0x10: h_res_load_model();     break;
            case 0x11: h_res_load_datafile();  break;
            case 0x12: h_res_load_extern();    break;
            case 0x13: h_res_load_dynamic();   break;
            case 0x1F: h_res_unload();         break;
            case 0x20: h_preproc_encode();     break;
            case 0x21: h_preproc_decode();     break;
            case 0x22: h_preproc_get_id();     break;
            case 0x2A: h_string_format();      break;
            case 0x30: h_tensor_create();      break;
            case 0x38: h_tensor_manipulate();  break;
            case 0x39: h_tensor_combine();     break;
            case 0x3A: h_tensor_info();        break;
            case 0x3B: h_tensor_extract();     break;
            case 0x59: h_sys_copy();           break;
            case 0x5F: h_sys_debug_print();    break;
            case 0x60: h_math_unary();         break;
            case 0x61: h_math_binary();        break;
            case 0x62: h_math_aggregate();     break;
            case 0x68: h_logic_compare();      break;
            case 0x70: h_analysis_top_k();     break;
            case 0x71: h_analysis_sample();    break;
            case 0x80: h_model_run_static();   break;
            case 0xA0: h_flow_loop_start();    break;
            case 0xA1: h_flow_loop_end();      break;
            case 0xA8: h_flow_branch_if();     break;
            case 0xA9: h_flow_break_loop_if(); break;
            case 0xE0: h_serialize_object();   break;
            case 0xF0: h_io_write();           break;
            case 0xFE: h_exec_return();        break;
            case 0xFF: h_exec_halt();          break;
            default:
                ESP_LOGW(TAG_MEP, "Unknown MEP opcode 0x%02X at ip=%u — skipping.", op, m_ip - 1);
                break;
        }
        Serial.printf("[MEP] ip=%3u op=0x%02X DONE  heap=%6u\n",
                      ip_before, op, (unsigned)esp_get_free_heap_size());
    }

    ESP_LOGI(TAG_MEP, "--- MEP execution finished ---");
    return m_return_val;
}

// =============================================================================
// 0x02  SRC_USER_PROMPT
// =============================================================================
void MEPInterpreter::h_src_user_prompt() {
    uint8_t out_key = ru8();
    /*data_type =*/ ru8();
    uint16_t prompt_id = ru16();

    const char* prompt_text = get_const(prompt_id).as_cstr();

    if (!m_pre_answers.empty()) {
        Serial.printf("[MEP] %s%s\n", prompt_text, m_pre_answers.front().c_str());
        release_slot(out_key);
        slot(out_key).set_str(m_pre_answers.front());
        m_pre_answers.erase(m_pre_answers.begin());
    } else {
        // No pre-answer: store empty string
        ESP_LOGW(TAG_MEP, "SRC_USER_PROMPT: no pre-answer available for '%s'", prompt_text);
        release_slot(out_key);
        slot(out_key).set_str("");
    }
}

// =============================================================================
// 0x04  SRC_CONSTANT
// =============================================================================
void MEPInterpreter::h_src_constant() {
    uint8_t  out_key  = ru8();
    uint16_t const_id = ru16();
    release_slot(out_key);
    m_ctx[out_key] = get_const(const_id); // value copy (strings are re-malloc'd if needed)
    if (get_const(const_id).type == MepValType::STRING) {
        m_ctx[out_key].set_str(get_const(const_id).as_cstr()); // deep copy the string
    }
}

// =============================================================================
// 0x10  RES_LOAD_MODEL
// Maps a model_id → primary_ctx.  On ESP32 only one model fits in RAM.
// =============================================================================
void MEPInterpreter::h_res_load_model() {
    /*model_id    =*/ ru8();
    /*path_const  =*/ ru16();
    // We already have the model loaded as m_primary_ctx.
    // Nothing to do; MODEL_RUN_STATIC always uses m_primary_ctx.
    ESP_LOGI(TAG_MEP, "RES_LOAD_MODEL: using already-loaded primary context.");
}

// =============================================================================
// 0x11  RES_LOAD_DATAFILE  (stub — .npy not commonly used on ESP32)
// =============================================================================
void MEPInterpreter::h_res_load_datafile() {
    uint8_t  out_key   = ru8();
    /*file_type =*/      ru8();
    /*path_id   =*/      ru16();
    (void)out_key;
    ESP_LOGW(TAG_MEP, "RES_LOAD_DATAFILE not implemented on ESP32 (use RES_LOAD_DYNAMIC).");
}

// =============================================================================
// 0x12  RES_LOAD_EXTERN  —  extract tokenizer from loaded model
// =============================================================================
void MEPInterpreter::h_res_load_extern() {
    uint8_t  out_key  = ru8();
    uint8_t  res_type = ru8();
    /*res_id_const_id =*/ ru16();

    if (res_type == 0) { // type 0 = extract TISAVM from model
        if (!m_primary_ctx->tokenizer) {
            ESP_LOGE(TAG_MEP, "RES_LOAD_EXTERN: primary_ctx has no tokenizer.");
            return;
        }
        release_slot(out_key);
        slot(out_key).set_opaque(m_primary_ctx->tokenizer.get());
        ESP_LOGI(TAG_MEP, "RES_LOAD_EXTERN: tokenizer extracted to slot %u.", out_key);
    } else {
        ESP_LOGW(TAG_MEP, "RES_LOAD_EXTERN: res_type %u not supported.", res_type);
    }
}

// =============================================================================
// 0x13  RES_LOAD_DYNAMIC  —  load image (file_type=3) from runtime path
// =============================================================================
void MEPInterpreter::h_res_load_dynamic() {
    uint8_t out_key   = ru8();
    uint8_t path_key  = ru8();
    uint8_t file_type = ru8();

    const char* path = slot(path_key).as_cstr();
    if (!path || !path[0]) {
        ESP_LOGE(TAG_MEP, "RES_LOAD_DYNAMIC: empty path in slot %u.", path_key);
        return;
    }

    if (file_type == 3) { // Image + ImageNet preprocessing → (1,3,224,224) INT8
        // Allocate target tensor
        Tensor* t = new_tensor(DataType::INT8, {1, 3, 224, 224});
        if (!t) { ESP_LOGE(TAG_MEP, "RES_LOAD_DYNAMIC: OOM for image tensor."); return; }

        // Use TJpg_Decoder path same as main.cpp
        extern Tensor* g_target_tensor_for_decode;
        extern bool tft_jpeg_output_to_tensor(int16_t,int16_t,uint16_t,uint16_t,uint16_t*);
        g_target_tensor_for_decode = t;

        // Open image from SD
        // (Requires TJpg_Decoder to be set up by main.cpp)
        ESP_LOGW(TAG_MEP, "RES_LOAD_DYNAMIC: image loading requires TJpgDec — ensure callback is set.");
        put_tensor(out_key, t);
    } else if (file_type == 2) { // .npy — not supported (no libnpy on ESP32)
        ESP_LOGW(TAG_MEP, "RES_LOAD_DYNAMIC: .npy loading not supported.");
    } else {
        ESP_LOGW(TAG_MEP, "RES_LOAD_DYNAMIC: unknown file_type %u.", file_type);
    }
}

// =============================================================================
// 0x1F  RES_UNLOAD
// =============================================================================
void MEPInterpreter::h_res_unload() {
    uint8_t res_type = ru8();
    uint8_t id_key   = ru8();
    if (res_type == 1) { // context slot
        release_slot(id_key);
    }
    // res_type == 0 (model): nothing to do — primary_ctx stays alive
}

// =============================================================================
// 0x20  PREPROC_ENCODE  —  tokenize text
// =============================================================================
void MEPInterpreter::h_preproc_encode() {
    uint8_t proc_key = ru8();
    uint8_t in_key   = ru8();
    uint8_t out_key  = ru8();

    TISAVM* tok = slot(proc_key).as_tokenizer();
    if (!tok) { ESP_LOGE(TAG_MEP, "PREPROC_ENCODE: slot %u is not a tokenizer.", proc_key); return; }

    const char* text = slot(in_key).as_cstr();
    std::string text_str(text ? text : "");

    // TISAVM::run(manifest, text) → std::vector<int32_t>
    std::vector<int32_t> ids =
        tok->run(m_primary_ctx->tisa_manifest, text_str);

    if (ids.empty()) {
        ESP_LOGW(TAG_MEP, "PREPROC_ENCODE: tokenizer returned empty result.");
        return;
    }

    int seq_len = (int)ids.size();
    Tensor* t = new_tensor(DataType::INT32, {1, seq_len});
    if (!t) { ESP_LOGE(TAG_MEP, "PREPROC_ENCODE: OOM."); return; }
    memcpy(t->data, ids.data(), ids.size() * sizeof(int32_t));
    put_tensor(out_key, t);

    // ── Вывод результата токенизации ─────────────────────────────────────────
    // Печатаем массив ID одной строкой, затем построчно каждый токен с текстом.
    // Текст берём через decode() на одиночный ID — требует одного SD-чтения
    // на токен через vidx.b + vocab.b (загружены в RSRC).
    Serial.printf("[TOKENIZE] seq_len=%d  ids=[", seq_len);
    for (int i = 0; i < seq_len; ++i)
        Serial.printf("%s%d", i ? ", " : "", ids[i]);
    Serial.println("]");

    // Побайтовый decode использует vocab_idx_for_decode — проверяем наличие.
    bool has_decode = (m_primary_ctx->tokenizer_resources.vocab_idx_for_decode != nullptr &&
                       m_primary_ctx->tokenizer_resources.vocab   != nullptr);
    for (int i = 0; i < seq_len; ++i) {
        Serial.printf("[TOKENIZE]   [%2d] %5d", i, ids[i]);
        if (has_decode) {
            // decode ожидает vector<int>, конвертируем один ID
            std::string tok_text = tok->decode({(int)ids[i]},
                                               /*skip_special_tokens=*/false);
            Serial.printf("  \"%s\"", tok_text.c_str());
        }
        Serial.println();
    }
    ESP_LOGI(TAG_MEP, "PREPROC_ENCODE: %d tokens → slot %u.", seq_len, out_key);
}

// =============================================================================
// 0x21  PREPROC_DECODE  —  detokenize
// =============================================================================
void MEPInterpreter::h_preproc_decode() {
    uint8_t proc_key = ru8();
    uint8_t in_key   = ru8();
    uint8_t out_key  = ru8();

    TISAVM* tok = slot(proc_key).as_tokenizer();
    if (!tok) { ESP_LOGE(TAG_MEP, "PREPROC_DECODE: slot %u is not a tokenizer.", proc_key); return; }

    Tensor* t = slot(in_key).tensor;
    if (!t || !t->data) { ESP_LOGE(TAG_MEP, "PREPROC_DECODE: no tensor in slot %u.", in_key); return; }

    // Use vector<int> to match TISA_VM.h decode() signature exactly.
    // int is stable across all TUs; int32_t is not (see TISA_VM.h note).
    std::vector<int> ids(t->num_elements);
    if (t->dtype == DataType::INT32) {
        const int32_t* src = static_cast<const int32_t*>(t->data);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(src[i]);
    } else if (t->dtype == DataType::FLOAT32) {
        const float* f = static_cast<const float*>(t->data);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(f[i]);
    }

    std::string decoded = tok->decode(ids);
    release_slot(out_key);
    slot(out_key).set_str(decoded);
}

// =============================================================================
// 0x22  PREPROC_GET_ID  —  get special token ID (e.g. <eos>)
// =============================================================================
void MEPInterpreter::h_preproc_get_id() {
    uint8_t  proc_key      = ru8();
    uint16_t item_const_id = ru16();
    uint8_t  out_key       = ru8();

    (void)proc_key; // tokenizer slot — lookup goes through vocab directly on ESP32

    const char* token_str = get_const(item_const_id).as_cstr();
    release_slot(out_key);

    int32_t tid = -1;
    if (token_str && token_str[0] != '\0' && m_primary_ctx->tokenizer_resources.vocab) {
        m_primary_ctx->tokenizer_resources.vocab->find(std::string(token_str), tid);
    }
    slot(out_key).set_i64((int64_t)tid);
    ESP_LOGD(TAG_MEP, "PREPROC_GET_ID: '%s' → %d", token_str ? token_str : "", tid);
}

// =============================================================================
// 0x2A  STRING_FORMAT
// =============================================================================
void MEPInterpreter::h_string_format() {
    uint8_t  out_key         = ru8();
    uint16_t format_const_id = ru16();
    uint8_t  count           = ru8();
    std::vector<uint8_t> keys(count);
    for (uint8_t i = 0; i < count; ++i) keys[i] = ru8();

    const char* fmt = get_const(format_const_id).as_cstr();
    if (!fmt) { release_slot(out_key); slot(out_key).set_str(""); return; }

    // Python-style positional replacement.
    // Supports: {}  {:<width>}  {:.Nf}  {:.N%}  {:d}  {:s}  etc.
    // The MEP compiler may emit bare {} or Python format specs inside braces.
    std::string result;
    result.reserve(256);
    int arg_idx = 0;
    const char* p = fmt;
    while (*p) {
        if (*p != '{') { result += *p++; continue; }

        // Find the matching closing brace
        const char* close = strchr(p + 1, '}');
        if (!close) { result += *p++; continue; }  // unmatched '{', copy literally

        // Parse the optional format spec between '{' and '}'
        // spec sits after an optional ':', e.g.  {}  {:d}  {:.2f}  {:.2%}
        const char* spec = p + 1;                   // points to ':' or '}'
        if (*spec == ':') spec++;                   // skip ':'

        // Precision — digits after '.', e.g. ".2" → 2
        int prec = 6;       // default for floats (matches old %.6g behaviour)
        bool has_prec = false;
        if (*spec == '.') {
            char* end_ptr = nullptr;
            long pv = strtol(spec + 1, &end_ptr, 10);
            if (end_ptr && end_ptr > spec + 1) { prec = (int)pv; has_prec = true; spec = end_ptr; }
        }

        // Conversion character (last non-'}' in spec)
        char conv = *spec;   // could be 'f', 'd', 's', '%', 'g', 'e', '\0', etc.

        if (arg_idx < (int)count) {
            char tmp[64];
            MepVal& v = slot(keys[arg_idx++]);

            switch (conv) {
                // ── Percentage  {:.2%}  multiply by 100, append '%' ──────────
                case '%': {
                    double fv = v.as_f64();
                    snprintf(tmp, sizeof(tmp), "%.*f%%", prec, fv * 100.0);
                    result += tmp; break;
                }
                // ── Integer  {:d}  or bare {} on INT64 ───────────────────────
                case 'd':
                case 'i': {
                    snprintf(tmp, sizeof(tmp), "%lld", (long long)v.as_i64());
                    result += tmp; break;
                }
                // ── String  {:s} ─────────────────────────────────────────────
                case 's': {
                    if (v.type == MepValType::STRING && v.str_p) result += v.str_p;
                    else { snprintf(tmp, sizeof(tmp), "%lld", (long long)v.as_i64()); result += tmp; }
                    break;
                }
                // ── Default: bare {}, {:.Nf}, {:g}, {:e}, {:.Ne} etc. ────────
                default: {
                    switch (v.type) {
                        case MepValType::INT64:
                            snprintf(tmp, sizeof(tmp), "%lld", (long long)v.i64);
                            result += tmp; break;
                        case MepValType::FLOAT64:
                            if (has_prec)
                                snprintf(tmp, sizeof(tmp), "%.*f", prec, v.f64);
                            else
                                snprintf(tmp, sizeof(tmp), "%.6g", v.f64);
                            result += tmp; break;
                        case MepValType::BOOL:
                            result += (v.b ? "True" : "False"); break;
                        case MepValType::STRING:
                            if (v.str_p) result += v.str_p; break;
                        case MepValType::TENSOR: {
                            // Single-element float tensor → print value
                            if (v.tensor && v.tensor->num_elements == 1
                                         && v.tensor->dtype == DataType::FLOAT32) {
                                double fv = (double)*static_cast<float*>(v.tensor->data);
                                if (conv == '%') {
                                    snprintf(tmp, sizeof(tmp), "%.*f%%", prec, fv * 100.0);
                                } else if (has_prec) {
                                    snprintf(tmp, sizeof(tmp), "%.*f", prec, fv);
                                } else {
                                    snprintf(tmp, sizeof(tmp), "%.6g", fv);
                                }
                            } else {
                                snprintf(tmp, sizeof(tmp), "Tensor[%u]",
                                         v.tensor ? (unsigned)v.tensor->num_elements : 0u);
                            }
                            result += tmp; break;
                        }
                        default: break;
                    }
                    break;
                }
            }
        }
        p = close + 1;  // advance past the closing '}'
    }
    release_slot(out_key);
    slot(out_key).set_str(result);
}

// =============================================================================
// 0x30  TENSOR_CREATE
// =============================================================================
void MEPInterpreter::h_tensor_create() {
    uint8_t out_key      = ru8();
    uint8_t dtype_code   = ru8();
    uint8_t creation_type = ru8();
    uint8_t src_key      = ru8();

    DataType dt = (dtype_code == 5) ? DataType::INT32 : DataType::FLOAT32;

    Tensor* t = nullptr;
    if (creation_type == 0) { // from_py: scalar/list → tensor(1, N)
        int64_t val = slot(src_key).as_i64();
        t = new_tensor(dt, {1, 1});
        if (t) {
            if (dt == DataType::INT32) *(int32_t*)t->data = (int32_t)val;
            else                       *(float*)  t->data = (float)  val;
        }
    } else if (creation_type == 1) { // arange(N)
        int64_t n = slot(src_key).as_i64();
        t = new_tensor(DataType::INT32, {1, (int)n});
        if (t) {
            int32_t* p = (int32_t*)t->data;
            for (int i = 0; i < n; ++i) p[i] = i;
        }
    } else if (creation_type == 2) { // ones(shape)
        int64_t n = slot(src_key).as_i64();
        t = new_tensor(DataType::FLOAT32, {1, (int)n});
        if (t) {
            float* p = (float*)t->data;
            for (int i = 0; i < n; ++i) p[i] = 1.0f;
        }
    }

    if (t) put_tensor(out_key, t);
    else   ESP_LOGE(TAG_MEP, "TENSOR_CREATE: failed (ctype=%u).", creation_type);
}

// =============================================================================
// 0x38  TENSOR_MANIPULATE  (op_type=1: left-pad)
// =============================================================================
void MEPInterpreter::h_tensor_manipulate() {
    uint8_t op_type      = ru8();
    uint8_t out_key      = ru8();
    uint8_t in_key       = ru8();
    uint8_t pad_width_key = ru8();
    uint8_t pad_val_key  = ru8();

    Tensor* in = slot(in_key).tensor;
    if (!in || !in->data) { ESP_LOGE(TAG_MEP, "TENSOR_MANIPULATE: no tensor in slot %u.", in_key); return; }

    if (op_type == 1) { // pad: left-pad last dimension
        int pad_w = (int)slot(pad_width_key).as_i64();
        int32_t pad_v = (int32_t)slot(pad_val_key).as_i64();

        if (in->dtype != DataType::INT32) {
            ESP_LOGW(TAG_MEP, "TENSOR_MANIPULATE: only INT32 padding implemented."); return;
        }
        size_t orig_len = in->num_elements;
        size_t new_len  = orig_len + pad_w;

        Tensor* t = new_tensor(DataType::INT32, {1, (int)new_len});
        if (!t) { ESP_LOGE(TAG_MEP, "TENSOR_MANIPULATE: OOM."); return; }

        int32_t* dst = (int32_t*)t->data;
        for (int i = 0; i < pad_w; ++i) dst[i] = pad_v;         // left padding
        memcpy(dst + pad_w, in->data, orig_len * sizeof(int32_t));
        put_tensor(out_key, t);
    } else {
        ESP_LOGW(TAG_MEP, "TENSOR_MANIPULATE: op_type %u not implemented.", op_type);
    }
}

// =============================================================================
// 0x39  TENSOR_COMBINE  (op_type=0: concat along axis)
// =============================================================================
void MEPInterpreter::h_tensor_combine() {
    uint8_t op_type  = ru8();
    uint8_t out_key  = ru8();
    uint8_t count    = ru8();
    std::vector<uint8_t> keys(count);
    for (uint8_t i = 0; i < count; ++i) keys[i] = ru8();
    uint8_t axis_key = ru8();
    (void)axis_key; // for 2-D (1, N) tensors, concat along the last dim is always correct

    if (op_type == 0) { // concat
        // Determine dtype from first valid input tensor
        DataType dt = DataType::INT32;
        for (uint8_t k : keys) {
            Tensor* t = slot(k).tensor;
            if (t) { dt = t->dtype; break; }
        }

        size_t total = 0;
        for (uint8_t k : keys) {
            Tensor* t = slot(k).tensor;
            if (t) total += t->num_elements;
        }
        Tensor* out = new_tensor(dt, {1, (int)total});
        if (!out) { ESP_LOGE(TAG_MEP, "TENSOR_COMBINE: OOM."); return; }

        size_t elem_size = out->get_element_byte_size();
        size_t offset = 0;
        for (uint8_t k : keys) {
            Tensor* t = slot(k).tensor;
            if (t && t->data) {
                memcpy((uint8_t*)out->data + offset, t->data,
                       t->num_elements * elem_size);
                offset += t->num_elements * elem_size;
            }
        }
        put_tensor(out_key, out);
    } else {
        ESP_LOGW(TAG_MEP, "TENSOR_COMBINE: op_type %u not implemented.", op_type);
    }
}

// =============================================================================
// 0x3A  TENSOR_INFO
// =============================================================================
void MEPInterpreter::h_tensor_info() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();

    Tensor* t = slot(in_key).tensor;

    // ── Compute BEFORE release to handle out_key == in_key safely ───────────
    // If release_slot(out_key) fired first and out_key == in_key, tensor->data
    // would be freed (nullptr) before we read any values from it.
    if (!t) {
        release_slot(out_key);
        slot(out_key).set_i64(0);
        return;
    }

    if (op_type == 0) { // shape → num_elements
        int64_t v = (int64_t)t->num_elements;
        release_slot(out_key);
        slot(out_key).set_i64(v);
    } else if (op_type == 1) { // dim[index]
        uint8_t dim_key = ru8();
        int64_t dim_idx = slot(dim_key).as_i64();
        int64_t v = (dim_idx < (int64_t)t->shape.size()) ? t->shape[(int)dim_idx] : 0;
        release_slot(out_key);
        slot(out_key).set_i64(v);
    } else if (op_type == 2) { // to_py scalar
        if (t->num_elements == 1) {
            if      (t->dtype == DataType::FLOAT32) { float   v = *static_cast<float*>  (t->data); release_slot(out_key); slot(out_key).set_f64(v); }
            else if (t->dtype == DataType::INT32)   { int32_t v = *static_cast<int32_t*>(t->data); release_slot(out_key); slot(out_key).set_i64(v); }
            else                                    { release_slot(out_key); slot(out_key).set_i64(0); }
        } else {
            release_slot(out_key);
            slot(out_key).set_i64(0);
        }
    } else {
        release_slot(out_key);
        slot(out_key).set_i64(0);
    }
}

// =============================================================================
// 0x3B  TENSOR_EXTRACT  —  extract slice at index
// =============================================================================
void MEPInterpreter::h_tensor_extract() {
    uint8_t out_key    = ru8();
    uint8_t tensor_key = ru8();
    uint8_t idx_key    = ru8();

    Tensor* t   = slot(tensor_key).tensor;
    int64_t idx = slot(idx_key).as_i64();
    if (!t || !t->data) { release_slot(out_key); return; }

    // For 3-D logits (1, seq_len, vocab): extract token at position idx
    size_t elem_size = t->get_element_byte_size();
    size_t stride    = 1;
    size_t offset    = 0;

    if (t->shape.size() == 3) {
        stride = t->shape[2];
        offset = (size_t)idx * stride;
    } else if (t->shape.size() == 2) {
        stride = t->shape[1];
        offset = (size_t)idx * stride;
    } else {
        stride = 1;
        offset = (size_t)idx;
    }

    Tensor* out = new_tensor(t->dtype, {1, (int)stride});
    if (!out) { ESP_LOGE(TAG_MEP, "TENSOR_EXTRACT: OOM."); return; }
    memcpy(out->data, (uint8_t*)t->data + offset * elem_size, stride * elem_size);
    put_tensor(out_key, out);
}

// =============================================================================
// 0x59  SYS_COPY
// =============================================================================
void MEPInterpreter::h_sys_copy() {
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();
    // Deep-copy: create new tensor with same data
    MepVal& src = slot(in_key);
    if (src.type == MepValType::TENSOR && src.tensor) {
        Tensor* t = new_tensor(src.tensor->dtype, src.tensor->shape);
        if (t && t->data && src.tensor->data) {
            memcpy(t->data, src.tensor->data, src.tensor->size);
        }
        put_tensor(out_key, t);
    } else {
        release_slot(out_key);
        m_ctx[out_key] = m_ctx[in_key];
        if (m_ctx[in_key].type == MepValType::STRING)
            m_ctx[out_key].set_str(m_ctx[in_key].as_cstr()); // deep copy str
    }
}

// =============================================================================
// 0x5F  SYS_DEBUG_PRINT
// =============================================================================
void MEPInterpreter::h_sys_debug_print() {
    uint8_t  key        = ru8();
    uint16_t msg_const  = ru16();
    const char* msg = get_const(msg_const).as_cstr();
    MepVal& v = slot(key);
    switch (v.type) {
        case MepValType::INT64:   Serial.printf("[MEP DBG] %s: %lld\n", msg, v.i64); break;
        case MepValType::FLOAT64: Serial.printf("[MEP DBG] %s: %.4f\n", msg, v.f64); break;
        case MepValType::BOOL:    Serial.printf("[MEP DBG] %s: %s\n", msg, v.b?"true":"false"); break;
        case MepValType::STRING:  Serial.printf("[MEP DBG] %s: '%s'\n", msg, v.as_cstr()); break;
        case MepValType::TENSOR:
            if (v.tensor) Serial.printf("[MEP DBG] %s: Tensor(ne=%u)\n", msg, (unsigned)v.tensor->num_elements);
            break;
        default: Serial.printf("[MEP DBG] %s: (none)\n", msg); break;
    }
}

// =============================================================================
// 0x60  MATH_UNARY  (op=0: softmax)
// =============================================================================
void MEPInterpreter::h_math_unary() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();

    Tensor* in = slot(in_key).tensor;
    if (!in || !in->data || in->dtype != DataType::FLOAT32) {
        ESP_LOGE(TAG_MEP, "MATH_UNARY: bad input tensor in slot %u.", in_key); return;
    }

    if (op_type == 0) { // softmax
        Tensor* out = new_tensor(DataType::FLOAT32, in->shape);
        if (!out) { ESP_LOGE(TAG_MEP, "MATH_UNARY: OOM."); return; }
        softmax(static_cast<float*>(in->data),
                static_cast<float*>(out->data),
                in->num_elements);
        put_tensor(out_key, out);
    }
}

// =============================================================================
// 0x61  MATH_BINARY  (0=add, 1=sub, 2=mul)
// =============================================================================
void MEPInterpreter::h_math_binary() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t k1      = ru8();
    uint8_t k2      = ru8();

    MepVal& v1 = slot(k1);
    MepVal& v2 = slot(k2);

    if (v1.type == MepValType::TENSOR && v2.type == MepValType::TENSOR) {
        // Tensor × Tensor element-wise (numpy broadcast for same shape)
        Tensor* t1 = v1.tensor;
        Tensor* t2 = v2.tensor;
        if (!t1 || !t2 || !t1->data || !t2->data) return;
        if (t1->dtype != DataType::FLOAT32 || t2->dtype != DataType::FLOAT32) {
            ESP_LOGW(TAG_MEP, "MATH_BINARY T×T: only FLOAT32 supported."); return;
        }
        if (t1->num_elements != t2->num_elements) {
            ESP_LOGW(TAG_MEP, "MATH_BINARY T×T: size mismatch %u vs %u.",
                     (unsigned)t1->num_elements, (unsigned)t2->num_elements); return;
        }
        Tensor* out = new_tensor(DataType::FLOAT32, t1->shape);
        if (!out) { ESP_LOGE(TAG_MEP, "MATH_BINARY T×T: OOM."); return; }
        float* s1  = (float*)t1->data;
        float* s2  = (float*)t2->data;
        float* dst = (float*)out->data;
        for (size_t i = 0; i < t1->num_elements; ++i) {
            if      (op_type == 0) dst[i] = s1[i] + s2[i];
            else if (op_type == 1) dst[i] = s1[i] - s2[i];
            else if (op_type == 2) dst[i] = s1[i] * s2[i];
        }
        put_tensor(out_key, out);

    } else if (v1.type == MepValType::TENSOR) {
        // Tensor op Scalar
        Tensor* in = v1.tensor;
        float   sc = (float)v2.as_f64();
        if (!in || !in->data || in->dtype != DataType::FLOAT32) return;

        Tensor* out = new_tensor(DataType::FLOAT32, in->shape);
        if (!out) return;
        float* src = (float*)in->data;
        float* dst = (float*)out->data;
        for (size_t i = 0; i < in->num_elements; ++i) {
            if      (op_type == 0) dst[i] = src[i] + sc;
            else if (op_type == 1) dst[i] = src[i] - sc;
            else if (op_type == 2) dst[i] = src[i] * sc;
        }
        put_tensor(out_key, out);

    } else {
        // Scalar op Scalar → FLOAT64
        double a = v1.as_f64(), b = v2.as_f64(), r = 0;
        if      (op_type == 0) r = a + b;
        else if (op_type == 1) r = a - b;
        else if (op_type == 2) r = a * b;
        release_slot(out_key);
        slot(out_key).set_f64(r);
    }
}

// =============================================================================
// 0x62  MATH_AGGREGATE  (op=0: argmax)
// =============================================================================
void MEPInterpreter::h_math_aggregate() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();

    Tensor* t = slot(in_key).tensor;

    // ── Compute BEFORE release to handle out_key == in_key safely ───────────
    // When out_key == in_key, release_slot() frees tensor->data before argmax
    // can read it, so argmax always returns 0 (index 0 = NEGATIVE class),
    // and the subsequent softmax slot stays unset → NaN confidence values.
    int64_t result = 0;
    if (t && t->data && t->dtype == DataType::FLOAT32) {
        if (op_type == 0) // argmax
            result = (int64_t)argmax(static_cast<float*>(t->data), t->num_elements);
    }
    release_slot(out_key);
    slot(out_key).set_i64(result);
}

// =============================================================================
// 0x68  LOGIC_COMPARE  (0=eq, 1=neq, 2=gt, 3=lt)
// =============================================================================
void MEPInterpreter::h_logic_compare() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t k1      = ru8();
    uint8_t k2      = ru8();

    int64_t a = slot(k1).as_i64();
    int64_t b = slot(k2).as_i64();
    bool    res;
    if      (op_type == 0) res = (a == b);
    else if (op_type == 1) res = (a != b);
    else if (op_type == 2) res = (a >  b);
    else                   res = (a <  b);

    release_slot(out_key);
    slot(out_key).set_bool(res);
}

// =============================================================================
// 0x70  ANALYSIS_TOP_K
// =============================================================================
void MEPInterpreter::h_analysis_top_k() {
    uint8_t in_key      = ru8();
    uint8_t k_key       = ru8();
    uint8_t out_idx_key = ru8();
    uint8_t out_val_key = ru8();

    Tensor* logits = slot(in_key).tensor;
    int64_t k      = slot(k_key).as_i64();
    if (!logits || !logits->data || logits->dtype != DataType::FLOAT32) return;

    size_t  n  = logits->num_elements;
    float*  lp = (float*)logits->data;
    if (k <= 0 || (size_t)k > n) k = (int64_t)n;

    // Partial sort indices descending
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [lp](size_t a, size_t b) { return lp[a] > lp[b]; });

    Tensor* t_idx = new_tensor(DataType::INT32,   {1, (int)k});
    Tensor* t_val = new_tensor(DataType::FLOAT32, {1, (int)k});
    if (!t_idx || !t_val) {
        if (t_idx) m_primary_ctx->tensor_pool.release(t_idx);
        if (t_val) m_primary_ctx->tensor_pool.release(t_val);
        return;
    }
    int32_t* pi = (int32_t*)t_idx->data;
    float*   pv = (float*  )t_val->data;
    for (int64_t i = 0; i < k; ++i) { pi[i] = (int32_t)idx[i]; pv[i] = lp[idx[i]]; }

    put_tensor(out_idx_key, t_idx);
    put_tensor(out_val_key, t_val);
}

// =============================================================================
// 0x71  ANALYSIS_SAMPLE  —  temperature + top-k + softmax + stochastic sample
// =============================================================================
void MEPInterpreter::h_analysis_sample() {
    uint8_t logits_key = ru8();
    uint8_t temp_key   = ru8();
    uint8_t topk_key   = ru8();
    uint8_t out_key    = ru8();

    Tensor* logits = slot(logits_key).tensor;
    if (!logits || !logits->data || logits->dtype != DataType::FLOAT32) {
        release_slot(out_key); slot(out_key).set_i64(0); return;
    }

    size_t  n    = logits->num_elements;
    float   temp = (float)slot(temp_key).as_f64();
    int64_t k    = slot(topk_key).as_i64();

    // Work on a local copy
    std::vector<float> lv(n);
    memcpy(lv.data(), logits->data, n * sizeof(float));

    // 1. Temperature
    if (temp > 0.0f) {
        for (float& x : lv) x /= temp;
    }

    // 2. Top-k masking
    if (k > 0 && (size_t)k < n) {
        std::vector<float> sorted_lv = lv;
        std::sort(sorted_lv.begin(), sorted_lv.end(), std::greater<float>());
        float threshold = sorted_lv[(size_t)k - 1];
        for (float& x : lv) { if (x < threshold) x = -1e30f; }
    }

    // 3. Softmax with stability
    float max_v = *std::max_element(lv.begin(), lv.end());
    float sum   = 0.0f;
    for (float& x : lv) { x = expf(x - max_v); sum += x; }
    for (float& x : lv) x /= sum;

    // 4. Stochastic sample
    float r = (float)rand() / (float)RAND_MAX;
    float acc = 0.0f;
    int64_t sampled = 0;
    for (size_t i = 0; i < n; ++i) {
        acc += lv[i];
        if (r <= acc) { sampled = (int64_t)i; break; }
    }

    release_slot(out_key);
    slot(out_key).set_i64(sampled);
}

// =============================================================================
// 0x80  MODEL_RUN_STATIC  —  synchronous NAC inference
// =============================================================================
void MEPInterpreter::h_model_run_static() {
    uint8_t model_id  = ru8();
    uint8_t count_in  = ru8();
    std::vector<uint8_t> in_keys(count_in);
    for (uint8_t i = 0; i < count_in; ++i) in_keys[i] = ru8();
    uint8_t count_out = ru8();
    std::vector<uint8_t> out_keys(count_out);
    for (uint8_t i = 0; i < count_out; ++i) out_keys[i] = ru8();

    (void)model_id; // always use primary_ctx on ESP32

    // Build input tensor list — transfer ownership from MEP slots to user_input_tensors.
    //
    // OWNERSHIP RULE: a Tensor* must have exactly ONE owner at any time.
    //
    // Problem: run_model_sync() consumes user_input_tensors one by one, storing
    // each pointer in ctx->results[idx], which is then subject to MMAP FREE.
    // After the model finishes, those pool slots have been released and possibly
    // REACQUIRED by later ops.  If m_ctx[k].tensor is left non-null pointing to
    // the now-recycled slot, ~MEPInterpreter calls tensor_pool.release() a SECOND
    // time on a slot that has already been freed and reassigned.  This:
    //   1. Frees the new op's data buffer prematurely -> NaN propagation.
    //   2. Double-returns the pool slot -> two acquire() calls get the same slot
    //      -> two ops alias the same Tensor* -> heap block header corrupted
    //      -> CORRUPT HEAP at teardown (detected by vSemaphoreDelete).
    //
    // Fix: null out m_ctx[k].tensor immediately after the transfer so that
    // ~MEPInterpreter won't try to release a tensor it no longer owns.
    // ~MEPInterpreter guards on both (type==TENSOR) AND (tensor!=nullptr),
    // so nulling the pointer alone is sufficient to prevent the double-release.
    m_primary_ctx->user_input_tensors.clear();
    for (uint8_t k : in_keys) {
        Tensor* t = slot(k).tensor;
        if (!t) { ESP_LOGE(TAG_MEP, "MODEL_RUN_STATIC: slot %u has no tensor.", k); }
        m_primary_ctx->user_input_tensors.push_back(t);
        m_ctx[k].tensor = nullptr; // transfer ownership — prevents ~MEPInterpreter double-release
    }

    // Run model synchronously
    std::vector<Tensor*> outputs;
    if (!run_model_sync(outputs)) {
        ESP_LOGE(TAG_MEP, "MODEL_RUN_STATIC: model execution failed.");
        return;
    }

    // Place outputs into context slots
    for (size_t i = 0; i < count_out && i < outputs.size(); ++i) {
        put_tensor(out_keys[i], outputs[i]);
    }
    // Free excess outputs not captured by out_keys
    for (size_t i = count_out; i < outputs.size(); ++i) {
        m_primary_ctx->tensor_pool.release(outputs[i]);
    }
}

// =============================================================================
// Flow control
// =============================================================================

void MEPInterpreter::h_flow_loop_start() {
    uint8_t counter_key = ru8();
    int64_t count = slot(counter_key).as_i64();
    MepLoopFrame frame{ m_ip, count };
    m_loop_stack.push_back(frame);

    if (count <= 0) {
        // Skip entire loop body: scan for matching 0xA1 FLOW_LOOP_END
        uint32_t search_ip = m_ip;
        int balance = 1;
        while (search_ip < m_plan_size && balance > 0) {
            uint8_t flag = m_plan[search_ip];
            if      (flag == 0xA0) balance++;
            else if (flag == 0xA1) balance--;
            if (balance == 0) break;
            search_ip += instr_len(search_ip);
        }
        m_ip = search_ip + instr_len(search_ip); // past the LOOP_END
        m_loop_stack.pop_back();
    }
}

void MEPInterpreter::h_flow_loop_end() {
    int16_t jump_offset = ri16();
    if (m_loop_stack.empty()) {
        ESP_LOGE(TAG_MEP, "FLOW_LOOP_END without matching FLOW_LOOP_START.");
        return;
    }
    MepLoopFrame& frame = m_loop_stack.back();
    frame.remaining--;
    if (frame.remaining > 0) {
        m_ip += jump_offset; // jump_offset is negative → back to loop body
    } else {
        m_loop_stack.pop_back();
    }
}

void MEPInterpreter::h_flow_branch_if() {
    uint8_t  cond_key    = ru8();
    int16_t  jump_offset = ri16();
    if (slot(cond_key).as_bool()) {
        m_ip += jump_offset;
    }
}

void MEPInterpreter::h_flow_break_loop_if() {
    uint8_t cond_key    = ru8();
    int16_t jump_offset = ri16();
    if (slot(cond_key).as_bool()) {
        if (!m_loop_stack.empty()) m_loop_stack.pop_back();
        m_ip += jump_offset;
    }
}

// =============================================================================
// 0xE0  SERIALIZE_OBJECT
// =============================================================================
void MEPInterpreter::h_serialize_object() {
    uint8_t out_key     = ru8();
    uint8_t in_key      = ru8();
    uint8_t format_type = ru8();

    if (format_type == 0) { // UTF8_STRING
        char buf[64];
        MepVal& v = slot(in_key);
        if (v.type == MepValType::INT64)        snprintf(buf, sizeof(buf), "%lld", v.i64);
        else if (v.type == MepValType::FLOAT64)  snprintf(buf, sizeof(buf), "%.4f", v.f64);
        else if (v.type == MepValType::BOOL)     snprintf(buf, sizeof(buf), "%s", v.b ? "true" : "false");
        else if (v.type == MepValType::TENSOR && v.tensor)
            snprintf(buf, sizeof(buf), "Tensor[%u]", (unsigned)v.tensor->num_elements);
        else strncpy(buf, v.as_cstr(), sizeof(buf) - 1);
        release_slot(out_key);
        slot(out_key).set_str(buf);
    } else {
        ESP_LOGW(TAG_MEP, "SERIALIZE_OBJECT: format_type %u not implemented.", format_type);
    }
}

// =============================================================================
// 0xF0  IO_WRITE
// =============================================================================
void MEPInterpreter::h_io_write() {
    uint8_t in_key    = ru8();
    uint8_t dest_type = ru8();
    /*dest_key =*/      ru8();
    uint8_t write_mode = ru8();

    const char* text = nullptr;
    char tmp[64];
    MepVal& v = slot(in_key);

    switch (v.type) {
        case MepValType::STRING:  text = v.as_cstr(); break;
        case MepValType::INT64:   snprintf(tmp, sizeof(tmp), "%lld", v.i64);  text = tmp; break;
        case MepValType::FLOAT64: snprintf(tmp, sizeof(tmp), "%.4f", v.f64);  text = tmp; break;
        case MepValType::BOOL:    text = v.b ? "true" : "false"; break;
        default: text = ""; break;
    }
    if (!text) text = "";

    if (dest_type == 0) { // STDOUT → Serial (+ TFT if enabled)
        if (write_mode == 2) { // STREAM_CHUNK: no newline
            Serial.print(text);
#if MEP_IO_USE_TFT
            tft.print(text);
#endif
        } else {
            Serial.println(text);
#if MEP_IO_USE_TFT
            tft.println(text);
#endif
        }
    } else if (dest_type == 1) { // STDERR → Serial only
        Serial.printf("[MEP ERR] %s\n", text);
    }
    // dest_type == 2 (FILE) not implemented
}

// =============================================================================
// 0xFE  EXEC_RETURN
// =============================================================================
void MEPInterpreter::h_exec_return() {
    uint8_t count = ru8();
    if (count == 1) {
        uint8_t k = ru8();
        m_return_val = m_ctx[k];
        if (m_ctx[k].type == MepValType::STRING)
            m_return_val.set_str(m_ctx[k].as_cstr());
        // Tensor: transfer ownership — clear the slot so destructor won't release
        if (m_ctx[k].type == MepValType::TENSOR) {
            m_ctx[k].tensor = nullptr; // prevent release in ~MEPInterpreter
        }
    } else {
        for (uint8_t i = 0; i < count; ++i) ru8(); // consume keys
    }
    m_running = false;
}

void MEPInterpreter::h_exec_halt() { m_running = false; }

// =============================================================================
// Synchronous model execution helpers
// =============================================================================

Tensor* MEPInterpreter::load_param_tensor(uint16_t param_id) {
    if (param_id >= m_primary_ctx->param_locations.size() ||
        !m_primary_ctx->param_present[param_id]) return nullptr;

    const TensorLocation& loc = m_primary_ctx->param_locations[param_id];
    if (loc.data_size == 0) return nullptr;

    Tensor* t = m_primary_ctx->tensor_pool.acquire();
    if (!t) return nullptr;

    if (loc.meta_len > 0) {
        uint8_t hdr[2]; // dtype_id, rank
        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
        bool ok = (sdcard.seek(loc.meta_offset) && sdcard.readData(hdr, 2) == 2);
        uint8_t dtype_id = hdr[0];
        uint8_t rank     = hdr[1];

        if (ok && rank <= 8) {
            std::vector<uint32_t> dims(rank);
            ok = (sdcard.readData(reinterpret_cast<uint8_t*>(dims.data()), rank * sizeof(uint32_t)) == rank * sizeof(uint32_t));
            if (ok) {
                switch (dtype_id) {
                    case 4: t->dtype = DataType::INT32; break;
                    case 7: t->dtype = DataType::INT8;  break;
                    default: t->dtype = DataType::FLOAT32; break;
                }
                t->shape.resize(rank);
                for (uint8_t r = 0; r < rank; ++r) t->shape[r] = (int)dims[r];
                t->update_from_shape();
            }
        }
        
        if (ok) {
            uint8_t qtype = 0;
            ok = (sdcard.readData(&qtype, 1) == 1);
            t->quant_meta.quant_type = qtype;

            if (ok && qtype == 2) { // INT8_TENSOR: scale(4)
                ok = (sdcard.readData(reinterpret_cast<uint8_t*>(&t->quant_meta.scale), 4) == 4);
            } else if (ok && qtype == 3) { // INT8_CHANNEL
                uint8_t axis; uint32_t num_scales;
                ok = (sdcard.readData(&axis, 1) == 1 && sdcard.readData(reinterpret_cast<uint8_t*>(&num_scales), 4) == 4);
                if (ok) {
                    t->quant_meta.axis = axis;
                    // --- НАЧАЛО КЛЮЧЕВОГО ИСПРАВЛЕНИЯ ---
                    // Восстанавливаем 2D форму для эмбеддингов из 1D
                    if (axis == 0 && t->shape.size() == 1 && num_scales > 0) {
                        size_t total_elements = t->num_elements;
                        if (total_elements > num_scales && total_elements % num_scales == 0) {
                            int hidden_size = total_elements / num_scales;
                            t->shape = {(int)num_scales, hidden_size};
                            t->update_from_shape(); // Пересчитываем num_elements и size
                            Serial.printf("[MEP] param%u INT8_CH shape recovered [%d,%d]\n", param_id, (int)num_scales, hidden_size);
                        }
                    }
                    // Для стриминга мы не читаем все `scales`, но нам нужна одна общая `scale`.
                    // Используем стандартное значение для симметричной квантизации.
                    t->quant_meta.scale = 1.0f / 127.0f;
                    // --- КОНЕЦ КЛЮЧЕВОГО ИСПРАВЛЕНИЯ ---
                }
            }
        }
        xSemaphoreGive(g_sd_card_mutex);

        if (!ok) {
            Serial.printf("[MEP] param%u metadata read failed\n", param_id);
            m_primary_ctx->tensor_pool.release(t);
            return nullptr;
        }
    }

    // Выделяем буфер. Если памяти недостаточно — освобождаем слот пула и
    // возвращаем nullptr. Kernels проверяют nullptr и логируют ошибку.
    // Возврат Tensor* с data=nullptr приводил к разыменованию нулевого указателя
    // в kernel и crash abort() — это критический баг при нехватке SRAM/PSRAM.
    t->data = alloc_fast(loc.data_size);
    if (!t->data) {
        Serial.printf("[MEP] param%u OOM: need %llu B, SRAM free=%u, PSRAM free=%u\n",
                      param_id,
                      (unsigned long long)loc.data_size,
                      (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                      (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        m_primary_ctx->tensor_pool.release(t);
        return nullptr;
    }

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(loc.file_offset);
    sdcard.readData(static_cast<uint8_t*>(t->data), loc.data_size);
    xSemaphoreGive(g_sd_card_mutex);

    if (m_primary_ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
        if (!dequantize_tensor(t)) {
            m_primary_ctx->tensor_pool.release(t);
            return nullptr;
        }
    }
    t->param_location = loc;
    return t;
}

// =============================================================================
// mmap_tick_sync  —  SUPERSEDED, не вызывается
// =============================================================================
// Эта функция была inline-заменителем потока nac_memory_task для MEP-пути,
// когда задача не запускалась. Теперь nac_memory_task активен всегда (для
// обоих путей, MEP и легаси), и notify отправляется из run_model_sync после
// записи каждого results[idx]. Функция оставлена как reference-реализация
// семантики MMAP — не удалять без веской причины.
// Если когда-либо потребуется single-threaded режим без FreeRTOS — вызвать
// mmap_tick_sync(idx) вместо xTaskNotifyGive в run_model_sync.
void MEPInterpreter::mmap_tick_sync(uint32_t tick) {
    auto it = m_primary_ctx->mmap_schedule.find(tick);
    if (it == m_primary_ctx->mmap_schedule.end()) return;

    for (const MmapCommand& cmd : it->second) {
        switch (cmd.action) {
            case MmapAction::PRELOAD: {
                // cmd.target_id = instruction index of the B=1 INPUT op to preload.
                // ── Index by instruction index, NOT by param_id ──────────────────
                // Previously stored at fast_memory_cache[param_id].  param_id is an
                // index into the parameter table (0..num_params), while SAVE_RESULT
                // stores at fast_memory_cache[op_tick].  When param_id == some tick
                // that also has a SAVE_RESULT, the preloaded weight gets silently
                // overwritten with a computed result → B=1 load receives the wrong
                // tensor → kernel returns nullptr → cascade of nulls → -nan%.
                //
                // Fix: use the TARGET INSTRUCTION INDEX as the cache key.
                // Both PRELOAD and B=1 load now use the same key (target_op_idx /
                // my_op_idx), which is the op's own position in decoded_ops.
                uint16_t target_op_idx = cmd.target_id;
                Serial.printf("  [MMAP] tick=%u PRELOAD target_op=%u\n", tick, target_op_idx);
                if (target_op_idx >= m_primary_ctx->decoded_ops.size()) break;
                const ParsedInstruction& ins = m_primary_ctx->decoded_ops[target_op_idx];
                if (ins.A != 2 || ins.B != 1 || ins.C.size() < 2) break;
                uint16_t param_id = ins.C[1];

                // Skip if already preloaded at this instruction's cache slot
                if (target_op_idx < m_primary_ctx->fast_memory_cache.size() &&
                    m_primary_ctx->fast_memory_cache[target_op_idx]) {
                    ESP_LOGD(TAG_MEP, "PRELOAD: op%u already cached, skipping.", target_op_idx);
                    break;
                }

                Tensor* t = load_param_tensor(param_id);
                if (t) {
                    if (target_op_idx >= m_primary_ctx->fast_memory_cache.size())
                        m_primary_ctx->fast_memory_cache.resize(target_op_idx + 1, nullptr);
                    m_primary_ctx->fast_memory_cache[target_op_idx] = t;
                    ESP_LOGD(TAG_MEP, "PRELOAD: op%u param%u preloaded (%u B).",
                             target_op_idx, param_id, (unsigned)t->size);
                } else {
                    Serial.printf("  [MMAP] PRELOAD: op%u param%u OOM (will load on demand)\n",
                                  target_op_idx, param_id);
                }
                break;
            }
            case MmapAction::FREE: {
                uint16_t tid = cmd.target_id;
                Serial.printf("  [MMAP] tick=%u FREE target=%u ptr=%p\n",
                              tick, tid,
                              (tid < m_primary_ctx->results.size()) ? (void*)m_primary_ctx->results[tid] : nullptr);
                if (tid < m_primary_ctx->results.size() && m_primary_ctx->results[tid]) {
                    Tensor* to_free = m_primary_ctx->results[tid];
                    // ── Deduplicate aliases BEFORE releasing ───────────────────────
                    // results[] can contain the same Tensor* in multiple slots due to:
                    //   • FORWARD:      results[dst] = results[src]  (without nulling src)
                    //   • SAVE_RESULT:  fast_memory_cache[s] = results[tick]
                    //   • op_nac_pass:  returns args[0] directly
                    // Releasing a pointer that's still live in another slot turns it
                    // into a dangling pointer.  The tensor pool may reuse the slot,
                    // so the next read from that slot gets wrong data → NaN.
                    // At teardown the destructor tries to release it again → double-free
                    // → CORRUPT HEAP.
                    //
                    // Solution: null every alias across results[] AND fast_memory_cache[]
                    // before the single authoritative release here.
                    for (auto& rp : m_primary_ctx->results) {
                        if (rp == to_free) rp = nullptr;
                    }
                    for (auto& cp : m_primary_ctx->fast_memory_cache) {
                        if (cp == to_free) cp = nullptr;
                    }
                    m_primary_ctx->tensor_pool.release(to_free);
                    // (results[tid] was already nulled in the loop above)
                }
                break;
            }
            case MmapAction::SAVE_RESULT: {
                Serial.printf("  [MMAP] tick=%u SAVE_RESULT slot=%u result_ptr=%p\n",
                              tick, cmd.target_id,
                              (tick < m_primary_ctx->results.size()) ? (void*)m_primary_ctx->results[tick] : nullptr);
                if (tick < m_primary_ctx->results.size() && m_primary_ctx->results[tick]) {
                    uint16_t slot_id = cmd.target_id;
                    if (slot_id >= m_primary_ctx->fast_memory_cache.size())
                        m_primary_ctx->fast_memory_cache.resize(slot_id + 1, nullptr);
                    // ── Never release the previous cache occupant ────────────────
                    // fast_memory_cache[] is shared between two naming spaces:
                    //   PRELOAD  uses param_id  (0..num_params) as index
                    //   SAVE_RESULT uses op tick (0..num_ops) as index
                    // These ranges overlap (param_id can equal some tick value).
                    // Releasing the previous occupant here could evict a preloaded
                    // weight that has not been consumed yet, causing the B=1 load
                    // handler to get the wrong tensor.  FREE's deduplication loop
                    // scans the whole cache and handles cleanup correctly.
                    m_primary_ctx->fast_memory_cache[slot_id] = m_primary_ctx->results[tick];
                }
                break;
            }
            case MmapAction::FORWARD: {
                uint16_t src = (uint16_t)tick;
                uint16_t dst = cmd.target_id;
                Serial.printf("  [MMAP] tick=%u FORWARD src=%u->dst=%u src_ptr=%p\n",
                              tick, src, dst,
                              (src < m_primary_ctx->results.size()) ? (void*)m_primary_ctx->results[src] : nullptr);
                if (src < m_primary_ctx->results.size() &&
                    m_primary_ctx->results[src] &&
                    dst < m_primary_ctx->results.size()) {
                    // ── Set alias, do NOT null results[src] ───────────────────────
                    // gather_arguments for op dst reads results[src] via D-field offset
                    // (ancestor_idx = dst + d_val = src).  Nulling results[src] here
                    // makes gather_arguments return nullptr for every op that relies on
                    // this forwarded tensor → null propagates as NaN through all
                    // subsequent ops.
                    //
                    // Correct semantics: FORWARD is a *hint* that the result will have
                    // a single consumer — it does not transfer ownership.  The alias
                    // results[dst] = results[src] is set so that any code that happens
                    // to index by dst also works.  FREE's full deduplication loop
                    // (which scans all of results[] + fast_memory_cache[]) handles both
                    // aliases atomically: nulls every copy, releases once.  No double-
                    // free, no dangling pointers.
                    if (m_primary_ctx->results[dst] &&
                        m_primary_ctx->results[dst] != m_primary_ctx->results[src]) {
                        m_primary_ctx->tensor_pool.release(m_primary_ctx->results[dst]);
                    }
                    m_primary_ctx->results[dst] = m_primary_ctx->results[src];
                    // results[src] intentionally left non-null — gather_arguments needs it.
                }
                break;
            }
        }
    }
}

bool MEPInterpreter::run_model_sync(std::vector<Tensor*>& out_tensors) {
    NacRuntimeContext* ctx = m_primary_ctx;

    // Reset result slots from any previous run
    for (auto*& r : ctx->results) {
        if (r) { ctx->tensor_pool.release(r); r = nullptr; }
    }
    if (ctx->results.size() != ctx->decoded_ops.size())
        ctx->results.assign(ctx->decoded_ops.size(), nullptr);

    ctx->stop_flag.store(false);
    size_t user_input_idx = 0;

    // ── DEBUG HELPERS ────────────────────────────────────────────────────────
    // Inline lambda: print tensor state (shape, dtype, data ptr, first value).
    auto dbg_tensor = [](const char* tag, const Tensor* t) {
        if (!t) { Serial.printf("  [DBG] %s = nullptr\n", tag); return; }
        const char* dtype_s = (t->dtype==DataType::FLOAT32)?"f32":
                              (t->dtype==DataType::INT32)  ?"i32":
                              (t->dtype==DataType::INT8)   ?"i8" : "???";
        Serial.printf("  [DBG] %s  ptr=%p  ne=%u  dtype=%s  shape=[",
                      tag, t->data, (unsigned)t->num_elements, dtype_s);
        for (size_t si = 0; si < t->shape.size(); ++si)
            Serial.printf("%s%d", si?",":"", t->shape[si]);
        Serial.printf("]");
        if (t->data && t->num_elements > 0) {
            if (t->dtype == DataType::FLOAT32) {
                const float* fp = static_cast<const float*>(t->data);
                // Print first 4 values and scan for NaN/Inf
                Serial.printf("  first=[");
                for (int vi=0; vi<(int)std::min((size_t)4, t->num_elements); ++vi)
                    Serial.printf("%s%.4g", vi?",":"", fp[vi]);
                Serial.printf("]");
                bool has_nan=false, has_inf=false;
                size_t check = std::min(t->num_elements, (size_t)256);
                for (size_t vi=0; vi<check; ++vi) {
                    if (std::isnan(fp[vi])) { has_nan=true; break; }
                    if (std::isinf(fp[vi])) { has_inf=true; break; }
                }
                if (has_nan) Serial.printf(" *** NaN DETECTED ***");
                if (has_inf) Serial.printf(" *** Inf DETECTED ***");
            } else if (t->dtype == DataType::INT32) {
                const int32_t* ip = static_cast<const int32_t*>(t->data);
                Serial.printf("  first=[");
                for (int vi=0; vi<(int)std::min((size_t)4, t->num_elements); ++vi)
                    Serial.printf("%s%d", vi?",":"", ip[vi]);
                Serial.printf("]");
            } else if (t->dtype == DataType::INT8) {
                const int8_t* ip = static_cast<const int8_t*>(t->data);
                Serial.printf("  first=[");
                for (int vi=0; vi<(int)std::min((size_t)4, t->num_elements); ++vi)
                    Serial.printf("%s%d", vi?",":"", (int)ip[vi]);
                Serial.printf("]");
            }
        } else {
            Serial.printf("  (no data)");
        }
        Serial.printf("\n");
    };

    for (uint32_t idx = 0; idx < ctx->decoded_ops.size() && !ctx->stop_flag.load(); ++idx) {

        // Per-op diagnostic
        {
            const ParsedInstruction& dbg = ctx->decoded_ops[idx];
            uint32_t h  = (uint32_t)esp_get_free_heap_size();
            uint32_t lb = (uint32_t)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
            // Name from id_to_name_map if available, else "?"
            const char* op_name = "?";
            auto nit = ctx->id_to_name_map.find(dbg.A);
            if (nit != ctx->id_to_name_map.end()) op_name = nit->second.c_str();
            Serial.printf("[MODEL] op%3u A=%u B=%u heap=%u lb=%u  (%s)\n",
                          idx, dbg.A, dbg.B, h, lb, op_name);
            // Log all arguments this op will receive
            if (dbg.A >= 10) {
                for (size_t di = 0; di < dbg.D.size(); ++di) {
                    int32_t anc = (int32_t)idx + (int32_t)dbg.D[di];
                    char argname[32];
                    snprintf(argname, sizeof(argname), "  arg[%u] (op%u)", (unsigned)di, (unsigned)anc);
                    if (anc >= 0 && (uint32_t)anc < ctx->results.size())
                        dbg_tensor(argname, ctx->results[anc]);
                    else
                        Serial.printf("  [DBG] %s = OUT_OF_RANGE\n", argname);
                }
            }
        }

        // ── Compute ──────────────────────────────────────────────────────────
        const ParsedInstruction& ins = ctx->decoded_ops[idx];
        Tensor* result_tensor = nullptr;

        if (ins.A == 2) { // <INPUT>
            if (ins.B == 0) { // User / external input
                if (user_input_idx < ctx->user_input_tensors.size()) {
                    result_tensor = ctx->user_input_tensors[user_input_idx];
                    ctx->user_input_tensors[user_input_idx] = nullptr;
                    user_input_idx++;
                } else {
                    ESP_LOGE(TAG_MEP, "run_model_sync: user input vector exhausted at op %u.", idx);
                    ctx->stop_flag.store(true);
                }
            } else if (ins.B == 1) { // Weight
                if (ins.C.size() < 2) continue;
                uint16_t param_id = ins.C[1];
                // ── Lookup by instruction index under cache_mutex ─────────────
                // nac_memory_task PRELOAD stores at fast_memory_cache[target_op_idx]
                // where target_op_idx == idx (the B=1 instruction's own position).
                // cache_mutex is required: nac_memory_task writes under it, so
                // concurrent reads here without it are a data race.
                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (idx < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[idx]) {
                    result_tensor = ctx->fast_memory_cache[idx];
                    ctx->fast_memory_cache[idx] = nullptr;
                    ESP_LOGD(TAG_MEP, "Weight op%u: served from preload cache.", idx);
                }
                xSemaphoreGive(ctx->cache_mutex);
                if (!result_tensor) {
                    result_tensor = load_param_tensor(param_id);
                    if (!result_tensor)
                        Serial.printf("  [MODEL] op%u B=1 param%u load FAILED (OOM? data_size=?)\n",
                                      idx, param_id);
                }
            } else if (ins.B == 3) { // Model constant (vector indexed by const_id)
                if (ins.C.size() < 2) continue;
                uint16_t cid = ins.C[1];
                if (cid < ctx->constants.size() && ctx->constants[cid]) {
                    // MUST copy — never store the raw unique_ptr pointer in results[].
                    // MMAP FREE calls TensorPool::release() on results[idx], which frees
                    // tensor->data then calls delete tensor on the unique_ptr-owned struct.
                    // unique_ptr later fires ~Tensor() on the freed memory -> double-free
                    // -> CORRUPT HEAP at the same address every run (CNST allocs are
                    // first, so the address is deterministic).
                    Tensor* src = ctx->constants[cid].get();
                    if (src && src->data) {
                        Tensor* copy = ctx->tensor_pool.acquire();
                        if (copy) {
                            copy->dtype        = src->dtype;
                            copy->shape        = src->shape;
                            copy->num_elements = src->num_elements;
                            copy->size         = src->size;
                            copy->data         = alloc_fast(src->size);
                            if (copy->data) {
                                memcpy(copy->data, src->data, src->size);
                                result_tensor = copy;
                            } else {
                                ctx->tensor_pool.release(copy);
                                ESP_LOGE(TAG_MEP, "run_model_sync: OOM copying constant %u at op %u.", cid, idx);
                            }
                        }
                    }
                }
            }
        } else if (ins.A == 3) { // <o> OUTPUT node
            for (int16_t offset : ins.D) {
                int32_t src_idx = (int32_t)idx + (int32_t)offset;
                if (src_idx >= 0 && src_idx < (int32_t)ctx->results.size() &&
                    ctx->results[src_idx]) {
                    out_tensors.push_back(ctx->results[src_idx]);
                    ctx->results[src_idx] = nullptr; // transfer ownership
                }
            }
            if (ins.B == 1) break; // RETURN (not just WRITE)
        } else if (ins.A >= 10) { // Regular ops
            Tensor* arguments[MAX_INSTRUCTION_ARITY] = {};
            size_t argc = gather_arguments(*ctx, ins, idx, arguments);

            // Dequantise if needed
            Tensor* dq_args[MAX_INSTRUCTION_ARITY] = {};
            std::vector<Tensor*> temp_dq;

            if (ctx->quant_mode != QuantExecMode::DEQUANT_ON_LOAD) {
                for (size_t i = 0; i < argc; ++i) {
                    if (arguments[i] && arguments[i]->dtype != DataType::FLOAT32) {
                        Tensor* fp = create_fp32_copy_from_quantized(*ctx, arguments[i]);
                        dq_args[i] = fp ? fp : arguments[i];
                        if (fp) temp_dq.push_back(fp);
                    } else {
                        dq_args[i] = arguments[i];
                    }
                }
            } else {
                for (size_t i = 0; i < argc; ++i) dq_args[i] = arguments[i];
            }

            auto kit = g_op_kernels.find(ins.A);
            if (kit != g_op_kernels.end()) {
                result_tensor = kit->second(ctx, ins, dq_args, argc);
            } else {
                result_tensor = op_nac_pass(ctx, ins, dq_args, argc);
            }

            for (Tensor* tmp : temp_dq) ctx->tensor_pool.release(tmp);

            if (result_tensor && ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT &&
                result_tensor->dtype == DataType::FLOAT32) {
                requantize_result_tensor(*ctx, result_tensor);
            }
        }
        if (!result_tensor) {
            // Log null results but continue — some ops legitimately produce null
            // (e.g. when streaming fallback fails for non-embedding ops).
            // Aborting here causes double-release: the emergency cleanup releases
            // all results[], then ~NacRuntimeContext releases them again.
            ESP_LOGW(TAG_MEP, "op%u result=nullptr (OOM or bad args)", idx);
        }

        if (idx < ctx->results.size()) {
            ctx->results[idx] = result_tensor;
            // Нотифицируем nac_memory_task ПОСЛЕ записи results[idx].
            // SAVE_RESULT и FREE читают results[tick] — они должны видеть
            // уже заполненный слот, а не результат предыдущего шага.
            ctx->current_instruction_idx.store(idx, std::memory_order_release);
            if (g_nac_memory_task_handle)
                xTaskNotifyGive(g_nac_memory_task_handle);
        } else if (result_tensor) {
            ctx->tensor_pool.release(result_tensor);
        }
    }

    // ── Сбор выходных тензоров ───────────────────────────────────────────
    if (out_tensors.empty()) {
        for (int i = (int)ctx->results.size() - 1; i >= 0; --i) {
            if (ctx->results[i]) {
                out_tensors.push_back(ctx->results[i]);
                ctx->results[i] = nullptr;
                break;
            }
        }
    }

    // ── 1. Декантизация выходных тензоров (при необходимости) ────────────
    for (size_t oi = 0; oi < out_tensors.size(); ++oi) {
        Tensor* t = out_tensors[oi];
        if (!t) continue;

        if (t->dtype != DataType::FLOAT32 && t->quant_meta.quant_type != 0) {
            Tensor* fp = create_fp32_copy_from_quantized(*ctx, t);
            if (fp) {
                ctx->tensor_pool.release(t);   // освобождаем старый (квантованный) тензор
                out_tensors[oi] = fp;
            }
        } else if (t->dtype == DataType::INT8 && t->quant_meta.quant_type == 0) {
            size_t n = t->num_elements;
            Tensor* fp = ctx->tensor_pool.acquire();
            if (fp) {
                fp->dtype = DataType::FLOAT32;
                fp->shape = t->shape;
                fp->update_from_shape();
                fp->data = alloc_fast(fp->size);
                if (fp->data) {
                    const int8_t* src = static_cast<const int8_t*>(t->data);
                    float* dst = static_cast<float*>(fp->data);
                    float sc = 1.0f;
                    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * sc;
                    ctx->tensor_pool.release(t);
                    out_tensors[oi] = fp;
                } else {
                    ctx->tensor_pool.release(fp);
                }
            }
        }
    }

    // ── 2. Множество тензоров, которые должны остаться (выходные) ─────
    std::set<Tensor*> keep_set(out_tensors.begin(), out_tensors.end());

    // ── 3. Обнуляем выходные тензоры в results и fast_memory_cache ─────
    for (auto*& r : ctx->results) {
        if (r && keep_set.count(r)) {
            r = nullptr;   // больше не владеем, владеет out_tensors
        } else if (r) {
            // Не в keep_set — оставляем как есть, они будут освобождены
            // позже в деструкторе NacRuntimeContext (или FREE).
            // Не вызываем release() здесь!
        }
    }

    for (auto*& c : ctx->fast_memory_cache) {
        if (c && keep_set.count(c)) {
            c = nullptr;   // больше не владеем
        }
        // остальные не трогаем
    }

    return !out_tensors.empty();
}

// =============================================================================
// Top-level entry point
// =============================================================================
bool mep_run_orchestrated(NacRuntimeContext*               primary_ctx,
                           const std::vector<std::string>&  answers)
{
    std::vector<uint8_t>       bytecode;
    std::map<uint16_t, MepVal> constants;

    if (!mep_load_from_nac(bytecode, constants, primary_ctx)) {
        return false; // No ORCH section — caller should use legacy path
    }

    MEPInterpreter interp(bytecode.data(), bytecode.size(), constants, primary_ctx);
    interp.set_pre_answers(answers);
    interp.run();
    return true;
}
