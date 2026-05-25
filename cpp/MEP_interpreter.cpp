// =============================================================================
// MEP_interpreter.cpp  —  MEP ISA v1.0 interpreter for ESP32
//
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
// =============================================================================
#include "MEP_interpreter.h"

#ifdef ARDUINO
#  include "CYD28_SD.h"
#  include <Arduino.h>
#  include <esp_heap_caps.h>
#else
#  include "platform.h"
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// External SD card object declared in main.cpp
#ifdef ARDUINO
extern CYD28_SD sdcard;
#endif

// ABI note: TISAVM::decode() signature was changed from vector<int32_t> to
// vector<int> in TISA_VM.h. On xtensa-esp-elf gcc 14, int32_t resolves to
// `long` in this TU but to `int` in TISA_VM.cpp — different mangled symbols.
// `int` is always stable. See TISA_VM.h for the canonical explanation.
static_assert(sizeof(int) == 4, "int must be 32-bit on ESP32/Xtensa");

// TFT for IO_WRITE display output (optional — controlled by MEP_IO_USE_TFT)
#ifndef MEP_IO_USE_TFT
#  ifdef ARDUINO
#    define MEP_IO_USE_TFT 1
#  else
#    define MEP_IO_USE_TFT 0
#  endif
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
#ifdef DBG
    ESP_LOGI(TAG_MEP, "ORCH loaded: %u bytes bytecode, %u constants.", bytecode_len, constants_count);
#endif
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
    m_models[0] = primary_ctx; // By default 0 is primary, but the script can overwrite this
}

MEPInterpreter::~MEPInterpreter() {
    for (int k = 0; k < 256; ++k) release_slot(k);
    
    // Clearing dynamic contexts (e.g. t5-encoder)
    for (auto& pair : m_models) {
        if (pair.second != m_primary_ctx && pair.second != nullptr) {
            delete pair.second;
        }
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
        case 0x05: return 2;   // SRC_EXEC_MODE: out(1)
        case 0x10: return 4;   // model_id(1)+path_const_id(2)
        case 0x11: return 5;   // out(1)+file_type(1)+path_const_id(2)
        case 0x12: return 5;   // out(1)+res_type(1)+res_id_const_id(2)
        case 0x13: return 4;   // out(1)+path_key(1)+file_type(1)
        case 0x14: return 4;   // RES_LOAD_ARRAY: out(1)+name_cid(2)
        case 0x1F: return 3;   // res_type(1)+id_or_key(1)
        case 0x20: return 4;   // proc(1)+in(1)+out(1)
        case 0x21: return 4;
        case 0x22: return 5;   // proc(1)+item_const_id(2)+out(1)
        case 0x2A: {            // out(1)+fmt_const_id(2)+count(1)+N*key(1)
            if (ip + 5 > m_plan_size) return 1;
            return 1 + 1 + 2 + 1 + m_plan[ip + 4];
        }
        case 0x30: return 5;   // out(1)+dtype(1)+ctype(1)+src_key(1)
        case 0x38: {
            if (ip + 1 >= m_plan_size) return 1;
            uint8_t op_type = m_plan[ip + 1];
            if (op_type == 1) return 6; // 1(op) + 1(type) + 1(out) + 1(in) + 2(args)
            if (op_type == 2 || op_type == 3) return 5; // 1(op) + 1(type) + 1(out) + 1(in) + 1(arg)
            return 4; // fallback
        }
        case 0x39: {
            if (ip + 3 >= m_plan_size) return 1;
            uint8_t sub_op = m_plan[ip + 1];
            uint8_t count = m_plan[ip + 3];
            return 4 + count + (sub_op == 0 ? 1 : 0);
        }
        case 0x3A: {
            if (ip + 2 > m_plan_size) return 1;
            return 4 + (m_plan[ip + 1] == 1 ? 1 : 0);
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
            return 4 + ci + co;
        }
        case 0x82: {           // MODEL_TRAIN_STEP
            if (ip + 3 >= m_plan_size) return 1;
            uint8_t ci = m_plan[ip + 3];
            if (ip + 4 + ci >= m_plan_size) return 1;
            uint8_t co = m_plan[ip + 4 + ci];
            // 1(op) + 1(mod) + 1(loss) + 1(ci) + ci + 1(co) + co + 1(loss_k) + 1(lr) + 1(log_k) + 2(w_id) + 2(b_id)
            return 12 + ci + co; 
        }
        case 0x83: return 2;   // MODEL_ZERO_GRAD: mod(1)
        case 0x85: return 4;   // MODEL_SAVE_WEIGHTS: mod(1)+path(1)+type(1)
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
#ifdef DBG
static const char* mep_opcode_name(uint8_t op) {
    switch (op) {
        case 0x02: return "SRC_USER_PROMPT";
        case 0x04: return "SRC_CONSTANT";
        case 0x05: return "SRC_EXEC_MODE";
        case 0x10: return "RES_LOAD_MODEL";
        case 0x11: return "RES_LOAD_DATAFILE";
        case 0x12: return "RES_LOAD_EXTERN";
        case 0x13: return "RES_LOAD_DYNAMIC";
        case 0x14: return "RES_LOAD_ARRAY";
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
        case 0x82: return "MODEL_TRAIN_STEP";
        case 0x83: return "MODEL_ZERO_GRAD";
        case 0x85: return "MODEL_SAVE_WEIGHTS";
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
#endif
// =============================================================================
// Main execution loop
// =============================================================================
MepVal MEPInterpreter::run() {
#ifdef DBG
    ESP_LOGI(TAG_MEP, "--- MEP execution started ---");
#endif
    m_ip      = 0;
    m_running = true;

    while (m_ip < m_plan_size && m_running) {
        uint8_t op     = ru8();
#ifdef DBG
        uint32_t ip_before = m_ip - 1;  // ip before reading op
        uint32_t heap  = (uint32_t)esp_get_free_heap_size();
        uint32_t stack = (uint32_t)uxTaskGetStackHighWaterMark(NULL);
        Serial.printf("[MEP] ip=%3u op=0x%02X %-22s heap=%6u stack_hwm=%5u\n",ip_before, op, mep_opcode_name(op), heap, stack);
#endif
        switch (op) {
            case 0x02: h_src_user_prompt();    break;
            case 0x04: h_src_constant();       break;
            case 0x05: h_src_exec_mode();      break;
            case 0x10: h_res_load_model();     break;
            case 0x11: h_res_load_datafile();  break;
            case 0x12: h_res_load_extern();    break;
            case 0x13: h_res_load_dynamic();   break;
            case 0x14: h_res_load_array();     break;
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
            case 0x82: h_model_train_step();   break;
            case 0x83: h_model_zero_grad();    break;
            case 0x85: h_model_save_weights(); break;
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
#ifdef DBG
        Serial.printf("[MEP] ip=%3u op=0x%02X DONE  heap=%6u\n",ip_before, op, (unsigned)esp_get_free_heap_size());
#endif
    }
#ifdef DBG
    ESP_LOGI(TAG_MEP, "--- MEP execution finished ---");
#endif
    return m_return_val;
}

// =============================================================================
// 0x02  SRC_USER_PROMPT
// =============================================================================
void MEPInterpreter::h_src_user_prompt() {
    uint8_t out_key = ru8();
    uint8_t dtype   = ru8();
    uint16_t prompt_id = ru16();

    const char* prompt_text = get_const(prompt_id).as_cstr();
    std::string ans;

    if (!m_pre_answers.empty()) {
        ans = m_pre_answers.front();
        m_pre_answers.erase(m_pre_answers.begin());
        printf("%s%s\n", prompt_text ? prompt_text : "", ans.c_str());
    } else {
        // Clear input wait indicator to distinguish between a hang and a prompt
        printf("[WAITING FOR INPUT] %s", prompt_text ? prompt_text : "");
        fflush(stdout); 
        std::getline(std::cin, ans);
    }

    if (dtype == 1) { // int
        long long val = 0;
        try { val = std::stoll(ans); } catch(...) {}
        release_slot(out_key);
        slot(out_key).set_i64(val);
    } else if (dtype == 2) { // float
        double val = 0.0;
        try { val = std::stod(ans); } catch(...) {}
        release_slot(out_key);
        slot(out_key).set_f64(val);
    } else { // string
        release_slot(out_key);
        slot(out_key).set_str(ans);
    }
}

// =============================================================================
// 0x04  SRC_CONSTANT
// =============================================================================
void MEPInterpreter::h_src_constant() {
    uint8_t  out_key  = ru8();
    uint16_t const_id = ru16();
    release_slot(out_key);
    const MepVal& src = get_const(const_id);
    switch (src.type) {
        case MepValType::INT64:   slot(out_key).set_i64(src.i64); break;
        case MepValType::FLOAT64: slot(out_key).set_f64(src.f64); break;
        case MepValType::BOOL:    slot(out_key).set_bool(src.b); break;
        case MepValType::STRING:
            slot(out_key).set_str(src.str_p ? std::string(src.str_p, src.str_len) : std::string());
            break;
        case MepValType::OPAQUE:  slot(out_key).set_opaque(src.ptr); break;
        case MepValType::TENSOR:
            if (src.tensor) {
                Tensor* t = new_tensor(src.tensor->dtype, src.tensor->shape);
                if (t && t->data && src.tensor->data) memcpy(t->data, src.tensor->data, src.tensor->size);
                if (t) slot(out_key).set_tensor(t);
            }
            break;
        default:
            slot(out_key).set_none();
            break;
    }
}

// =============================================================================
// 0x05  SRC_EXEC_MODE
// =============================================================================
void MEPInterpreter::h_src_exec_mode() {
    uint8_t out_key = ru8();
    release_slot(out_key);
    slot(out_key).set_str(m_exec_mode);
}

// =============================================================================
// 0x10  RES_LOAD_MODEL
// Maps a model_id → primary_ctx.  On ESP32 only one model fits in RAM.
// =============================================================================
void MEPInterpreter::h_res_load_model() {
    uint8_t model_id = ru8();
    uint16_t path_cid = ru16();
    const char* path = get_const(path_cid).as_cstr();
    
    if (m_primary_ctx->model_path == path) {
        m_models[model_id] = m_primary_ctx;
        return;
    }
    
    NacRuntimeContext* new_ctx = new NacRuntimeContext();
    new_ctx->model_path = path;
    
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.openFile(path);
    xSemaphoreGive(g_sd_card_mutex);

    if (initialize_nac_context(*new_ctx)) {
        m_models[model_id] = new_ctx;
    } else {
        delete new_ctx;
    }
}

// =============================================================================
// 0x11  RES_LOAD_DATAFILE  (stub — .npy not commonly used on ESP32)
// =============================================================================
void MEPInterpreter::h_res_load_datafile() {
    uint8_t  out_key   = ru8();
    uint8_t  file_type = ru8();
    uint16_t path_id   = ru16();
    
    const char* path_cstr = get_const(path_id).as_cstr();
    
    // Emulating loading a numpy mask file from CompileTest.py for GPT-2.
    // Causal mask: BOOL tensor [1, 1, 64, 64] (lower triangular matrix)
    if (file_type == 2 && path_cstr && strstr(path_cstr, "gpt2_causal_mask.npy")) { 
        Tensor* t = new_tensor(DataType::BOOL, {1, 1, 64, 64});
        if (t && t->data) {
            uint8_t* p = (uint8_t*)t->data;
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    p[i * 64 + j] = (j <= i) ? 1 : 0;
                }
            }
            put_tensor(out_key, t);
#ifdef DBG
            ESP_LOGI(TAG_MEP, "RES_LOAD_DATAFILE: Generated causal mask [1,1,64,64] for '%s'.", path_cstr);
#endif
            return;
        } else {
            ESP_LOGE(TAG_MEP, "RES_LOAD_DATAFILE: OOM generating mask.");
        }
    } else {
        ESP_LOGW(TAG_MEP, "RES_LOAD_DATAFILE: Not fully implemented for file_type=%u, path='%s'", 
                 file_type, path_cstr ? path_cstr : "null");
    }
}


// =============================================================================
// 0x12  RES_LOAD_EXTERN  —  extract tokenizer from loaded model
// =============================================================================
void MEPInterpreter::h_res_load_extern() {
    uint8_t  out_key  = ru8();
    uint8_t  res_type = ru8();
    uint16_t res_cid  = ru16(); 

    if (res_type == 0) { // type 0 = extract Context for TISAVM
        int64_t model_id_val = get_const(res_cid).as_i64();
        uint8_t model_id = (uint8_t)model_id_val;
        
        NacRuntimeContext* target_ctx = m_models.count(model_id) ? m_models[model_id] : m_primary_ctx;

        if (!target_ctx || !target_ctx->tokenizer) {
            ESP_LOGE(TAG_MEP, "RES_LOAD_EXTERN: Context %u has no tokenizer.", model_id);
            return;
        }
        release_slot(out_key);
        // IMPORTANT: Save a pointer to the CONTEXT itself in order to get the manifest from it
        slot(out_key).set_opaque(target_ctx); 
        ESP_LOGD(TAG_MEP, "RES_LOAD_EXTERN: context %u extracted to slot %u.", model_id, out_key);
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

    const char* dyn_path = slot(path_key).as_cstr();
    if (!dyn_path || !dyn_path[0]) {
#ifdef DBG
        ESP_LOGE(TAG_MEP, "RES_LOAD_DYNAMIC: empty path in slot %u.", path_key);
#endif
        return;
    }

    if (file_type == 3) {
#ifdef DBG
        ESP_LOGI(TAG_MEP, "Loading and preprocessing image: '%s'", dyn_path);
#endif
        int width, height, channels;
        // Load as FLOAT32 (the STB will automatically convert 0-255 to 0.0-1.0)
        float* img_data = stbi_loadf(dyn_path, &width, &height, &channels, 3);
        if (!img_data) {
#ifdef DBG
            ESP_LOGE(TAG_MEP, "RES_LOAD_DYNAMIC: Failed to load image '%s'.", dyn_path);
#endif
            release_slot(out_key);
            return;
        }

        int resize_w = 256, resize_h = 256;
        float* resized_data = (float*)alloc_fast(resize_w * resize_h * 3 * sizeof(float));
        
        if (!resized_data) {
            stbi_image_free(img_data);
            release_slot(out_key);
            return;
        }

        stbir_resize_float_linear(img_data, width, height, 0, 
                                  resized_data, resize_w, resize_h, 0, 
                                  STBIR_RGB);
        stbi_image_free(img_data); // Freeing up STB memory

        // FLOAT32 is a must!
        Tensor* t = new_tensor(DataType::FLOAT32, {1, 3, 224, 224});
        if (!t || !t->data) {
            heap_caps_free(resized_data); // Freeing up memory
            release_slot(out_key);
            return;
        }

        float* dst = (float*)t->data;
        int crop_size = 224;
        int crop_x = (resize_w - crop_size) / 2;
        int crop_y = (resize_h - crop_size) / 2;

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3]  = {0.229f, 0.224f, 0.225f};

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < crop_size; ++y) {
                for (int x = 0; x < crop_size; ++x) {
                    int src_idx = ((crop_y + y) * resize_w + (crop_x + x)) * 3 + c;
                    dst[c * (crop_size * crop_size) + y * crop_size + x] = (resized_data[src_idx] - mean[c]) / std[c];
                }
            }
        }

        heap_caps_free(resized_data);
#ifdef DBG
        ESP_LOGI(TAG_MEP, "Image preprocessed successfully: [1, 3, 224, 224] Float32.");
#endif
        put_tensor(out_key, t);
#ifdef DBG
    } else {
        ESP_LOGW(TAG_MEP, "RES_LOAD_DYNAMIC: file_type %u not supported.", file_type);
#endif
    }
}

// =============================================================================
// 0x14  RES_LOAD_ARRAY
// =============================================================================
void MEPInterpreter::h_res_load_array() {
    uint8_t out_key = ru8();
    uint16_t name_cid = ru16();
    const char* arr_name = get_const(name_cid).as_cstr();
    
    std::string key_str = arr_name ? arr_name : "";
    auto it = m_primary_ctx->arrays.find(key_str);
    
    if (it != m_primary_ctx->arrays.end()) {
        Tensor* src = it->second;
        // Copy the tensor to the pool to avoid mutations of the original
        Tensor* t = new_tensor(src->dtype, src->shape);
        if (t && t->data && src->data) {
            memcpy(t->data, src->data, src->size);
            put_tensor(out_key, t);
        } else {
            if (t) m_primary_ctx->tensor_pool.release(t);
            ESP_LOGE(TAG_MEP, "RES_LOAD_ARRAY: OOM copying array '%s'", key_str.c_str());
        }
    } else {
        ESP_LOGE(TAG_MEP, "RES_LOAD_ARRAY: Array '%s' not found in ARRS section.", key_str.c_str());
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

    NacRuntimeContext* target_ctx = static_cast<NacRuntimeContext*>(slot(proc_key).ptr);
    if (!target_ctx || !target_ctx->tokenizer) { 
        ESP_LOGE(TAG_MEP, "PREPROC_ENCODE: slot %u does not contain a valid context.", proc_key); 
        return; 
    }

    TISAVM* tok = target_ctx->tokenizer.get();
    const char* text = slot(in_key).as_cstr();
    std::string text_str(text ? text : "");

    std::vector<int32_t> ids = tok->run(target_ctx->tisa_manifest, text_str);

    if (ids.empty()) {
        ESP_LOGW(TAG_MEP, "PREPROC_ENCODE: tokenizer returned empty result. Fallback to [0].");
        ids.push_back(0); // Safe fallback to prevent falls Attention
    }

    int seq_len = (int)ids.size();
    Tensor* t = new_tensor(DataType::INT32, {1, seq_len});
    if (!t) { ESP_LOGE(TAG_MEP, "PREPROC_ENCODE: OOM."); return; }
    memcpy(t->data, ids.data(), ids.size() * sizeof(int32_t));
    put_tensor(out_key, t);

#ifdef DBG
    Serial.printf("[TOKENIZE] seq_len=%d  ids=[", seq_len);
    for (int i = 0; i < seq_len; ++i) Serial.printf("%s%d", i ? ", " : "", ids[i]);
    Serial.println("]");
#endif
}

// =============================================================================
// 0x21  PREPROC_DECODE  —  detokenize
// =============================================================================
void MEPInterpreter::h_preproc_decode() {
    uint8_t proc_key = ru8();
    uint8_t in_key   = ru8();
    uint8_t out_key  = ru8();

    NacRuntimeContext* target_ctx = static_cast<NacRuntimeContext*>(slot(proc_key).ptr);
    if (!target_ctx || !target_ctx->tokenizer) { 
        ESP_LOGE(TAG_MEP, "PREPROC_DECODE: slot %u is not a valid context.", proc_key); 
        return; 
    }

    TISAVM* tok = target_ctx->tokenizer.get();
    Tensor* t = slot(in_key).tensor;
    std::vector<int> ids;
    bool is_scalar = false;

    if (t && t->data) {
        ids.resize(t->num_elements);
        if (t->dtype == DataType::INT32) {
            const int32_t* src = static_cast<const int32_t*>(t->data);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(src[i]);
        } else if (t->dtype == DataType::FLOAT32) {
            const float* f = static_cast<const float*>(t->data);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(f[i]);
        }
        if (t->num_elements == 1) is_scalar = true;
    } else {
        MepVal& sv = slot(in_key);
        if (sv.type == MepValType::INT64 || sv.type == MepValType::FLOAT64 || sv.type == MepValType::BOOL) {
            ids.push_back(static_cast<int>(sv.as_i64()));
            is_scalar = true;
        } else {
            ESP_LOGE(TAG_MEP, "PREPROC_DECODE: no tensor or scalar in slot %u.", in_key);
            return;
        }
    }

    std::string decoded = tok->decode(ids, !is_scalar);
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

    NacRuntimeContext* target_ctx = static_cast<NacRuntimeContext*>(slot(proc_key).ptr);
    if (!target_ctx) target_ctx = m_primary_ctx; 

    const char* token_str = get_const(item_const_id).as_cstr();
    release_slot(out_key);

    int32_t tid = -1;
    if (token_str && token_str[0] != '\0' && target_ctx->tokenizer_resources.vocab) {
        target_ctx->tokenizer_resources.vocab->find(std::string(token_str), tid);
    }
    slot(out_key).set_i64((int64_t)tid);
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
                            result += tmp;
                            break;
                        case MepValType::FLOAT64:
                            if (has_prec)
                                snprintf(tmp, sizeof(tmp), "%.*f", prec, v.f64);
                            else
                                snprintf(tmp, sizeof(tmp), "%.6g", v.f64);
                            result += tmp;
                            break;
                        case MepValType::BOOL:
                            result += (v.b ? "True" : "False");
                            break;
                        case MepValType::STRING:
                            if (v.str_p) result += v.str_p;
                            break;
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
                            result += tmp;
                            break;
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
    uint8_t out_key       = ru8();
    uint8_t dtype_code    = ru8();
    uint8_t creation_type = ru8();
    uint8_t src_key       = ru8();

    DataType dt = (dtype_code == 5) ? DataType::INT32 : DataType::FLOAT32;
    Tensor* t = nullptr;
    const MepVal& src = slot(src_key);

    if (creation_type == 0) {
        // from_py
        if (src.type == MepValType::TENSOR && src.tensor && src.tensor->data) {
            Tensor* src_t = src.tensor;
            int n = (int)src_t->num_elements;
            t = new_tensor(dt, {1, n});
            if (t) {
                if (dt == src_t->dtype) {
                    memcpy(t->data, src_t->data, src_t->size);
                } else if (dt == DataType::INT32 && src_t->dtype == DataType::FLOAT32) {
                    const float* s = static_cast<const float*>(src_t->data);
                    int32_t* d = static_cast<int32_t*>(t->data);
                    for (int i = 0; i < n; ++i) d[i] = (int32_t)s[i];
                } else if (dt == DataType::FLOAT32 && src_t->dtype == DataType::INT32) {
                    const int32_t* s = static_cast<const int32_t*>(src_t->data);
                    float* d = static_cast<float*>(t->data);
                    for (int i = 0; i < n; ++i) d[i] = (float)s[i];
                } else {
                    memcpy(t->data, src_t->data, std::min(t->size, src_t->size));
                }
            }
        } else {
            // scalar
            int64_t val = src.as_i64();
            t = new_tensor(dt, {1, 1});
            if (t) {
                if (dt == DataType::INT32) *(int32_t*)t->data = (int32_t)val;
                else                       *(float*)  t->data = (float)  val;
            }
        }
    } else if (creation_type == 1) { 
        // arange(N)
        int64_t n = src.as_i64();
        t = new_tensor(DataType::INT32, {1, (int)n});
        if (t) {
            int32_t* p = (int32_t*)t->data;
            for (int i = 0; i < n; ++i) p[i] = i;
        }
    } else if (creation_type == 2 || creation_type == 3) { 
        std::vector<int> shape;
        if (src.type == MepValType::TENSOR && src.tensor && src.tensor->data) {
            if (src.tensor->dtype == DataType::INT32) {
                int32_t* p = (int32_t*)src.tensor->data;
                for (size_t i = 0; i < src.tensor->num_elements; ++i) shape.push_back(p[i]);
            } else if (src.tensor->dtype == DataType::INT64) {
                int64_t* p = (int64_t*)src.tensor->data;
                for (size_t i = 0; i < src.tensor->num_elements; ++i) shape.push_back((int)p[i]);
            }
        } else {
            shape = {1, (int)src.as_i64()};
        }
        
        t = new_tensor(dt, shape);
        if (t && t->data) {
            if (creation_type == 3) { // ZEROS
                memset(t->data, 0, t->size);
            } else { // ONES
                if (dt == DataType::FLOAT32) {
                    float* p = (float*)t->data;
                    for (size_t i = 0; i < t->num_elements; ++i) p[i] = 1.0f;
                } else if (dt == DataType::INT32) {
                    int32_t* p = (int32_t*)t->data;
                    for (size_t i = 0; i < t->num_elements; ++i) p[i] = 1;
                } else {
                    memset(t->data, 1, t->size);
                }
            }
        }
    }

    if (t) put_tensor(out_key, t);
    else   ESP_LOGE(TAG_MEP, "TENSOR_CREATE: failed (ctype=%u).", creation_type);
}

// =============================================================================
// 0x38  TENSOR_MANIPULATE  (op_type=1: left-pad)
// =============================================================================
void MEPInterpreter::h_tensor_manipulate() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();

    uint8_t extra1 = 0, extra2 = 0;
    if (op_type == 1) {
        extra1 = ru8(); // pad_width_key
        extra2 = ru8(); // pad_val_key
    } else if (op_type == 2 || op_type == 3) {
        extra1 = ru8(); // dim_key
    }

    Tensor* in = slot(in_key).tensor;
    if (!in || !in->data) {
        ESP_LOGE(TAG_MEP, "TENSOR_MANIPULATE: no tensor in slot %u.", in_key);
        return;
    }

    if (op_type == 1) { // pad: left-pad last dimension
        int pad_w = (int)slot(extra1).as_i64();
        int32_t pad_v = (int32_t)slot(extra2).as_i64();

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
    } else if (op_type == 2) { // unsqueeze
        int dim = (int)slot(extra1).as_i64();
        int ndim = in->shape.size();
        if (dim < 0) dim += ndim + 1;
        if (dim < 0) dim = 0;
        if (dim > ndim) dim = ndim;

        // GCC 15 Warning: Manually building a form without vector::insert
        std::vector<int> out_shape;
        out_shape.reserve(ndim + 1);
        for (int i = 0; i < dim; ++i) out_shape.push_back(in->shape[i]);
        out_shape.push_back(1);
        for (int i = dim; i < ndim; ++i) out_shape.push_back(in->shape[i]);

        Tensor* t = new_tensor(in->dtype, out_shape);
        if (t && t->data) {
            memcpy(t->data, in->data, in->size);
            put_tensor(out_key, t);
        } else {
            if (t) m_primary_ctx->tensor_pool.release(t);
            ESP_LOGE(TAG_MEP, "TENSOR_MANIPULATE (unsqueeze): OOM.");
        }
    } else if (op_type == 3) { // squeeze
        int dim = (int)slot(extra1).as_i64();
        int ndim = in->shape.size();
        if (dim < 0) dim += ndim;

        std::vector<int> out_shape;
        for (int i = 0; i < ndim; ++i) {
            if (i == dim && in->shape[i] == 1) continue;
            out_shape.push_back(in->shape[i]);
        }
        if (out_shape.empty()) out_shape.push_back(1);

        Tensor* t = new_tensor(in->dtype, out_shape);
        if (t && t->data) {
            memcpy(t->data, in->data, in->size);
            put_tensor(out_key, t);
        } else {
            if (t) m_primary_ctx->tensor_pool.release(t);
            ESP_LOGE(TAG_MEP, "TENSOR_MANIPULATE (squeeze): OOM.");
        }
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
        DataType dt = DataType::INT32;
        int max_dims = 0;
        for (uint8_t k : keys) {
            Tensor* t = slot(k).tensor;
            if (t) { 
                if (t->dtype == DataType::FLOAT32) dt = DataType::FLOAT32; 
                if ((int)t->shape.size() > max_dims) max_dims = t->shape.size();
            }
        }

        // Calculate the general shape taking into account the "stretching" of dimensions
        std::vector<int> final_shape;
        int concat_dim = 0; // Default (for 1D)

        // Obtain the concatenation axis
        if (slot(axis_key).type == MepValType::INT64) concat_dim = (int)slot(axis_key).i64;
        
        for (uint8_t k : keys) {
            Tensor* t = slot(k).tensor;
            if (!t) continue;
            
            // If the tensor is lower in rank (e.g. [59] vs. [1, 5]), add 1 in front
            std::vector<int> t_shape = t->shape;
            while ((int)t_shape.size() < max_dims) {
                t_shape.insert(t_shape.begin(), 1);
            }
            
            if (final_shape.empty()) {
                final_shape = t_shape;
            } else {
                final_shape[concat_dim] += t_shape[concat_dim];
            }
        }
        
        Tensor* out = new_tensor(dt, final_shape);
        if (!out) { ESP_LOGE(TAG_MEP, "TENSOR_COMBINE: OOM."); return; }

        size_t elem_size = out->get_element_byte_size();
        
        // Robust N-dimensional concatenation
        size_t outer = 1;
        for (int i = 0; i < concat_dim; ++i) outer *= final_shape[i];
        size_t inner = 1;
        for (size_t i = concat_dim + 1; i < final_shape.size(); ++i) inner *= final_shape[i];

        size_t out_offset = 0;
        for (int o = 0; o < outer; ++o) {
            for (uint8_t k : keys) {
                Tensor* t = slot(k).tensor;
                if (!t || !t->data) continue;
                
                int t_concat_size = 1;
                if (!t->shape.empty()) {
                    int eff_dim = concat_dim - (final_shape.size() - t->shape.size());
                    if (eff_dim >= 0 && eff_dim < (int)t->shape.size()) 
                        t_concat_size = t->shape[eff_dim];
                }
                
                size_t copy_bytes = t_concat_size * inner * elem_size;
                size_t src_offset = o * copy_bytes;
                
                memcpy((uint8_t*)out->data + out_offset, 
                       (uint8_t*)t->data + src_offset, copy_bytes);
                out_offset += copy_bytes;
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

    if (!t) {
        if (op_type == 2 || op_type == 0) {
            MepVal& src = slot(in_key);
            if (src.type == MepValType::INT64  || src.type == MepValType::FLOAT64 || src.type == MepValType::BOOL) {
                if (out_key != in_key) {
                    release_slot(out_key);
                    m_ctx[out_key] = m_ctx[in_key]; 
                }
                return;
            }
        }
        release_slot(out_key);
        slot(out_key).set_i64(0);
        return;
    }

    if (op_type == 0) { // shape returns a Tensor (array)
        int ndim = t->shape.size();
        Tensor* shape_t = new_tensor(DataType::INT32, {1, ndim});
        if (shape_t && shape_t->data) {
            int32_t* p = (int32_t*)shape_t->data;
            for (int i = 0; i < ndim; ++i) p[i] = t->shape[i];
            put_tensor(out_key, shape_t);
        } else {
            release_slot(out_key);
        }
    } else if (op_type == 1) { 
        uint8_t dim_key = ru8();
        int64_t dim_idx = slot(dim_key).as_i64();
        int64_t v = (dim_idx < (int64_t)t->shape.size()) ? t->shape[(int)dim_idx] : 0;
        release_slot(out_key);
        slot(out_key).set_i64(v);
    } else if (op_type == 2) { 
        if (t->num_elements == 1) {
            if      (t->dtype == DataType::FLOAT32) { float   v = *static_cast<float*>  (t->data); release_slot(out_key); slot(out_key).set_f64(v); }
            else if (t->dtype == DataType::INT32)   { int32_t v = *static_cast<int32_t*>(t->data); release_slot(out_key); slot(out_key).set_i64(v); }
            else if (t->dtype == DataType::INT64)   { int64_t v = *static_cast<int64_t*>(t->data); release_slot(out_key); slot(out_key).set_i64(v); }
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
//
// Semantics depend on tensor rank:
//   3D [B, S, D]: idx selects position along S → output [1, D]
//   2D [1, N]:    idx selects element within N → output [1, 1]  (scalar probability/logit)
//   2D [N, M]:    idx selects row → output [1, M]
//   1D [N]:       idx selects element → output [1, 1]
//
// The [1,N] special case is critical for extracting a single vocabulary probability
// from softmax output: TENSOR_EXTRACT([1,vocab_size], idx=token_id) → scalar prob.
// Without this, 2D indexing uses stride=N which puts idx*N out-of-bounds for idx>0.
// =============================================================================
void MEPInterpreter::h_tensor_extract() {
    uint8_t out_key    = ru8();
    uint8_t tensor_key = ru8();
    uint8_t idx_key    = ru8();

    Tensor* t   = slot(tensor_key).tensor;
    int64_t idx = slot(idx_key).as_i64();
    if (!t || !t->data) { release_slot(out_key); return; }

    size_t elem_size = t->get_element_byte_size();
    size_t extract_len = 1; 
    size_t offset = 0; 

    if (t->shape.size() == 3) {
        // [B, S, D]: extract slice S[idx] → [1, D]
        extract_len = (size_t)t->shape[2]; 
        offset = (size_t)idx * extract_len; 
    } else if (t->shape.size() == 2) {
        if (t->shape[0] == 1) {
            // [1, N] OR we extract an element from a flat array (eg mask [1, 9])
            // Strict fallback to 1D!
            extract_len = 1;
            offset = (size_t)idx;
        } else {
            // [N, M]: select row idx → [1, M]
            extract_len = (size_t)t->shape[1];
            offset = (size_t)idx * extract_len;
        }
    } else {
        // 1D [N]: element extraction
        extract_len = 1;
        offset = (size_t)idx;
    }

    if (offset + extract_len > t->num_elements) {
        ESP_LOGE(TAG_MEP, "TENSOR_EXTRACT: idx=%lld out of bounds (ne=%zu extract_len=%zu offset=%zu)",
                 (long long)idx, t->num_elements, extract_len, offset);
        release_slot(out_key);
        return;
    }

    Tensor* out = new_tensor(t->dtype, {1, (int)extract_len});
    if (!out) { ESP_LOGE(TAG_MEP, "TENSOR_EXTRACT: OOM."); return; }
    memcpy(out->data, (const uint8_t*)t->data + offset * elem_size, extract_len * elem_size);

    put_tensor(out_key, out);
}

// =============================================================================
// 0x59  SYS_COPY
// =============================================================================
void MEPInterpreter::h_sys_copy() {
    uint8_t out_key = ru8();
    uint8_t in_key  = ru8();
    if (out_key == in_key) return;

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
        switch (src.type) {
            case MepValType::INT64:   slot(out_key).set_i64(src.i64); break;
            case MepValType::FLOAT64: slot(out_key).set_f64(src.f64); break;
            case MepValType::BOOL:    slot(out_key).set_bool(src.b); break;
            case MepValType::STRING:
                slot(out_key).set_str(src.str_p ? std::string(src.str_p, src.str_len) : std::string());
                break;
            case MepValType::OPAQUE:  slot(out_key).set_opaque(src.ptr); break;
            default:                  slot(out_key).set_none(); break;
        }
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
    if (!in || !in->data) {
        ESP_LOGE(TAG_MEP, "MATH_UNARY: bad input tensor in slot %u.", in_key); return;
    }

    Tensor* out = new_tensor(DataType::FLOAT32, in->shape);
    if (!out) { ESP_LOGE(TAG_MEP, "MATH_UNARY: OOM."); return; }

    float* dst = static_cast<float*>(out->data);
    
    // Safely reading values ​​by casting to float
    auto get_val = [in](size_t i) -> float {
        if (in->dtype == DataType::FLOAT32) return static_cast<float*>(in->data)[i];
        if (in->dtype == DataType::INT32)   return (float)static_cast<int32_t*>(in->data)[i];
        if (in->dtype == DataType::INT64)   return (float)static_cast<int64_t*>(in->data)[i];
        return (float)static_cast<uint8_t*>(in->data)[i];
    };

    if (op_type == 0) { // softmax
        // Softmax requires a float array as input, so copy
        float* src_f32 = (float*)alloc_fast(in->num_elements * sizeof(float));
        if (src_f32) {
            for(size_t i=0; i<in->num_elements; ++i) src_f32[i] = get_val(i);
            softmax(src_f32, dst, in->num_elements);
            heap_caps_free(src_f32);
        }
    } else {
        for(size_t i = 0; i < in->num_elements; ++i) {
            float v = get_val(i);
            if      (op_type == 1) dst[i] = sqrtf(v);
            else if (op_type == 2) dst[i] = fabsf(v);
            else if (op_type == 3) dst[i] = -v;
            else if (op_type == 4) dst[i] = expf(v);
            else if (op_type == 5) dst[i] = (v != 0.0f) ? (1.0f / v) : 0.0f;
        }
    }
    
    put_tensor(out_key, out);
}

// =============================================================================
// 0x61  MATH_BINARY  (0=add, 1=sub, 2=mul, 3=div, 4=pow, 5=max, 6=min)
// =============================================================================
void MEPInterpreter::h_math_binary() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t k1      = ru8();
    uint8_t k2      = ru8();

    MepVal& v1 = slot(k1);
    MepVal& v2 = slot(k2);

    auto get_tensor_val = [](Tensor* t, size_t idx) -> float {
        if (t->dtype == DataType::FLOAT32) return ((float*)t->data)[idx];
        if (t->dtype == DataType::INT32)   return (float)((int32_t*)t->data)[idx];
        if (t->dtype == DataType::INT64)   return (float)((int64_t*)t->data)[idx];
        return (float)((uint8_t*)t->data)[idx]; 
    };

    if (v1.type == MepValType::TENSOR && v2.type == MepValType::TENSOR) {
        Tensor* t1 = v1.tensor;
        Tensor* t2 = v2.tensor;
        if (!t1 || !t2 || !t1->data || !t2->data) return;

        std::vector<int> out_shape;
        int n1 = t1->shape.size(), n2 = t2->shape.size();
        int m = std::max(n1, n2);
        for (int i = 0; i < m; ++i) {
            int d1 = (i < m - n1) ? 1 : t1->shape[i - (m - n1)];
            int d2 = (i < m - n2) ? 1 : t2->shape[i - (m - n2)];
            out_shape.push_back(std::max(d1, d2));
        }

        DataType out_dt = DataType::FLOAT32;
        if (t1->dtype == DataType::BOOL && t2->dtype == DataType::BOOL && op_type != 3) {
            out_dt = DataType::BOOL;
        } else if ((t1->dtype == DataType::INT32 || t1->dtype == DataType::INT64 || t1->dtype == DataType::BOOL) && 
                   (t2->dtype == DataType::INT32 || t2->dtype == DataType::INT64 || t2->dtype == DataType::BOOL) && op_type != 3) {
            out_dt = DataType::INT32;
        }

        Tensor* out = new_tensor(out_dt, out_shape);
        if (!out || !out->data) { if(out) m_primary_ctx->tensor_pool.release(out); return; }
        
        // --- Safe Precomputed Steps (STRIDES) ---
        std::vector<int> stride_out(m, 1), stride_a(m, 0), stride_b(m, 0);
        int sa = 1, sb = 1;
        for (int i = m - 1; i >= 0; --i) {
            if (i < m - 1) stride_out[i] = stride_out[i+1] * out_shape[i+1];
            int d1 = (i < m - n1) ? 1 : t1->shape[i - (m - n1)];
            int d2 = (i < m - n2) ? 1 : t2->shape[i - (m - n2)];
            stride_a[i] = (d1 == 1) ? 0 : sa;
            stride_b[i] = (d2 == 1) ? 0 : sb;
            if (d1 > 1) sa *= d1;
            if (d2 > 1) sb *= d2;
        }

        for (size_t i = 0; i < out->num_elements; ++i) {
            size_t idx_a = 0, idx_b = 0, temp = i;
            for (int d = 0; d < m; ++d) {
                int coord = temp / stride_out[d];
                temp %= stride_out[d];
                idx_a += coord * stride_a[d];
                idx_b += coord * stride_b[d];
            }
            // -------------------------------------------------------------

            float v_a = get_tensor_val(t1, idx_a);
            float v_b = get_tensor_val(t2, idx_b);
                        
            float res = 0.0f;
            if      (op_type == 0) res = v_a + v_b;
            else if (op_type == 1) res = v_a - v_b;
            else if (op_type == 2) res = v_a * v_b;
            else if (op_type == 3) res = (v_b != 0.0f) ? (v_a / v_b) : 0.0f;
            else if (op_type == 4) res = powf(v_a, v_b);
            else if (op_type == 5) res = std::max(v_a, v_b);
            else if (op_type == 6) res = std::min(v_a, v_b);

            if (out_dt == DataType::FLOAT32) ((float*)out->data)[i] = res;
            else if (out_dt == DataType::BOOL) ((uint8_t*)out->data)[i] = (res != 0.0f) ? 1 : 0;
            else ((int32_t*)out->data)[i] = (int32_t)res;
        }
        put_tensor(out_key, out);

    } else if (v1.type == MepValType::TENSOR) {
        Tensor* in = v1.tensor;
        if (!in || !in->data) return;
        
        float sc = 0.0f;
        if (v2.type == MepValType::INT64) sc = (float)v2.i64;
        else if (v2.type == MepValType::FLOAT64) sc = (float)v2.f64;
        else if (v2.type == MepValType::BOOL) sc = v2.b ? 1.0f : 0.0f;

        DataType out_dt = DataType::FLOAT32;
        if (in->dtype == DataType::BOOL && v2.type == MepValType::BOOL && op_type != 3) {
            out_dt = DataType::BOOL;
        } else if ((in->dtype == DataType::INT32 || in->dtype == DataType::INT64 || in->dtype == DataType::BOOL) && 
                   (v2.type == MepValType::INT64 || v2.type == MepValType::BOOL) && op_type != 3) {
            out_dt = DataType::INT32;
        }

        Tensor* out = new_tensor(out_dt, in->shape);
        if (!out || !out->data) { if(out) m_primary_ctx->tensor_pool.release(out); return; }
        
        for (size_t i = 0; i < in->num_elements; ++i) {
            float src_val = get_tensor_val(in, i);
            
            float res = 0.0f;
            if      (op_type == 0) res = src_val + sc;
            else if (op_type == 1) res = src_val - sc;
            else if (op_type == 2) res = src_val * sc;
            else if (op_type == 3) res = (sc != 0.0f) ? (src_val / sc) : 0.0f;
            else if (op_type == 4) res = powf(src_val, sc);
            else if (op_type == 5) res = std::max(src_val, sc);
            else if (op_type == 6) res = std::min(src_val, sc);

            if (out_dt == DataType::FLOAT32) ((float*)out->data)[i] = res;
            else if (out_dt == DataType::BOOL) ((uint8_t*)out->data)[i] = (res != 0.0f) ? 1 : 0;
            else ((int32_t*)out->data)[i] = (int32_t)res;
        }
        put_tensor(out_key, out);

    } else {
        double a = (v1.type == MepValType::INT64) ? (double)v1.i64 : v1.f64;
        double b = (v2.type == MepValType::INT64) ? (double)v2.i64 : v2.f64;
        
        double r = 0;
        if      (op_type == 0) r = a + b;
        else if (op_type == 1) r = a - b;
        else if (op_type == 2) r = a * b;
        else if (op_type == 3) r = (b != 0.0) ? (a / b) : 0.0;
        else if (op_type == 4) r = pow(a, b);
        else if (op_type == 5) r = std::max(a, b);
        else if (op_type == 6) r = std::min(a, b);
        
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
    if (!t || !t->data || t->dtype != DataType::FLOAT32) {
        release_slot(out_key);
        slot(out_key).set_i64(0);
        return;
    }

    if (op_type == 0) { // argmax
        std::vector<int> out_shape = t->shape;
        if (!out_shape.empty()) out_shape.pop_back();
        if (out_shape.empty()) out_shape.push_back(1);

        Tensor* out = new_tensor(DataType::INT32, out_shape);
        if (!out || !out->data) {
            release_slot(out_key);
            return;
        }

        int last_dim = t->shape.back();
        int outer_dims = out->num_elements;
        
        float* src = (float*)t->data;
        int32_t* dst = (int32_t*)out->data;

        for (int i = 0; i < outer_dims; ++i) {
            float* row = src + i * last_dim;
            float max_val = row[0];
            int32_t max_idx = 0;
            for (int j = 1; j < last_dim; ++j) {
                if (row[j] > max_val) {
                    max_val = row[j];
                    max_idx = j;
                }
            }
            dst[i] = max_idx;
        }
        put_tensor(out_key, out);
    } else {
        release_slot(out_key);
        slot(out_key).set_i64(0);
    }
}

// =============================================================================
// 0x68  LOGIC_COMPARE  (0=eq, 1=neq, 2=gt, 3=lt)
//
// Two modes depending on the TYPE of slot k1:
//
//   Scalar mode (k1 is INT64/FLOAT64/BOOL/NONE):
//     Compare two scalar values. Result: bool stored in out_key.
//     Used by DistilBERT ORCH to compare argmax class index against 0.
//
//   Tensor mode (k1 type == TENSOR):
//     Element-wise compare of tensor k1 against scalar k2.
//     Result: FLOAT32 tensor with 1.0 where condition holds, 0.0 elsewhere.
//     Used by RoBERTa ORCH: LOGIC_COMPARE(input_ids == mask_id) produces a
//     binary mask over the sequence; MATH_AGGREGATE(argmax) then finds the
//     mask token's position so the correct vocabulary logits are extracted.
//
// IMPORTANT: check slot(k1).type == TENSOR, NOT slot(k1).tensor != nullptr.
// set_i64() / set_f64() do NOT clear the tensor field, leaving stale pointers
// that would falsely trigger tensor mode on slots that hold scalars.
// =============================================================================
void MEPInterpreter::h_logic_compare() {
    uint8_t op_type = ru8();
    uint8_t out_key = ru8();
    uint8_t k1      = ru8();
    uint8_t k2      = ru8();

    if (slot(k1).type == MepValType::TENSOR && slot(k1).tensor && slot(k1).tensor->num_elements > 1) {
        Tensor* t      = slot(k1).tensor;
        int64_t scalar = slot(k2).as_i64();

        if (!t->data) goto scalar_mode;

        // IMPORTANT: Return DataType::FLOAT32, since masks are multiplied algebraically
        // within transformer graphs (e.g. (1.0 - mask) * -inf)
        Tensor* result = new_tensor(DataType::FLOAT32, std::vector<int>(t->shape));
        if (!result) goto scalar_mode;

        float* dst = static_cast<float*>(result->data);
        for (size_t i = 0; i < t->num_elements; ++i) {
            int64_t elem = 0;
            if      (t->dtype == DataType::INT32)   elem = (int64_t)static_cast<const int32_t*>(t->data)[i];
            else if (t->dtype == DataType::FLOAT32) elem = (int64_t)static_cast<const float*>(t->data)[i];
            else if (t->dtype == DataType::INT8)    elem = (int64_t)static_cast<const int8_t*>(t->data)[i];
            else if (t->dtype == DataType::UINT8 || t->dtype == DataType::BOOL) elem = (int64_t)static_cast<const uint8_t*>(t->data)[i];
            else if (t->dtype == DataType::INT64)   elem = (int64_t)static_cast<const int64_t*>(t->data)[i];
            
            bool cond;
            if      (op_type == 0) cond = (elem == scalar);
            else if (op_type == 1) cond = (elem != scalar);
            else if (op_type == 2) cond = (elem >  scalar);
            else                   cond = (elem <  scalar);
            
            dst[i] = cond ? 1.0f : 0.0f;
        }
        put_tensor(out_key, result);
        return;
    }

scalar_mode: {
        MepVal& v1 = slot(k1);
        MepVal& v2 = slot(k2);
        bool res = false;

        if (v1.type == MepValType::STRING || v2.type == MepValType::STRING) {
            std::string s1 = v1.to_string();
            std::string s2 = v2.to_string();
            if      (op_type == 0) res = (s1 == s2);
            else if (op_type == 1) res = (s1 != s2);
            else if (op_type == 2) res = (s1 >  s2);
            else                   res = (s1 <  s2);
        } else {
            int64_t a = v1.as_i64();
            int64_t b = v2.as_i64();
            if      (op_type == 0) res = (a == b);
            else if (op_type == 1) res = (a != b);
            else if (op_type == 2) res = (a >  b);
            else                   res = (a <  b);
        }

        release_slot(out_key);
        slot(out_key).set_bool(res);
    }
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

    NacRuntimeContext* ctx = m_models.count(model_id) ? m_models[model_id] : m_primary_ctx;

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.openFile(ctx->model_path.c_str());
    xSemaphoreGive(g_sd_card_mutex);

    ctx->user_input_tensors.clear();
    for (uint8_t k : in_keys) {
        Tensor* src = slot(k).tensor;
        if (src && src->data) {
            Tensor* copy = ctx->tensor_pool.acquire(); 
            if (copy) {
                copy->dtype = src->dtype;
                copy->shape = src->shape;
                copy->update_from_shape();
                copy->data = alloc_fast(copy->size);
                if (copy->data) {
                    memcpy(copy->data, src->data, src->size);
                    ctx->user_input_tensors.push_back(copy);
                } else {
                    ctx->tensor_pool.release(copy);
                    ctx->user_input_tensors.push_back(nullptr);
                }
            } else {
                ctx->user_input_tensors.push_back(nullptr);
            }
        } else {
            ctx->user_input_tensors.push_back(nullptr);
        }
    }
#ifdef DBG
    uint32_t start_ms = millis();
#endif
    std::vector<Tensor*> outputs;
    
    if (!run_model_sync(ctx, outputs)) {
        ESP_LOGE(TAG_MEP, "MODEL_RUN_STATIC: model execution failed.");
        return;
    }
#ifdef DBG
    uint32_t end_ms = millis();
    // If the model execution took more than 1 second, notify the user
    if (end_ms - start_ms > 1000) {
        printf("[MEP] Model '%s' executed in %.2f seconds.\n", 
               ctx->model_path.c_str(), (end_ms - start_ms) / 1000.0f);
        fflush(stdout);
    }
#endif
    for (size_t i = 0; i < count_out && i < outputs.size(); ++i) {
        Tensor* out_src = outputs[i];
        if (out_src && out_src->data) {
            Tensor* out_copy = new_tensor(out_src->dtype, out_src->shape); 
            if (out_copy && out_copy->data) {
                memcpy(out_copy->data, out_src->data, out_src->size);
                put_tensor(out_keys[i], out_copy);
            } else {
                if (out_copy) m_primary_ctx->tensor_pool.release(out_copy);
                release_slot(out_keys[i]);
            }
        } else {
            release_slot(out_keys[i]);
        }
        ctx->tensor_pool.release(out_src); 
    }

    for (size_t i = count_out; i < outputs.size(); ++i) {
        ctx->tensor_pool.release(outputs[i]);
    }
    
    if (ctx != m_primary_ctx) {
        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
        sdcard.openFile(m_primary_ctx->model_path.c_str());
        xSemaphoreGive(g_sd_card_mutex);
    }
}

// =============================================================================
// Training Stubs (0x82, 0x83, 0x85)
// =============================================================================
// Since the C++ runtime targets Edge Inference, the heavy learning (Backward Pass)
// is replaced here with stubs to prevent the parser from crashing when reading generic plans.
// ─── Helper function for TRNG ─────────────────────────────────────────
static size_t gather_trng_arguments(NacRuntimeContext* ctx, const ParsedInstruction& ins, uint32_t idx, const std::vector<Tensor*>& trng_results, Tensor** args_out) {
    size_t argc = 0, c_idx = 1;
    bool use_consts = !ins.C.empty() && ins.C[0] > 0;
    for (int16_t d : ins.D) {
        if (argc >= MAX_INSTRUCTION_ARITY) break;
        if (d != 0) {
            uint32_t anc = idx + d;
            args_out[argc++] = (anc < trng_results.size()) ? trng_results[anc] : nullptr;
        } else {
            if (use_consts && c_idx < ins.C.size()) {
                uint16_t cid = ins.C[c_idx++];
                args_out[argc++] = (cid < ctx->constants.size() && ctx->constants[cid]) ? ctx->constants[cid].get() : nullptr;
            } else {
                args_out[argc++] = nullptr;
            }
        }
    }
    return argc;
}

// =============================================================================
// Training Stubs (0x82, 0x83, 0x85) -> Full TRNG implementation
// =============================================================================
void MEPInterpreter::h_model_train_step() {
    uint8_t model_id = ru8();
    uint8_t loss_type = ru8();
    uint8_t count_in = ru8();
    std::vector<uint8_t> in_keys(count_in);
    for (int i=0; i<count_in; ++i) in_keys[i] = ru8();

    uint8_t count_tgt = ru8();
    std::vector<uint8_t> tgt_keys(count_tgt);
    for (int i=0; i<count_tgt; ++i) tgt_keys[i] = ru8();

    uint8_t out_loss_key = ru8();
    uint8_t lr_key = ru8();
    uint8_t logits_key = ru8();
    uint16_t head_w_id = ru16();
    uint16_t head_b_id = ru16();

    NacRuntimeContext* ctx = m_models.count(model_id) ? m_models[model_id] : m_primary_ctx;
    if (!ctx) {
        release_slot(out_loss_key); slot(out_loss_key).set_f64(0.0); return;
    }

    ctx->targets.clear();
    for (uint8_t k : tgt_keys) {
        Tensor* tgt = slot(k).tensor;
        if (tgt) ctx->targets.push_back(tgt);
    }

    Tensor* target_t = ctx->targets.empty() ? nullptr : ctx->targets[0];
    if (!target_t || !target_t->data) {
        release_slot(out_loss_key); slot(out_loss_key).set_f64(0.0); return;
    }
    
    int64_t max_target = 0;
    if (target_t->dtype == DataType::INT32) {
        int32_t* p = (int32_t*)target_t->data; for(size_t i=0; i<target_t->num_elements; ++i) if(p[i] > max_target) max_target = p[i];
    } else if (target_t->dtype == DataType::INT64) {
        int64_t* p = (int64_t*)target_t->data; for(size_t i=0; i<target_t->num_elements; ++i) if(p[i] > max_target) max_target = p[i];
    } else if (target_t->dtype == DataType::FLOAT32) {
        float* p = (float*)target_t->data; for(size_t i=0; i<target_t->num_elements; ++i) if((int64_t)p[i] > max_target) max_target = (int64_t)p[i];
    }

    std::string head_w_name = head_w_id ? get_const(head_w_id).to_string() : "";
    std::string head_b_name = head_b_id ? get_const(head_b_id).to_string() : "";
    
    auto get_editable_param = [&](const std::string& name) -> std::pair<uint16_t, Tensor*> {
        if (name.empty()) return {0, nullptr};
        uint16_t pid = 0; bool found = false;
        for (const auto& pair : ctx->param_id_to_name) {
            if (pair.second == name) { pid = pair.first; found = true; break; }
        }
        if (!found) return {0, nullptr};
        if (ctx->updated_parameters.count(pid)) return {pid, ctx->updated_parameters[pid]};

        Tensor* t = load_param_tensor(ctx, pid);
        if (t) {
            Tensor* heap_t = new Tensor();
            heap_t->dtype = t->dtype; heap_t->shape = t->shape; heap_t->update_from_shape();
            heap_t->data = alloc_fast(heap_t->size);
            if (heap_t->data) memcpy(heap_t->data, t->data, t->size);
            ctx->tensor_pool.release(t);
            ctx->updated_parameters[pid] = heap_t;
            return {pid, heap_t};
        }
        return {0, nullptr};
    };

    int current_output_size = 0;
    auto w_pair = get_editable_param(head_w_name);
    if (w_pair.second && w_pair.second->shape.size() == 2) {
        current_output_size = w_pair.second->shape[0];
        int expand_dim = std::max(0, (int)max_target - current_output_size + 1);
        if (expand_dim > 0) {
            Tensor* p = w_pair.second; int in_feat = p->shape[1];
            Tensor* new_p = new Tensor();
            new_p->dtype = p->dtype; new_p->shape = {current_output_size + expand_dim, in_feat};
            new_p->update_from_shape(); new_p->data = alloc_fast(new_p->size);
            memset(new_p->data, 0, new_p->size);
            memcpy(new_p->data, p->data, p->size);
            delete p; ctx->updated_parameters[w_pair.first] = new_p;
            
            auto b_pair = get_editable_param(head_b_name);
            if (b_pair.second && b_pair.second->shape.size() == 1) {
                Tensor* bp = b_pair.second;
                Tensor* new_bp = new Tensor();
                new_bp->dtype = bp->dtype; new_bp->shape = {current_output_size + expand_dim};
                new_bp->update_from_shape(); new_bp->data = alloc_fast(new_bp->size);
                memset(new_bp->data, 0, new_bp->size);
                memcpy(new_bp->data, bp->data, bp->size);
                delete bp; ctx->updated_parameters[b_pair.first] = new_bp;
            }
            current_output_size += expand_dim;
        }
    }

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.openFile(ctx->model_path.c_str());
    xSemaphoreGive(g_sd_card_mutex);

    ctx->user_input_tensors.clear();
    for (uint8_t k : in_keys) {
        Tensor* src = slot(k).tensor;
        if (src && src->data) {
            Tensor* copy = ctx->tensor_pool.acquire();
            if (copy) {
                copy->dtype = src->dtype; copy->shape = src->shape; copy->update_from_shape();
                copy->data = alloc_fast(copy->size);
                if (copy->data) { memcpy(copy->data, src->data, src->size); ctx->user_input_tensors.push_back(copy); }
                else { ctx->tensor_pool.release(copy); ctx->user_input_tensors.push_back(nullptr); }
            } else ctx->user_input_tensors.push_back(nullptr);
        } else ctx->user_input_tensors.push_back(nullptr);
    }

    ctx->training_mode.store(true, std::memory_order_relaxed);
    std::vector<Tensor*> outputs;
    if (!run_model_sync(ctx, outputs)) {
        ctx->training_mode.store(false, std::memory_order_relaxed);
        ESP_LOGE(TAG_MEP, "MODEL_TRAIN_STEP: forward failed.");
        release_slot(out_loss_key); slot(out_loss_key).set_f64(0.0); return;
    }
    ctx->training_mode.store(false, std::memory_order_relaxed);

    if (ctx != m_primary_ctx) {
        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY); 
        sdcard.openFile(m_primary_ctx->model_path.c_str()); 
        xSemaphoreGive(g_sd_card_mutex);
    }

    Tensor* logits = nullptr;
    if (logits_key != 0 && slot(logits_key).type == MepValType::TENSOR) logits = slot(logits_key).tensor;
    else if (!outputs.empty()) logits = outputs[0];

    if (!logits || !logits->data || logits->dtype != DataType::FLOAT32) {
        for (Tensor* t : outputs) ctx->tensor_pool.release(t);
        release_slot(out_loss_key); slot(out_loss_key).set_f64(0.0); return;
    }

    if (current_output_size == 0) current_output_size = logits->shape.back();
    int batch_size = (logits->shape.size() >= 2) ? (logits->num_elements / current_output_size) : 1;

    float loss_scalar = 0.0f;
    Tensor* grad_logits_t = new_tensor(DataType::FLOAT32, logits->shape);
    if (!grad_logits_t || !grad_logits_t->data) {
        for (Tensor* t : outputs) ctx->tensor_pool.release(t);
        release_slot(out_loss_key); slot(out_loss_key).set_f64(0.0); return;
    }
    
    float* grad_logits = (float*)grad_logits_t->data;
    float* src_log = (float*)logits->data;
    float lr = (float)slot(lr_key).as_f64();

    if (loss_type == 0) { 
        double sum_loss = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            float* l_row = src_log + b * current_output_size;
            float* g_row = grad_logits + b * current_output_size;
            
            float max_l = l_row[0];
            for (int i=1; i<current_output_size; ++i) if(l_row[i] > max_l) max_l = l_row[i];

            float sum_exp = 0.0f;
            for (int i=0; i<current_output_size; ++i) { 
                float val = expf(l_row[i] - max_l);
                g_row[i] = val;
                sum_exp += val;
            }
            float log_sum = logf(sum_exp);
            
            // Absolutely safe reading of targets!
            int tgt_idx = (b < target_t->num_elements) ? b : (target_t->num_elements - 1);
            int64_t tgt = 0;
            if (tgt_idx >= 0) {
                if (target_t->dtype == DataType::INT32) tgt = ((int32_t*)target_t->data)[tgt_idx];
                else if (target_t->dtype == DataType::INT64) tgt = ((int64_t*)target_t->data)[tgt_idx];
                else if (target_t->dtype == DataType::FLOAT32) tgt = (int64_t)((float*)target_t->data)[tgt_idx];
            }
            
            if (tgt < 0 || tgt >= current_output_size) tgt = 0;
            
            sum_loss += -(l_row[tgt] - max_l - log_sum);

            for (int i=0; i<current_output_size; ++i) { g_row[i] /= sum_exp; }
            g_row[tgt] -= 1.0f;
        }
        loss_scalar = (float)(sum_loss / batch_size);
        for(size_t i=0; i<grad_logits_t->num_elements; ++i) grad_logits[i] /= batch_size;
    } else { 
        double sum_loss = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            float* l_row = src_log + b * current_output_size;
            float* g_row = grad_logits + b * current_output_size;
            
            int tgt_idx = (b < target_t->num_elements) ? b : (target_t->num_elements - 1);
            int64_t tgt = 0;
            if (tgt_idx >= 0) {
                if (target_t->dtype == DataType::INT32) tgt = ((int32_t*)target_t->data)[tgt_idx];
                else if (target_t->dtype == DataType::INT64) tgt = ((int64_t*)target_t->data)[tgt_idx];
                else if (target_t->dtype == DataType::FLOAT32) tgt = (int64_t)((float*)target_t->data)[tgt_idx];
            }
            
            for (int i=0; i<current_output_size; ++i) {
                float target_val = (i == tgt) ? 1.0f : 0.0f;
                float diff = l_row[i] - target_val;
                sum_loss += diff * diff;
                g_row[i] = 2.0f * diff / current_output_size;
            }
        }
        loss_scalar = (float)(sum_loss / batch_size);
        for(size_t i=0; i<grad_logits_t->num_elements; ++i) grad_logits[i] /= batch_size;
    }

    bool use_trng = false;
    if (!ctx->trng_operations.empty()) {
        if (m_train_mode == "trng" || head_w_name.empty()) {
            use_trng = true;
        }
    }

    if (use_trng) {
        execute_trng_block(ctx, grad_logits_t);
        
        int updated = 0;
        for (auto& pair : ctx->trng_param_updates) {
            uint16_t pid = pair.first;
            Tensor* grad = pair.second;
            if (!grad || !grad->data) continue;
            
            auto param_pair = get_editable_param(ctx->param_id_to_name[pid]);
            if (param_pair.second) {
                Tensor* p = param_pair.second;
                float* p_data = (float*)p->data;
                size_t update_len = std::min(p->num_elements, grad->num_elements);
                
                for (size_t i = 0; i < update_len; ++i) {
                    float g_val = 0.0f;
                    if (grad->dtype == DataType::FLOAT32) g_val = ((float*)grad->data)[i];
                    else if (grad->dtype == DataType::INT32) g_val = (float)((int32_t*)grad->data)[i];
                    else if (grad->dtype == DataType::INT8) g_val = (float)((int8_t*)grad->data)[i];
                    
                    p_data[i] -= lr * g_val;
                }
                updated++;
            }
            ctx->tensor_pool.release(grad);
        }
        ctx->trng_param_updates.clear();
    } else {
        // Smart input activation search (input_act) for classification head
        auto w_p = get_editable_param(head_w_name);
        if (w_p.second && w_p.second->shape.size() == 2) {
            Tensor* w = w_p.second;
            int in_features = w->shape[1];
            Tensor* input_act = nullptr;

            // Looking for instructions that included the weight w_p.first
            for (uint32_t i = 0; i < ctx->decoded_ops.size(); ++i) {
                const auto& ins = ctx->decoded_ops[i];
                if (ins.A == 2 && ins.B == 1 && ins.C.size() >= 2 && ins.C[1] == w_p.first) {
                    // Found the weight loading. Let's look for where it's being used.
                    for (uint32_t j = i + 1; j < ctx->decoded_ops.size(); ++j) {
                        const auto& u_ins = ctx->decoded_ops[j];
                        std::string op_name = ctx->id_to_name_map[u_ins.A];
                        if (op_name == "aten.matmul.default" || op_name == "aten.addmm.default" || op_name == "aten.linear.default") {
                            for (size_t d = 0; d < u_ins.D.size(); ++d) {
                                if (u_ins.D[d] != 0 && (j + u_ins.D[d] == i)) {
                                    // The weight is used here. The second argument will be the activation.
                                    int act_arg_idx = (d == 1) ? 0 : 1; 
                                    if (u_ins.D[act_arg_idx] != 0) {
                                        uint32_t act_ip = j + u_ins.D[act_arg_idx];
                                        if (act_ip < ctx->results.size()) {
                                            input_act = ctx->results[act_ip];
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        if (input_act) break;
                    }
                }
                if (input_act) break;
            }

            // Fallback (if can't find it by the graph, use the size)
            if (!input_act) {
                for (int i = (int)ctx->results.size() - 1; i >= 0; --i) {
                    if (ctx->results[i] && ctx->results[i]->num_elements == (size_t)(batch_size * in_features)) {
                        input_act = ctx->results[i]; break;
                    }
                }
            }

            if (input_act && input_act->dtype == DataType::FLOAT32) {
                float* w_data = (float*)w->data;
                float* in_data = (float*)input_act->data;
                for (int i=0; i<current_output_size; ++i) {
                    for (int j=0; j<in_features; ++j) {
                        float g = 0.0f;
                        for (int b=0; b<batch_size; ++b) {
                            g += grad_logits[b * current_output_size + i] * in_data[b * in_features + j];
                        }
                        w_data[i * in_features + j] -= lr * g;
                    }
                }
            } else {
                ESP_LOGE(TAG_MEP, "TRNG: Could not locate FP32 input_act for layer '%s'", head_w_name.c_str());
            }
        }
        
        auto b_p = get_editable_param(head_b_name);
        if (b_p.second && b_p.second->shape.size() == 1) {
            float* b_data = (float*)b_p.second->data;
            for (int i=0; i<current_output_size; ++i) {
                float g = 0.0f;
                for (int b=0; b<batch_size; ++b) g += grad_logits[b * current_output_size + i];
                b_data[i] -= lr * g;
            }
        }
    }

    ctx->tensor_pool.release(grad_logits_t);
    for (Tensor* t : outputs) ctx->tensor_pool.release(t);
    release_slot(out_loss_key);
    slot(out_loss_key).set_f64((double)loss_scalar);
}

void MEPInterpreter::execute_trng_block(NacRuntimeContext* ctx, Tensor* grad_logits) {
    std::vector<Tensor*> trng_results(ctx->trng_operations.size(), nullptr);
    ctx->trng_param_updates.clear();

    for (size_t ip = 0; ip < ctx->trng_operations.size(); ++ip) {
        const auto& op = ctx->trng_operations[ip];
        
        std::string op_name = "";
        auto it = ctx->id_to_name_map.find(op.A);
        if (it != ctx->id_to_name_map.end()) op_name = it->second;

        // Workaround for broken _log_softmax_backward
        if (op_name == "aten._log_softmax_backward_data.default" || op_name == "aten._softmax_backward_data.default") {
            Tensor* t = ctx->tensor_pool.acquire();
            if (t) {
                t->dtype = grad_logits->dtype; t->shape = grad_logits->shape; t->update_from_shape();
                t->data = alloc_fast(t->size);
                if (t->data) memcpy(t->data, grad_logits->data, grad_logits->size);
                trng_results[ip] = t;
            }
            break;
        }
    }

    for (size_t ip = 0; ip < ctx->trng_operations.size(); ++ip) {
        if (trng_results[ip] != nullptr) continue; 

        const auto& op = ctx->trng_operations[ip];

        if (op.A == 2) { // <INPUT>
            uint8_t B = op.B;
            if (B == 1 && op.C.size() >= 2) { 
                uint16_t pid = op.C[1];
                if (ctx->updated_parameters.count(pid)) {
                    Tensor* heap_t = ctx->updated_parameters[pid];
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = heap_t->dtype; t->shape = heap_t->shape; t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) memcpy(t->data, heap_t->data, t->size);
                    }
                    trng_results[ip] = t;
                } else {
                    trng_results[ip] = load_param_tensor(ctx, pid);
                }
            } else if (B == 3 && op.C.size() >= 2) { 
                uint16_t cid = op.C[1];
                if (cid < ctx->constants.size() && ctx->constants[cid]) {
                    Tensor* src = ctx->constants[cid].get();
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = src->dtype; t->shape = src->shape; t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) memcpy(t->data, src->data, src->size);
                    }
                    trng_results[ip] = t;
                }
            } else if (B == 4 && op.C.size() >= 2) { 
                uint16_t fw_idx = op.C[1];
                if (fw_idx < ctx->results.size() && ctx->results[fw_idx]) {
                    Tensor* src = ctx->results[fw_idx];
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = src->dtype; t->shape = src->shape; t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) memcpy(t->data, src->data, src->size);
                    }
                    trng_results[ip] = t;
                }
            } else if (B == 5 && op.C.size() >= 2) { 
                uint16_t tgt_idx = op.C[1];
                if (tgt_idx < ctx->targets.size() && ctx->targets[tgt_idx]) {
                    Tensor* src = ctx->targets[tgt_idx];
                    Tensor* t = ctx->tensor_pool.acquire();
                    if (t) {
                        t->dtype = src->dtype; t->shape = src->shape; t->update_from_shape();
                        t->data = alloc_fast(t->size);
                        if (t->data) memcpy(t->data, src->data, src->size);
                    }
                    trng_results[ip] = t;
                }
            }
        } else if (op.A == 3) { // <OUTPUT>
            if (op.B == 3 && op.C.size() >= 2 && op.D.size() >= 2) { // SGD Step
                uint16_t param_id = op.C[1];
                int32_t grad_idx = (int32_t)ip + op.D[0];
                if (grad_idx >= 0 && grad_idx < (int32_t)trng_results.size() && trng_results[grad_idx]) {
                    Tensor* src = trng_results[grad_idx];
                    Tensor* grad_copy = ctx->tensor_pool.acquire();
                    if (grad_copy) {
                        grad_copy->dtype = src->dtype; grad_copy->shape = src->shape; grad_copy->update_from_shape();
                        grad_copy->data = alloc_fast(grad_copy->size);
                        if (grad_copy->data) memcpy(grad_copy->data, src->data, src->size);
                        ctx->trng_param_updates[param_id] = grad_copy;
                    }
                }
            } else if (op.B == 4) {
                break; 
            }
        } else {
            Tensor* args[MAX_INSTRUCTION_ARITY] = {nullptr};
            size_t argc = gather_trng_arguments(ctx, op, ip, trng_results, args);
            std::vector<Tensor*> temp_tensors;
            
            // Restoring tensor loss in AOTAutograd (to scalar 1.0)
            if (op.A == 22 && argc >= 2) { // 22 = nac.matmul
                for (int i = 0; i < 2; ++i) {
                    bool is_scalar_one = false;
                    if (args[i] && args[i]->num_elements == 1) {
                        if (args[i]->dtype == DataType::FLOAT32 && ((float*)args[i]->data)[0] == 1.0f) is_scalar_one = true;
                        if (args[i]->dtype == DataType::INT32 && ((int32_t*)args[i]->data)[0] == 1) is_scalar_one = true;
                    }
                    
                    if (is_scalar_one && args[1-i] && args[1-i]->shape.size() >= 2) {
                        int expected_dim = (i == 1) ? args[0]->shape.back() : args[1]->shape[args[1]->shape.size()-2];
                        Tensor* found_tensor = nullptr;
                        
                        // Looking in the updated parameters
                        for (const auto& pair : ctx->updated_parameters) {
                            if (pair.second && pair.second->shape.size() >= 2) {
                                int p_dim = (i == 1) ? pair.second->shape[pair.second->shape.size()-2] : pair.second->shape.back();
                                if (p_dim == expected_dim) { found_tensor = pair.second; break; }
                            }
                        }
                        
                        // Searching for FW in saved activations
                        if (!found_tensor) {
                            for (int j = (int)ctx->results.size() - 1; j >= 0; --j) {
                                Tensor* fw_res = ctx->results[j];
                                if (fw_res && fw_res->shape.size() >= 2) {
                                    int fw_dim = (i == 1) ? fw_res->shape[fw_res->shape.size()-2] : fw_res->shape.back();
                                    if (fw_dim == expected_dim) { found_tensor = fw_res; break; }
                                }
                            }
                        }
                        
                        // Fallback (if nothing is found, create an Identity)
                        if (found_tensor) {
                            args[i] = found_tensor;
                        } else {
                            Tensor* identity = ctx->tensor_pool.acquire();
                            if (identity) {
                                identity->dtype = DataType::FLOAT32; identity->shape = {expected_dim, expected_dim};
                                identity->update_from_shape();
                                identity->data = alloc_fast(identity->size);
                                if (identity->data) {
                                    memset(identity->data, 0, identity->size);
                                    for(int j=0; j<expected_dim; ++j) ((float*)identity->data)[j * expected_dim + j] = 1.0f;
                                }
                                args[i] = identity; 
                                temp_tensors.push_back(identity);
                            }
                        }
                    }
                }
            }

            auto kit = g_op_kernels.find(op.A);
            if (kit != g_op_kernels.end()) {
                trng_results[ip] = kit->second(ctx, op, args, argc);
            }
            
            for (Tensor* t : temp_tensors) ctx->tensor_pool.release(t);
        }
    }

    for (size_t i = 0; i < trng_results.size(); ++i) {
        if (trng_results[i]) ctx->tensor_pool.release(trng_results[i]);
    }
}

void MEPInterpreter::h_model_zero_grad() { ru8(); }

// =============================================================================
// 0x85  MODEL_SAVE_WEIGHTS
// =============================================================================
void MEPInterpreter::h_model_save_weights() { 
    uint8_t model_id = ru8();
    ru8(); ru8(); // Skipping path_key and save_type

    NacRuntimeContext* ctx = m_models.count(model_id) ? m_models[model_id] : m_primary_ctx;
    if (!ctx || ctx->updated_parameters.empty()) {
        ESP_LOGD(TAG_MEP, "MODEL_SAVE_WEIGHTS: No parameters were updated.");
        return;
    }

#ifdef ARDUINO
    ESP_LOGW(TAG_MEP, "Dynamic weight saving is not supported on ESP32.");
    return;
#else
    ESP_LOGI(TAG_MEP, "Saving %zu updated weights to disk...", ctx->updated_parameters.size());

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.closeFile();
    FILE* f = fopen(ctx->model_path.c_str(), "rb");
    if (!f) {
        ESP_LOGE(TAG_MEP, "Failed to open file for reading: %s", ctx->model_path.c_str());
        sdcard.openFile(ctx->model_path.c_str());
        xSemaphoreGive(g_sd_card_mutex);
        return;
    }

    // 1. Reading the file header
    uint8_t header[100];
    fread(header, 1, 100, f);
    
    uint16_t d_model; memcpy(&d_model, header + 10, 2);
    uint64_t offsets[11]; memcpy(offsets, header + 12, sizeof(offsets));
    uint64_t data_off = offsets[5];

    if (data_off == 0) {
        fclose(f);
        sdcard.openFile(ctx->model_path.c_str());
        xSemaphoreGive(g_sd_card_mutex);
        ESP_LOGE(TAG_MEP, "No DATA section found in .nac");
        return;
    }

    // 2. Find the start and end of the tensor block inside the DATA section
    fseek(f, data_off, SEEK_SET);
    char tag[4]; fread(tag, 1, 4, f);
    
    uint32_t num_p; fread(&num_p, 1, 4, f);
    for (uint32_t i = 0; i < num_p; ++i) { fseek(f, 2, SEEK_CUR); uint16_t l; fread(&l, 1, 2, f); fseek(f, l, SEEK_CUR); }
    uint32_t num_in; fread(&num_in, 1, 4, f);
    for (uint32_t i = 0; i < num_in; ++i) { fseek(f, 2, SEEK_CUR); uint16_t l; fread(&l, 1, 2, f); fseek(f, l, SEEK_CUR); }
    
    uint64_t tensors_start_offset = ftell(f);

    uint64_t tensors_end_offset = 0;
    for (int i = 6; i < 11; ++i) {
        if (offsets[i] > data_off) {
            tensors_end_offset = (tensors_end_offset == 0) ? offsets[i] : std::min(tensors_end_offset, offsets[i]);
        }
    }
    
    fseek(f, 0, SEEK_END);
    uint64_t file_size = ftell(f);
    if (tensors_end_offset == 0) tensors_end_offset = file_size;

    // 3. Reading blocks before and after tensors
    std::vector<uint8_t> chunk_before(tensors_start_offset);
    fseek(f, 0, SEEK_SET);
    fread(chunk_before.data(), 1, tensors_start_offset, f);

    std::vector<uint8_t> chunk_after(file_size - tensors_end_offset);
    fseek(f, tensors_end_offset, SEEK_SET);
    fread(chunk_after.data(), 1, file_size - tensors_end_offset, f);
    fclose(f);

    // 4. Forming a new block of tensors
    std::vector<uint8_t> tensor_block_bytes;
    
    // Count the ACTUAL number of tensors present
    uint32_t actual_num_tensors = 0;
    for (bool present : ctx->param_present) {
        if (present) actual_num_tensors++;
    }
    
    uint8_t* p = (uint8_t*)&actual_num_tensors;
    tensor_block_bytes.insert(tensor_block_bytes.end(), p, p + 4);

    uint32_t num_tensors = ctx->param_locations.size();
    for (uint16_t pid = 0; pid < num_tensors; ++pid) {
        if (!ctx->param_present[pid]) continue;

        Tensor* t = ctx->updated_parameters.count(pid) ? ctx->updated_parameters[pid] : nullptr;
        
        if (!t) {
            const TensorLocation& loc = ctx->param_locations[pid];
            uint32_t ml = loc.meta_len;
            uint64_t dl = loc.data_size;

            uint8_t hdr_bytes[14];
            memcpy(hdr_bytes, &pid, 2);
            memcpy(hdr_bytes + 2, &ml, 4);
            memcpy(hdr_bytes + 6, &dl, 8);
            tensor_block_bytes.insert(tensor_block_bytes.end(), hdr_bytes, hdr_bytes + 14);

            FILE* read_f = fopen(ctx->model_path.c_str(), "rb");
            if (read_f) {
                std::vector<uint8_t> tmp(ml + dl);
                fseek(read_f, loc.meta_offset, SEEK_SET);
                fread(tmp.data(), 1, ml, read_f);
                fseek(read_f, loc.file_offset, SEEK_SET);
                fread(tmp.data() + ml, 1, dl, read_f);
                tensor_block_bytes.insert(tensor_block_bytes.end(), tmp.begin(), tmp.end());
                fclose(read_f);
            }
            continue;
        }

        // If the tensor HAS BEEN UPDATED, requantize and store the new data
        std::vector<uint8_t> new_meta_bytes;
        std::vector<uint8_t> new_data_bytes;
        float* src = (float*)t->data;
        int n = t->num_elements;
        
        uint8_t orig_dtype = 0; // Float32
        new_meta_bytes.push_back(orig_dtype);
        new_meta_bytes.push_back((uint8_t)t->shape.size());
        for (int d : t->shape) {
            uint32_t ud = d; uint8_t* bp = (uint8_t*)&ud;
            new_meta_bytes.insert(new_meta_bytes.end(), bp, bp + 4);
        }
        new_meta_bytes.push_back(0); // quant_type = none

        new_data_bytes.resize(n * 4);
        memcpy(new_data_bytes.data(), src, n * 4);

        uint32_t ml = new_meta_bytes.size();
        uint64_t dl = new_data_bytes.size();
        uint8_t hdr_bytes[14];
        memcpy(hdr_bytes, &pid, 2);
        memcpy(hdr_bytes + 2, &ml, 4);
        memcpy(hdr_bytes + 6, &dl, 8);

        tensor_block_bytes.insert(tensor_block_bytes.end(), hdr_bytes, hdr_bytes + 14);
        tensor_block_bytes.insert(tensor_block_bytes.end(), new_meta_bytes.begin(), new_meta_bytes.end());
        tensor_block_bytes.insert(tensor_block_bytes.end(), new_data_bytes.begin(), new_data_bytes.end());
    }

    // 5. Calculate the difference and edit the title
    int64_t diff = (int64_t)tensor_block_bytes.size() - (int64_t)(tensors_end_offset - tensors_start_offset);

    if (diff != 0) {
        ESP_LOGI(TAG_MEP, "Model architecture changed. Expanding .nac sections (Delta: %lld bytes).", (long long)diff);
        uint64_t new_offsets[11];
        for (int i = 0; i < 11; ++i) {
            if (offsets[i] > data_off) new_offsets[i] = offsets[i] + diff;
            else new_offsets[i] = offsets[i];
        }
        memcpy(chunk_before.data() + 12, new_offsets, sizeof(new_offsets));
    } else {
        ESP_LOGI(TAG_MEP, "Dimensions unchanged. Performing fast update.");
    }

    // 6. Overwriting the file
    FILE* out_f = fopen(ctx->model_path.c_str(), "wb");
    if (out_f) {
        fwrite(chunk_before.data(), 1, chunk_before.size(), out_f);
        fwrite(tensor_block_bytes.data(), 1, tensor_block_bytes.size(), out_f);
        fwrite(chunk_after.data(), 1, chunk_after.size(), out_f);
        fclose(out_f);
        ESP_LOGI(TAG_MEP, "Successfully saved weights directly into '%s'", ctx->model_path.c_str());
    } else {
        ESP_LOGE(TAG_MEP, "Failed to write updated file.");
    }

    sdcard.openFile(ctx->model_path.c_str());
    xSemaphoreGive(g_sd_card_mutex);
#endif
}

// =============================================================================
// Flow control
// =============================================================================

void MEPInterpreter::h_flow_loop_start() {
    uint8_t counter_key = ru8();
    int64_t count = slot(counter_key).as_i64();
    MepLoopFrame frame{ m_ip, count };
    m_loop_stack.push_back(frame);

#ifdef DBG
    printf("\n[MEP] Loop started: %lld iterations\n", (long long)count);
    fflush(stdout);
#endif
    if (count <= 0) {
        uint32_t search_ip = m_ip;
        int balance = 1;
        while (search_ip < m_plan_size && balance > 0) {
            uint8_t flag = m_plan[search_ip];
            if      (flag == 0xA0) balance++;
            else if (flag == 0xA1) balance--;
            if (balance == 0) break;
            search_ip += instr_len(search_ip);
        }
        m_ip = search_ip + instr_len(search_ip); 
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
#ifdef DBG
    printf("[MEP] Loop step completed. Remaining: %lld\n", (long long)frame.remaining);
    fflush(stdout);
#endif
    if (frame.remaining > 0) {
        m_ip += jump_offset; 
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
        if (v.type == MepValType::INT64)        snprintf(buf, sizeof(buf), "%lld", (long long)v.i64);
        else if (v.type == MepValType::FLOAT64)  snprintf(buf, sizeof(buf), "%.4f", v.f64);
        else if (v.type == MepValType::BOOL)     snprintf(buf, sizeof(buf), "%s", v.b ? "true" : "false");
        else if (v.type == MepValType::TENSOR && v.tensor)
            snprintf(buf, sizeof(buf), "Tensor[%u]", (unsigned)v.tensor->num_elements);
        else strncpy(buf, v.as_cstr(), sizeof(buf) - 1);
        release_slot(out_key);
        slot(out_key).set_str(buf);
    } 
    else if (format_type == 2) { // PNG (Universal Implementation)
        MepVal& v = slot(in_key);
        if (v.type == MepValType::TENSOR && v.tensor && v.tensor->data) {
            Tensor* t = v.tensor;
            
            int C = 1, H = 1, W = 1;
            bool is_chw = true;

            // Universal dimensional parsing (2D, 3D, 4D)
            if (t->shape.size() == 4) {
                C = t->shape[1]; H = t->shape[2]; W = t->shape[3];
            } else if (t->shape.size() == 3) {
                if (t->shape[0] <= 4) { C = t->shape[0]; H = t->shape[1]; W = t->shape[2]; is_chw = true; }
                else { H = t->shape[0]; W = t->shape[1]; C = t->shape[2]; is_chw = false; }
            } else if (t->shape.size() == 2) {
                H = t->shape[0]; W = t->shape[1]; C = 1;
            } else {
                ESP_LOGE(TAG_MEP, "SERIALIZE_OBJECT(PNG): Invalid tensor rank %zu.", t->shape.size());
                goto png_error;
            }

            if (C < 1 || C > 4) {
                ESP_LOGE(TAG_MEP, "SERIALIZE_OBJECT(PNG): Invalid channels count %d. Must be 1-4.", C);
                goto png_error;
            }

            // Buffer for STB (HWC format, 8-bit)
            uint8_t* rgb_data = (uint8_t*)alloc_fast(H * W * C);
            if (rgb_data) {
                if (t->dtype == DataType::FLOAT32) {
                    float* src = (float*)t->data;
                    
                    // Universal Auto-Ranging (determine the distribution of values)
                    float t_min = src[0], t_max = src[0];
                    for(size_t i = 1; i < t->num_elements; ++i) {
                        if (src[i] < t_min) t_min = src[i];
                        if (src[i] > t_max) t_max = src[i];
                    }

                    bool is_symmetric = (t_min < -0.1f); // For example [-1.0, 1.0]
                    bool is_255       = (t_max > 2.0f);  // For example [0.0, 255.0]

                    for (int c = 0; c < C; ++c) {
                        for (int h = 0; h < H; ++h) {
                            for (int w = 0; w < W; ++w) {
                                // Take the first image from the batch if it is a 4D tensor
                                int src_idx = is_chw ? (c * (H * W) + h * W + w) : (h * W * C + w * C + c);
                                float val = src[src_idx];
                                
                                if (is_255) {
                                    // It's already 0-255, not doing anything.
                                } else if (is_symmetric) {
                                    val = (val + 1.0f) * 127.5f; // [-1, 1] -> [0, 255]
                                } else {
                                    val = val * 255.0f; // [0, 1] -> [0, 255]
                                }
                                
                                val = std::max(0.0f, std::min(255.0f, val));
                                rgb_data[(h * W + w) * C + c] = (uint8_t)val;
                            }
                        }
                    }
                } else if (t->dtype == DataType::INT32 || t->dtype == DataType::INT8 || t->dtype == DataType::UINT8) {
                    // Processing integer tensors
                    for (int c = 0; c < C; ++c) {
                        for (int h = 0; h < H; ++h) {
                            for (int w = 0; w < W; ++w) {
                                int src_idx = is_chw ? (c * (H * W) + h * W + w) : (h * W * C + w * C + c);
                                float val = 0.0f;
                                if (t->dtype == DataType::INT32) val = (float)((int32_t*)t->data)[src_idx];
                                else if (t->dtype == DataType::INT8) val = (float)((int8_t*)t->data)[src_idx];
                                else if (t->dtype == DataType::UINT8) val = (float)((uint8_t*)t->data)[src_idx];
                                
                                val = std::max(0.0f, std::min(255.0f, val));
                                rgb_data[(h * W + w) * C + c] = (uint8_t)val;
                            }
                        }
                    }
                } else {
                    ESP_LOGE(TAG_MEP, "SERIALIZE_OBJECT(PNG): Unsupported tensor dtype.");
                    heap_caps_free(rgb_data);
                    goto png_error;
                }

                // Writing to memory in PNG format
                int png_len = 0;
                unsigned char* png_data = stbi_write_png_to_mem(rgb_data, 0, W, H, C, &png_len);
                heap_caps_free(rgb_data);

                if (png_data && png_len > 0) {
                    release_slot(out_key);
                    // The result is the raw bytes of the PNG binary in the (MepValType::STRING) slot.
                    slot(out_key).type = MepValType::STRING;
                    slot(out_key).str_len = png_len;
                    slot(out_key).str_p = (char*)malloc(png_len + 1);
                    if (slot(out_key).str_p) {
                        memcpy(slot(out_key).str_p, png_data, png_len);
                        slot(out_key).str_p[png_len] = '\0';
                    }
                    STBIW_FREE(png_data);
                    return;
                }
            }
        } else {
            ESP_LOGE(TAG_MEP, "SERIALIZE_OBJECT(PNG): Input slot %u is not a valid tensor.", in_key);
        }
    png_error:
        release_slot(out_key);
        slot(out_key).set_str(""); 
    }
    else {
        ESP_LOGW(TAG_MEP, "SERIALIZE_OBJECT: format_type %u not implemented.", format_type);
    }
}

// =============================================================================
// 0xF0  IO_WRITE
// =============================================================================
void MEPInterpreter::h_io_write() {
    uint8_t in_key    = ru8();
    uint8_t dest_type = ru8();
    uint8_t dest_key  = ru8();
    uint8_t write_mode = ru8();

    const char* text = nullptr;
    char tmp[64];
    MepVal& v = slot(in_key);

    switch (v.type) {
        case MepValType::STRING:  text = v.as_cstr(); break;
        case MepValType::INT64:   snprintf(tmp, sizeof(tmp), "%lld", (long long)v.i64);  text = tmp; break;
        case MepValType::FLOAT64: snprintf(tmp, sizeof(tmp), "%.4f", v.f64);  text = tmp; break;
        case MepValType::BOOL:    text = v.b ? "true" : "false"; break;
        default: text = ""; break;
    }
    if (!text) text = "";

    if (dest_type == 0 || dest_type == 1) { // STDOUT / STDERR
        FILE* stream = (dest_type == 0) ? stdout : stderr;
        if (write_mode == 2) {
            fprintf(stream, "%s", text);
        } else {
            fprintf(stream, "%s\n", text);
        }
        fflush(stream);
    } else if (dest_type == 2) { // Write a binary file to SD
        const char* filename = slot(dest_key).as_cstr();
        if (filename && filename[0]) {
            const char* mode = (write_mode == 0) ? "wb" : "ab";
            FILE* f = fopen(filename, mode);
            if (f) {
                if (v.type == MepValType::STRING && v.str_p) {
                    fwrite(v.str_p, 1, v.str_len, f); // Secure binary save
                } else {
                    fwrite(text, 1, strlen(text), f);
                }
                if (write_mode != 2 && v.type != MepValType::STRING) {
                    fwrite("\n", 1, 1, f);
                }
                fclose(f);
            } else {
                ESP_LOGE(TAG_MEP, "IO_WRITE: Cannot open file '%s'", filename);
            }
        }
    }
}

// =============================================================================
// 0xFE  EXEC_RETURN
// =============================================================================
void MEPInterpreter::h_exec_return() {
    uint8_t count = ru8();
    if (count == 1) {
        uint8_t k = ru8();
        m_return_val._free_str();
        const MepVal& src = m_ctx[k];
        switch (src.type) {
            case MepValType::INT64:   m_return_val.set_i64(src.i64); break;
            case MepValType::FLOAT64: m_return_val.set_f64(src.f64); break;
            case MepValType::BOOL:    m_return_val.set_bool(src.b); break;
            case MepValType::STRING:
                m_return_val.set_str(src.str_p ? std::string(src.str_p, src.str_len) : std::string());
                break;
            case MepValType::OPAQUE:  m_return_val.set_opaque(src.ptr); break;
            case MepValType::TENSOR:
                m_return_val.set_tensor(src.tensor);
                m_ctx[k].tensor = nullptr; // prevent release in ~MEPInterpreter
                break;
            default:
                m_return_val.set_none();
                break;
        }
        // Tensor: transfer ownership — clear the slot so destructor won't release
    } else {
        for (uint8_t i = 0; i < count; ++i) ru8(); // consume keys
    }
    m_running = false;
}

void MEPInterpreter::h_exec_halt() { m_running = false; }

// =============================================================================
// Synchronous model execution helpers
// =============================================================================

Tensor* MEPInterpreter::load_param_tensor(NacRuntimeContext* ctx, uint16_t param_id) {
    // If the weight was updated during training, take it from RAM
    if (ctx->updated_parameters.count(param_id)) {
        Tensor* heap_t = ctx->updated_parameters[param_id];
        Tensor* t = ctx->tensor_pool.acquire();
        if (t) {
            t->dtype = heap_t->dtype; 
            t->shape = heap_t->shape; 
            t->update_from_shape();
            t->data = alloc_fast(t->size);
            if (t->data) memcpy(t->data, heap_t->data, t->size);
            else { ctx->tensor_pool.release(t); return nullptr; }
        }
        return t;
    }

    if (param_id >= ctx->param_locations.size() || !ctx->param_present[param_id]) return nullptr;
    const TensorLocation& loc = ctx->param_locations[param_id];
    if (loc.data_size == 0) return nullptr;
    Tensor* t = ctx->tensor_pool.acquire();
    if (!t) return nullptr;

    if (loc.meta_len > 0) {
        std::vector<uint8_t> meta_buf(loc.meta_len);
        xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
        bool ok = sdcard.seek(loc.meta_offset) && (sdcard.readData(meta_buf.data(), loc.meta_len) == loc.meta_len);
        xSemaphoreGive(g_sd_card_mutex);

        if (!ok || !parse_quant_meta_from_buffer(t, meta_buf.data(), meta_buf.size())) {
            ctx->tensor_pool.release(t); return nullptr;
        }
        if (t->quant_meta.quant_type == 4 && t->quant_meta.original_shape.empty()) t->quant_meta.original_shape = t->shape;
    }

    t->data = alloc_fast(loc.data_size);
    if (!t->data) { ctx->tensor_pool.release(t); return nullptr; }

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(loc.file_offset);
    sdcard.readData(static_cast<uint8_t*>(t->data), loc.data_size);
    xSemaphoreGive(g_sd_card_mutex);

    if (ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
        if (!dequantize_tensor(t)) { ctx->tensor_pool.release(t); return nullptr; }
    }
    t->param_location = loc;
    return t;
}

// =============================================================================
// mmap_tick_sync  —  SUPERSEDED, not called
// =============================================================================
// This function was an inline replacement for the nac_memory_task thread for the MEP path,
// when the task was not running. Now nac_memory_task is always active (for
// both paths, MEP and legacy), and notify is sent from run_model_sync after
// each results[idx] is written. This function is left as a reference implementation
// of MMAP semantics - do not remove without a good reason.
// If single-threaded mode is ever needed without FreeRTOS, call
// mmap_tick_sync(idx) instead of xTaskNotifyGive in run_model_sync.
void MEPInterpreter::mmap_tick_sync(NacRuntimeContext* ctx, uint32_t tick) {
    auto it = ctx->mmap_schedule.find(tick);
    if (it == ctx->mmap_schedule.end()) return;

    for (const MmapCommand& cmd : it->second) {
        switch (cmd.action) {
            case MmapAction::PRELOAD: {
                uint16_t target_op_idx = cmd.target_id;
                if (target_op_idx >= ctx->decoded_ops.size()) break;
                const ParsedInstruction& ins = ctx->decoded_ops[target_op_idx];
                if (ins.A != 2 || ins.B != 1 || ins.C.size() < 2) break;
                uint16_t param_id = ins.C[1];
                if (target_op_idx < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[target_op_idx]) break;

                Tensor* t = load_param_tensor(ctx, param_id);
                if (t) {
                    if (target_op_idx >= ctx->fast_memory_cache.size()) ctx->fast_memory_cache.resize(target_op_idx + 1, nullptr);
                    ctx->fast_memory_cache[target_op_idx] = t;
                }
                break;
            }
            case MmapAction::FREE: {
                // Stopping garbage collection during training to preserve activations
                if (ctx->training_mode.load(std::memory_order_relaxed)) break;
                
                uint16_t tid = cmd.target_id;
                if (tid < ctx->results.size() && ctx->results[tid]) {
                    Tensor* to_free = ctx->results[tid];
                    for (auto& rp : ctx->results) { if (rp == to_free) rp = nullptr; }
                    for (auto& cp : ctx->fast_memory_cache) { if (cp == to_free) cp = nullptr; }
                    ctx->tensor_pool.release(to_free);
                }
                break;
            }
            case MmapAction::SAVE_RESULT: {
                if (tick < ctx->results.size() && ctx->results[tick]) {
                    uint16_t slot_id = cmd.target_id;
                    if (slot_id >= ctx->fast_memory_cache.size()) ctx->fast_memory_cache.resize(slot_id + 1, nullptr);
                    ctx->fast_memory_cache[slot_id] = ctx->results[tick];
                }
                break;
            }
            case MmapAction::FORWARD: {
                uint16_t src = (uint16_t)tick;
                uint16_t dst = cmd.target_id;
                if (src < ctx->results.size() && ctx->results[src] && dst < ctx->results.size()) {
                    if (ctx->results[dst] && ctx->results[dst] != ctx->results[src]) {
                        ctx->tensor_pool.release(ctx->results[dst]);
                    }
                    ctx->results[dst] = ctx->results[src];
                }
                break;
            }
        }
    }
}

bool MEPInterpreter::run_model_sync(NacRuntimeContext* ctx, std::vector<Tensor*>& out_tensors) {
    {
        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
        std::set<Tensor*> to_release;
        for (Tensor* r : ctx->results)           { if (r) to_release.insert(r); }
        for (Tensor* c : ctx->fast_memory_cache) { if (c) to_release.insert(c); }
        for (auto*& r : ctx->results)           r = nullptr;
        for (auto*& c : ctx->fast_memory_cache) c = nullptr;
        xSemaphoreGive(ctx->cache_mutex);
        for (Tensor* t : to_release) ctx->tensor_pool.release(t);
    }

    if (ctx->results.size() != ctx->decoded_ops.size()) {
        ctx->results.assign(ctx->decoded_ops.size(), nullptr);
    }
    ctx->stop_flag.store(false);
    size_t user_input_idx = 0;
    Tensor* last_user_input = nullptr;

#ifdef DBG
    ESP_LOGI(TAG_MEP, "--- Starting synchronous execution of model '%s' (%zu ops) ---", 
             ctx->model_path.c_str(), ctx->decoded_ops.size());
#endif

    for (uint32_t idx = 0; idx < ctx->decoded_ops.size() && !ctx->stop_flag.load(); ++idx) {
        const ParsedInstruction& ins = ctx->decoded_ops[idx];
        Tensor* result_tensor = nullptr;

#ifdef DBG
        uint32_t op_start_ms = millis();
#endif

        if (ins.A == 2) { 
            if (ins.B == 0) { 
                if (user_input_idx < ctx->user_input_tensors.size()) {
                    result_tensor = ctx->user_input_tensors[user_input_idx];
                    ctx->user_input_tensors[user_input_idx] = nullptr;
                    user_input_idx++;
                    
                    if (last_user_input) ctx->tensor_pool.release(last_user_input);
                    last_user_input = ctx->tensor_pool.acquire();
                    if (last_user_input) {
                        last_user_input->dtype = result_tensor->dtype;
                        last_user_input->shape = result_tensor->shape;
                        last_user_input->update_from_shape();
                        last_user_input->data = alloc_fast(last_user_input->size);
                        if (last_user_input->data) memcpy(last_user_input->data, result_tensor->data, result_tensor->size);
                    }
                } else {
                    if (last_user_input && last_user_input->data) {
                        result_tensor = ctx->tensor_pool.acquire();
                        if (result_tensor) {
                            result_tensor->dtype = last_user_input->dtype;
                            result_tensor->shape = last_user_input->shape;
                            result_tensor->update_from_shape();
                            result_tensor->data = alloc_fast(result_tensor->size);
                            if (result_tensor->data) memcpy(result_tensor->data, last_user_input->data, result_tensor->size);
                        }
                    } else {
                        result_tensor = ctx->tensor_pool.acquire();
                        if (result_tensor) {
                            result_tensor->dtype = DataType::FLOAT32;
                            result_tensor->shape = {1};
                            result_tensor->update_from_shape();
                            result_tensor->data = alloc_fast(result_tensor->size);
                            if (result_tensor->data) memset(result_tensor->data, 0, result_tensor->size);
                        }
                    }
                }
            } else if (ins.B == 1) { 
                if (ins.C.size() >= 2) {
                    uint16_t param_id = ins.C[1];
                    xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                    if (idx < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[idx]) {
                        result_tensor = ctx->fast_memory_cache[idx];
                        ctx->fast_memory_cache[idx] = nullptr;
                    }
                    xSemaphoreGive(ctx->cache_mutex);
                    if (!result_tensor) {
                        result_tensor = load_param_tensor(ctx, param_id);
                    }
                }
            } else if (ins.B == 3) { 
                if (ins.C.size() >= 2) {
                    uint16_t cid = ins.C[1];
                    if (cid < ctx->constants.size() && ctx->constants[cid]) {
                        Tensor* src = ctx->constants[cid].get();
                        if (src && src->data) {
                            Tensor* copy = ctx->tensor_pool.acquire();
                            if (copy) {
                                copy->dtype        = src->dtype;
                                copy->shape        = src->shape;
                                copy->update_from_shape();
                                copy->data         = alloc_fast(copy->size);
                                if (copy->data) {
                                    memcpy(copy->data, src->data, copy->size);
                                    result_tensor = copy;
                                } else {
                                    ctx->tensor_pool.release(copy);
                                }
                            }
                        }
                    }
                }
            }
        } else if (ins.A == 3) { 
            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            for (int16_t offset : ins.D) {
                int32_t src_idx = (int32_t)idx + (int32_t)offset;
                if (src_idx >= 0 && src_idx < (int32_t)ctx->results.size() && ctx->results[src_idx]) {
                    out_tensors.push_back(ctx->results[src_idx]);
                    ctx->results[src_idx] = nullptr; 
                }
            }
            xSemaphoreGive(ctx->cache_mutex);
            if (ins.B == 1) break; 
        } else if (ins.A >= 10) { 
            Tensor* arguments[MAX_INSTRUCTION_ARITY] = {};
            size_t argc = gather_arguments(*ctx, ins, idx, arguments);
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
            
            if (result_tensor && ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT && result_tensor->dtype == DataType::FLOAT32) {
                requantize_result_tensor(*ctx, result_tensor);
            }
        }

#ifdef DBG
        uint32_t op_end_ms = millis();
        uint32_t duration = op_end_ms - op_start_ms;
        
        std::string op_name = "Unknown";
        auto it = ctx->id_to_name_map.find(ins.A);
        if (it != ctx->id_to_name_map.end()) op_name = it->second;
        else if (ins.A == 2) op_name = "<INPUT>";
        else if (ins.A == 3) op_name = "<OUTPUT>";
        
        // Print operations that took more than 100 ms to catch SD freezes
        if (duration >= 100 || (idx % 10 == 0) || idx == ctx->decoded_ops.size() - 1) {
            ESP_LOGI("PERF", "[Model %s] Op %u (A=%u %s) took %u ms", 
                     ctx->model_path.c_str(), idx, ins.A, op_name.c_str(), duration);
        }
#endif

        if (idx < ctx->results.size()) {
            ctx->results[idx] = result_tensor;
            ctx->current_instruction_idx.store(idx, std::memory_order_release);
            
            if (ctx == m_primary_ctx && g_nac_memory_task_handle) {
                if (ctx->mmap_sync_event) xEventGroupClearBits(ctx->mmap_sync_event, ctx->MMAP_DONE_BIT);
                xTaskNotifyGive(g_nac_memory_task_handle);
                if (ctx->mmap_sync_event) xEventGroupWaitBits(ctx->mmap_sync_event, ctx->MMAP_DONE_BIT, pdTRUE, pdFALSE, portMAX_DELAY);
            } else {
                mmap_tick_sync(ctx, idx);
            }
        } else if (result_tensor) {
            ctx->tensor_pool.release(result_tensor);
        }
    }

#ifdef DBG
    ESP_LOGI(TAG_MEP, "--- Finished synchronous execution of model '%s' ---", ctx->model_path.c_str());
#endif

    if (last_user_input) ctx->tensor_pool.release(last_user_input);
    
    if (out_tensors.empty()) {
        for (int i = (int)ctx->results.size() - 1; i >= 0; --i) {
            if (ctx->results[i]) { 
                out_tensors.push_back(ctx->results[i]); 
                ctx->results[i] = nullptr; 
                break; 
            }
        }
    }
    
    for (size_t oi = 0; oi < out_tensors.size(); ++oi) {
        Tensor* t = out_tensors[oi];
        if (!t) continue;
        if (t->dtype != DataType::FLOAT32 && t->quant_meta.quant_type != 0) {
            Tensor* fp = create_fp32_copy_from_quantized(*ctx, t);
            if (fp) { ctx->tensor_pool.release(t); out_tensors[oi] = fp; }
        } else if (t->dtype == DataType::INT8 && t->quant_meta.quant_type == 0) {
            size_t n = t->num_elements;
            Tensor* fp = ctx->tensor_pool.acquire();
            if (fp) {
                fp->dtype = DataType::FLOAT32; fp->shape = t->shape; fp->update_from_shape();
                fp->data = alloc_fast(fp->size);
                if (fp->data) {
                    const int8_t* src = static_cast<const int8_t*>(t->data); float* dst = static_cast<float*>(fp->data);
                    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * 1.0f;
                    ctx->tensor_pool.release(t); out_tensors[oi] = fp;
                } else { ctx->tensor_pool.release(fp); }
            }
        }
    }

    std::set<Tensor*> keep_set(out_tensors.begin(), out_tensors.end());
    {
        xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
        
        if (!ctx->training_mode.load(std::memory_order_relaxed)) {
            std::set<Tensor*> to_release;
            for (Tensor* r : ctx->results) {
                if (r && keep_set.count(r) == 0) to_release.insert(r);
            }
            for (Tensor* c : ctx->fast_memory_cache) {
                if (c && keep_set.count(c) == 0) to_release.insert(c);
            }
            
            for (auto*& r : ctx->results) r = nullptr;
            for (auto*& c : ctx->fast_memory_cache) c = nullptr;
            
            for (Tensor* t : to_release) {
                ctx->tensor_pool.release(t);
            }
        }
        
        xSemaphoreGive(ctx->cache_mutex);
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
