// =============================================================================
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
// main.cpp  —  x64 NAC Model Executor
//
// Replaces the Arduino/ESP32 main_cpp.ino.
//
// Architecture (mirrors ESP32 two-task design):
//   • Main thread:   parses the NAC file, drives MEP execution (or legacy
//                    compute loop), reads stdin prompts, prints results.
//   • Memory thread: runs nac_memory_task() — identical logic to the ESP32
//                    FreeRTOS task, uses std::thread + TaskNotify primitives
//                    defined in platform.h.
//
// Usage:
//   ./nac_executor <path/to/model.nac> [prompt]
//   If [prompt] is omitted the program reads one line from stdin.
// =============================================================================

#include "platform.h"
#include "NacFile.h"
#include "types.h"
#include "op_kernels.h"
#include "MEP_interpreter.h"

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cstring>
#include <atomic>
#include <climits>
#include <limits>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <numeric>
#include <filesystem>
#include <iostream>
#include <thread>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

// ─── Global state ─────────────────────────────────────────────────────────────
// Defined in op_kernels.cpp — referenced here via op_kernels.h extern declarations
extern std::map<std::string, KernelFunc> g_kernel_string_map;
extern std::map<uint8_t,     KernelFunc> g_op_kernels;

#define MAX_INSTRUCTION_ARITY 8

EventGroupHandle_t g_system_events      = nullptr;
SemaphoreHandle_t  g_sd_card_mutex      = nullptr;
TaskHandle_t       g_nac_memory_task_handle = nullptr;

const uint32_t EVT_START_EXECUTION   = BIT0;
const uint32_t EVT_COMPUTE_TASK_DONE = BIT1;
const uint32_t EVT_MEMORY_TASK_DONE  = BIT2;

// Target tensor used by image-decode callback (vision models).
// On x64 the JPEG decode path is stubbed out (no TJpgDec); set to nullptr.
Tensor* g_target_tensor_for_decode = nullptr;

static const char* TAG_MAIN    = "MAIN";
static const char* TAG_COMPUTE = "NAC_COMPUTE";
static const char* TAG_MEM     = "NAC_MEM";

// ─── alloc_fast ───────────────────────────────────────────────────────────────
// On x64 there is no PSRAM; use 16-byte aligned malloc for all buffers.
void* alloc_fast(size_t size) {
    if (!size) return nullptr;
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, 16);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 16, size) != 0) ptr = nullptr;
#endif
    if (!ptr) ESP_LOGE("alloc_fast", "OOM: %zu bytes requested", size);
    return ptr;
}

size_t argmax(const float* data, size_t len) {
    if (!len) return 0;
    size_t max_idx = 0;
    for (size_t i = 1; i < len; ++i)
        if (data[i] > data[max_idx]) max_idx = i;
    return max_idx;
}

void softmax(const float* input, float* output, size_t len) {
    if (!len) return;
    float max_val = *std::max_element(input, input + len);
    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) { output[i] = expf(input[i] - max_val); sum += output[i]; }
    for (size_t i = 0; i < len; ++i) output[i] /= sum;
}

// ─── NAC instruction parser (unchanged from ESP32) ───────────────────────────
bool parse_instruction_at(const uint8_t* buffer, size_t buffer_size, size_t offset,
                           const std::vector<std::string>& permutations,
                           uint16_t num_outputs, ParsedInstruction& ins) {
    ins.C.clear(); ins.D.clear(); ins.bytes_consumed = 0;
    if (offset + 2 > buffer_size) return false;

    const uint8_t* p_start = buffer + offset;
    const uint8_t* p       = p_start;
    const uint8_t* buf_end = buffer + buffer_size;

    ins.A = *p++; ins.B = *p++;

    if (ins.A < 10) {
        if (ins.A == 2) {
            if (ins.B >= 1 && ins.B <= 5) {
                if (p + 4 > buf_end) return false;
                int16_t v0, v1;
                memcpy(&v0, p, 2); p += 2;
                memcpy(&v1, p, 2); p += 2;
                ins.C.push_back(v0);
                ins.C.push_back(v1);
            }
        } else if (ins.A == 3) {
            if (ins.B == 0 || ins.B == 1) { // inference output / subgraph
                size_t nC = num_outputs + 1;
                if (p + nC * 2 > buf_end) return false;
                ins.C.reserve(nC);
                for (size_t i = 0; i < nC; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.C.push_back(v); }
                size_t nD = num_outputs;
                if (p + nD * 2 > buf_end) return false;
                ins.D.reserve(nD);
                for (size_t i = 0; i < nD; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.D.push_back(v); }
            } else if (ins.B == 3) { // fused_sgd_step
                size_t nC = 2; // len(1) + param_id
                if (p + nC * 2 > buf_end) return false;
                for (size_t i = 0; i < nC; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.C.push_back(v); }
                size_t nD = 2; // grad_offset, lr_offset
                if (p + nD * 2 > buf_end) return false;
                for (size_t i = 0; i < nD; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.D.push_back(v); }
            } else if (ins.B == 4) { // return_trng
                size_t nD = 1;
                if (p + nD * 2 > buf_end) return false;
                for (size_t i = 0; i < nD; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.D.push_back(v); }
            }
        } else if (ins.A == 7) {
            if (p + 2 > buf_end) return false;
            int16_t nc; memcpy(&nc, p, 2); p += 2;
            ins.C.push_back(nc);
            if (p + nc * 2 > buf_end) return false;
            for (int i = 0; i < nc; ++i) { int16_t v; memcpy(&v, p, 2); p += 2; ins.C.push_back(v); }
            if (p + 2 > buf_end) return false;
            int16_t nd; memcpy(&nd, p, 2); p += 2;
            ins.D.push_back(nd);
        }
    } else {
        if (ins.B >= permutations.size() || permutations[ins.B].empty()) {
            ins.bytes_consumed = 2; return true;
        }
        const std::string& perm = permutations[ins.B];

        bool needs_consts = false;
        for (char c : perm) { if (strchr("SAfibsc", c)) { needs_consts = true; break; } }

        if (needs_consts) {
            if (p + 2 > buf_end) return false;
            int16_t nc; memcpy(&nc, p, 2); p += 2;
            ins.C.push_back(nc);
            if (nc > 0) {
                if (p + (size_t)nc * 2 > buf_end) return false;
                ins.C.reserve(nc + 1);
                for (int i = 0; i < nc; ++i) { uint16_t id; memcpy(&id, p, 2); p += 2; ins.C.push_back(id); }
            }
        }
        size_t nD = perm.length();
        if (p + nD * 2 > buf_end) return false;
        ins.D.reserve(nD);
        for (size_t i = 0; i < nD; ++i) { int16_t d; memcpy(&d, p, 2); p += 2; ins.D.push_back(d); }
    }
    ins.bytes_consumed = (size_t)(p - p_start);
    return true;
}

size_t gather_arguments(NacRuntimeContext& ctx, const ParsedInstruction& ins,
                        uint32_t idx, Tensor** args_out) {
    if (ins.D.size() > MAX_INSTRUCTION_ARITY) return 0;
    size_t argc = 0, c_idx = 1;
    bool use_consts = !ins.C.empty() && ins.C[0] > 0;
    for (int16_t d : ins.D) {
        if (argc >= MAX_INSTRUCTION_ARITY) break;
        if (d != 0) {
            uint32_t anc = idx + d;
            args_out[argc++] = (anc < ctx.results.size()) ? ctx.results[anc] : nullptr;
        } else {
            if (use_consts && c_idx < ins.C.size()) {
                uint16_t cid = ins.C[c_idx++];
                Tensor* ct = (cid < ctx.constants.size() && ctx.constants[cid])
                             ? ctx.constants[cid].get() : nullptr;
#ifdef DBG
                if (!ct) {
                    printf("[GATHER] op%u arg[%zu]: const_id=%u nullptr (pool=%zu) C=[",
                           idx, (size_t)argc, (unsigned)cid, ctx.constants.size());
                    for (size_t _ci=0; _ci<ins.C.size(); ++_ci) printf("%s%d", _ci?",":"", (int)ins.C[_ci]);
                    printf("] D=[");
                    for (size_t _di=0; _di<ins.D.size(); ++_di) printf("%s%d", _di?",":"", (int)ins.D[_di]);
                    printf("]\n");
                }
#endif
                args_out[argc++] = ct;
            } else {
#ifdef DBG
                printf("[GATHER] op%u: arg[%zu] D=0 but no const slot (use_consts=%d c_idx=%zu C.size=%zu)\n",
                       idx, argc, (int)use_consts, c_idx, ins.C.size());
#endif
                args_out[argc++] = nullptr;
            }
        }
    }
    return argc;
}

// ─── Quantization helpers ─────────────────────────────

static DataType nac_dtype_id_to_datatype(uint8_t id) {
    switch (id) {
        case 0: return DataType::FLOAT32;
        case 1: return DataType::FLOAT64;
        case 2: return DataType::FLOAT16;
        case 3: return DataType::BFLOAT16;
        case 4: return DataType::INT32;
        case 5: return DataType::INT64;
        case 6: return DataType::INT16;
        case 7: return DataType::INT8;
        case 8: return DataType::UINT8;
        case 9: return DataType::BOOL;
        default: return DataType::FLOAT32;
    }
}

bool parse_quant_meta_from_buffer(Tensor* tensor, const uint8_t* buf, size_t len) {
    if (!len) { tensor->quant_meta.quant_type = 0; return true; }
    const uint8_t* p = buf, *end = buf + len;
    if (p + 2 > end) return false;
    uint8_t dtype_id = *p++, rank = *p++;
    tensor->dtype = nac_dtype_id_to_datatype(dtype_id);
    if (p + rank * 4 > end) return false;
    tensor->shape.resize(rank);
    for (uint8_t r = 0; r < rank; ++r) {
        uint32_t dim; memcpy(&dim, p, 4); p += 4;
        tensor->shape[r] = (int)dim;
    }
    tensor->update_from_shape();
    if (p >= end) return false;
    tensor->quant_meta.quant_type = *p++;
    switch (tensor->quant_meta.quant_type) {
        case 2:
            if (p + 4 > end) return false;
            memcpy(&tensor->quant_meta.scale, p, 4);
            break;
        case 3: {
            if (p + 5 > end) return false;
            uint32_t ns; memcpy(&tensor->quant_meta.axis, p, 1); p += 1;
            memcpy(&ns, p, 4); p += 4;
            if (p + ns * 4 > end) return false;
            tensor->quant_meta.scales.resize(ns);
            memcpy(tensor->quant_meta.scales.data(), p, ns * 4);
            break;
        }
        case 4: {
            if (p + 3 > end) return false;
            uint8_t orig_rank;
            memcpy(&tensor->quant_meta.block_size, p, 2); p += 2;
            orig_rank = *p++;
            if (p + orig_rank * 4 > end) return false;
            tensor->quant_meta.original_shape.resize(orig_rank);
            if (orig_rank) memcpy(tensor->quant_meta.original_shape.data(), p, orig_rank * 4);
            p += orig_rank * 4;
            if (p + 4 > end) return false;
            uint32_t nbs; memcpy(&nbs, p, 4); p += 4;
            if (p + nbs * 4 > end) return false;
            tensor->quant_meta.block_scales.resize(nbs);
            memcpy(tensor->quant_meta.block_scales.data(), p, nbs * 4);
            break;
        }
        default: break;
    }
    return true;
}

static bool load_tensor_from_disk(Tensor* tensor, const TensorLocation& loc) {
    std::vector<uint8_t> meta_buf;
    if (loc.meta_len > 0) meta_buf.resize(loc.meta_len);

    tensor->update_from_byte_size(loc.data_size);
    tensor->data = alloc_fast(loc.data_size);
    if (!tensor->data) {
        ESP_LOGE(TAG_MEM, "load_tensor_from_disk: OOM for %llu bytes",
                 (unsigned long long)loc.data_size);
        return false;
    }

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    bool ok = true;
    if (loc.meta_len > 0) {
        ok &= sdcard.seek(loc.meta_offset);
        ok &= (sdcard.readData(meta_buf.data(), loc.meta_len) == loc.meta_len);
    }
    ok &= sdcard.seek(loc.file_offset);
    ok &= (sdcard.readData((uint8_t*)tensor->data, loc.data_size) == loc.data_size);
    xSemaphoreGive(g_sd_card_mutex);

    if (!ok) {
        ESP_LOGE(TAG_MEM, "load_tensor_from_disk: read error");
        heap_caps_free(tensor->data);
        tensor->data = nullptr;
        return false;
    }
    return parse_quant_meta_from_buffer(tensor, meta_buf.data(), meta_buf.size());
}

bool dequantize_tensor(Tensor* tensor) {
    if (!tensor || !tensor->data) return false;
    uint8_t qt = tensor->quant_meta.quant_type;
    if (qt == 0 || qt == 1) { tensor->quant_meta.clear(); return true; }

    size_t n = tensor->num_elements;
    float* fp = static_cast<float*>(alloc_fast(n * sizeof(float)));
    if (!fp) { ESP_LOGE("dequant", "OOM (%zu floats)", n); return false; }

    if (qt == 2) {
        int8_t* s = static_cast<int8_t*>(tensor->data);
        float sc = tensor->quant_meta.scale;
        for (size_t i = 0; i < n; ++i) fp[i] = s[i] * sc;
    } else if (qt == 3) {
        int8_t* s = static_cast<int8_t*>(tensor->data);
        uint8_t ax = tensor->quant_meta.axis;
        const auto& scs = tensor->quant_meta.scales;
        size_t ax_sz = (ax < tensor->shape.size()) ? tensor->shape[ax] : 1;
        if (!ax_sz) ax_sz = 1;
        size_t stride = 1;
        for (size_t i = ax + 1; i < tensor->shape.size(); ++i) stride *= tensor->shape[i];
        for (size_t i = 0; i < n; ++i) {
            float sc = ((i / stride) % ax_sz < scs.size()) ? scs[(i / stride) % ax_sz] : 1.0f;
            fp[i] = s[i] * sc;
        }
    } else if (qt == 4) {
        int8_t* s = static_cast<int8_t*>(tensor->data);
        uint16_t bs = tensor->quant_meta.block_size;
        const auto& bsc = tensor->quant_meta.block_scales;
        const auto& os  = tensor->quant_meta.original_shape;
        size_t orig_n = 1; for (int d : os) orig_n *= d;
        if (orig_n < n) {
            heap_caps_free(fp);
            fp = static_cast<float*>(alloc_fast(orig_n * sizeof(float)));
            if (!fp) { heap_caps_free(tensor->data); tensor->data = nullptr; return false; }
        }
        size_t out = 0;
        for (size_t bi = 0; bi < bsc.size() && out < orig_n; ++bi) {
            float sc = bsc[bi];
            for (size_t i = 0; i < bs && out < orig_n; ++i) fp[out++] = s[bi * bs + i] * sc;
        }
        tensor->shape = os; tensor->num_elements = orig_n;
    } else {
        ESP_LOGW("dequant", "Unknown quant type %u", qt);
        heap_caps_free(fp);
        tensor->quant_meta.clear();
        return false;
    }
    heap_caps_free(tensor->data);
    tensor->data = fp;
    tensor->dtype = DataType::FLOAT32;
    tensor->size  = tensor->num_elements * sizeof(float);
    tensor->quant_meta.clear();
    return true;
}

Tensor* create_fp32_copy_from_quantized(NacRuntimeContext& ctx, Tensor* src) {
    if (!src || !src->data || src->quant_meta.quant_type == 0
        || src->dtype == DataType::FLOAT32) return nullptr;

    size_t n = src->num_elements;
    Tensor* t = ctx.tensor_pool.acquire();
    if (!t) return nullptr;
    t->shape = src->shape; t->num_elements = n;
    t->dtype = DataType::FLOAT32; t->size = n * sizeof(float);
    t->data  = alloc_fast(t->size);
    if (!t->data) { ctx.tensor_pool.release(t); return nullptr; }

    float*  fp  = static_cast<float*>(t->data);
    uint8_t qt  = src->quant_meta.quant_type;

    if (qt == 2) {
        int8_t* s = static_cast<int8_t*>(src->data);
        float sc = src->quant_meta.scale;
        float mv = src->quant_meta.scales.empty() ? 0.0f : src->quant_meta.scales[0];
        for (size_t i = 0; i < n; ++i) fp[i] = (s[i] + 128.0f) * sc + mv;
    } else if (qt == 3) {
        int8_t* s = static_cast<int8_t*>(src->data);
        uint8_t ax = src->quant_meta.axis;
        const auto& scs = src->quant_meta.scales;
        size_t ax_sz = (ax < src->shape.size()) ? src->shape[ax] : 1;
        size_t stride = 1;
        for (size_t i = ax + 1; i < src->shape.size(); ++i) stride *= src->shape[i];
        for (size_t i = 0; i < n; ++i) {
            float sc = ((i / stride) % ax_sz < scs.size()) ? scs[(i / stride) % ax_sz] : 1.0f;
            fp[i] = s[i] * sc;
        }
    } else if (qt == 4) {
        int8_t* s = static_cast<int8_t*>(src->data);
        uint16_t bs = src->quant_meta.block_size;
        const auto& bsc = src->quant_meta.block_scales;
        size_t out = 0;
        for (size_t bi = 0; bi < bsc.size() && out < n; ++bi)
            for (size_t i = 0; i < bs && out < n; ++i) fp[out++] = s[bi * bs + i] * bsc[bi];
    } else {
        ctx.tensor_pool.release(t); return nullptr;
    }
    return t;
}

bool requantize_result_tensor(NacRuntimeContext& ctx, Tensor* tensor) {
    if (!tensor || !tensor->data || tensor->dtype != DataType::FLOAT32) return false;
    size_t n = tensor->num_elements;
    float* fp = static_cast<float*>(tensor->data);
    float mn = fp[0], mx = fp[0];
    for (size_t i = 1; i < n; ++i) { mn = std::min(mn, fp[i]); mx = std::max(mx, fp[i]); }
    float rng = mx - mn, sc = rng > 0.0f ? rng / 255.0f : 1.0f;
    int8_t* q = static_cast<int8_t*>(alloc_fast(n));
    if (!q) return false;
    for (size_t i = 0; i < n; ++i) {
        int v = (int)roundf((fp[i] - mn) / sc) - 128;
        q[i]  = (int8_t)std::max(-128, std::min(127, v));
    }
    heap_caps_free(tensor->data);
    tensor->data = q; tensor->dtype = DataType::INT8; tensor->size = n;
    tensor->quant_meta.clear();
    tensor->quant_meta.quant_type = 2;
    tensor->quant_meta.scale = sc;
    tensor->quant_meta.scales.resize(1); tensor->quant_meta.scales[0] = mn;
    return true;
}

bool read_and_parse_quant_metadata(Tensor* tensor, const TensorLocation& loc) {
    if (!loc.meta_len) { tensor->quant_meta.quant_type = 0; return true; }
    std::vector<uint8_t> buf(loc.meta_len);
    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    sdcard.seek(loc.meta_offset);
    size_t got = sdcard.readData(buf.data(), loc.meta_len);
    xSemaphoreGive(g_sd_card_mutex);
    if (got != loc.meta_len) { ESP_LOGE("META", "Short read"); return false; }
    return parse_quant_meta_from_buffer(tensor, buf.data(), buf.size());
}

// ─── MMAP tick processor (identical to ESP32, no platform dependencies) ──────
static void process_mmap_tick(NacRuntimeContext* ctx, uint32_t tick) {
    auto it = ctx->mmap_schedule.find(tick);
    if (it == ctx->mmap_schedule.end()) return;

    for (const auto& cmd : it->second) {
        switch (cmd.action) {

        case MmapAction::PRELOAD: {
            uint16_t op = cmd.target_id;
            
            // If the main thread has already passed this instruction, then it has loaded the weight itself.
            if (ctx->current_instruction_idx.load(std::memory_order_relaxed) >= op) continue;
            
            if (op >= ctx->decoded_ops.size()) continue;
            const auto& ins = ctx->decoded_ops[op];
            if (ins.A != 2 || ins.B != 1 || ins.C.size() < 2) continue;
            uint16_t pid = ins.C[1];
            if (pid >= ctx->param_present.size() || !ctx->param_present[pid]) continue;

            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            bool exists = (op < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[op] != nullptr);
            xSemaphoreGive(ctx->cache_mutex);
            if (exists) continue;

            Tensor* t = ctx->tensor_pool.acquire();
            if (!t) continue;
            
            if (ctx->updated_parameters.count(pid) && ctx->updated_parameters[pid]) {
                Tensor* heap_t = ctx->updated_parameters[pid];
                t->dtype = heap_t->dtype; 
                t->shape = heap_t->shape; 
                t->update_from_shape();
                t->data = alloc_fast(t->size);
                if (t->data) memcpy(t->data, heap_t->data, t->size);
                else { ctx->tensor_pool.release(t); continue; }
            } else {
                if (!load_tensor_from_disk(t, ctx->param_locations[pid])) {
                    ESP_LOGE(TAG_MEM, "PRELOAD: read failed param %u op %u", pid, op);
                    ctx->tensor_pool.release(t); continue;
                }
                if (ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
                    if (!dequantize_tensor(t)) { ctx->tensor_pool.release(t); continue; }
                }
            }

            // Final check: Did the main thread overtake us while we were reading from the SD card?
            if (ctx->current_instruction_idx.load(std::memory_order_relaxed) >= op) {
                ctx->tensor_pool.release(t);
                continue;
            }

            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            if (op >= ctx->fast_memory_cache.size()) ctx->fast_memory_cache.resize(op + 1, nullptr);
            if (ctx->fast_memory_cache[op]) ctx->tensor_pool.release(ctx->fast_memory_cache[op]);
            ctx->fast_memory_cache[op] = t;
            xSemaphoreGive(ctx->cache_mutex);
            break;
        }
        case MmapAction::FREE: {
            if (ctx->training_mode.load(std::memory_order_relaxed)) break;
            uint16_t tid = cmd.target_id;
            
            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            Tensor* tf = nullptr;
            if (tid < ctx->results.size() && ctx->results[tid]) {
                tf = ctx->results[tid];
            } else if (tid < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[tid]) {
                tf = ctx->fast_memory_cache[tid];
            }

            if (tf) {
                for (auto& r : ctx->results)           { if (r  == tf) r  = nullptr; }
                for (auto& c : ctx->fast_memory_cache) { if (c  == tf) c  = nullptr; }
            }
            xSemaphoreGive(ctx->cache_mutex);
            
            if (tf) ctx->tensor_pool.release(tf);
            break;
        }
        case MmapAction::SAVE_RESULT: {
            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            if (tick < ctx->results.size() && ctx->results[tick]) {
                uint16_t s = cmd.target_id;
                if (s >= ctx->fast_memory_cache.size()) ctx->fast_memory_cache.resize(s + 1, nullptr);
                ctx->fast_memory_cache[s] = ctx->results[tick];
            }
            xSemaphoreGive(ctx->cache_mutex);
            break;
        }
        case MmapAction::FORWARD: {
            uint16_t src = (uint16_t)tick, dst = cmd.target_id;
            xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
            if (src < ctx->results.size() && ctx->results[src]) {
                if (dst >= ctx->results.size()) ctx->results.resize(dst + 1, nullptr);
                if (ctx->results[dst] && ctx->results[dst] != ctx->results[src])
                    ctx->tensor_pool.release(ctx->results[dst]);
                ctx->results[dst] = ctx->results[src];
            }
            xSemaphoreGive(ctx->cache_mutex);
            break;
        }
        }
    }
}

// ─── Memory management thread ─────────────────────────────────────────────────
// Runs concurrently with the compute / MEP thread.
// Notified via xTaskNotifyGive after every results[idx] write.
// Drains all skipped ticks (same tick-skipping fix as on ESP32).
void nac_memory_task(void* param) {
    auto* ctx = static_cast<NacRuntimeContext*>(param);
    uint32_t last = UINT32_MAX;

    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (ctx->stop_flag.load()) break;

        uint32_t cur = ctx->current_instruction_idx.load(std::memory_order_acquire);
        uint32_t from = (last == UINT32_MAX || cur <= last) ? 0u : last + 1u;
        
        for (uint32_t t = from; t <= cur; ++t) {
            process_mmap_tick(ctx, t);
        }
        last = cur;
        
        xEventGroupSetBits(ctx->mmap_sync_event, ctx->MMAP_DONE_BIT);
    }
    xEventGroupSetBits(g_system_events, EVT_MEMORY_TASK_DONE);
}

// ─── Compute task (legacy non-MEP path) ──────────────────────────────────────
// Same logic as ESP32 nac_compute_task; runs in a std::thread.
void nac_compute_task(void* param) {
    auto* ctx = static_cast<NacRuntimeContext*>(param);
    size_t user_input_idx = 0;

    for (uint32_t idx = 0; idx < ctx->decoded_ops.size() && !ctx->stop_flag.load(); ++idx) {
        const auto& ins = ctx->decoded_ops[idx];
        Tensor* result = nullptr;

        if (ins.A == 2) { 
            Tensor* src = nullptr;
            if (ins.B == 1) { 
                if (ins.C.size() < 2) { ctx->results[idx] = nullptr; continue; }
                uint16_t pid = ins.C[1], ck = (uint16_t)idx;

                xSemaphoreTake(ctx->cache_mutex, portMAX_DELAY);
                if (ck < ctx->fast_memory_cache.size() && ctx->fast_memory_cache[ck]) {
                    src = ctx->fast_memory_cache[ck]; ctx->fast_memory_cache[ck] = nullptr;
                }
                xSemaphoreGive(ctx->cache_mutex);

                if (!src && pid < ctx->param_present.size() && ctx->param_present[pid]) {
                    src = ctx->tensor_pool.acquire();
                    if (!load_tensor_from_disk(src, ctx->param_locations[pid])) {
                        ctx->tensor_pool.release(src); src = nullptr;
                    } else if (ctx->quant_mode == QuantExecMode::DEQUANT_ON_LOAD) {
                        if (!dequantize_tensor(src)) { ctx->tensor_pool.release(src); src = nullptr; }
                    }
                }
            } else if (ins.B == 0) { 
                if (user_input_idx < ctx->user_input_tensors.size()) {
                    src = ctx->user_input_tensors[user_input_idx];
                    ctx->user_input_tensors[user_input_idx++] = nullptr;
                } else {
                    ESP_LOGE(TAG_COMPUTE, "User input exhausted");
                    ctx->stop_flag.store(true);
                }
            }
            result = src;
        } else if (ins.A >= 10) {
            Tensor* args[MAX_INSTRUCTION_ARITY] = {};
            size_t  argc = gather_arguments(*ctx, ins, idx, args);

            Tensor* dq[MAX_INSTRUCTION_ARITY] = {};
            std::vector<Tensor*> tmp;
            if (ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_PASS_FP
             || ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT) {
                for (size_t i = 0; i < argc; ++i) {
                    if (args[i] && args[i]->dtype != DataType::FLOAT32) {
                        Tensor* fp = create_fp32_copy_from_quantized(*ctx, args[i]);
                        dq[i] = fp ? fp : args[i];
                        if (fp) tmp.push_back(fp);
                    } else { dq[i] = args[i]; }
                }
            } else {
                for (size_t i = 0; i < argc; ++i) dq[i] = args[i];
            }

            auto it = g_op_kernels.find(ins.A);
            result = (it != g_op_kernels.end())
                     ? it->second(ctx, ins, dq, argc)
                     : op_nac_pass(ctx, ins, dq, argc);

            for (Tensor* t : tmp) ctx->tensor_pool.release(t);

            if (result && ctx->quant_mode == QuantExecMode::DEQUANT_EXEC_REQUANT
                       && result->dtype == DataType::FLOAT32)
                requantize_result_tensor(*ctx, result);
        }

        if (idx < ctx->results.size()) {
            ctx->results[idx] = result;
            ctx->current_instruction_idx.store(idx, std::memory_order_release);
            
            if (g_nac_memory_task_handle) {
                if (ctx->mmap_sync_event) xEventGroupClearBits(ctx->mmap_sync_event, ctx->MMAP_DONE_BIT);
                xTaskNotifyGive(g_nac_memory_task_handle);
                if (ctx->mmap_sync_event) xEventGroupWaitBits(ctx->mmap_sync_event, ctx->MMAP_DONE_BIT, pdTRUE, pdFALSE, portMAX_DELAY);
            }
        } else if (result) {
            ctx->tensor_pool.release(result);
        }
    }

    xEventGroupSetBits(g_system_events, EVT_COMPUTE_TASK_DONE);
}

// ─── NAC context initializer (no SD mutex contention in single-threaded init) ─
bool initialize_nac_context(NacRuntimeContext& ctx) {
    register_kernels();

    if (!sdcard.isFileOpen()) {
#ifdef DBG
       ESP_LOGE(TAG_MAIN, "NAC file not open.");
#endif
       return false;
    }

    uint8_t hdr[100];
    sdcard.seek(0);
    if (sdcard.readData(hdr, 100) != 100) { ESP_LOGE(TAG_MAIN, "Header read failed"); return false; }
    if (memcmp(hdr, "NAC", 3) != 0)       { ESP_LOGE(TAG_MAIN, "Bad NAC magic");      return false; }
    if (hdr[3] != 2)                      { ESP_LOGE(TAG_MAIN, "Unsupported NAC version %d", hdr[3]); return false; }

    uint8_t  quant_byte           = hdr[4];
    bool     store_weights_intern = (quant_byte & 0x80) != 0;
    memcpy(&ctx.num_inputs,  hdr + 5, 2);
    memcpy(&ctx.num_outputs, hdr + 7, 2);
#ifdef DBG
    printf("[MAIN] IO: %u inputs, %u outputs | internal storage: %s\n",
           ctx.num_inputs, ctx.num_outputs, store_weights_intern ? "YES" : "NO");
#endif
    uint64_t offsets[11] = {};
    memcpy(offsets, hdr + 12, sizeof(offsets));
    uint64_t mmap_off = offsets[0], ops_off = offsets[1], cmap_off = offsets[2], cnst_off = offsets[3],
             perm_off = offsets[4], data_off = offsets[5], proc_off = offsets[6],
             orch_off = offsets[7], trng_off = offsets[8], rsrc_off = offsets[9], arrs_off = offsets[10];

    // Suppressing compiler warnings
    (void)orch_off;
    (void)trng_off;

    // MMAP
    if (mmap_off) {
        sdcard.seek(mmap_off + 4);
        uint32_t n; sdcard.readData((uint8_t*)&n, 4);
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t tid; uint8_t nc;
            sdcard.readData((uint8_t*)&tid, 2); sdcard.readData(&nc, 1);
            std::vector<MmapCommand> cmds; cmds.reserve(nc);
            for (uint8_t j = 0; j < nc; ++j) {
                MmapCommand c; uint8_t a;
                sdcard.readData(&a, 1); sdcard.readData((uint8_t*)&c.target_id, 2);
                c.action = static_cast<MmapAction>(a); cmds.push_back(c);
            }
            ctx.mmap_schedule[tid] = std::move(cmds);
        }
    }

    // CMAP
    if (cmap_off) {
        sdcard.seek(cmap_off + 4);
        uint32_t n; sdcard.readData((uint8_t*)&n, 4);
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t oid; uint8_t nl;
            sdcard.readData((uint8_t*)&oid, 2); sdcard.readData(&nl, 1);
            std::string name(nl, '\0'); sdcard.readData((uint8_t*)name.data(), nl);
            ctx.id_to_name_map[oid] = name;
            auto it = g_kernel_string_map.find(name);
            if (it != g_kernel_string_map.end()) g_op_kernels[oid] = it->second;
        }
    }

    // PERM
    if (perm_off) {
        sdcard.seek(perm_off + 4);
        uint32_t n; sdcard.readData((uint8_t*)&n, 4);
        uint16_t max_id = 0;
        std::map<uint16_t, std::string> tmp;
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t id; uint8_t len;
            sdcard.readData((uint8_t*)&id, 2); sdcard.readData(&len, 1);
            std::string s(len, '\0'); sdcard.readData((uint8_t*)s.data(), len);
            tmp[id] = s; if (id > max_id) max_id = id;
        }
        ctx.permutations.resize(max_id + 1);
        for (auto& p : tmp) ctx.permutations[p.first] = std::move(p.second);
    }

    // CNST
    if (cnst_off) {
        sdcard.seek(cnst_off + 4);
        uint32_t cc; sdcard.readData((uint8_t*)&cc, 4);
        struct RC { uint16_t id; uint8_t type; uint16_t length; std::vector<uint8_t> data; };
        std::vector<RC> raws(cc); uint16_t max_id = 0;
        for (uint32_t i = 0; i < cc; ++i) {
            RC& rc = raws[i];
            sdcard.readData((uint8_t*)&rc.id, 2); sdcard.readData(&rc.type, 1);
            sdcard.readData((uint8_t*)&rc.length, 2);
            size_t bc = 0;
            switch (rc.type) {
                case 1: bc = 1; break; case 2: case 3: bc = 8; break;
                case 4: bc = rc.length; break;
                case 5: case 6: bc = (size_t)rc.length * 4; break;
            }
            if (bc) { rc.data.resize(bc); sdcard.readData(rc.data.data(), bc); }
            if (rc.id > max_id) max_id = rc.id;
        }
        ctx.constants.resize((size_t)max_id + 1);
        for (const RC& rc : raws) {
            if (rc.type == 0) continue;
            
            auto tu = std::make_unique<Tensor>();
            Tensor* t = tu.get();
            switch (rc.type) {
                case 1: { t->dtype = DataType::INT32; t->shape = {1}; t->update_from_shape();
                          t->data = alloc_fast(4); if (!t->data) { ESP_LOGE("INIT", "CNST 1 OOM"); return false; }
                          *static_cast<int32_t*>(t->data) = rc.data[0] ? 1 : 0; break; }
                case 2: { int64_t v; memcpy(&v, rc.data.data(), 8);
                          t->dtype = DataType::INT32; t->shape = {1}; t->update_from_shape();
                          t->data = alloc_fast(4); if (!t->data) { ESP_LOGE("INIT", "CNST 2 OOM"); return false; }
                          *static_cast<int32_t*>(t->data) = (int32_t)v; break; }
                case 3: { double v; memcpy(&v, rc.data.data(), 8);
                          t->dtype = DataType::FLOAT32; t->shape = {1}; t->update_from_shape();
                          t->data = alloc_fast(4); if (!t->data) { ESP_LOGE("INIT", "CNST 3 OOM"); return false; }
                          *static_cast<float*>(t->data) = (float)v; break; }
                case 4: { // STRING (e.g. 'float32', 'cpu')
                          // Create an empty INT8 tensor, just so the slot isn't nullptr
                          // Kernels that need strings (to_copy, arange) ignore them
                          t->dtype = DataType::INT8; t->shape = {0}; t->update_from_shape();
                          t->data = nullptr; 
                          break; }
                case 5: { int n = rc.length; t->dtype = DataType::INT32; t->shape = {n}; t->update_from_shape();
                          if (n > 0) {
                              t->data = alloc_fast(n * 4);
                              if (!t->data) { ESP_LOGE("INIT", "CNST 5 OOM"); return false; }
                              memcpy(t->data, rc.data.data(), n * 4);
                          } else { t->data = nullptr; }
                          break; }
                case 6: { int n = rc.length; t->dtype = DataType::FLOAT32; t->shape = {n}; t->update_from_shape();
                          if (n > 0) {
                              t->data = alloc_fast(n * 4);
                              if (!t->data) { ESP_LOGE("INIT", "CNST 6 OOM"); return false; }
                              memcpy(t->data, rc.data.data(), n * 4);
                          } else { t->data = nullptr; }
                          break; }
                default: continue;
            }
            ctx.constants[rc.id] = std::move(tu);
        }
#ifdef DBG
        printf("[MAIN] CNST: %u constants (max_id %u)\n", cc, max_id);
#endif
    }

    // OPS
    if (ops_off) {
        sdcard.seek(ops_off + 4);
        uint32_t nops; sdcard.readData((uint8_t*)&nops, 4);
        uint64_t ops_start = sdcard.getPosition();

        // Find end of OPS section
        uint64_t next = sdcard.size();
        for (int i = 0; i < 9; ++i)
            if (offsets[i] > ops_off && offsets[i] < next) next = offsets[i];

        size_t ops_sz = (size_t)(next - ops_start);
        std::vector<uint8_t> ops_buf(ops_sz);
        sdcard.seek(ops_start);
        sdcard.readData(ops_buf.data(), ops_sz);

        ctx.decoded_ops.reserve(nops);
        size_t ptr = 0;
        while (ptr < ops_buf.size() && ctx.decoded_ops.size() < nops) {
            ParsedInstruction ins;
            if (!parse_instruction_at(ops_buf.data(), ops_buf.size(), ptr,
                                      ctx.permutations, ctx.num_outputs, ins)
                || ins.bytes_consumed == 0){
                ESP_LOGE("INIT", "Parse op failed at ptr=%zu", ptr);
                return false;
            }
            ctx.decoded_ops.push_back(std::move(ins));
            ptr += ins.bytes_consumed;
        }
        ctx.results.resize(ctx.decoded_ops.size(), nullptr);
#ifdef DBG
        printf("[MAIN] OPS: %zu instructions loaded\n", ctx.decoded_ops.size());
#endif
    }

    // DATA (weight locations)
    if (data_off) {
        sdcard.seek(data_off + 4);
        uint32_t pnc, inc;
        sdcard.readData((uint8_t*)&pnc, 4);
        for (uint32_t i = 0; i < pnc; ++i) {
            uint16_t id, len; 
            sdcard.readData((uint8_t*)&id, 2); 
            sdcard.readData((uint8_t*)&len, 2);
            std::string pname(len, '\0');
            sdcard.readData((uint8_t*)pname.data(), len);
            ctx.param_id_to_name[id] = pname; // Save the parameter name
        }
        sdcard.readData((uint8_t*)&inc, 4);
        for (uint32_t i = 0; i < inc; ++i) {
            uint16_t id, len; 
            sdcard.readData((uint8_t*)&id, 2); 
            sdcard.readData((uint8_t*)&len, 2);
            sdcard.seek(sdcard.getPosition() + len); // Skip the input names
        }
        if (store_weights_intern) {
            uint32_t nt; sdcard.readData((uint8_t*)&nt, 4);
#ifdef DBG
            printf("[MAIN] DATA: %u tensors\n", nt);
#endif
            for (uint32_t i = 0; i < nt; ++i) {
                uint16_t pid; uint32_t ml; uint64_t dl;
                sdcard.readData((uint8_t*)&pid, 2);
                sdcard.readData((uint8_t*)&ml, 4);
                sdcard.readData((uint8_t*)&dl, 8);
                uint64_t mo = sdcard.getPosition();
                sdcard.seek(mo + ml);
                uint64_t fo = sdcard.getPosition();
                if (pid >= ctx.param_locations.size()) {
                    ctx.param_locations.resize(pid + 1);
                    ctx.param_present.resize(pid + 1, false);
                }
                ctx.param_locations[pid] = {mo, ml, fo, dl};
                ctx.param_present[pid]   = true;
                sdcard.seek(fo + dl);
            }
        }
    }

    // PROC (TISA tokenizer manifest)
    if (proc_off) {
        sdcard.seek(proc_off + 4);
        uint32_t ms; sdcard.readData((uint8_t*)&ms, 4);
        ctx.tisa_manifest.resize(ms);
        sdcard.readData(ctx.tisa_manifest.data(), ms);
    }

    // TRNG (Training Graph)
    if (trng_off) {
        sdcard.seek(trng_off);
        char magic[4];
        sdcard.readData((uint8_t*)magic, 4);
        if (memcmp(magic, "TRNG", 4) == 0) {
            uint32_t nops; 
            sdcard.readData((uint8_t*)&nops, 4);
            uint64_t trng_start = sdcard.getPosition();
            
            uint64_t next = sdcard.size();
            for (int i = 0; i < 11; ++i)
                if (offsets[i] > trng_off && offsets[i] < next) next = offsets[i];
            
            size_t trng_sz = (size_t)(next - trng_start);
            std::vector<uint8_t> trng_buf(trng_sz);
            sdcard.seek(trng_start);
            sdcard.readData(trng_buf.data(), trng_sz);

            ctx.trng_operations.reserve(nops);
            size_t ptr = 0;
            while (ptr < trng_buf.size() && ctx.trng_operations.size() < nops) {
                ParsedInstruction ins;
                if (!parse_instruction_at(trng_buf.data(), trng_buf.size(), ptr,
                                          ctx.permutations, ctx.num_outputs, ins) || ins.bytes_consumed == 0) {
                    ESP_LOGE("INIT", "Parse TRNG op failed at ptr=%zu", ptr);
                    break;
                }
                ctx.trng_operations.push_back(std::move(ins));
                ptr += ins.bytes_consumed;
            }
#ifdef DBG
            ESP_LOGI(TAG_MAIN, "TRNG: %zu instructions loaded", ctx.trng_operations.size());
#endif
        }
    }

    // RSRC (binary vocab, merges, etc.)
    if (rsrc_off) {
        sdcard.seek(rsrc_off + 4);
        uint32_t nf; sdcard.readData((uint8_t*)&nf, 4);
#ifdef DBG
        printf("[MAIN] RSRC: %u files\n", nf);
#endif
        for (uint32_t i = 0; i < nf; ++i) {
            uint16_t nl; sdcard.readData((uint8_t*)&nl, 2);
            std::string fname(nl, '\0'); sdcard.readData((uint8_t*)fname.data(), nl);
            uint32_t dl; sdcard.readData((uint8_t*)&dl, 4);
            uint64_t doff = sdcard.getPosition();
#ifdef DBG
            printf("[MAIN]  resource '%s' %u bytes\n", fname.c_str(), dl);
#endif
            if      (fname == "vocab.b")  ctx.tokenizer_resources.vocab   = std::make_unique<BinaryVocabView>(doff, dl);
            else if (fname == "vidx.b")   ctx.tokenizer_resources.vocab_idx_for_decode = std::make_unique<BinaryVocabIndexView>(doff, dl);
            else if (fname == "merges.b") ctx.tokenizer_resources.merges  = std::make_unique<BinaryMergesView>(doff, dl);
            sdcard.seek(doff + dl);
        }
    }

    if (arrs_off) {
        sdcard.seek(arrs_off + 4); // Skipping the 'ARRS' tag
        uint32_t num_arrays; 
        sdcard.readData((uint8_t*)&num_arrays, 4);
        
        for (uint32_t i = 0; i < num_arrays; ++i) {
            uint16_t name_len; 
            sdcard.readData((uint8_t*)&name_len, 2);
            std::string arr_name(name_len, '\0');
            sdcard.readData((uint8_t*)arr_name.data(), name_len);

            uint8_t dtype_id, rank;
            sdcard.readData(&dtype_id, 1);
            sdcard.readData(&rank, 1);

            std::vector<int> shape(rank);
            if (rank > 0) {
                std::vector<uint32_t> u_shape(rank);
                sdcard.readData((uint8_t*)u_shape.data(), rank * 4);
                for(int r = 0; r < rank; ++r) shape[r] = (int)u_shape[r];
            }

            uint64_t data_len; 
            sdcard.readData((uint8_t*)&data_len, 8);
            
            Tensor* t = ctx.tensor_pool.acquire();
            t->dtype = nac_dtype_id_to_datatype(dtype_id);
            t->shape = shape;
            t->update_from_shape();
            
            t->data = alloc_fast(data_len);
            if (t->data) {
                sdcard.readData((uint8_t*)t->data, data_len);
            } else {
                ESP_LOGE(TAG_MAIN, "OOM loading ARRS tensor '%s'", arr_name.c_str());
                sdcard.seek(sdcard.getPosition() + data_len); 
            }
            ctx.arrays[arr_name] = t;
        }
#ifdef DBG
        printf("[MAIN] ARRS: %u arrays loaded\n", num_arrays);
#endif
    }

    // ── GPT-2 byte_map (required for BPE BYTE_ENCODE opcode 0x15) ────────────
    // IMPORTANT: This must happen BEFORE TISAVM initialization, 
    // so that the tokenizer gets a filled byte map to restore the spaces.
    if (ctx.tokenizer_resources.merges) {
        auto& bm = ctx.tokenizer_resources.byte_map;
        if (bm.empty()) {
            // Collect "printable" bytes: ! (33) to ~ (126), ¡ (161) to ¬ (172), ® (174) to ÿ (255)
            std::vector<uint8_t> bs;
            for (int b = 33;  b <= 126; ++b) bs.push_back((uint8_t)b);
            for (int b = 161; b <= 172; ++b) bs.push_back((uint8_t)b);
            for (int b = 174; b <= 255; ++b) bs.push_back((uint8_t)b);
            // cs starts as a copy of bs
            std::vector<uint32_t> cs(bs.begin(), bs.end());
            // Fill remaining 256 - |bs| bytes with codepoints 256, 257, ...
            uint32_t next_cp = 256;
            for (int b = 0; b < 256; ++b) {
                bool found = false;
                for (uint8_t x : bs) if (x == (uint8_t)b) { found = true; break; }
                if (!found) {
                    bs.push_back((uint8_t)b);
                    cs.push_back(next_cp++);
                }
            }
            // Build map byte → UTF-8 string of codepoint
            for (size_t i = 0; i < bs.size(); ++i) {
                uint32_t cp = cs[i];
                std::string utf8;
                if      (cp < 0x80)   { utf8 += (char)cp; }
                else if (cp < 0x800)  { utf8 += (char)(0xC0|(cp>>6)); utf8 += (char)(0x80|(cp&0x3F)); }
                else if (cp < 0x10000){ utf8 += (char)(0xE0|(cp>>12)); utf8 += (char)(0x80|((cp>>6)&0x3F)); utf8 += (char)(0x80|(cp&0x3F)); }
                else                  { utf8 += (char)(0xF0|(cp>>18)); utf8 += (char)(0x80|((cp>>12)&0x3F)); utf8 += (char)(0x80|((cp>>6)&0x3F)); utf8 += (char)(0x80|(cp&0x3F)); }
                bm[bs[i]] = utf8;
            }
#ifdef DBG
            ESP_LOGI("MAIN", "Built GPT-2 byte_map (%u entries).", (unsigned)bm.size());
#endif
        }
    }

    if (!ctx.tisa_manifest.empty() && ctx.tokenizer_resources.vocab)
        ctx.tokenizer = std::make_unique<TISAVM>(ctx.tokenizer_resources);

    ctx.cache_mutex = xSemaphoreCreateMutex();
    return true;
}

// ─── Execution orchestrator ───────────────────────────────────────────────────
// Mirrors the EXECUTION_MODE block in the ESP32 loop().
// Returns 0 on success, non-zero on error.
static int run_nac_file(const std::string& nac_path, const std::string& exec_mode, const std::string& train_mode,
                        const std::vector<std::string>& pre_answers,
                        const std::map<std::string, std::string>& patches,
                        const std::map<std::string, std::string>& rewrites) {
#ifdef DBG
    printf("[MAIN] Opening '%s'\n", nac_path.c_str());
#endif
    
    // --- REWRITE SUPPORT ---
    if (!rewrites.empty()) {
        printf("[MAIN] Permanent rewrite not implemented in C++ runtime yet. Use Python for rewrites.\n");
    }

    xSemaphoreTake(g_sd_card_mutex, portMAX_DELAY);
    bool opened = sdcard.openFile(nac_path.c_str());
    xSemaphoreGive(g_sd_card_mutex);

    if (!opened) {
        fprintf(stderr, "[MAIN][ERROR] Cannot open '%s'\n", nac_path.c_str());
        return 1;
    }

    NacRuntimeContext* ctx = new NacRuntimeContext();
    ctx->model_path = nac_path;
    if (!initialize_nac_context(*ctx)) {
        fprintf(stderr, "[MAIN][ERROR] initialize_nac_context failed\n");
        sdcard.closeFile(); delete ctx; return 1;
    }

    if (getenv("NAC_REQUANT")) {
        ctx->quant_mode = QuantExecMode::DEQUANT_EXEC_REQUANT;
    } else {
        ctx->quant_mode = QuantExecMode::DEQUANT_ON_LOAD;
    }

    // ── Starting the MMAP (Memory Management) thread ──────────────────────────────
    xEventGroupClearBits(g_system_events, EVT_MEMORY_TASK_DONE);
    g_nac_memory_task_handle = new TaskNotify();

    std::thread mem_thread(nac_memory_task, (void*)ctx);

    // ── Finding and Executing MEP Orchestrator ──────────────────────────────────
    std::vector<uint8_t>       mep_bc;
    std::map<uint16_t, MepVal> mep_consts;
    bool has_orch = mep_load_from_nac(mep_bc, mep_consts, ctx);

    if (has_orch) {
        // БЛОК IN-MEMORY PATCH
        for (const auto& patch : patches) {
            bool patched = false;
            for (auto& pair : mep_consts) {
                if (pair.second.to_string() == patch.first) {
                    pair.second = MepVal::from_string(patch.second); // Using the right converter
                    printf("[Patch] In-memory: constant '%s' -> '%s'\n", patch.first.c_str(), patch.second.c_str());
                    patched = true;
                    break;
                }
            }
            if (!patched) printf("[Patch] WARNING: constant '%s' not found.\n", patch.first.c_str());
        }

#ifdef DBG
        printf("[MAIN] ORCH section found — MEP path. Mode: %s\n", exec_mode.c_str());
#endif
        MEPInterpreter mep(mep_bc.data(), mep_bc.size(), mep_consts, ctx);
        mep.set_pre_answers(pre_answers);
        mep.set_exec_mode(exec_mode); // Set the mode (infer, train, infer_train)
        mep.run();
        
    } else {
        // ── Fallback (Legacy compute path) ───────────────────────────────────
#ifdef DBG
        printf("[MAIN] No ORCH section — legacy compute path.\n");
#endif
        // Passing the prompt from the first pre_answer, if any
        std::string prompt = pre_answers.empty() ? "" : pre_answers[0];

        if (ctx->tokenizer && !ctx->tisa_manifest.empty()) {
            auto ids = ctx->tokenizer->run(ctx->tisa_manifest, prompt);
            if (!ids.empty()) {
                Tensor* inp = ctx->tensor_pool.acquire();
                inp->dtype = DataType::INT32;
                inp->shape = {1, (int)ids.size()};
                inp->update_from_shape();
                inp->data = alloc_fast(inp->size);
                if (inp->data) {
                    int32_t* d = static_cast<int32_t*>(inp->data);
                    for (size_t i = 0; i < ids.size(); ++i) d[i] = ids[i];
                    ctx->user_input_tensors.push_back(inp);
                }
            }
        }

        xEventGroupClearBits(g_system_events, EVT_COMPUTE_TASK_DONE);
        std::thread cmp_thread(nac_compute_task, (void*)ctx);
        xEventGroupWaitBits(g_system_events, EVT_COMPUTE_TASK_DONE, pdTRUE, pdFALSE, -1);
        cmp_thread.join();

        // Print the result
        for (int i = (int)ctx->results.size() - 1; i >= 0; --i) {
            Tensor* t = ctx->results[i];
            if (t && t->data && t->dtype == DataType::FLOAT32 && t->num_elements > 0) {
                size_t best = argmax(static_cast<float*>(t->data), t->num_elements);
                if (ctx->tokenizer && !ctx->tisa_manifest.empty()) {
                    std::string out = ctx->tokenizer->decode({(int)best});
                    printf("[OUTPUT] %s\n", out.c_str());
                } else {
                    printf("[OUTPUT] argmax=%zu (logit=%.4f)\n", best, static_cast<float*>(t->data)[best]);
                }
                break;
            }
        }
    }

    // ── Completing the MMAP stream ────────────────────────────────────────────────
    ctx->stop_flag.store(true);
    if (g_nac_memory_task_handle) g_nac_memory_task_handle->give(); 
    xEventGroupWaitBits(g_system_events, EVT_MEMORY_TASK_DONE, pdTRUE, pdFALSE, 10000);
    mem_thread.join();

    delete g_nac_memory_task_handle;
    g_nac_memory_task_handle = nullptr;

    sdcard.closeFile();
    delete ctx;
    return 0;
}

// ─── Scan directory for .nac files ───────────────────────────────────────────
[[maybe_unused]] static std::vector<std::string> scan_nac_files(const std::string& dir) {
    std::vector<std::string> out;
    try {
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            if (e.is_regular_file()) {
                std::string p = e.path().string();
                if (p.size() > 4 && p.substr(p.size() - 4) == ".nac")
                    out.push_back(p);
            }
        }
    } catch (...) {}
    std::sort(out.begin(), out.end());
    return out;
}

// ─── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    Serial.begin(0);

    g_system_events = xEventGroupCreate();
    g_sd_card_mutex = xSemaphoreCreateMutex();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.nac> [--mode infer|train|infer_train] [--train-mode head_only|trng] [--patch K=V] [args...]\n", argv[0]);
        return 1;
    }

    std::string nac_path = argv[1];
    std::string exec_mode = "infer_train";
    std::string train_mode = "head_only";
    std::vector<std::string> pre_answers;
    std::map<std::string, std::string> patches;
    std::map<std::string, std::string> rewrites;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            exec_mode = argv[++i];
        } else if (arg == "--train-mode" && i + 1 < argc) {
            train_mode = argv[++i];
        } else if (arg == "--patch" && i + 1 < argc) {
            std::string kv = argv[++i];
            size_t pos = kv.find('=');
            if (pos != std::string::npos) {
                patches[kv.substr(0, pos)] = kv.substr(pos + 1);
            }
        } else if (arg == "--rewrite" && i + 1 < argc) {
            std::string kv = argv[++i];
            size_t pos = kv.find('=');
            if (pos != std::string::npos) {
                rewrites[kv.substr(0, pos)] = kv.substr(pos + 1);
            }
        } else {
            pre_answers.push_back(arg);
        }
    }

    int ret = run_nac_file(nac_path, exec_mode, train_mode, pre_answers, patches, rewrites);

    vSemaphoreDelete(g_sd_card_mutex);
    delete g_system_events;
    return ret;
}
