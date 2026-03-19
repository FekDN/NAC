// =============================================================================
// MEP_interpreter.h  —  MEP ISA v1.0 interpreter for ESP32 / NAC runtime
//
// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
// Licensed under the Apache License, Version 2.0
//
// Mirrors Python MEP_interpreter.py for embedded execution.
// Works with the existing NacRuntimeContext from types.h.
// Model inference is executed SYNCHRONOUSLY (no new FreeRTOS tasks).
// =============================================================================
#pragma once

#include "types.h"
#include "op_kernels.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <string>

// MAX_INSTRUCTION_ARITY is defined in main.cpp; guard here so MEP_interpreter.cpp compiles standalone
#ifndef MAX_INSTRUCTION_ARITY
#define MAX_INSTRUCTION_ARITY 8
#endif

// ── External symbols provided by main.cpp ────────────────────────────────────
// (Already declared via types.h, but spelled out here for clarity)
extern SemaphoreHandle_t             g_sd_card_mutex;
extern std::map<uint8_t, KernelFunc> g_op_kernels;
// Handle used by run_model_sync() to notify nac_memory_task after each op.
// Set by main.cpp before MEP execution; NULL when task is not running.
extern TaskHandle_t                  g_nac_memory_task_handle;

bool    initialize_nac_context(NacRuntimeContext& ctx);
bool    dequantize_tensor(Tensor* tensor);
Tensor* create_fp32_copy_from_quantized(NacRuntimeContext& ctx, Tensor* src);
bool    read_and_parse_quant_metadata(Tensor* tensor, const TensorLocation& loc);
bool    requantize_result_tensor(NacRuntimeContext& ctx, Tensor* tensor);
void    softmax(const float* in, float* out, size_t len);
size_t  argmax(const float* data, size_t len);
void*   alloc_fast(size_t size);
size_t  gather_arguments(NacRuntimeContext& ctx,
                          const ParsedInstruction& ins,
                          uint32_t idx, Tensor** out);
Tensor* op_nac_pass(NacRuntimeContext* ctx,
                     const ParsedInstruction& ins,
                     Tensor** args, size_t argc);

// ── MEP context-slot value type ───────────────────────────────────────────────

enum class MepValType : uint8_t {
    NONE = 0,
    TENSOR,   // Tensor* in primary_ctx->tensor_pool
    INT64,
    FLOAT64,
    BOOL,
    STRING,   // heap-alloc'd char* (malloc/free)
    OPAQUE    // void* borrowed pointer (not owned — tokenizer, etc.)
};

/**
 * Tagged-union value for one of the 256 MEP context slots.
 *
 * Ownership rules:
 *   TENSOR  — pointer is in primary_ctx->tensor_pool; released via pool.release().
 *   STRING  — char* allocated with malloc(); freed in set_*() / ~MepVal().
 *   OPAQUE  — borrowed; never freed by MepVal.
 */
struct MepVal {
    MepValType type  = MepValType::NONE;
    Tensor*    tensor = nullptr;
    int64_t    i64    = 0;
    double     f64    = 0.0;
    bool       b      = false;
    char*      str_p  = nullptr;   // TYPE == STRING
    void*      ptr    = nullptr;   // TYPE == OPAQUE

    // ── setters ──────────────────────────────────────────────────────────────
    void _free_str() {
        if (type == MepValType::STRING && str_p) { free(str_p); str_p = nullptr; }
    }

    void set_none()   { _free_str(); type = MepValType::NONE;   tensor = nullptr; }
    void set_i64  (int64_t v)  { _free_str(); type = MepValType::INT64;   i64 = v; }
    void set_f64  (double  v)  { _free_str(); type = MepValType::FLOAT64; f64 = v; }
    void set_bool (bool    v)  { _free_str(); type = MepValType::BOOL;    b   = v; }
    void set_opaque(void*  p)  { _free_str(); type = MepValType::OPAQUE;  ptr = p; }

    void set_tensor(Tensor* t) { _free_str(); type = MepValType::TENSOR;  tensor = t; }

    void set_str(const char* s) {
        _free_str();
        type = MepValType::STRING;
        if (s) {
            size_t n = strlen(s) + 1;
            str_p = static_cast<char*>(malloc(n));
            if (str_p) memcpy(str_p, s, n);
        }
    }
    void set_str(const std::string& s) { set_str(s.c_str()); }

    // ── getters ───────────────────────────────────────────────────────────────
    const char* as_cstr() const {
        if (type == MepValType::STRING) return str_p ? str_p : "";
        return "";
    }

    int64_t as_i64() const {
        switch (type) {
            case MepValType::INT64:   return i64;
            case MepValType::FLOAT64: return (int64_t)f64;
            case MepValType::BOOL:    return b ? 1 : 0;
            case MepValType::TENSOR:
                if (tensor && tensor->num_elements == 1) {
                    switch (tensor->dtype) {
                        case DataType::FLOAT32: return (int64_t)*static_cast<float  *>(tensor->data);
                        case DataType::INT32:   return (int64_t)*static_cast<int32_t*>(tensor->data);
                        case DataType::INT8:    return (int64_t)*static_cast<int8_t *>(tensor->data);
                        default: break;
                    }
                }
                return 0;
            default: return 0;
        }
    }

    double as_f64() const {
        switch (type) {
            case MepValType::FLOAT64: return f64;
            case MepValType::INT64:   return (double)i64;
            case MepValType::BOOL:    return b ? 1.0 : 0.0;
            case MepValType::TENSOR:
                if (tensor && tensor->num_elements == 1 && tensor->dtype == DataType::FLOAT32)
                    return (double)*static_cast<float*>(tensor->data);
                return 0.0;
            default: return 0.0;
        }
    }

    bool as_bool() const {
        switch (type) {
            case MepValType::BOOL:    return b;
            case MepValType::INT64:   return i64 != 0;
            case MepValType::FLOAT64: return f64 != 0.0;
            case MepValType::TENSOR:
                if (tensor && tensor->num_elements == 1) {
                    if (tensor->dtype == DataType::FLOAT32) return *static_cast<float  *>(tensor->data) != 0.0f;
                    if (tensor->dtype == DataType::INT32)   return *static_cast<int32_t*>(tensor->data) != 0;
                }
                return false;
            default: return false;
        }
    }

    TISAVM* as_tokenizer() const {
        return (type == MepValType::OPAQUE) ? static_cast<TISAVM*>(ptr) : nullptr;
    }
};

// ── Loop frame ────────────────────────────────────────────────────────────────
struct MepLoopFrame {
    uint32_t body_ip;   ///< IP of first byte INSIDE the loop body
    int64_t  remaining; ///< Iterations remaining (decremented at LOOP_END)
};

// =============================================================================
// MEPInterpreter
// =============================================================================

/**
 * Synchronous MEP ISA v1.0 interpreter for ESP32.
 *
 *  Typical usage (from inside the EXECUTION_MODE branch):
 *
 *    // 1. Load ORCH bytecode from the open SD file
 *    std::vector<uint8_t>      mep_bc;
 *    std::map<uint16_t,MepVal> mep_consts;
 *    if (mep_load_from_nac(mep_bc, mep_consts)) {
 *        // 2. Create interpreter, wire up primary context
 *        MEPInterpreter mep(mep_bc.data(), mep_bc.size(), mep_consts, context);
 *        mep.set_pre_answers({ g_user_prompt });
 *        // 3. Run — handles tokenise → infer → decode internally
 *        mep.run();
 *    }
 */
class MEPInterpreter {
public:
    /**
     * @param bytecode      Raw MEP bytecode payload (ORCH section).
     * @param bytecode_len  Length in bytes.
     * @param constants     Pre-parsed constants pool (from mep_load_from_nac).
     * @param primary_ctx   Already-loaded NacRuntimeContext (model ID 0).
     *                      Borrowed — do NOT free while interpreter is alive.
     */
    MEPInterpreter(const uint8_t*                   bytecode,
                   size_t                            bytecode_len,
                   const std::map<uint16_t, MepVal>& constants,
                   NacRuntimeContext*                primary_ctx);
    ~MEPInterpreter();

    /// Strings popped FIFO by SRC_USER_PROMPT (e.g. { g_user_prompt }).
    void set_pre_answers(const std::vector<std::string>& answers) {
        m_pre_answers.assign(answers.begin(), answers.end());
    }

    /// Execute the MEP plan.  Returns the value from EXEC_RETURN, or NONE on HALT.
    MepVal run();

private:
    // ── VM state ──────────────────────────────────────────────────────────────
    const uint8_t*             m_plan;
    size_t                     m_plan_size;
    std::map<uint16_t, MepVal> m_consts;
    NacRuntimeContext*         m_primary_ctx;

    uint32_t                   m_ip      = 0;
    bool                       m_running = false;
    MepVal                     m_ctx[256];
    std::vector<MepLoopFrame>  m_loop_stack;
    std::vector<std::string>   m_pre_answers;
    MepVal                     m_return_val;

    // ── helpers ───────────────────────────────────────────────────────────────
    uint8_t  ru8();
    uint16_t ru16();
    int16_t  ri16();

    MepVal& slot(uint8_t k)             { return m_ctx[k]; }
    const MepVal& get_const(uint16_t id) const;

    /// Release tensor in slot k back to pool, then reset slot to NONE.
    void release_slot(uint8_t k);

    /// Place tensor in slot k, releasing previous occupant if needed.
    void put_tensor(uint8_t k, Tensor* t);

    /// Create a new Tensor from the primary pool with alloc_fast data.
    Tensor* new_tensor(DataType dt, const std::vector<int>& shape);

    /// Total instruction length in bytes (opcode included) — for loop-skip.
    uint32_t instr_len(uint32_t at_ip) const;

    // ── instruction handlers ─────────────────────────────────────────────────
    void h_src_user_prompt();
    void h_src_constant();

    void h_res_load_model();
    void h_res_load_datafile();
    void h_res_load_extern();
    void h_res_load_dynamic();
    void h_res_unload();

    void h_preproc_encode();
    void h_preproc_decode();
    void h_preproc_get_id();
    void h_string_format();

    void h_tensor_create();
    void h_tensor_manipulate();
    void h_tensor_combine();
    void h_tensor_info();
    void h_tensor_extract();

    void h_sys_copy();
    void h_sys_debug_print();

    void h_math_unary();
    void h_math_binary();
    void h_math_aggregate();
    void h_logic_compare();
    void h_analysis_top_k();
    void h_analysis_sample();

    void h_model_run_static();

    void h_flow_loop_start();
    void h_flow_loop_end();
    void h_flow_branch_if();
    void h_flow_break_loop_if();

    void h_serialize_object();
    void h_io_write();
    void h_exec_return();
    void h_exec_halt();

    // ── synchronous NAC model execution ─────────────────────────────────────
    /**
     * Runs primary_ctx's NAC graph synchronously in the calling thread.
     * Inputs must already be in primary_ctx->user_input_tensors.
     * On return, output tensors are in `out_tensors` (ownership transferred
     * from primary_ctx->results — caller must release via pool).
     */
    bool run_model_sync(std::vector<Tensor*>& out_tensors);

    /// Inline memory-task tick: processes mmap_schedule commands for `tick`.
    void mmap_tick_sync(uint32_t tick);

    /// Load a weight tensor on-demand (shared with mmap preload path).
    Tensor* load_param_tensor(uint16_t param_id);
};

// ── Free functions ────────────────────────────────────────────────────────────

/**
 * Read the ORCH section from the currently open SD card file (sdcard global).
 * Call AFTER sdcard is open and the file is seekable.
 *
 * @param out_bytecode  Receives MEP bytecode bytes.
 * @param out_consts    Receives pre-parsed constants pool.
 * @return true if ORCH section is present and successfully parsed.
 */
bool mep_load_from_nac(std::vector<uint8_t>&       out_bytecode,
                        std::map<uint16_t, MepVal>&  out_consts,
                        NacRuntimeContext*           ctx);

/**
 * All-in-one helper used from the EXECUTION_MODE block in main.cpp.
 *
 * Checks whether the open NAC file has an ORCH section.  If yes, loads it,
 * creates an MEPInterpreter, and runs the plan.  Returns false if the file
 * has no ORCH section (caller should fall back to manual execution).
 *
 * @param primary_ctx  Fully initialised NacRuntimeContext for the model.
 * @param answers      Pre-answers for SRC_USER_PROMPT (typically {g_user_prompt}).
 * @return true  if ORCH was found and execution completed (success or halt).
 * @return false if no ORCH section present — use legacy task-based path.
 */
bool mep_run_orchestrated(NacRuntimeContext*               primary_ctx,
                           const std::vector<std::string>&  answers);
