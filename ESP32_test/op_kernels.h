// op_kernels.h

#ifndef OP_KERNELS_H
#define OP_KERNELS_H

#include "types.h"

using KernelFunc = Tensor* (*)(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

void register_kernels();

// --- Стандартные NAC ядра ---
Tensor* op_nac_pass(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_add(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_sub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_view(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_arange(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_neg(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_where(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

// --- ATen ядра ---
Tensor* op_aten_mul_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_div_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_silu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_relu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_gelu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_linear_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_layer_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_softmax_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_rsqrt_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_embedding_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_transpose_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_permute_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_slice_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_conv2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_native_batch_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_max_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_adaptive_avg_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

extern std::map<std::string, KernelFunc> g_kernel_string_map;
extern std::map<uint8_t, KernelFunc> g_op_kernels;

#endif