// op_kernels.h
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

#ifndef OP_KERNELS_H
#define OP_KERNELS_H

#include "types.h"
#include "platform.h"

using KernelFunc = Tensor* (*)(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

void register_kernels();

// --- Standard NAC kernels ---
Tensor* op_nac_pass(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_add(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_sub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_gt(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_neg(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_where(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_clone(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_view(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_le(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_arange(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_mul(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_div(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_transpose(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_nac_ones(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

// --- ATen kernels ---
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
Tensor* op_aten_expand_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_ne_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten__to_copy_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_cumsum_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_type_as_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_zeros_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_select_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_unsqueeze_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_sym_size_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_tanh_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_cos_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_sin_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_exp_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_log_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_pow_Tensor_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_eq_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_lt_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_lt_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_min_other(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_zeros_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_ones_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_full_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_mean_dim(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_triu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_cat_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_group_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_upsample_nearest2d_vec(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_slice_scatter_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_addmm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_full_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_scaled_dot_product_attention_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_getitem(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_split_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_hardsigmoid_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_hardswish_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_sigmoid_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_sum_dim_IntList(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_scalar_tensor_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

// TRNG Backwards Stub
Tensor* op_backward_stub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_hardswish_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_threshold_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_select_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_native_layer_norm_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_gelu_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten_tanh_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);
Tensor* op_aten__scaled_dot_product_flash_attention_for_cpu_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

Tensor* op_aten_lstm_cell_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc);

extern std::map<std::string, KernelFunc> g_kernel_string_map;
extern std::map<uint8_t, KernelFunc> g_op_kernels;

#endif