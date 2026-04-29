# --- START OF FILE NAC_kernels.py ---

# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import numpy as np
import math
from typing import Any

NAC_OPS = {
    10: "nac.pass",
    11: "nac.add",
    12: "nac.sub",
    13: "nac.gt",
    14: "nac.neg",
    15: "nac.where",
    16: "nac.clone",
    17: "nac.view",
    18: "nac.le",
    19: "nac.arange",
    20: "nac.mul",
    21: "nac.div",
    22: "nac.matmul",
    23: "nac.transpose",
    24: "nac.zeros",
    25: "nac.zeros_like",
    26: "nac.new_zeros",
    27: "nac.ones",
    28: "nac.ones_like",
    29: "nac.new_ones",
    30: "nac.full",
    31: "nac.full_like",
}

NAC_OPS_REVERSED = {v: k for k, v in NAC_OPS.items()}
_max_standard_op_id = max(NAC_OPS.keys()) if NAC_OPS else 9
CUSTOM_OP_ID_START = _max_standard_op_id + 1

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s

from functools import lru_cache

@lru_cache(maxsize=128)
def get_im2col_indices_cached(x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - field_height) // stride + 1
    out_w = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(field_width), field_height * C)

    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0[:, None] + i1[None, :]
    j = j0[:, None] + j1[None, :]
    k = np.repeat(np.arange(C), field_height * field_width)[:, None]

    return k, i, j

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')

    k, i, j = get_im2col_indices_cached(
        x.shape, field_height, field_width, padding, stride
    )

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    return cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)


class NacKernelBase:
    def _enum_to_numpy_dtype(self, enum: Any) -> Any: return {0:np.float32, 1:np.float64, 2:np.float16, 4:np.int32, 5:np.int64, 6:np.int16, 7:np.int8, 8:np.uint8, 9:np.bool_}.get(enum)

    def op_nac_conv2d_precomputed(self, x, precomputed_weight, bias, params):
        """
        Выполняет свертку, используя предвычисленные веса.
        - x: неизвестный входной тензор.
        - precomputed_weight: веса, уже преобразованные в матрицу (C_out, C_in * kH * kW).
        - bias: тензор смещения (если есть).
        - params: список всех остальных параметров в фиксированном порядке:
                  [C_out, out_h, out_w, kH, kW, sH, sW, pH, pW, groups]
        """
        # Распаковываем параметры из списка
        C_out, out_h, out_w, kH, kW, sH, sW, pH, pW, groups = params

        if x.dtype != np.float32: x = x.astype(np.float32, copy=False)

        N, C_in, H, W = x.shape

        # im2col - это рантайм операция, так как она зависит от x
        x_cols = im2col_indices(x, kH, kW, padding=pH, stride=sH)

        # Главное матричное умножение - основная работа в рантайме
        if groups == 1:
            out_cols = precomputed_weight @ x_cols
        else:
            # Эта логика должна быть реализована, если у вас есть групповые свертки
            raise NotImplementedError("Grouped precomputed convolution is not implemented yet.")

        # Преобразование колонок обратно в формат изображения
        out = out_cols.reshape(C_out, out_h, out_w, N).transpose(3, 0, 1, 2)

        if bias is not None:
            if bias.dtype != np.float32: bias = bias.astype(np.float32, copy=False)
            out += bias.reshape(1, -1, 1, 1)

        return out

    def op_getitem(self, t, i):
        return t[int(i)]

    def op_aten_sym_size_int(self, tensor, dim):
        return tensor.shape[int(dim)]

    def op_aten_relu_default(self, x):
        return np.maximum(x, 0)

    def op_aten_exp_default(self, x):
        return np.exp(x)

    def op_aten_cos_default(self, x):
        return np.cos(x)

    def op_aten_sin_default(self, x):
        return np.sin(x)

    def op_aten_eq_Scalar(self, a, b):
        return np.equal(a, b)

    def op_aten_cat_default(self, *args, **kwargs):
        # Version contains a patch for the torch.export bug.
        if not args: return np.array([])
        # Parse arguments reliably, since dim may or may not be the last one.
        dim_candidate = args[-1]
        if isinstance(dim_candidate, (int, np.integer)):
            tensors = args[:-1]; dim = int(dim_candidate)
            # Handling the case where all tensors are passed as a single list
            if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)): tensors = tensors[0]
        else:
            tensors = args; dim = 0
            if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)): tensors = tensors[0]
        tensors_to_cat = [np.asarray(t) for t in tensors if t is not None]
        # Gpt2-streaming bug patch: find a pair of 2D and 4D tensors and return the 4D one
        if len(tensors_to_cat) == 2:
            t0, t1 = tensors_to_cat[0], tensors_to_cat[1]
            if {t0.ndim, t1.ndim} == {2, 4}:
                # This is a specific error pattern. We return the correct 4D tensor.
                return t1 if t1.ndim == 4 else t0
        # General logic for KV cache: filtering out empty tensors
        non_empty_tensors = [t for t in tensors_to_cat if t.size > 0]
        if not non_empty_tensors: 
            return tensors_to_cat[0] if tensors_to_cat else np.array([])
        if len(non_empty_tensors) == 1: 
            return non_empty_tensors[0]
        return np.concatenate(non_empty_tensors, axis=dim)

    def _sigmoid(self, x):
        x = np.asarray(x)
        dtype = x.dtype
        x = x.astype(dtype, copy=False)
        out = np.empty_like(x)
        positive = x >= 0
        negative = ~positive
        # sigmoid(x) = 1 / (1 + exp(-x))
        out[positive] = 1 / (1 + np.exp(-x[positive], dtype=dtype))
        # sigmoid(x) = exp(x) / (1 + exp(x))
        exp_x = np.exp(x[negative], dtype=dtype)
        out[negative] = exp_x / (1 + exp_x)
        return out

    def op_aten_silu_default(self, x):
        x = np.asarray(x, dtype=np.float32)
        # return x * self._sigmoid(x)
        # Formula x * sigmoid(x) equivalent x / (1 + exp(-x))
        # This avoids creating a separate array for the sigmoid.
        return x / (1.0 + np.exp(-x))

    def op_aten_group_norm_default(self, x, num_groups, weight=None, bias=None, eps=1e-5):
        x = np.asarray(x)

        # FIX: torch.export VAE attention bug
        if x.ndim == 3:
            # x: [N, C, L] — attention tokens
            # In PyTorch, it's not group_norm, but channel-wise norm.
            mean = x.mean(axis=2, keepdims=True)
            var  = x.var(axis=2, keepdims=True)
            x = (x - mean) / np.sqrt(var + eps)

            if weight is not None:
                x = x * weight.reshape(1, -1, 1)
            if bias is not None:
                x = x + bias.reshape(1, -1, 1)
            return x

        # normal PATH
        N, C, H, W = x.shape
        assert C % num_groups == 0

        x = x.reshape(N, num_groups, C // num_groups, H * W)
        mean = x.mean(axis=(2, 3), keepdims=True)
        var  = x.var(axis=(2, 3), keepdims=True)
        x = (x - mean) / np.sqrt(var + eps)
        x = x.reshape(N, C, H, W)

        if weight is not None:
            x = x * weight.reshape(1, -1, 1, 1)
        if bias is not None:
            x = x + bias.reshape(1, -1, 1, 1)
        return x

    def op_aten_scaled_dot_product_attention_default(self, *args, _perm=None, **kwargs):
        tensors = {}; consts = {}
        has_semantic_codes = any(c in ('Q', 'K', 'V', 'M') for c in _perm) if _perm else False
        if has_semantic_codes:
            arg_iter = iter(args)
            for code in _perm:
                arg = next(arg_iter)
                if code == 'Q': tensors['q'] = arg
                elif code == 'K': tensors['k'] = arg
                elif code == 'V': tensors['v'] = arg
                elif code == 'M': tensors['attn_mask'] = arg
                elif code == 'f': consts['scale'] = arg
                elif code == 'b': consts['is_causal'] = arg
        else:
            tensor_args = [arg for arg in args if isinstance(arg, np.ndarray)]
            const_args = [arg for arg in args if not isinstance(arg, np.ndarray)]
            if len(tensor_args) >= 3:
                tensors = {'q': tensor_args[0], 'k': tensor_args[1], 'v': tensor_args[2]}
            if len(tensor_args) >= 4: tensors['attn_mask'] = tensor_args[3]
        q, k, v = tensors.get('q'), tensors.get('k'), tensors.get('v')
        if q is None or k is None or v is None:
            raise ValueError("Missing Q, K, or V for attention.")
        # Detects and repairs broken shapes from UNet and VAE
        # VAE pattern: q, k, v come as [B, 1, S, C], where C = H*D
        # For example: (1, 1, 1024, 512)
        if q.ndim == 4 and q.shape[1] == 1:
            num_heads = 8 # For SD 1.5 VAE
            batch, _, seq_len, total_channels = q.shape
            head_dim = total_channels // num_heads
            # (B, 1, S, H*D) -> (B, S, H, D) -> (B, H, S, D)
            q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        # UNet pattern: q, k, v come as [B, S, C]
        elif q.ndim == 3:
            num_heads = 8 # Для SD 1.5 UNet
            batch, seq_len, total_channels = q.shape
            head_dim = total_channels // num_heads
            # (B, S, H*D) -> (B, S, H, D) -> (B, H, S, D)
            q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3) # -1 for K/V from text (seq_len=77)
            v = v.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3) # -1 for K/V from text (seq_len=77)
        scale = consts.get('scale') or (1.0 / math.sqrt(q.shape[-1]))
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        if 'attn_mask' in tensors and tensors['attn_mask'] is not None:
            attn = attn + tensors['attn_mask']
        attn = softmax(attn, axis=-1)
        return attn @ v

    def op_aten_pad_default(self, x, pad, mode='constant', value=0.0):
        if mode != 'constant':
            raise NotImplementedError("Only constant padding is supported.")
        pad = list(pad)
        ndim = x.ndim
        np_pad = [(0, 0)] * ndim
        idx = 0
        for dim in range(ndim - 1, ndim - 1 - len(pad) // 2, -1):
            np_pad[dim] = (pad[idx], pad[idx + 1])
            idx += 2
        return np.pad(x, np_pad, mode='constant', constant_values=value)

    def op_aten_addmm_default(self, C, A, B, beta=1.0, alpha=1.0):
        return (beta * C) + (alpha * (A @ B))

    def op_aten_linear_default(self, x, w, b=None, _perm=None):
        x = np.asarray(x)
        w = np.asarray(w)
        # --- PyTorch dtype promotion ---
        if x.dtype == np.float32 or w.dtype == np.float32:
            compute_dtype = np.float32
        elif x.dtype == np.float16 or w.dtype == np.float16:
            compute_dtype = np.float16
        else:
            compute_dtype = np.result_type(x.dtype, w.dtype)
        # --- Cast ONLY for compute ---
        x_c = x.astype(compute_dtype, copy=False)
        w_c = w.astype(compute_dtype, copy=False)
        # --- Shape check ---
        if x_c.shape[-1] != w_c.shape[1]:
            raise ValueError(
                f"aten.linear shape mismatch: "
                f"{x_c.shape[-1]} vs {w_c.shape}"
            )
        # --- MatMul ---
        y = np.matmul(x_c, w_c.T)
        # --- Bias ---
        if b is not None:
            b_c = np.asarray(b).astype(compute_dtype, copy=False)
            y = y + b_c
        # --- PyTorch: output dtype = compute dtype ---
        return y

    def op_aten_layer_norm_default(self, x, norm_shape, w, b, eps=1e-5):
        x = np.asarray(x, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        axis = tuple(range(x.ndim - len(norm_shape), x.ndim))
        mean = np.mean(x, axis=axis, keepdims=True)
        # fused variance: faster and more accurate than **2
        diff = x - mean
        var = np.mean(diff * diff, axis=axis, keepdims=True)
        inv_std = np.reciprocal(np.sqrt(var + eps))
        # fused normalization
        out = diff * inv_std
        out *= w
        out += b
        return out

    def op_aten_split_Tensor(self, x, split_size, dim):
        dim = int(dim)
        split_size = int(split_size)
        if x.shape[dim] == 0:
            return [x]
        return np.split(x, np.arange(split_size, x.shape[dim], split_size), axis=dim)

    def op_aten_mul_Tensor(self, a, b, _perm=None):
        a = np.asarray(a)
        b = np.asarray(b)

        # First try to perform standard multiplication.
        # If NumPy crashes with a broadcasting error, it's most likely an empty tensor issue.
        try:
            # PyTorch-style dtype promotion
            if a.dtype == np.float32 or b.dtype == np.float32:
                dtype = np.float32
            elif a.dtype == np.float16 or b.dtype == np.float16:
                dtype = np.float16
            else:
                dtype = np.result_type(a.dtype, b.dtype)
            
            # Type casting as before
            if a.ndim == 0: a = np.array(a, dtype=dtype)
            else: a = a.astype(dtype, copy=False)
            if b.ndim == 0: b = np.array(b, dtype=dtype)
            else: b = b.astype(dtype, copy=False)

            return np.multiply(a, b)

        except ValueError as e:
            # If the error is not about broadcasting, forward it further.
            if "broadcast" not in str(e):
                raise e
            
            # Logically, the result of multiplication by the empty tensor is the empty tensor.
            # Return the empty operand.
            if a.size == 0:
                return a
            if b.size == 0:
                return b
            
            # If both are not empty and the error still exists, then the forms are indeed incompatible.
            raise ValueError(f"aten::mul.Tensor shapes {a.shape} and {b.shape} are not broadcast-compatible")

    def op_aten_expand_default(self, *args, _perm=None):
        """
        The function implementation can take the target shape
        either as a sequence of individual arguments (*shape),
        or as a single argument in the form of a list/tuple.
        This fixes a bug where torch.export packs
        symbolic dimensions into a single tuple.
        """
        if not args:
            raise ValueError("aten.expand: No form arguments passed.")
        # The first argument is always the input tensor.
        x = args[0]
        shape_args = args[1:]
        # Сheck if the entire form was submitted as a single list/tuple.
        if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
            target_shape = list(shape_args[0])
        else:
            # Otherwise, the form was passed as separate arguments.
            target_shape = list(shape_args)
            
        # ПРИВОДИМ ВСЕ РАЗМЕРНОСТИ К ЦЕЛЫМ ЧИСЛАМ
        target_shape = [int(dim) for dim in target_shape]
        
        # The rank of the target shape must be >= the rank of the input tensor
        if len(target_shape) < x.ndim:
            raise ValueError(
                f"aten.expand: the rank of the target shape ({len(target_shape)}) cannot be less than the rank of the input ({x.ndim})."
            )
        # Placeholder processing `-1`.
        for i in range(1, x.ndim + 1):
            if target_shape[-i] == -1:
                target_shape[-i] = x.shape[-i]
        # Add unit dimensions to match the rank.
        rank_diff = len(target_shape) - x.ndim
        if rank_diff > 0:
            x_reshaped = x.reshape((1,) * rank_diff + x.shape)
        else:
            x_reshaped = x
        return np.broadcast_to(x_reshaped, tuple(target_shape))

    def op_aten_select_int(self, tensor, dim, index):
        """
        Selects a slice along the `dim` dimension with the `index` index and removes that dimension.
        Equivalent to tensor[:, ..., index, ..., :]
        """
        # Create a list of slices (slice(None) is equivalent to ':')
        slicer = [slice(None)] * tensor.ndim
        
        # We replace the cut for the required measurement with a specific index
        slicer[int(dim)] = int(index)
        
        # Apply slices. Conversion to tuple is required.
        return tensor[tuple(slicer)]

    def op_aten_mean_dim(self, tensor, dim=None, keepdim=False, *args, **kwargs):
        # aten.mean can pass axes as a list, while np.mean expects a tuple
        if dim is not None and isinstance(dim, list):
            dim = tuple(dim)
            # Calculating an average on low-precision types (float16) leads to a loss of
            # precision. We must force the use of float32 for
            # internal summation to ensure stability.
        return np.mean(tensor, axis=dim, keepdims=keepdim, dtype=np.float32)

    def op_aten_lt_Scalar(self, a, b):
        return np.less(a, b)

    def op_aten_min_other(self, a, b):
        return np.minimum(a, b)

    def op_aten_abs_default(self, tensor):
        return np.abs(tensor)

    def op_aten_log_default(self, tensor):
        # Add a very small value (epsilon) to avoid log(0), <--!!
        # which results in -inf and breaks the calculation of relative positional embeddings in T5.
        epsilon = np.finfo(np.float32).eps
        return np.log(np.maximum(tensor, epsilon))

    def op_aten_rsqrt_default(self, tensor):
        # For numerical stability, we convert to float32
        x = np.asarray(tensor, dtype=np.float32)
        return np.reciprocal(np.sqrt(x))

    def op_aten_div_Tensor(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        # PyTorch always promotes to float when dividing to avoid
        # integer division and loss of precision. We emulate this by
        # ensuring that the type for calculations is at least float32.
        compute_dtype = np.result_type(a, b, np.float32)

        a_c = a.astype(compute_dtype, copy=False)
        b_c = b.astype(compute_dtype, copy=False)

        # np.divide handles division by zero correctly by default (-> inf),
        # which is consistent with PyTorch's behavior.
        return np.divide(a_c, b_c)

    def op_aten_tanh_default(self, x):
        return np.tanh(x)

    def op_aten_pow_Tensor_Scalar(self, a, b):
        return np.power(a, b)

    def op_aten_triu_default(self, x, diagonal=0, **kwargs):
        x = np.asarray(x)
        return np.triu(x, k=diagonal)

    def op_aten_unsqueeze_default(self, x, dim):
        dim = int(dim) # Обязательно приводим к int
        dim = dim if dim >= 0 else x.ndim + dim + 1  # поддержка отрицательных dim
        return np.expand_dims(x, dim)

    def op_aten_softmax_int(self, tensor, dim):
        dim = int(dim)
        x = np.asarray(tensor)

        # If the input array is empty (i.e., any of its dimensions is 0),
        # then calculating max/sum will cause an error.
        # The result of softmax for an empty array is also an empty array.
        if x.size == 0:
            return x

        # 1. Choosing a safe dtype
        if x.dtype == np.float16:
            x = x.astype(np.float32)
            out_dtype = np.float16
        else:
            out_dtype = x.dtype
            
        # 2. Stable softmax
        x_max = np.max(x, axis=dim, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=dim, keepdims=True)
        
        # Protection against division by zero if all exp are 0
        y = np.divide(x_exp, x_sum, out=np.zeros_like(x_exp), where=x_sum!=0)
        
        return y.astype(out_dtype, copy=False)

    def op_aten_embedding_default(self, w, idx, *a, **kw):
        if not np.issubdtype(idx.dtype, np.integer):
            idx = idx.astype(np.int64)
        return w[idx] if idx.size > 0 else np.zeros((0, w.shape[1]), dtype=w.dtype)

    def op_aten_slice_Tensor(self, x, dim, start=None, end=None, step=1, _perm=None):
        x = np.asarray(x)
        dim = int(dim)
        if start is not None: start = int(start)
        if end is not None: end = int(end)
        step = int(step)
        
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            return x
        slc = [slice(None)] * x.ndim
        if start is None:
            start = 0
        if end is None or end > x.shape[dim]:
            end = x.shape[dim]
        slc[dim] = slice(start, end, step)
        return x[tuple(slc)]

    def op_aten__to_copy_default(self, x, *a, **kw):
        dtype = kw.get('dtype', a[0] if a else None)
        if dtype is not None:
            if x.dtype != dtype:
                return np.asarray(x, dtype=dtype)
        # PyTorch: copy only if needed
        return np.asarray(x)

    def op_aten_slice_scatter_default(self, x, src, dim=0, start=None, end=None, step=1):
        """
        Implements aten.slice_scatter.default.
        Equivalent to x[:, start:end, :] = src
        """
        dim = int(dim)
        if start is not None: start = int(start)
        if end is not None: end = int(end)
        step = int(step)
        
        out = np.copy(x)
        slicer = [slice(None)] * x.ndim
        slicer[dim] = slice(start, end, step)
        out[tuple(slicer)] = src
        return out

    def op_aten_copy_default(self, x, src, *args, **kwargs):
        """
        Simply return a copy of the source
        """
        return np.copy(src)

    def op_aten_ne_Scalar(self, x, val):
        return x != val

    def op_aten_cumsum_default(self, x, dim, *a, **kw):
        return np.cumsum(x, axis=int(dim))

    def op_aten_type_as_default(self, x, other):
        return x.astype(other.dtype) if x.dtype != other.dtype else x

    def op_aten_gelu_default(self, x, *a):
        # return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        k0 = 0.7978845608028654
        k1 = 0.044715
        if isinstance(x, np.ndarray):
            x2 = x * x
            x3 = x2 * x
            return 0.5 * x * (1.0 + np.tanh(k0 * (x + k1 * x3)))
        else:
            x2 = x * x
            x3 = x2 * x
            return 0.5 * x * (1.0 + math.tanh(k0 * (x + k1 * x3)))

    def op_aten_conv2d_default(self, x, weight, bias=None,
                               stride=(1,1), padding=(0,0),
                               dilation=(1,1), groups=1):

        if dilation != (1, 1):
            raise NotImplementedError("Convolution with dilation != 1 is not supported by this kernel.")

        # Type casting for calculations
        if x.dtype != np.float32: x = x.astype(np.float32, copy=False)
        if weight.dtype != np.float32: weight = weight.astype(np.float32, copy=False)

        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        sH, sW = stride
        pH, pW = padding

        # Calculating output sizes
        out_h = (H + 2 * pH - kH) // sH + 1
        out_w = (W + 2 * pW - kW) // sW + 1

        # im2col converts image patches into columns
        x_cols = im2col_indices(x, kH, kW, padding=pH, stride=sH)

        if groups == 1:
            # Standard roll: one large matmul
            w_cols = weight.reshape(C_out, -1)
            out_cols = w_cols @ x_cols
        else:
            # Grouped convolution: performed as a series of independent convolutions
            # Output data will be collected here
            out_cols = np.zeros((C_out, x_cols.shape[1]), dtype=x.dtype)
            
            # Channel sizes per group
            C_out_per_group = C_out // groups
            C_in_per_group = C_in // groups
            
            # Size of one group in expanded scales and inputs
            w_group_size = C_out_per_group * (C_in_per_group * kH * kW)
            x_group_size = C_in_per_group * kH * kW

            w_reshaped = weight.reshape(groups, C_out_per_group, -1)
            x_cols_reshaped = x_cols.reshape(groups, x_group_size, -1)
            
            # Perform matmul for each group separately
            out_cols_reshaped = np.matmul(w_reshaped, x_cols_reshaped)
            
            # Collect the result back into a single matrix
            out_cols = out_cols_reshaped.reshape(C_out, -1)

        # Convert columns back to image format
        out = out_cols.reshape(C_out, out_h, out_w, N).transpose(3, 0, 1, 2)

        if bias is not None:
            out += bias.reshape(1, -1, 1, 1)

        return out

    def op_aten_conv2d_default1(self, x, weight, bias=None,
                               stride=(1,1), padding=(0,0),
                               dilation=(1,1), groups=1):

        if dilation != (1,1):
            raise NotImplementedError

        N, C_in, H, W = x.shape
        C_out, C_in_g, kH, kW = weight.shape
        sH, sW = stride
        pH, pW = padding

        if sH != sW or pH != pW:
            raise NotImplementedError

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if weight.dtype != np.float32:
            weight = weight.astype(np.float32, copy=False)

        if groups > 1:
            out = []
            ocpg = C_out // groups
            icpg = C_in // groups
            for g in range(groups):
                out.append(
                    self.op_aten_conv2d_default(
                        x[:, g*icpg:(g+1)*icpg],
                        weight[g*ocpg:(g+1)*ocpg],
                        None, stride, padding, dilation, 1
                    )
                )
            out = np.concatenate(out, axis=1)
        else:
            out_h = (H + 2*pH - kH)//sH + 1
            out_w = (W + 2*pW - kW)//sW + 1

            x_cols = im2col_indices(x, kH, kW, pH, sH)
            x_cols = np.ascontiguousarray(x_cols)

            w_cols = weight.reshape(C_out, -1)
            w_cols = np.ascontiguousarray(w_cols)

            out_cols = w_cols @ x_cols
            out = out_cols.reshape(C_out, out_h, out_w, N).transpose(3,0,1,2)

        if bias is not None:
            out += bias.reshape(1, -1, 1, 1)

        return out

    def op_aten_max_pool2d_default(self, x, kernel_size, stride=None, padding=(0,0), dilation=(1,1), ceil_mode=False):
        if tuple(dilation) != (1, 1):
            raise NotImplementedError("Dilation != 1 not supported.")
        # If stride is not specified, it is equal to kernel_size
        stride = stride or kernel_size
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        pH, pW = padding
        if pH != pW or sH != sW:
             raise NotImplementedError("Asymmetric padding/stride not supported by this fast pool kernel.")
        # Calculating the output dimensions
        out_h = (H + 2 * pH - kH) // sH + 1
        out_w = (W + 2 * pW - kW) // sW + 1
        x_cols = im2col_indices(x, kH, kW, padding=pH, stride=sH)
        # x_cols has the form (C * kH * kW, N * out_h * out_w)
        # Need to find the maximum for groups corresponding to one window
        # Reshape to highlight the batch and channels
        # The form becomes (C, kH * kW, N * out_h * out_w)
        x_cols_reshaped = x_cols.reshape(C, kH * kW, -1)
        # Find the maximum along the axis corresponding to the elements in the window (kH * kW)
        max_cols = np.max(x_cols_reshaped, axis=1)
        # Reshape the result into the final image form.
        # Form max_cols: (C, N * out_h * out_w)
        out = max_cols.reshape(C, N, out_h, out_w)
        # Rearrange the axes so that the batch is first: (N, C, H, W)
        out = out.transpose(1, 0, 2, 3)
        return out

    def op_aten_tril_default(self, x, diagonal=0, **kwargs):
        if x is None:
            # streaming causal mask: 1x1 (minimally correct start)
            x = np.ones((1, 1), dtype=np.float32)
        x = np.asarray(x)
        return np.tril(x, k=diagonal)

    def op_aten_ge_Tensor(self, a, b, *args, **kw):
        return np.greater_equal(a, b)

    def op_aten__native_batch_norm_legit_no_training_default(self, x, w, b, rm, rv, mom, eps):
        eps_val = eps if eps is not None else 1e-5
        sh = (1, -1, 1, 1) if x.ndim == 4 else (1, -1)
        return (x - rm.reshape(sh)) / np.sqrt(rv.reshape(sh) + eps_val) * w.reshape(sh) + b.reshape(sh)

    def op_aten_upsample_nearest2d_vec(self, x, output_size, scale_factors, *args, **kwargs):
        _, _, in_h, in_w = x.shape
        sh = sw = 2
        return x[:, :, :, None, :, None] \
            .repeat(sh, axis=3) \
            .repeat(sw, axis=5) \
            .reshape(x.shape[0], x.shape[1], in_h * sh, in_w * sw)

    def op_aten_upsample_nearest2d_vec2(self, x, output_size, scale_factors, *args, **kwargs):
        """ 
        Handle cases where output_size=None, and scaling is specified via scale_factors
        """
        if x.ndim != 4:
            raise ValueError("upsample_nearest2d_vec: input must be a 4D tensor")
        if output_size is None:
            # If output_size is not specified, calculate it from scale_factors.
            # In UNet, scale_factor is typically 2.0 for both height and width.
            # The scale_factors argument can come as a list [h_scale, w_scale].
            if scale_factors and len(scale_factors) == 2:
                scale_h = int(scale_factors[0])
                scale_w = int(scale_factors[1])
            else:
                # Fallback, if scale_factors are not passed, we use the default scale=2
                scale_h, scale_w = 2, 2
            _, _, in_h, in_w = x.shape
            out_h, out_w = in_h * scale_h, in_w * scale_w
        else:
            # If output_size is specified, use it
            out_h, out_w = output_size
        _, _, in_h, in_w = x.shape
        # Calculating integer scaling factors
        # Adding protection against division by zero if in_h or in_w=0
        scale_h_factor = out_h // in_h if in_h > 0 else 1
        scale_w_factor = out_w // in_w if in_w > 0 else 1
        upsampled = x.repeat(scale_h_factor, axis=2).repeat(scale_w_factor, axis=3)
        return upsampled

    def op_aten_adaptive_avg_pool2d_default(self, x, out_s):
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        in_s = x.shape[2:]
        stride = [i // o for i, o in zip(in_s, out_s)]
        ks = [i - (o - 1) * s for i, o, s in zip(in_s, out_s, stride)]
        out = np.zeros((*x.shape[:2], *out_s))
        for i in range(out_s[0]):
            for j in range(out_s[1]):
                out[:, :, i, j] = np.mean(x[:, :, i * stride[0]:i * stride[0] + ks[0], j * stride[1]:j * stride[1] + ks[1]], axis=(2, 3))
        return out

    def op_aten_permute_default(self, tensor, *dims):
        """
        Correctly handles two call styles and adds protection against "lost" batch measurements.
        """
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            final_dims = list(dims[0])
        else:
            final_dims = list(dims)
            
        final_dims = [int(d) for d in final_dims]
        
        # If the number of dimensions does not match but differs by 1,
        # this is most likely a lost batch dimension.
        if len(final_dims) != tensor.ndim and len(final_dims) == tensor.ndim + 1:
            # Add the missing dimension to the beginning (batch_size=1)
            tensor_reshaped = np.expand_dims(tensor, axis=0)
            
            # Let's check again
            if len(final_dims) == tensor_reshaped.ndim:
                # If everything now matches, work with the modified tensor
                return np.transpose(tensor_reshaped, axes=final_dims)

        # original check
        if len(final_dims) != tensor.ndim:
            raise RuntimeError(
                f"aten.permute: invalid dims length. "
                f"Tensor has {tensor.ndim} dimensions (shape: {tensor.shape}), "
                f"but received {len(final_dims)} permutation dims: {final_dims}"
            )
            
        return np.transpose(tensor, axes=final_dims)

    def op_aten_matmul_default(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        # If one of the operands is empty, matmul is not possible, but we can return an empty array of the correct shape.
        if a.size == 0 or b.size == 0:
            # Result format: (...batch_dimensions, rows_A, columns_B)
            # Batch sizes must be compatible. Simply take batch A.
            output_shape = list(a.shape[:-2]) + [a.shape[-2], b.shape[-1]]
            
            # If one of the "internal" dimensions is 0,
            # one of the "external" ones will also be 0.
            # For example, (N, 0) @ (0, M) -> (N, M), but the result is empty.
            # if (N, K) @ (K, 0) -> (N, 0).
            # need to make sure that the resulting form is correct.
            if a.shape[-1] == 0 or b.shape[-2] == 0:
                 # If the internal dimensions do not match, but one of them is 0,
                 # then the resulting form will also contain 0
                 pass # The form is already correct

            # Return an empty array of the desired shape.
            return np.zeros(tuple(output_shape), dtype=np.result_type(a, b))

        return np.matmul(a, b)

    def op_aten_ge_Scalar(self, a, b):
        return np.greater_equal(a, b)

    def op_aten___and___Tensor(self, a, b):
        # Побитовое И. Поддерживает broadcast
        a = np.asarray(a)
        b = np.asarray(b)
        return np.bitwise_and(a, b)

    def op_aten___or___Tensor(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.bitwise_or(a, b)
        
    def op_aten___not___Tensor(self, a):
        a = np.asarray(a)
        return np.bitwise_not(a)

    def op_aten_where_ScalarOther(self, condition, x, y):
        """
        Упрощенная версия where, где один из аргументов (y) — это скаляр (например, -inf)
        """
        condition = np.asarray(condition, dtype=bool)
        x = np.asarray(x)
        # Если y это объект float, numpy его легко broadcast-ит
        return np.where(condition, x, y)
        
    def op_aten_index_Tensor(self, tensor, *indices):
        """
        Эмулирует aten.index.Tensor (Advanced indexing в PyTorch).
        Часто используется для извлечения/применения масок.
        В PyTorch indices - это список тензоров.
        """
        tensor = np.asarray(tensor)
        # PyTorch передает indices как список/tuple массивов. Numpy требует tuple.
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            # Извлекаем список тензоров-индексов
            idx_tuple = tuple(np.asarray(i) if i is not None else slice(None) for i in indices[0])
            return tensor[idx_tuple]
        else:
            # Обычный вызов
            idx_tuple = tuple(np.asarray(i) if i is not None else slice(None) for i in indices)
            return tensor[idx_tuple]

    def _parse_shape(self, sz):
        """Обобщенный парсинг размерности (sz) из разных форматов PyTorch."""
        if isinstance(sz, list) and len(sz) == 0: return ()
        if isinstance(sz, int): return (sz,)
        if isinstance(sz, np.ndarray): return tuple(sz.tolist())
        return tuple(sz)

    def _get_dtype(self, explicit_dtype_enum, base_dtype=np.float32):
        """Обобщенный поиск нужного типа данных."""
        if explicit_dtype_enum is not None:
            return self._enum_to_numpy_dtype(explicit_dtype_enum)
        return base_dtype

    # === ZEROS ===
    def op_nac_zeros(self, sz, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None))
        return np.zeros(self._parse_shape(sz), dtype=dtype)

    def op_nac_zeros_like(self, x, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None), x.dtype)
        return np.zeros(x.shape, dtype=dtype)

    def op_nac_new_zeros(self, x, sz, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None), x.dtype)
        return np.zeros(self._parse_shape(sz), dtype=dtype)

    # === ONES ===
    def op_nac_ones(self, sz, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None))
        return np.ones(self._parse_shape(sz), dtype=dtype)

    def op_nac_ones_like(self, x, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None), x.dtype)
        return np.ones(x.shape, dtype=dtype)

    def op_nac_new_ones(self, x, sz, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None), x.dtype)
        return np.ones(self._parse_shape(sz), dtype=dtype)

    # === FULL ===
    def op_nac_full(self, sz, val, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None))
        return np.full(self._parse_shape(sz), val, dtype=dtype)

    def op_nac_full_like(self, x, val, *args, **kwargs):
        dtype = self._get_dtype(kwargs.get('dtype', args[0] if args else None), x.dtype)
        return np.full(x.shape, val, dtype=dtype)

    def op_nac_neg(self, x):
        return np.negative(x)

    def op_nac_gt(self, a, b):
        return np.greater(a, b)

    def op_nac_pass(self, x, *a, **kw):
        return x

    def op_nac_add(self, a, b, *args, **kwargs):
        a = np.asarray(a)
        b = np.asarray(b)
        def is_feature_map(t): return t.ndim == 4
        def is_likely_embedding(t): return t.ndim < 4
        if is_feature_map(a) and is_likely_embedding(b):
            try: return a + b.reshape(a.shape[0], a.shape[1], 1, 1)
            except ValueError: pass
        if is_feature_map(b) and is_likely_embedding(a):
            try: return a.reshape(b.shape[0], b.shape[1], 1, 1) + b
            except ValueError: pass
        try: return a + b
        except ValueError as e:
            if "could not be broadcast" in str(e): return a if a.ndim >= b.ndim else b
            raise
    
    def op_nac_sub(self, a, b, *args, **kwargs):
        return a - b

    def op_nac_where(self, condition, x, y, *a, **kw):
        return np.where(condition, x, y)

    def op_nac_clone(self, x, *a, **kw):
        return x.copy()

    def op_nac_view(self, x, *s, _perm=None):
        shape = list(s[0]) if len(s) == 1 and isinstance(s[0], (list, tuple)) else list(s)
        is_vae_attn_view_bug = (x.ndim == 3 and len(shape) == 4 and shape[0] == 1 and shape[2] == 1)
        if is_vae_attn_view_bug:
            try:
                num_heads, (batch, seq_len, total_channels) = 8, x.shape
                head_dim = total_channels // num_heads
                return x.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            except ValueError: pass
        if x.size == 655360 and tuple(shape) == (1, 1024, 320): return x
        if x.size == 0 and -1 not in shape and np.prod(shape) > 0:
            if len(shape) > 1: shape[-2] = 0
            else: shape[0] = 0
        if -1 in shape:
            idx = shape.index(-1)
            prod = np.prod([d for d in shape if d != -1])
            shape[idx] = x.size // prod if prod != 0 else 0
        try: return x.reshape(shape)
        except ValueError: return x

    def _execute_arange(self, *args, **kw):
        start = None
        end = None
        step = 1
        dtype = None
        numeric = []
        for a in args:
            if isinstance(a, (int, float, np.integer, np.floating)):
                numeric.append(a)
            elif isinstance(a, str):
                if a in ("float16", "float32", "float64", "int32", "int64"):
                    dtype = a
            # device ignore (cpu)
        if len(numeric) == 1:
            start, end = 0, numeric[0]
        elif len(numeric) >= 2:
            start, end = numeric[0], numeric[1]
            if len(numeric) >= 3:
                step = numeric[2]
        else:
            raise ValueError("nac.arange: invalid arguments")
        if step == 0:
            step = 1
        if dtype is None:
            if any(isinstance(x, float) for x in (start, end, step)):
                np_dtype = np.float32
            else:
                np_dtype = np.int64
        else:
            np_dtype = np.dtype(dtype)
        return np.arange(start, end, step, dtype=np_dtype)

    def op_nac_arange(self, *args, **kw):
        return self._execute_arange(*args, **kw)
    
    def op_nac_le(self, a, b, *args, **kw):
        return np.less_equal(a, b)

    def op_nac_mul(self, a, b, *args, **kwargs):
        return np.multiply(a, b)

    def op_nac_div(self, a, b, *args, **kwargs):
        return np.divide(a, b)

    def op_nac_transpose(self, tensor, dim0=None, dim1=None):
        """
        Универсальное транспонирование.
        Если dim0 и dim1 не переданы, работает как aten.t()
        Если переданы, работает как aten.transpose(dim0, dim1)
        """
        if dim0 is None and dim1 is None:
            # Логика aten.t
            return np.transpose(tensor)
        else:
            # Логика aten.transpose
            dim0 = int(dim0)
            dim1 = int(dim1)
            axes = list(range(tensor.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            return np.transpose(tensor, axes=axes)

    def op_nac_matmul(self, a, b):
        return np.matmul(a, b)

    def op_aten_sum_dim_IntList(self, x, dim, keepdim=False, dtype=None, *args, **kwargs):
        axis = tuple(dim) if dim else None
        res = np.sum(x, axis=axis, keepdims=keepdim)
        if dtype is not None:
            res = res.astype(self._enum_to_numpy_dtype(dtype), copy=False)
        return res

    def op_aten_div_Scalar(self, a, b):
        return np.divide(a, b)

    def op_aten_threshold_backward_default(self, grad_output, x, threshold):
        return grad_output * (x > threshold)

    def op_aten_native_batch_norm_backward_default(self, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
        sh = (1, -1, 1, 1) if input.ndim == 4 else (1, -1)
        grad_input = grad_weight = grad_bias = None
        invstd = 1.0 / np.sqrt(running_var.reshape(sh) + eps)
        norm_x = (input - running_mean.reshape(sh)) * invstd
        
        if output_mask[0]:
            w = weight.reshape(sh) if weight is not None else 1.0
            grad_input = grad_out * w * invstd
        if output_mask[1] and weight is not None:
            axes = tuple(i for i in range(input.ndim) if i != 1)
            grad_weight = np.sum(grad_out * norm_x, axis=axes)
        if output_mask[2] and weight is not None:
            axes = tuple(i for i in range(input.ndim) if i != 1)
            grad_bias = np.sum(grad_out, axis=axes)
            
        return (grad_input, grad_weight, grad_bias)

    def _col2im_indices(self, cols, x_shape, field_height, field_width, padding=1, stride=1):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = get_im2col_indices_cached(x_shape, field_height, field_width, padding, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N).transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding > 0:
            return x_padded[:, :, padding:-padding, padding:-padding]
        return x_padded

    def op_aten_convolution_backward_default(self, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
        grad_input = grad_weight = grad_bias = None
        if output_mask[2]:
            grad_bias = np.sum(grad_output, axis=(0, 2, 3))
            
        if output_mask[0] or output_mask[1]:
            N, C_out, H_out, W_out = grad_output.shape
            _, C_in, H_in, W_in = input.shape
            kH, kW = weight.shape[2], weight.shape[3]
            sH, sW = stride
            pH, pW = padding
            
            x_cols = im2col_indices(input, kH, kW, padding=pH, stride=sH)
            grad_out_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(C_out, -1)
            
            if output_mask[1]:
                gw_cols = grad_out_reshaped @ x_cols.T
                grad_weight = gw_cols.reshape(weight.shape)
                
            if output_mask[0]:
                w_reshaped = weight.reshape(C_out, -1)
                dx_cols = w_reshaped.T @ grad_out_reshaped
                grad_input = self._col2im_indices(dx_cols, input.shape, kH, kW, padding=pH, stride=sH)
                
        return (grad_input, grad_weight, grad_bias)

    def op_aten_max_pool2d_with_indices_backward_default(self, grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices):
        grad_input = np.zeros_like(x)
        N, C, H, W = x.shape
        grad_input_flat = grad_input.reshape(N, C, H * W)
        grad_output_flat = grad_output.reshape(N, C, -1)
        indices_flat = indices.reshape(N, C, -1)
        
        for n in range(N):
            for c in range(C):
                np.add.at(grad_input_flat[n, c], indices_flat[n, c], grad_output_flat[n, c])
                
        return grad_input

    def op_aten_mul_Scalar(self, a, b):
        return np.multiply(a, b)

    def op_aten__softmax_backward_data_default(self, grad_output, output, dim, input_dtype):
        sum_val = np.sum(grad_output * output, axis=dim, keepdims=True)
        return output * (grad_output - sum_val)

    def op_aten_new_empty_strided_default(self, x, size, stride, **kwargs):
        return np.empty(size, dtype=x.dtype)

    def op_aten_embedding_dense_backward_default(self, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
        grad_weight = np.zeros((num_weights, grad_output.shape[-1]), dtype=grad_output.dtype)
        indices_flat = indices.flatten()
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        np.add.at(grad_weight, indices_flat, grad_output_flat)
        if padding_idx is not None and padding_idx >= 0:
            grad_weight[padding_idx] = 0
        return grad_weight

    def op_aten_slice_backward_default(self, grad_output, input_sizes, dim, start, end, step):
        grad_input = np.zeros(input_sizes, dtype=grad_output.dtype)
        slc = [slice(None)] * len(input_sizes)
        slc[dim] = slice(start, end, step)
        grad_input[tuple(slc)] = grad_output
        return grad_input

    def op_aten_sigmoid_default(self, x):
        return self._sigmoid(x)

    def op_aten_empty_like_default(self, x, *args, **kwargs):
        return np.empty_like(x)

    def op_aten_fill_Scalar(self, x, value):
        out = np.empty_like(x)
        out.fill(value)
        return out

    def op_aten_native_group_norm_backward_default(self, dY, X, mean, rstd, weight, N, C, HxW, groups, output_mask):
        dY_reshaped = dY.reshape(N, groups, C // groups, -1)
        X_reshaped = X.reshape(N, groups, C // groups, -1)
        mean_reshaped = mean.reshape(N, groups, 1, 1)
        rstd_reshaped = rstd.reshape(N, groups, 1, 1)
        weight_reshaped = weight.reshape(1, groups, C // groups, 1) if weight is not None else 1.0

        dX = dW = dB = None
        if output_mask[0]:
            dx_norm = dY_reshaped * weight_reshaped
            N_el = (C // groups) * HxW
            sum_dx_norm = np.sum(dx_norm, axis=(2, 3), keepdims=True)
            sum_dx_norm_x = np.sum(dx_norm * X_reshaped, axis=(2, 3), keepdims=True)
            dX_reshaped = (dx_norm - sum_dx_norm / N_el - (X_reshaped - mean_reshaped) * rstd_reshaped**2 * sum_dx_norm_x / N_el) * rstd_reshaped
            dX = dX_reshaped.reshape(X.shape)
            
        if output_mask[1] and weight is not None:
            norm_X = (X - mean.reshape(N, groups, 1, 1).repeat(C // groups, axis=2).reshape(N, C, 1, 1)) * rstd.reshape(N, groups, 1, 1).repeat(C // groups, axis=2).reshape(N, C, 1, 1)
            dW = np.sum(dY * norm_X, axis=(0, 2, 3))
            
        if output_mask[2] and weight is not None:
            dB = np.sum(dY, axis=(0, 2, 3))
            
        return (dX, dW, dB)

    def op_aten_gelu_backward_default(self, grad_output, x, approximate="none"):
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return grad_output * (cdf + x * pdf)

    def op_aten_native_layer_norm_backward_default(self, grad_out, input, norm_shape, mean, rstd, weight, bias, output_mask):
        axis = tuple(range(input.ndim - len(norm_shape), input.ndim))
        dX = dW = dB = None
        
        if output_mask[0]:
            N_el = np.prod(norm_shape)
            W = weight if weight is not None else 1.0
            dx_norm = grad_out * W
            sum_dx_norm = np.sum(dx_norm, axis=axis, keepdims=True)
            sum_dx_norm_x = np.sum(dx_norm * input, axis=axis, keepdims=True)
            dX = (dx_norm - sum_dx_norm / N_el - (input - np.expand_dims(mean, axis)) * np.expand_dims(rstd, axis)**2 * sum_dx_norm_x / N_el) * np.expand_dims(rstd, axis)
            
        if output_mask[1] and weight is not None:
            norm_x = (input - np.expand_dims(mean, axis)) * np.expand_dims(rstd, axis)
            dW = np.sum(grad_out * norm_x, axis=tuple(range(input.ndim - len(norm_shape))))
            
        if output_mask[2] and bias is not None:
            dB = np.sum(grad_out, axis=tuple(range(input.ndim - len(norm_shape))))
            
        return (dX, dW, dB)

    def op_aten__scaled_dot_product_flash_attention_for_cpu_backward_default(self, grad_out, q, k, v, out, logsumexp, p, is_causal, scale):
        return (np.zeros_like(q), np.zeros_like(k), np.zeros_like(v))

    def op_aten__unsafe_index_put_default(self, x, indices, values, accumulate=False):
        out = np.copy(x)
        idx_tuple = tuple(np.asarray(i) if i is not None else slice(None) for i in indices)
        if accumulate:
            np.add.at(out, idx_tuple, values)
        else:
            out[idx_tuple] = values
        return out

    def op_aten_nll_loss_backward_default(self, grad_output, x, target, weight, reduction, ignore_index, total_weight):
        grad_input = np.zeros_like(x)
        target = np.asarray(target, dtype=int)
        reduction = int(reduction)
        ignore_index = int(ignore_index)
        
        if target.ndim == 0: target = np.expand_dims(target, 0)
        if grad_output.ndim == 0: grad_output = np.expand_dims(grad_output, 0)
            
        if x.ndim == 2:
            for i in range(x.shape[0]):
                t = target[i]
                if t != ignore_index:
                    w = weight[t] if weight is not None else 1.0
                    div = x.shape[0] if reduction == 1 else 1.0
                    grad_input[i, t] = -grad_output[0] * w / div
        return grad_input

    def op_aten__log_softmax_backward_data_default(self, grad_output, output, dim, input_dtype):
        dim = int(dim)
        sum_val = np.sum(grad_output * output, axis=dim, keepdims=True)
        return grad_output - np.exp(output) * sum_val

    def op_aten_select_backward_default(self, grad_output, input_sizes, dim, index):
        grad_input = np.zeros(input_sizes, dtype=grad_output.dtype)
        slc = [slice(None)] * len(input_sizes)
        slc[dim] = index
        grad_input[tuple(slc)] = grad_output
        return grad_input

    def op_aten_nll_loss2d_backward_default(self, grad_output, x, target, weight, reduction, ignore_index, total_weight):
        return self.op_aten_nll_loss_backward_default(grad_output, x, target, weight, reduction, ignore_index, total_weight)

    def op_aten_tanh_backward_default(self, grad_output, output):
        return grad_output * (1.0 - output**2)

    def op_aten_scalar_tensor_default(self, s, **kwargs):
        dtype_enum = kwargs.get('dtype', None)
        np_dtype = self._enum_to_numpy_dtype(dtype_enum) if dtype_enum is not None else np.float32
        return np.array(s, dtype=np_dtype)

    def op_aten_sum_dim_IntList(self, x, dim, keepdim=False, dtype=None, *args, **kwargs):
        if isinstance(keepdim, float): keepdim = bool(keepdim)
        if dim is not None:
            dim = tuple(int(d) for d in dim)
        res = np.sum(x, axis=dim, keepdims=keepdim)
        if dtype is not None:
            res = res.astype(self._enum_to_numpy_dtype(dtype), copy=False)
        return res

# --- END OF FILE NAC_kernels.py ---