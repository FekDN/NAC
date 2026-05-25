// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
#include "op_kernels.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

std::map<std::string, KernelFunc> g_kernel_string_map;
std::map<uint8_t, KernelFunc> g_op_kernels;

struct Param2D {
    int h = 1, w = 1;
    Param2D(Tensor* t) {
        if (t && t->data) {
            if (t->num_elements == 1) {
                h = w = ((int*)t->data)[0];
            } else if (t->num_elements >= 2) {
                h = ((int*)t->data)[0];
                w = ((int*)t->data)[1];
            }
        }
    }
};

Tensor* create_result_tensor(NacRuntimeContext* ctx, const std::vector<int>& shape, DataType dtype) {
    Tensor* result = ctx->tensor_pool.acquire();
    result->shape = shape;
    result->dtype = dtype;
    result->num_elements = 1;
    for (int dim : shape) {
        if (dim == 0) { result->num_elements = 0; break; }
        result->num_elements *= dim;
    }
    result->size = result->get_byte_size();
    if (result->size > 0) {
        result->data = alloc_fast(result->size);
        if (!result->data) {
            ESP_LOGE("KERNELS", "OOM for result tensor! Size=%zu", result->size);
            ctx->tensor_pool.release(result);
            return nullptr;
        }
    } else {
        result->data = nullptr;
    }
    return result;
}

Tensor* op_nac_pass(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc > 0 && args[0] && args[0]->data) {
        Tensor* x = args[0];
        Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
        if (result && result->data)
            memcpy(result->data, x->data, x->size);
        return result;
    }
    return nullptr;
}

Tensor* op_nac_zeros(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    DataType out_dt = DataType::FLOAT32;
    
    if (argc > 0 && args[0] && args[0]->data) {
        if (args[0]->dtype == DataType::INT32) {
            int32_t* dims = (int32_t*)args[0]->data;
            for (size_t i = 0; i < args[0]->num_elements; ++i) shape.push_back(dims[i]);
        } else if (args[0]->dtype == DataType::INT64) {
            int64_t* dims = (int64_t*)args[0]->data;
            for (size_t i = 0; i < args[0]->num_elements; ++i) shape.push_back((int)dims[i]);
        }
    }
    
    // In nac.zeros the second argument can be a string like ('float32') - we ignore this
    // since the C++ runtime currently only uses float32 and int32
    if (shape.empty()) shape.push_back(1);
    
    Tensor* result = create_result_tensor(ctx, shape, out_dt);
    if (result && result->data) {
        memset(result->data, 0, result->size);
    }
    return result;
}

Tensor* op_nac_neg(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;

    long long N = (long long)x->num_elements;
    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data;
        float* dst = (float*)result->data;
        #pragma omp parallel for simd
        for (long long i = 0; i < N; ++i) dst[i] = -src[i];
    } else if (x->dtype == DataType::INT32) {
        int* src = (int*)x->data;
        int* dst = (int*)result->data;
        #pragma omp parallel for simd
        for (long long i = 0; i < N; ++i) dst[i] = -src[i];
    }
    return result;
}

template <typename Op>
Tensor* op_nac_comparison(NacRuntimeContext* ctx, Tensor** args, size_t argc, Op op) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* a = args[0]; Tensor* b = args[1];

    std::vector<int> out_shape;
    int n1 = a->shape.size(), n2 = b->shape.size();
    int m = std::max(n1, n2);
    for (int i = 0; i < m; ++i) {
        int d1 = (i < m - n1) ? 1 : a->shape[i - (m - n1)];
        int d2 = (i < m - n2) ? 1 : b->shape[i - (m - n2)];
        out_shape.push_back((d1 == 1) ? d2 : ((d2 == 1) ? d1 : std::max(d1, d2)));
    }

    Tensor* result = create_result_tensor(ctx, out_shape, DataType::INT32);
    if (!result || !result->data || a->num_elements == 0 || b->num_elements == 0) {
        if (result && result->data) memset(result->data, 0, result->size);
        return result;
    }

    std::vector<int> stride_out(m, 1), stride_a(m, 0), stride_b(m, 0);
    int sa = 1, sb = 1;
    for (int i = m - 1; i >= 0; --i) {
        if (i < m - 1) stride_out[i] = stride_out[i+1] * out_shape[i+1];
        int d1 = (i < m - n1) ? 1 : a->shape[i - (m - n1)];
        int d2 = (i < m - n2) ? 1 : b->shape[i - (m - n2)];
        stride_a[i] = (d1 == 1) ? 0 : sa;
        stride_b[i] = (d2 == 1) ? 0 : sb;
        if (d1 > 1) sa *= d1;
        if (d2 > 1) sb *= d2;
    }

    long long N = (long long)result->num_elements;
    
    // Parallel broadcasting calculation
    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        size_t idx_a = 0, idx_b = 0, temp = i;
        for (int d = 0; d < m; ++d) {
            int coord = temp / stride_out[d];
            temp %= stride_out[d];
            idx_a += coord * stride_a[d];
            idx_b += coord * stride_b[d];
        }
        float av = 0.f, bv = 0.f;
        if (a->dtype == DataType::FLOAT32) av = ((float*)a->data)[idx_a];
        else if (a->dtype == DataType::INT32) av = (float)((int*)a->data)[idx_a];
        else av = (float)((uint8_t*)a->data)[idx_a];

        if (b->dtype == DataType::FLOAT32) bv = ((float*)b->data)[idx_b];
        else if (b->dtype == DataType::INT32) bv = (float)((int*)b->data)[idx_b];
        else bv = (float)((uint8_t*)b->data)[idx_b];

        ((int*)result->data)[i] = op(av, bv);
    }
    return result;
}

Tensor* op_nac_gt(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_comparison(ctx, args, argc, [](float a, float b) { return a > b ? 1 : 0; });
}

Tensor* op_nac_le(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_comparison(ctx, args, argc, [](float a, float b) { return a <= b ? 1 : 0; });
}

Tensor* op_aten_lt_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_comparison(ctx, args, argc, [](float a, float b) { return a < b ? 1 : 0; });
}

Tensor* op_nac_view(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* input = args[0];
    std::vector<int> new_shape;

    for (size_t i = 1; i < argc; ++i) {
        if (!args[i] || !args[i]->data) continue;
        if (args[i]->num_elements > 1) {
            if (args[i]->dtype == DataType::INT32) {
                const int32_t* dims = static_cast<const int32_t*>(args[i]->data);
                for (size_t d = 0; d < args[i]->num_elements; ++d) new_shape.push_back((int)dims[d]);
            } else if (args[i]->dtype == DataType::INT64) {
                const int64_t* dims = static_cast<const int64_t*>(args[i]->data);
                for (size_t d = 0; d < args[i]->num_elements; ++d) new_shape.push_back((int)dims[d]);
            }
        } else {
            if (args[i]->dtype == DataType::INT32) new_shape.push_back(*(int32_t*)args[i]->data);
            else if (args[i]->dtype == DataType::INT64) new_shape.push_back((int)*(int64_t*)args[i]->data);
        }
    }

    if (new_shape.empty()) new_shape.push_back((int)input->num_elements);

    // PATCH: VAE (Very Aimable Attention) Defense
    if (input->shape.size() == 3 && new_shape.size() == 4 && new_shape[0] == 1 && new_shape[2] == 1) {
        int num_heads = 8;
        int batch = input->shape[0];
        int seq_len = input->shape[1];
        int total_channels = input->shape[2];
        int head_dim = total_channels / num_heads;
        
        if (total_channels % num_heads == 0) {
            Tensor* result = create_result_tensor(ctx, {batch, num_heads, seq_len, head_dim}, input->dtype);
            if (result && result->data) {
                size_t elem_size = input->get_element_byte_size();
                const uint8_t* src = (const uint8_t*)input->data;
                uint8_t* dst = (uint8_t*)result->data;
                
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch; ++b) {
                    for (int s = 0; s < seq_len; ++s) {
                        for (int h = 0; h < num_heads; ++h) {
                            for (int d = 0; d < head_dim; ++d) {
                                size_t src_idx = b * (seq_len * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim + d;
                                size_t dst_idx = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim + d;
                                memcpy(dst + dst_idx * elem_size, src + src_idx * elem_size, elem_size);
                            }
                        }
                    }
                }
            }
            return result;
        }
    }

    long long num_elements = (long long)input->num_elements;
    long long product = 1;
    int minus_one_idx = -1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) minus_one_idx = i;
        else if (new_shape[i] > 0) product *= new_shape[i];
    }
    
    if (minus_one_idx != -1) {
        if (product > 0) new_shape[minus_one_idx] = (int)(num_elements / product);
        else new_shape[minus_one_idx] = 0;
    }

    Tensor* result = create_result_tensor(ctx, new_shape, input->dtype);
    if (result && (long long)result->num_elements == num_elements) {
        memcpy(result->data, input->data, result->size);
    } else if (result) {
        ctx->tensor_pool.release(result);
        return nullptr;
    }
    return result;
}

Tensor* op_nac_clone(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* result = create_result_tensor(ctx, args[0]->shape, args[0]->dtype);
    if (result && args[0]->size > 0 && args[0]->data)
        memcpy(result->data, args[0]->data, args[0]->size);
    return result;
}

Tensor* op_nac_where(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* cond = args[0]; Tensor* x = args[1]; Tensor* y = args[2];
    
    std::vector<int> out_shape;
    int nc = cond->shape.size(), nx = x->shape.size(), ny = y->shape.size();
    int m = std::max({nc, nx, ny});
    for (int i = 0; i < m; ++i) {
        int dc = (i < m - nc) ? 1 : cond->shape[i - (m - nc)];
        int dx = (i < m - nx) ? 1 : x->shape[i - (m - nx)];
        int dy = (i < m - ny) ? 1 : y->shape[i - (m - ny)];
        out_shape.push_back(std::max({dc, dx, dy}));
    }
    
    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;

    // --- SAFE PRECOMPUTED STEPS (AS IN MATH_BINARY) ---
    std::vector<int> stride_out(m, 1), stride_c(m, 0), stride_x(m, 0), stride_y(m, 0);
    int sc = 1, sx = 1, sy = 1;
    for (int i = m - 1; i >= 0; --i) {
        if (i < m - 1) stride_out[i] = stride_out[i+1] * out_shape[i+1];
        int dc = (i < m - nc) ? 1 : cond->shape[i - (m - nc)];
        int dx = (i < m - nx) ? 1 : x->shape[i - (m - nx)];
        int dy = (i < m - ny) ? 1 : y->shape[i - (m - ny)];
        stride_c[i] = (dc == 1) ? 0 : sc;
        stride_x[i] = (dx == 1) ? 0 : sx;
        stride_y[i] = (dy == 1) ? 0 : sy;
        if (dc > 1) sc *= dc;
        if (dx > 1) sx *= dx;
        if (dy > 1) sy *= dy;
    }
    
    for (size_t i = 0; i < out->num_elements; ++i) {
        size_t idx_c = 0, idx_x = 0, idx_y = 0, temp = i;
        for (int d = 0; d < m; ++d) {
            int coord = temp / stride_out[d];
            temp %= stride_out[d];
            idx_c += coord * stride_c[d];
            idx_x += coord * stride_x[d];
            idx_y += coord * stride_y[d];
        }
        // -------------------------------------------------------------------------
        
        bool c_val = false;
        if (cond->dtype == DataType::FLOAT32) c_val = ((float*)cond->data)[idx_c] != 0.0f;
        else if (cond->dtype == DataType::INT32) c_val = ((int32_t*)cond->data)[idx_c] != 0;
        else if (cond->dtype == DataType::BOOL || cond->dtype == DataType::INT8 || cond->dtype == DataType::UINT8) 
            c_val = ((uint8_t*)cond->data)[idx_c] != 0;
            
        float x_val = (x->dtype == DataType::FLOAT32) ? ((float*)x->data)[idx_x] : (float)((int32_t*)x->data)[idx_x];
        float y_val = (y->dtype == DataType::FLOAT32) ? ((float*)y->data)[idx_y] : (float)((int32_t*)y->data)[idx_y];
        ((float*)out->data)[i] = c_val ? x_val : y_val;
    }
    return out;
}

Tensor* op_nac_arange(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc == 0) return nullptr;
    std::vector<float> numeric_args;
    for (size_t i = 0; i < argc; ++i) {
        Tensor* arg = args[i];
        // If encounter an "empty" tensor (like a string), it means run out of positional math arguments.
        if (!arg || !arg->data || arg->num_elements == 0) {
            break; 
        }
        if (arg->dtype == DataType::FLOAT32 || arg->dtype == DataType::FLOAT64)
            numeric_args.push_back(arg->dtype == DataType::FLOAT32 ? ((float*)arg->data)[0] : (float)((double*)arg->data)[0]);
        else if (arg->dtype == DataType::INT32 || arg->dtype == DataType::INT64)
            numeric_args.push_back(arg->dtype == DataType::INT32 ? (float)((int32_t*)arg->data)[0] : (float)((int64_t*)arg->data)[0]);
        else if (arg->dtype == DataType::BOOL)
            break;
    }
    
    if (numeric_args.empty()) return nullptr;

    float start = 0.0f, end = 0.0f, step = 1.0f;
    if (numeric_args.size() == 1) { end = numeric_args[0]; }
    else if (numeric_args.size() == 2) {
        start = numeric_args[0]; end = numeric_args[1];
    } else if (numeric_args.size() >= 3) {
        start = numeric_args[0]; end = numeric_args[1]; step = numeric_args[2];
    }
    if (step == 0.0f) step = 1.0f;

    int length = (int)ceil((end - start) / step);
    if (length < 0) length = 0;

    DataType out_dtype = DataType::INT32;
    if (floor(start) != start || floor(end) != end || floor(step) != step) out_dtype = DataType::FLOAT32;

    Tensor* out = create_result_tensor(ctx, {length}, out_dtype);
    if (out && out->data) {
        if (out_dtype == DataType::FLOAT32) {
            float* dst = (float*)out->data;
            for (int i = 0; i < length; ++i) dst[i] = start + step * i;
        } else {
            int32_t* dst = (int32_t*)out->data;
            for (int i = 0; i < length; ++i) dst[i] = (int32_t)(start + step * i);
        }
    }
    return out;
}

Tensor* op_aten_linear_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x = args[0]; 
    Tensor* w = args[1]; 
    Tensor* b = (argc > 2) ? args[2] : nullptr;

    if (x->dtype != DataType::FLOAT32 || w->dtype != DataType::FLOAT32) return nullptr;
    
    int in_features = w->shape.back();
    int out_features = w->shape.front();
    if (x->shape.back() != in_features) return nullptr;
    
    std::vector<int> out_shape = x->shape;
    out_shape.back() = out_features;
    
    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    const float* src_x = (const float*)x->data;
    const float* src_w = (const float*)w->data;
    const float* src_b = b && b->data ? (const float*)b->data : nullptr;
    float* dst         = (float*)out->data;
    
    int outer_dim = (int)(x->num_elements / in_features);

    // Parallelizing computations across batch rows
    #pragma omp parallel for
    for (int i = 0; i < outer_dim; ++i) {
        const float* x_row = src_x + i * in_features;
        float* out_row = dst + i * out_features;
        
        for (int j = 0; j < out_features; ++j) {
            const float* w_row = src_w + j * in_features;
            
            // Initialize the accumulator with the bias value (get rid of the extra cycle)
            float sum = src_b ? src_b[j] : 0.0f;
            
            // Vectorizing the Dot Product
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < in_features; ++k) {
                sum += x_row[k] * w_row[k];
            }
            
            out_row[j] = sum;
        }
    }
    return out;
}

Tensor* op_aten_matmul_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* a = args[0]; Tensor* b = args[1];
    if (a->dtype != DataType::FLOAT32 || b->dtype != DataType::FLOAT32) return nullptr;
    
    std::vector<int> a_shape = a->shape; if (a_shape.size() < 2) a_shape.insert(a_shape.begin(), 1);
    std::vector<int> b_shape = b->shape; if (b_shape.size() < 2) b_shape.push_back(1);

    if (a->num_elements == 0 || b->num_elements == 0) {
        std::vector<int> out_shape = (a_shape.size() >= b_shape.size()) ? a_shape : b_shape;
        out_shape.pop_back(); out_shape.pop_back();
        out_shape.push_back(a_shape[a_shape.size() - 2]);
        out_shape.push_back(b_shape.back());
        Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
        if (out && out->data) memset(out->data, 0, out->size);
        return out;
    }

    int m = a_shape[a_shape.size() - 2];
    int k = a_shape.back();
    int n = b_shape.back();
    if (b_shape[b_shape.size() - 2] != k) return nullptr;
    
    std::vector<int> out_shape = a->shape;
    if (out_shape.size() < 2) out_shape.push_back(n);
    else out_shape.back() = n; 
    
    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    float* src_a = (float*)a->data;
    float* src_b = (float*)b->data;
    float* dst   = (float*)out->data;
    int batches = a->num_elements / (m * k);
    int b_batch_stride = (b->num_elements > (size_t)(k * n)) ? (k * n) : 0;
    
    memset(dst, 0, out->size);
    int total_tasks = batches * m;

    #pragma omp parallel for
    for (int task = 0; task < total_tasks; ++task) {
        int b_idx = task / m;
        int i = task % m;
        
        float* a_row = src_a + b_idx * (m * k) + i * k;
        float* b_batch = src_b + b_idx * b_batch_stride;
        float* out_row = dst + b_idx * (m * n) + i * n;
        
        for (int l = 0; l < k; ++l) {
            float a_val = a_row[l];
            if (a_val == 0.0f) continue;
            
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                out_row[j] += a_val * b_batch[l * n + j];
            }
        }
    }
    return out;
}

Tensor* op_aten_addmm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* bias = args[0];
    Tensor* mat1 = args[1];
    Tensor* mat2 = args[2];
    if (mat1->dtype != DataType::FLOAT32 || mat2->dtype != DataType::FLOAT32) return nullptr;
    int M = 1, K = 1, N = 1;
    if (mat1->shape.size() >= 2) {
        M = mat1->shape[mat1->shape.size() - 2];
        K = mat1->shape.back();
    }
    if (mat2->shape.size() >= 2) {
        N = mat2->shape.back();
    }
    float beta = 1.0f, alpha = 1.0f;
    if (argc > 3 && args[3] && args[3]->data) {
        if (args[3]->dtype == DataType::FLOAT32) beta = ((float*)args[3]->data)[0];
        else if (args[3]->dtype == DataType::INT32) beta = (float)((int32_t*)args[3]->data)[0];
    }
    if (argc > 4 && args[4] && args[4]->data) {
        if (args[4]->dtype == DataType::FLOAT32) alpha = ((float*)args[4]->data)[0];
        else if (args[4]->dtype == DataType::INT32) alpha = (float)((int32_t*)args[4]->data)[0];
    }
    std::vector<int> out_shape = mat1->shape;
    out_shape.back() = N;
    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    const float* a_ptr = (const float*)mat1->data;
    const float* b_ptr = (const float*)mat2->data;
    const float* bias_ptr = bias->data ? (const float*)bias->data : nullptr;
    float* out_ptr = (float*)out->data;
    bool bias_is_1d = (bias->shape.size() == 1);
    bool bias_is_scalar = (bias->num_elements == 1);
    int outer = (int)(out->num_elements / (M * N));
    int total_rows = outer * M;
    // Parallelize all the rows of all the batches to maximize core load.
    #pragma omp parallel for
    for (int idx = 0; idx < total_rows; ++idx) {
        int o = idx / M;
        int i = idx % M;
        const float* a_row = a_ptr + o * (M * K) + i * K;
        float* out_row = out_ptr + o * (M * N) + i * N;
        // 1. Initializing the result string with bias values
        if (bias_ptr) {
            if (bias_is_1d && bias->num_elements == (size_t)N) {
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    out_row[j] = beta * bias_ptr[j];
                }
            } else if (bias_is_scalar) {
                float b_val = beta * bias_ptr[0];
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    out_row[j] = b_val;
                }
            } else {
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    out_row[j] = 0.0f;
                }
            }
        } else {
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                out_row[j] = 0.0f;
            }
        }
        // 2. Accumulating the result of matrix multiplication (Order i, k, j for perfect caching)
        for (int k = 0; k < K; ++k) {
            float a_val = alpha * a_row[k];
            const float* b_row = b_ptr + k * N;
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                out_row[j] += a_val * b_row[j];
            }
        }
    }
    return out;
}

Tensor* op_aten_layer_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 4 || !args[0] || !args[2] || !args[3]) return nullptr;
    Tensor* in = args[0]; Tensor* w = args[2]; Tensor* b = args[3];
    
    float eps = 1e-5f;
    if (argc > 4 && args[4] && args[4]->dtype == DataType::FLOAT32) eps = *(float*)args[4]->data;

    Tensor* out = create_result_tensor(ctx, in->shape, DataType::FLOAT32);
    if (!out || !out->data) return out;

    size_t inner_dim = w->num_elements;
    if (inner_dim == 0) inner_dim = 1;
    long long outer_dim = (long long)(in->num_elements / inner_dim);

    float* src = (float*)in->data; float* dst = (float*)out->data;
    float* w_ptr = (float*)w->data; float* b_ptr = (float*)b->data;

    #pragma omp parallel for
    for (long long i = 0; i < outer_dim; ++i) {
        float* src_row = src + i * inner_dim;
        float* dst_row = dst + i * inner_dim;
        
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < inner_dim; ++j) sum += src_row[j];
        float mean = (float)(sum / inner_dim);
        
        double var_sum = 0.0;
        #pragma omp simd reduction(+:var_sum)
        for (size_t j = 0; j < inner_dim; ++j) {
            float diff = src_row[j] - mean;
            var_sum += (double)diff * (double)diff;
        }
        float inv_std = 1.0f / sqrtf((float)(var_sum / inner_dim) + eps);
        
        #pragma omp simd
        for (size_t j = 0; j < inner_dim; ++j) {
            dst_row[j] = (src_row[j] - mean) * inv_std * w_ptr[j] + b_ptr[j];
        }
    }
    return out;
}

Tensor* op_aten_softmax_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    int dim = *(static_cast<int*>(args[1]->data));
    if (dim < 0) dim += x->shape.size();
    if (dim != x->shape.size() - 1) return nullptr; 

    int last_dim_size = x->shape.back();
    if (last_dim_size == 0) return op_nac_clone(ctx, ins, args, argc);

    long long outer_dims_size = (long long)(x->num_elements / last_dim_size);
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result || !result->data) return result;

    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    
    #pragma omp parallel for
    for (long long i = 0; i < outer_dims_size; ++i) {
        float* row_start = x_data + i * last_dim_size;
        float* res_row_start = res_data + i * last_dim_size;
        
        float max_val = row_start[0];
        for (int j = 1; j < last_dim_size; ++j) {
            if (row_start[j] > max_val) max_val = row_start[j];
        }
        
        float sum_exp = 0.0f;
        #pragma omp simd reduction(+:sum_exp)
        for (int j = 0; j < last_dim_size; ++j) {
            float val = expf(row_start[j] - max_val);
            res_row_start[j] = val;
            sum_exp += val;
        }
        
        if (sum_exp != 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            #pragma omp simd
            for (int j = 0; j < last_dim_size; ++j) res_row_start[j] *= inv_sum;
        } else {
            for (int j = 0; j < last_dim_size; ++j) res_row_start[j] = 0.0f;
        }
    }
    return result;
}

template<typename Op>
Tensor* broadcast_binary_op(NacRuntimeContext* ctx, Tensor* a, Tensor* b, Op op) {
    if (!a || !b) return nullptr;
    
    std::vector<int> out_shape;
    int n1 = a->shape.size(), n2 = b->shape.size();
    int m = std::max(n1, n2);
    for (int i = 0; i < m; ++i) {
        int d1 = (i < m - n1) ? 1 : a->shape[i - (m - n1)];
        int d2 = (i < m - n2) ? 1 : b->shape[i - (m - n2)];
        out_shape.push_back((d1 == 1) ? d2 : ((d2 == 1) ? d1 : std::max(d1, d2)));
    }

    DataType dt = (a->dtype == DataType::INT32 && b->dtype == DataType::INT32) ? DataType::INT32 : DataType::FLOAT32;
    Tensor* out = create_result_tensor(ctx, out_shape, dt);
    if (!out || !out->data || a->num_elements == 0 || b->num_elements == 0) {
        if (out && out->data) memset(out->data, 0, out->size);
        return out;
    }
    
    std::vector<int> stride_out(m, 1), stride_a(m, 0), stride_b(m, 0);
    int sa = 1, sb = 1;
    for (int i = m - 1; i >= 0; --i) {
        if (i < m - 1) stride_out[i] = stride_out[i+1] * out_shape[i+1];
        int d1 = (i < m - n1) ? 1 : a->shape[i - (m - n1)];
        int d2 = (i < m - n2) ? 1 : b->shape[i - (m - n2)];
        stride_a[i] = (d1 == 1) ? 0 : sa; 
        stride_b[i] = (d2 == 1) ? 0 : sb;
        if (d1 > 1) sa *= d1;
        if (d2 > 1) sb *= d2;
    }
    
    long long N = (long long)out->num_elements;

    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        size_t idx_a = 0, idx_b = 0, temp = i;
        for (int d = 0; d < m; ++d) {
            int coord = temp / stride_out[d];
            temp %= stride_out[d];
            idx_a += coord * stride_a[d];
            idx_b += coord * stride_b[d];
        }
        
        float va = (a->dtype == DataType::FLOAT32) ? ((float*)a->data)[idx_a] : 
                   (a->dtype == DataType::INT32) ? (float)((int32_t*)a->data)[idx_a] : (float)((uint8_t*)a->data)[idx_a];

        float vb = (b->dtype == DataType::FLOAT32) ? ((float*)b->data)[idx_b] : 
                   (b->dtype == DataType::INT32) ? (float)((int32_t*)b->data)[idx_b] : (float)((uint8_t*)b->data)[idx_b];
        
        if (out->dtype == DataType::FLOAT32) ((float*)out->data)[i] = op(va, vb);
        else ((int32_t*)out->data)[i] = (int32_t)op(va, vb);
    }
    return out;
}

Tensor* op_nac_add(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return broadcast_binary_op(ctx, args[0], args[1], [](float a, float b){ return a + b; });
}
Tensor* op_nac_sub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return broadcast_binary_op(ctx, args[0], args[1], [](float a, float b){ return a - b; });
}
Tensor* op_nac_mul(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return broadcast_binary_op(ctx, args[0], args[1], [](float a, float b){ return a * b; });
}
Tensor* op_nac_div(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return broadcast_binary_op(ctx, args[0], args[1], [](float a, float b){ return b != 0.0f ? a / b : 0.0f; });
}

Tensor* op_aten_mul_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_mul(ctx, ins, args, argc);
}
Tensor* op_aten_div_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_div(ctx, ins, args, argc);
}

Tensor* op_aten_relu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result || !result->data) return result;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for(size_t i = 0; i < x->num_elements; ++i) res_data[i] = std::max(0.0f, x_data[i]);
    return result;
}

Tensor* op_aten_silu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result || !result->data) return result;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for (size_t i = 0; i < x->num_elements; ++i) {
        float val = x_data[i];
        res_data[i] = val / (1.0f + expf(-val));
    }
    return result;
}

Tensor* op_aten_gelu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (!argc || !args[0] || !args[0]->data || args[0]->dtype != DataType::FLOAT32) return nullptr;
    Tensor* in = args[0];
    Tensor* out = create_result_tensor(ctx, in->shape, in->dtype);
    if (!out || !out->data) return out;
    
    float* src = (float*)in->data;
    float* dst = (float*)out->data;
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (size_t i = 0; i < out->num_elements; ++i) {
        float x = src[i];
        float x3 = x * x * x;
        dst[i] = 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
    }
    return out;
}

Tensor* op_aten_hardsigmoid_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* x = args[0];
    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data; float* dst = (float*)out->data;
        for (size_t i = 0; i < x->num_elements; ++i) {
            float val = src[i] + 3.0f;
            val = val < 0.0f ? 0.0f : (val > 6.0f ? 6.0f : val);
            dst[i] = val / 6.0f;
        }
    }
    return out;
}

Tensor* op_aten_hardswish_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* x = args[0];
    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data; float* dst = (float*)out->data;
        for (size_t i = 0; i < x->num_elements; ++i) {
            float val = src[i] + 3.0f;
            val = val < 0.0f ? 0.0f : (val > 6.0f ? 6.0f : val);
            dst[i] = src[i] * (val / 6.0f);
        }
    }
    return out;
}

Tensor* op_aten_sigmoid_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* x = args[0];
    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data; float* dst = (float*)out->data;
        for (size_t i = 0; i < x->num_elements; ++i) {
            dst[i] = 1.0f / (1.0f + expf(-src[i]));
        }
    }
    return out;
}

Tensor* op_aten_rsqrt_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result || !result->data) return result;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    const float kEps = 1e-12f;
    for (size_t i = 0; i < x->num_elements; ++i) {
        float v = x_data[i] < kEps ? kEps : x_data[i];
        res_data[i] = 1.0f / sqrtf(v);
    }
    return result;
}

Tensor* op_aten_transpose_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[0]->data || !args[1] || !args[1]->data || !args[2] || !args[2]->data) return nullptr;

    Tensor* x = args[0];
    int dim0 = *(int*)args[1]->data;
    int dim1 = *(int*)args[2]->data;

    int ndim = x->shape.size();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 >= ndim || dim1 >= ndim) return nullptr;

    std::vector<int> out_shape = x->shape;
    std::swap(out_shape[dim0], out_shape[dim1]);

    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result || !result->data) return result;

    std::vector<int> in_stride(ndim), out_stride(ndim);
    in_stride[ndim-1] = 1;
    out_stride[ndim-1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        in_stride[i]  = in_stride[i+1]  * x->shape[i+1];
        out_stride[i] = out_stride[i+1] * out_shape[i+1];
    }

    float* src = (float*)x->data;
    float* dst = (float*)result->data;
    std::vector<int> idx(ndim);

    for (int i = 0; i < x->num_elements; ++i) {
        int tmp = i;
        for (int d = 0; d < ndim; ++d) {
            idx[d] = tmp / in_stride[d];
            tmp %= in_stride[d];
        }
        std::swap(idx[dim0], idx[dim1]);
        int out_off = 0;
        for (int d = 0; d < ndim; ++d) out_off += idx[d] * out_stride[d];
        dst[out_off] = src[i];
    }
    return result;
}

Tensor* op_nac_transpose(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc >= 3 && args[1] && args[2]) return op_aten_transpose_int(ctx, ins, args, argc);
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    std::vector<int> out_shape = x->shape;
    std::reverse(out_shape.begin(), out_shape.end());
    Tensor* out = create_result_tensor(ctx, out_shape, x->dtype);
    if (!out || !out->data) return out;
    
    if (x->shape.size() == 2) {
        int R = x->shape[0], C = x->shape[1];
        if (x->dtype == DataType::FLOAT32) {
            float* src = (float*)x->data; float* dst = (float*)out->data;
            for(int r=0; r<R; ++r) for(int c=0; c<C; ++c) dst[c*R + r] = src[r*C + c];
        } else if (x->dtype == DataType::INT32) {
            int32_t* src = (int32_t*)x->data; int32_t* dst = (int32_t*)out->data;
            for(int r=0; r<R; ++r) for(int c=0; c<C; ++c) dst[c*R + r] = src[r*C + c];
        }
    } else {
        memcpy(out->data, x->data, out->size);
    }
    return out;
}

Tensor* op_aten_unsqueeze_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    int dim = (args[1]->dtype == DataType::INT64) ? (int)((int64_t*)args[1]->data)[0] : ((int32_t*)args[1]->data)[0];
    int ndim = x->shape.size();
    if (dim < 0) dim += ndim + 1;
    std::vector<int> out_shape = x->shape;
    if (dim < 0) dim = 0;
    if (dim > ndim) dim = ndim;
    out_shape.insert(out_shape.begin() + dim, 1);
    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result || !result->data) return result;
    memcpy(result->data, x->data, x->get_byte_size());
    return result;
}

Tensor* op_aten_sym_size_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    int dim = *(int*)args[1]->data;
    if (dim < 0) dim += x->shape.size();
    if (dim >= (int)x->shape.size()) return nullptr;

    Tensor* result = create_result_tensor(ctx, {1}, DataType::INT32);
    if (!result || !result->data) return result;
    *(int*)result->data = x->shape[dim];
    return result;
}

Tensor* op_aten_ne_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    Tensor* sc = args[1];

    Tensor* result = create_result_tensor(ctx, x->shape, DataType::INT32);
    if (!result || !result->data) return result;

    int32_t* dst = (int32_t*)result->data;
    size_t N = x->num_elements;

    if (x->dtype == DataType::INT32) {
        int32_t scalar = (sc->dtype == DataType::INT32) ? *(int32_t*)sc->data : (int32_t)(*(float*)sc->data);
        const int32_t* src = (const int32_t*)x->data;
        for (size_t i = 0; i < N; ++i) dst[i] = (src[i] != scalar) ? 1 : 0;
    } else {
        float scalar = (sc->dtype == DataType::FLOAT32) ? *(float*)sc->data : (float)(*(int32_t*)sc->data);
        const float* src = (const float*)x->data;
        for (size_t i = 0; i < N; ++i) dst[i] = (src[i] != scalar) ? 1 : 0;
    }
    return result;
}

Tensor* op_aten__to_copy_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* in = args[0];
    Tensor* out = create_result_tensor(ctx, in->shape, in->dtype);
    if (!out || !out->data) return out;
    memcpy(out->data, in->data, out->size);
    return out;
}

Tensor* op_aten_cumsum_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x = args[0];
    int dim = *(int32_t*)args[1]->data;
    if (dim < 0) dim += (int)x->shape.size();

    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result || !result->data) return result;

    int stride = 1;
    for (int i = dim + 1; i < (int)x->shape.size(); ++i) stride *= x->shape[i];
    int block = x->shape[dim] * stride;
    int outer  = (int)(x->num_elements / block);

    if (x->dtype == DataType::INT32) {
        const int32_t* src = (const int32_t*)x->data;
        int32_t*       dst = (int32_t*)      result->data;
        for (int o = 0; o < outer; ++o) {
            for (int s = 0; s < stride; ++s) {
                int32_t acc = 0;
                for (int i = 0; i < x->shape[dim]; ++i) {
                    int idx = o * block + i * stride + s;
                    acc += src[idx];
                    dst[idx] = acc;
                }
            }
        }
    } else {
        const float* src = (const float*)x->data;
        float*       dst = (float*)      result->data;
        for (int o = 0; o < outer; ++o) {
            for (int s = 0; s < stride; ++s) {
                float acc = 0.f;
                for (int i = 0; i < x->shape[dim]; ++i) {
                    int idx = o * block + i * stride + s;
                    acc += src[idx];
                    dst[idx] = acc;
                }
            }
        }
    }
    return result;
}

Tensor* op_aten_type_as_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x   = args[0];
    Tensor* ref = args[1];

    Tensor* result = create_result_tensor(ctx, x->shape, ref->dtype);
    if (!result || !result->data) return result;

    size_t N = x->num_elements;
    if (x->dtype == ref->dtype) {
        memcpy(result->data, x->data, x->size);
    } else if (x->dtype == DataType::INT32 && ref->dtype == DataType::FLOAT32) {
        const int32_t* src = (const int32_t*)x->data;
        float*         dst = (float*)result->data;
        for (size_t i = 0; i < N; ++i) dst[i] = (float)src[i];
    } else if (x->dtype == DataType::FLOAT32 && ref->dtype == DataType::INT32) {
        const float* src = (const float*)x->data;
        int32_t*     dst = (int32_t*)result->data;
        for (size_t i = 0; i < N; ++i) dst[i] = (int32_t)src[i];
    } else {
        memcpy(result->data, x->data, std::min(x->size, result->size));
    }
    return result;
}

Tensor* op_aten_zeros_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    for (size_t i = 0; i < argc; ++i) {
        if (!args[i] || !args[i]->data) continue;
        if (args[i]->num_elements != 1) continue;
        if (args[i]->dtype != DataType::INT32) continue;
        int32_t v = *(int32_t*)args[i]->data;
        if (v > 0) shape.push_back(v);
    }
    if (shape.empty()) return nullptr;

    Tensor* result = create_result_tensor(ctx, shape, DataType::FLOAT32);
    if (!result || !result->data) return result;

    memset(result->data, 0, result->size);
    return result;
}

Tensor* op_nac_ones(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    DataType out_dt = DataType::FLOAT32;
    
    if (argc > 0 && args[0] && args[0]->data) {
        if (args[0]->dtype == DataType::INT32) {
            int32_t* dims = (int32_t*)args[0]->data;
            for (size_t i = 0; i < args[0]->num_elements; ++i) shape.push_back(dims[i]);
        } else if (args[0]->dtype == DataType::INT64) {
            int64_t* dims = (int64_t*)args[0]->data;
            for (size_t i = 0; i < args[0]->num_elements; ++i) shape.push_back((int)dims[i]);
        }
    }
    
    if (shape.empty()) shape.push_back(1);
    
    Tensor* result = create_result_tensor(ctx, shape, out_dt);
    if (result && result->data) {
        float* p = (float*)result->data;
        for (size_t i = 0; i < result->num_elements; ++i) p[i] = 1.0f;
    }
    return result;
}

Tensor* op_nac_new_zeros(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    // The data type is inherited from the first argument x
    DataType out_dt = (argc > 0 && args[0]) ? args[0]->dtype : DataType::FLOAT32;
    
    //The form is in the second argument
    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::INT32) {
            int32_t* dims = (int32_t*)args[1]->data;
            for (size_t i = 0; i < args[1]->num_elements; ++i) shape.push_back(dims[i]);
        } else if (args[1]->dtype == DataType::INT64) {
            int64_t* dims = (int64_t*)args[1]->data;
            for (size_t i = 0; i < args[1]->num_elements; ++i) shape.push_back((int)dims[i]);
        }
    }
    
    if (shape.empty()) shape.push_back(1);
    
    Tensor* result = create_result_tensor(ctx, shape, out_dt);
    if (result && result->data) {
        memset(result->data, 0, result->size);
    }
    return result;
}

Tensor* op_nac_new_ones(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    // The data type is inherited from the first argument x
    DataType out_dt = (argc > 0 && args[0]) ? args[0]->dtype : DataType::FLOAT32;
    
    // The form is in the second argument
    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::INT32) {
            int32_t* dims = (int32_t*)args[1]->data;
            for (size_t i = 0; i < args[1]->num_elements; ++i) shape.push_back(dims[i]);
        } else if (args[1]->dtype == DataType::INT64) {
            int64_t* dims = (int64_t*)args[1]->data;
            for (size_t i = 0; i < args[1]->num_elements; ++i) shape.push_back((int)dims[i]);
        }
    }
    
    if (shape.empty()) shape.push_back(1);
    
    Tensor* result = create_result_tensor(ctx, shape, out_dt);
    if (result && result->data) {
        if (out_dt == DataType::FLOAT32) {
            float* p = (float*)result->data;
            for (size_t i = 0; i < result->num_elements; ++i) p[i] = 1.0f;
        } else if (out_dt == DataType::INT32) {
            int32_t* p = (int32_t*)result->data;
            for (size_t i = 0; i < result->num_elements; ++i) p[i] = 1;
        }
    }
    return result;
}

Tensor* op_aten_full_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape = {1}; 
    if (argc > 0 && args[0] && args[0]->data && args[0]->dtype == DataType::INT32) {
        int32_t* dims = (int32_t*)args[0]->data;
        shape.clear();
        for (size_t i = 0; i < args[0]->num_elements; ++i) shape.push_back(dims[i]);
    }
    
    DataType out_dt = DataType::FLOAT32;
    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::FLOAT32 || args[1]->dtype == DataType::FLOAT64) out_dt = DataType::FLOAT32;
        else if (args[1]->dtype == DataType::INT32 || args[1]->dtype == DataType::INT64) out_dt = DataType::INT32;
    }

    Tensor* out = create_result_tensor(ctx, shape, out_dt);
    if (!out || !out->data) return nullptr;

    if (out_dt == DataType::FLOAT32) {
        float val = 0.0f;
        if (argc > 1 && args[1] && args[1]->data) {
            if (args[1]->dtype == DataType::FLOAT32) val = ((float*)args[1]->data)[0];
            else if (args[1]->dtype == DataType::FLOAT64) val = (float)((double*)args[1]->data)[0];
            else if (args[1]->dtype == DataType::INT32) val = (float)((int32_t*)args[1]->data)[0];
        }
        float* dst = (float*)out->data;
        for (size_t i = 0; i < out->num_elements; ++i) dst[i] = val;
    } else {
        int32_t val = 0;
        if (argc > 1 && args[1] && args[1]->data) {
            if (args[1]->dtype == DataType::INT32) val = ((int32_t*)args[1]->data)[0];
            else if (args[1]->dtype == DataType::INT64) val = (int32_t)((int64_t*)args[1]->data)[0];
            else if (args[1]->dtype == DataType::FLOAT32) val = (int32_t)((float*)args[1]->data)[0];
        }
        int32_t* dst = (int32_t*)out->data;
        for (size_t i = 0; i < out->num_elements; ++i) dst[i] = val;
    }
    return out;
}

Tensor* op_aten_select_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[0]->data || !args[1] || !args[1]->data || !args[2] || !args[2]->data) return nullptr;
    Tensor* x = args[0];
    int dim = (args[1]->dtype == DataType::INT64) ? (int)((int64_t*)args[1]->data)[0] : ((int32_t*)args[1]->data)[0];
    int idx = (args[2]->dtype == DataType::INT64) ? (int)((int64_t*)args[2]->data)[0] : ((int32_t*)args[2]->data)[0];
    int ndim = (int)x->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return nullptr;
    int dim_size = x->shape[dim];
    if (idx < 0) idx += dim_size;
    if (idx < 0 || idx >= dim_size) return nullptr;
    std::vector<int> out_shape;
    out_shape.reserve(ndim - 1);
    for (int i = 0; i < ndim; ++i) if (i != dim) out_shape.push_back(x->shape[i]);
    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result || !result->data) return result;
    int inner = 1; for (int i = dim + 1; i < ndim; ++i) inner *= x->shape[i];
    int outer = (int)(x->num_elements / (dim_size * inner));
    size_t elem_bytes = (x->dtype == DataType::INT32) ? sizeof(int32_t) : sizeof(float);
    const uint8_t* src = static_cast<const uint8_t*>(x->data);
    uint8_t* dst = static_cast<uint8_t*>(result->data);
    for (int o = 0; o < outer; ++o) {
        const uint8_t* row_src = src + (o * dim_size + idx) * inner * elem_bytes;
        memcpy(dst + o * inner * elem_bytes, row_src, inner * elem_bytes);
    }
    return result;
}

Tensor* op_aten_slice_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 4 || !args[0] || !args[1] || !args[1]->data || !args[2] || !args[2]->data || !args[3] || !args[3]->data) return nullptr;
    
    Tensor* x = args[0];
    if (x->shape.empty() || x->num_elements == 0) {
        Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
        if (out && out->data && x->data) memcpy(out->data, x->data, out->size);
        return out;
    }

    int dim = 0, start = 0, end = -1, step = 1;
    if (args[1]->dtype == DataType::INT64) dim = (int)((int64_t*)args[1]->data)[0]; else dim = ((int32_t*)args[1]->data)[0];
    
    if (args[2]->dtype == DataType::INT64) {
        int64_t v = ((int64_t*)args[2]->data)[0];
        start = (v > INT32_MAX) ? INT32_MAX : (v < -INT32_MAX ? -INT32_MAX : (int)v);
    } else start = ((int32_t*)args[2]->data)[0];

    if (args[3]->dtype == DataType::INT64) {
        int64_t v = ((int64_t*)args[3]->data)[0];
        end = (v >= INT32_MAX) ? -1 : (v <= -INT32_MAX ? -1 : (int)v);
    } else end = ((int32_t*)args[3]->data)[0];

    if (argc > 4 && args[4] && args[4]->data) {
        if (args[4]->dtype == DataType::INT64) step = (int)((int64_t*)args[4]->data)[0];
        else step = ((int32_t*)args[4]->data)[0];
    }

    int ndim = x->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return nullptr;

    int dim_size = x->shape[dim];
    if (start < 0) start += dim_size;
    if (start < 0) start = 0;
    if (start > dim_size) start = dim_size;

    if (end < 0) {
        if (end == -1) end = dim_size;
        else end += dim_size;
    }
    if (end > dim_size) end = dim_size;
    if (end < start) end = start;

    std::vector<int> out_shape = x->shape;
    out_shape[dim] = (end - start + step - 1) / step; 

    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result) return nullptr;
    if (result->num_elements == 0 || !result->data) return result;

    int inner = 1; for (int i = dim + 1; i < ndim; ++i) inner *= x->shape[i];
    int outer = 1; for (int i = 0; i < dim; ++i) outer *= x->shape[i];

    size_t elem_size = x->get_element_byte_size();
    uint8_t* src = (uint8_t*)x->data;
    uint8_t* dst = (uint8_t*)result->data;

    if (src && dst) {
        for (int o = 0; o < outer; ++o) {
            for (int s = 0; s < out_shape[dim]; ++s) {
                int src_idx = start + s * step;
                memcpy(
                    dst + (o * out_shape[dim] + s) * inner * elem_size,
                    src + (o * dim_size + src_idx) * inner * elem_size,
                    inner * elem_size
                );
            }
        }
    }
    return result;
}

Tensor* op_aten_embedding_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* weight  = args[0];
    Tensor* indices = args[1];

    if (weight->shape.size() < 2 || indices->shape.empty()) return nullptr;

    int vocab_size  = weight->shape[0];
    int hidden_size = weight->shape[1];

    std::vector<int> out_shape = indices->shape;
    out_shape.push_back(hidden_size);

    Tensor* result = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!result || !result->data) return result;

    const float*   w_data   = static_cast<const float*>(weight->data);
    const int32_t* idx_data = static_cast<const int32_t*>(indices->data);
    float*         res_data = static_cast<float*>(result->data);

    int n_tokens = (int)indices->num_elements;
    for (int i = 0; i < n_tokens; ++i) {
        int token_id = idx_data[i];
        if (token_id >= 0 && token_id < vocab_size) {
            memcpy(res_data + i * hidden_size, w_data + token_id * hidden_size, hidden_size * sizeof(float));
        } else {
            memset(res_data + i * hidden_size, 0, hidden_size * sizeof(float));
        }
    }
    return result;
}

Tensor* op_aten_expand_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* src = args[0];

    std::vector<int> tgt_shape;
    for (size_t i = 1; i < argc; ++i) {
        if (args[i] && args[i]->data) {
            if (args[i]->num_elements > 1) {
                if (args[i]->dtype == DataType::INT32) {
                    const int32_t* dims = static_cast<const int32_t*>(args[i]->data);
                    for (size_t d = 0; d < args[i]->num_elements; ++d) tgt_shape.push_back((int)dims[d]);
                } else if (args[i]->dtype == DataType::INT64) {
                    const int64_t* dims = static_cast<const int64_t*>(args[i]->data);
                    for (size_t d = 0; d < args[i]->num_elements; ++d) tgt_shape.push_back((int)dims[d]);
                }
            } else {
                if (args[i]->dtype == DataType::INT32) tgt_shape.push_back(*(int32_t*)args[i]->data);
                else if (args[i]->dtype == DataType::INT64) tgt_shape.push_back((int)*(int64_t*)args[i]->data);
            }
        }
    }

    if (tgt_shape.empty()) tgt_shape = src->shape;

    for (size_t i = 0; i < tgt_shape.size(); ++i) {
        if (tgt_shape[i] == -1) {
            int src_idx = (int)src->shape.size() - ((int)tgt_shape.size() - i);
            if (src_idx >= 0) tgt_shape[i] = src->shape[src_idx];
        }
    }

    Tensor* result = create_result_tensor(ctx, tgt_shape, src->dtype);
    if (!result || !result->data) return result;

    if (result->num_elements == src->num_elements) {
        memcpy(result->data, src->data, src->size);
        return result;
    }

    size_t elem_bytes = src->get_element_byte_size();
    const uint8_t* srcp = (const uint8_t*)src->data;
    uint8_t*       dstp = (uint8_t*)result->data;

    int src_ndim = (int)src->shape.size();
    int dst_ndim = (int)tgt_shape.size();
    
    std::vector<int> src_padded(dst_ndim, 1);
    for (int i = 0; i < src_ndim; ++i) src_padded[dst_ndim - src_ndim + i] = src->shape[i];

    std::vector<size_t> src_strides(dst_ndim, 0);
    size_t stride = 1;
    for (int i = dst_ndim - 1; i >= 0; --i) {
        src_strides[i] = (src_padded[i] == 1) ? 0 : stride;
        stride *= src_padded[i];
    }

    size_t total = result->num_elements;
    for (size_t flat = 0; flat < total; ++flat) {
        size_t tmp = flat;
        size_t src_flat = 0;
        for (int i = dst_ndim - 1; i >= 0; --i) {
            size_t coord = tmp % tgt_shape[i];
            tmp /= tgt_shape[i];
            src_flat += coord * src_strides[i];
        }
        memcpy(dstp + flat * elem_bytes, srcp + src_flat * elem_bytes, elem_bytes);
    }
    return result;
}

Tensor* op_aten_scaled_dot_product_attention_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* Q = args[0];
    Tensor* K = args[1];
    Tensor* V = args[2];
    Tensor* attn_mask = (argc > 3) ? args[3] : nullptr; 
    bool is_causal = false;
    if (argc > 5 && args[5] && args[5]->data) {
        is_causal = ((uint8_t*)args[5]->data)[0] != 0;
    }
    float scale_val = -1.0f;
    if (argc > 6 && args[6] && args[6]->data) {
        scale_val = (args[6]->dtype == DataType::FLOAT32) ? ((float*)args[6]->data)[0] : (float)((double*)args[6]->data)[0];
    }
    if (Q->num_elements == 0 || K->num_elements == 0 || V->num_elements == 0) {
        std::vector<int> out_shape = Q->shape;
        if (!out_shape.empty() && !V->shape.empty()) out_shape.back() = V->shape.back();
        Tensor* out = create_result_tensor(ctx, out_shape, Q->dtype);
        if (out && out->data) memset(out->data, 0, out->size);
        return out;
    }
    // SDPA Dynamic Head Separation (Patches for VAE and UNet)
    bool needs_free = false;
    if (Q->shape.size() == 4 && Q->shape[1] == 1) { // VAE pattern (B, 1, S, H*D)
        int num_heads = 8, b = Q->shape[0], s = Q->shape[2], c = Q->shape[3];
        int hd = c / num_heads;
        auto split = [&](Tensor* t) -> Tensor* {
            Tensor* r = create_result_tensor(ctx, {b, num_heads, s, hd}, t->dtype);
            for(int bi=0; bi<b; ++bi) for(int si=0; si<s; ++si) for(int hi=0; hi<num_heads; ++hi) for(int di=0; di<hd; ++di)
                ((float*)r->data)[bi*(num_heads*s*hd) + hi*(s*hd) + si*hd + di] = ((float*)t->data)[bi*(s*c) + si*c + hi*hd + di];
            return r;
        };
        Q = split(Q); K = split(K); V = split(V);
        needs_free = true;
    } else if (Q->shape.size() == 3) { // UNet pattern (B, S, H*D)
        int num_heads = 8, b = Q->shape[0], sq = Q->shape[1], cq = Q->shape[2];
        int sk = K->shape[1], hd = cq / num_heads;
        auto split = [&](Tensor* t, int seq_len) -> Tensor* {
            Tensor* r = create_result_tensor(ctx, {b, num_heads, seq_len, hd}, t->dtype);
            for(int bi=0; bi<b; ++bi) for(int si=0; si<seq_len; ++si) for(int hi=0; hi<num_heads; ++hi) for(int di=0; di<hd; ++di)
                ((float*)r->data)[bi*(num_heads*seq_len*hd) + hi*(seq_len*hd) + si*hd + di] = ((float*)t->data)[bi*(seq_len*cq) + si*cq + hi*hd + di];
            return r;
        };
        Q = split(Q, sq); K = split(K, sk); V = split(V, sk);
        needs_free = true;
    }
    int ndim = Q->shape.size();
    if (ndim < 2) {
        if (needs_free) { ctx->tensor_pool.release(Q); ctx->tensor_pool.release(K); ctx->tensor_pool.release(V); }
        return nullptr;
    }
    int D = Q->shape.back();
    int T = Q->shape[ndim - 2];
    int T_k = K->shape[ndim-2];
    int outer = (int)(Q->num_elements / (T * D));
    int num_heads = (ndim >= 3) ? Q->shape[ndim - 3] : 1;
    int m_ndim = attn_mask ? attn_mask->shape.size() : 0;
    std::vector<int> m_strides(m_ndim, 1);
    if (attn_mask) {
        for (int d = m_ndim - 2; d >= 0; --d) m_strides[d] = m_strides[d+1] * attn_mask->shape[d+1];
    }
    Tensor* output = create_result_tensor(ctx, Q->shape, Q->dtype);
    if (!output || !output->data) {
        if (needs_free) { ctx->tensor_pool.release(Q); ctx->tensor_pool.release(K); ctx->tensor_pool.release(V); }
        return output;
    }
    const float* q_data = (const float*)Q->data;
    const float* k_data = (const float*)K->data;
    const float* v_data = (const float*)V->data;
    float* out_data = (float*)output->data;
    float scale = (scale_val > 0.0f) ? scale_val : (1.0f / sqrtf((float)D));
    // Let's parallelize calculations across heads (or batches)
    #pragma omp parallel for
    for (int o = 0; o < outer; ++o) {
        int b = o / num_heads;
        int h = o % num_heads;
        const float* q_batch = q_data + o * T * D;
        const float* k_batch = k_data + o * T_k * D;
        const float* v_batch = v_data + o * T_k * D;
        float* out_batch = out_data + o * T * D;
        // Local buffer for caching the transposed K matrix
        // If T_k * D is not too large, this will dramatically speed up the inner loop.
        float* k_transposed = (float*)alloc_fast(T_k * D * sizeof(float));
        bool can_transpose = (k_transposed != nullptr);
        if (can_transpose) {
            for (int j = 0; j < T_k; ++j) {
                for (int d = 0; d < D; ++d) {
                    k_transposed[d * T_k + j] = k_batch[j * D + d];
                }
            }
        }
        // Local buffer for raw logits
        float* row_scores = (float*)alloc_fast(T_k * sizeof(float));
        if (!row_scores) {
            if (can_transpose) heap_caps_free(k_transposed);
            continue; 
        }
        for (int i = 0; i < T; ++i) {
            const float* q_i = q_batch + i * D;
            float* out_i = out_batch + i * D;
            float max_score = -1e9f;
            // 1. Calculate the scalar products Q * K^T
            if (can_transpose) {
                #pragma omp simd
                for (int j = 0; j < T_k; ++j) row_scores[j] = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float q_val = q_i[d];
                    const float* k_row = k_transposed + d * T_k;
                    #pragma omp simd
                    for (int j = 0; j < T_k; ++j) {
                        row_scores[j] += q_val * k_row[j];
                    }
                }
            } else {
                for (int j = 0; j < T_k; ++j) {
                    float score = 0.0f;
                    const float* k_j = k_batch + j * D;
                    #pragma omp simd reduction(+:score)
                    for (int d = 0; d < D; ++d) score += q_i[d] * k_j[d];
                    row_scores[j] = score;
                }
            }
            // 2. Apply Scale, Mask, and Causal, while simultaneously searching for max_score
            for (int j = 0; j < T_k; ++j) {
                if (is_causal && j > i) {
                    row_scores[j] = -1e9f;
                } else {
                    float score = row_scores[j] * scale;
                    float mask_val = 0.0f;
                    if (attn_mask && attn_mask->data) {
                        int coords[4] = {b, h, i, j};
                        int m_idx = 0;
                        for (int m_d = 0; m_d < m_ndim; ++m_d) {
                            int c = coords[4 - m_ndim + m_d];
                            if (attn_mask->shape[m_d] == 1) c = 0; // Broadcasting
                            m_idx += c * m_strides[m_d];
                        }
                        float raw_mask = (attn_mask->dtype == DataType::FLOAT32) ? ((float*)attn_mask->data)[m_idx] : 
                                         (attn_mask->dtype == DataType::INT32) ? (float)((int32_t*)attn_mask->data)[m_idx] :
                                         (float)((uint8_t*)attn_mask->data)[m_idx];
                        if (attn_mask->dtype == DataType::BOOL || attn_mask->dtype == DataType::INT8 || attn_mask->dtype == DataType::UINT8) {
                            mask_val = (raw_mask > 0.0f) ? 0.0f : -1e9f;
                        } else {
                            mask_val = raw_mask;
                        }
                    }
                    row_scores[j] = score + mask_val;
                }
                if (row_scores[j] > max_score) max_score = row_scores[j];
            }
            // 3. Calculate Softmax (Exp + Normalize)
            float sum_exp = 0.0f;
            #pragma omp simd reduction(+:sum_exp)
            for (int j = 0; j < T_k; ++j) {
                float val = expf(row_scores[j] - max_score);
                row_scores[j] = val;
                sum_exp += val;
            }
            float inv_sum_exp = (sum_exp != 0.0f) ? (1.0f / sum_exp) : 0.0f;
            #pragma omp simd
            for (int j = 0; j < T_k; ++j) row_scores[j] *= inv_sum_exp;
            // 4. Multiply the weights by V (Softmax * V)
            #pragma omp simd
            for (int d = 0; d < D; ++d) out_i[d] = 0.0f;
            for (int j = 0; j < T_k; ++j) {
                float weight = row_scores[j];
                const float* v_j = v_batch + j * D;
                #pragma omp simd
                for (int d = 0; d < D; ++d) {
                    out_i[d] += weight * v_j[d];
                }
            }
        }
        heap_caps_free(row_scores);
        if (can_transpose) heap_caps_free(k_transposed);
    }
    if (needs_free) {
        ctx->tensor_pool.release(Q); 
        ctx->tensor_pool.release(K); 
        ctx->tensor_pool.release(V);
    }
    return output;
}

Tensor* op_aten_conv2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    
    Tensor* x = args[0]; Tensor* w = args[1]; Tensor* bias = (argc > 2) ? args[2] : nullptr;
    if (x->dtype != DataType::FLOAT32 || w->dtype != DataType::FLOAT32) return nullptr;
    
    auto get_pair = [](Tensor* t, int& v1, int& v2) {
        if (!t || !t->data) return;
        if (t->dtype == DataType::INT32) {
            int32_t* p = (int32_t*)t->data;
            v1 = p[0]; v2 = (t->num_elements >= 2) ? p[1] : p[0];
        } else if (t->dtype == DataType::INT64) {
            int64_t* p = (int64_t*)t->data;
            v1 = (int)p[0]; v2 = (int)((t->num_elements >= 2) ? p[1] : p[0]);
        } else if (t->dtype == DataType::FLOAT32) {
            float* p = (float*)t->data;
            v1 = (int)p[0]; v2 = (int)((t->num_elements >= 2) ? p[1] : p[0]);
        }
    };

    int sH = 1, sW = 1; if (argc > 3) get_pair(args[3], sH, sW);
    int pH = 0, pW = 0; if (argc > 4) get_pair(args[4], pH, pW);
    int dH = 1, dW = 1; if (argc > 5) get_pair(args[5], dH, dW);
    
    int groups = 1;
    if (argc > 6 && args[6] && args[6]->data) {
        if (args[6]->dtype == DataType::INT32) groups = *(int32_t*)args[6]->data;
        else if (args[6]->dtype == DataType::INT64) groups = (int)*(int64_t*)args[6]->data;
        else if (args[6]->dtype == DataType::FLOAT32) groups = (int)*(float*)args[6]->data;
    }

    if (x->shape.size() != 4 || w->shape.size() != 4) return nullptr;
    
    int N = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    int C_out = w->shape[0], kH = w->shape[2], kW = w->shape[3];
    
    if (N == 0 || C_in == 0 || H == 0 || W == 0 || C_out == 0) {
        Tensor* out = create_result_tensor(ctx, {N, C_out, H, W}, DataType::FLOAT32);
        if (out && out->data) memset(out->data, 0, out->size);
        return out;
    }

    int eff_kH = (kH - 1) * dH + 1;
    int eff_kW = (kW - 1) * dW + 1;

    int out_h = (H + 2 * pH - eff_kH) / sH + 1;
    int out_w = (W + 2 * pW - eff_kW) / sW + 1;
    if (out_h <= 0 || out_w <= 0) return nullptr;

    Tensor* out = create_result_tensor(ctx, {N, C_out, out_h, out_w}, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    const float* src_x = (const float*)x->data;
    const float* src_w = (const float*)w->data;
    const float* b_ptr = bias ? (const float*)bias->data : nullptr;
    float* dst         = (float*)out->data;

    int C_in_g = C_in / groups;
    int C_out_g = C_out / groups;
    int total_tasks = N * C_out;

    // -------------------------------------------------------------------------
    // FAST PATH: 1x1 Convolution (Pointwise)
    // Extreme Speedup for ResNet/MobileNet Bottlenecks
    // -------------------------------------------------------------------------
    if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && pH == 0 && pW == 0 && groups == 1) {
        int HW = H * W; // H == out_h, W == out_w
        
        #pragma omp parallel for
        for (int task = 0; task < total_tasks; ++task) {
            int n = task / C_out;
            int c_out = task % C_out;
            
            float b_val = b_ptr ? b_ptr[c_out] : 0.0f;
            float* out_row = dst + n * (C_out * HW) + c_out * HW;
            
            // Fast vectorized initialization
            #pragma omp simd
            for (int p = 0; p < HW; ++p) out_row[p] = b_val;
            
            const float* w_row = src_w + c_out * C_in;
            for (int c_in = 0; c_in < C_in; ++c_in) {
                float w_val = w_row[c_in];
                if (w_val == 0.0f) continue; // Sparsity skip
                
                const float* x_row = src_x + n * (C_in * HW) + c_in * HW;
                
                // Vectorized FMA (Fused Multiply-Add) over the entire image plane
                #pragma omp simd
                for (int p = 0; p < HW; ++p) {
                    out_row[p] += x_row[p] * w_val;
                }
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // GENERAL VECTORIZED PATH: Any convolutions (including Depthwise)
    // -------------------------------------------------------------------------
    #pragma omp parallel for
    for (int task = 0; task < total_tasks; ++task) {
        int n = task / C_out;
        int c_out = task % C_out;
        int g = c_out / C_out_g;
        
        float b_val = b_ptr ? b_ptr[c_out] : 0.0f;
        float* out_channel = dst + n * (C_out * out_h * out_w) + c_out * (out_h * out_w);
        const float* w_row = src_w + c_out * (C_in_g * kH * kW);
        
        // Channel initialization with offset
        for (int p = 0; p < out_h * out_w; ++p) out_channel[p] = b_val;

        for (int ic = 0; ic < C_in_g; ++ic) {
            int c_in = g * C_in_g + ic;
            const float* x_channel = src_x + n * (C_in * H * W) + c_in * (H * W);
            const float* w_ic = w_row + ic * (kH * kW);
            
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    float w_val = w_ic[kh * kW + kw];
                    if (w_val == 0.0f) continue; // Sparsity skip: skipping a plane
                    
                    // Pre-calculate safe bounds for the ow loop
                    // to avoid branching if (iw >= 0 && iw < W) inside
                    int offset_w = kw * dW - pW;
                    int ow_start = 0;
                    if (offset_w < 0) {
                        ow_start = (-offset_w + sW - 1) / sW; // ceil
                    }
                    int ow_end = out_w;
                    if (W - offset_w - 1 >= 0) {
                        ow_end = (W - offset_w - 1) / sW + 1;
                    } else {
                        ow_end = 0;
                    }
                    
                    // Output tensor overflow protection
                    ow_start = std::max(0, std::min(out_w, ow_start));
                    ow_end   = std::max(ow_start, std::min(out_w, ow_end));

                    for (int oh = 0; oh < out_h; ++oh) {
                        int ih = oh * sH - pH + kh * dH;
                        if (ih < 0 || ih >= H) continue;
                        
                        float* out_row = out_channel + oh * out_w;
                        const float* x_row = x_channel + ih * W;

                        // NO BRANCHING! The compiler will generate SIMD instructions here.
                        #pragma omp simd
                        for (int ow = ow_start; ow < ow_end; ++ow) {
                            out_row[ow] += x_row[ow * sW + offset_w] * w_val;
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_native_batch_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 5 || !args[0] || !args[1] || !args[2] || !args[3] || !args[4]) return nullptr;
    
    Tensor* x = args[0]; Tensor* w = args[1]; Tensor* b = args[2]; Tensor* rm = args[3]; Tensor* rv = args[4];
    if (x->dtype != DataType::FLOAT32) return nullptr;
    
    float eps = 1e-5f;
    if (argc > 6 && args[6] && args[6]->dtype == DataType::FLOAT32) eps = *(float*)args[6]->data;
    
    Tensor* out = create_result_tensor(ctx, x->shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    float* src = (float*)x->data; float* dst = (float*)out->data;
    float* w_p = (float*)w->data; float* b_p = (float*)b->data;
    float* rm_p = (float*)rm->data; float* rv_p = (float*)rv->data;
    
    int N = x->shape[0]; int C = x->shape[1];
    int spatial = 1; for (size_t i = 2; i < x->shape.size(); ++i) spatial *= x->shape[i];
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float mean = rm_p[c], var = rv_p[c], weight = w_p[c], bias = b_p[c];
            float inv_std = weight / sqrtf(var + eps);
            for (int s = 0; s < spatial; ++s) {
                int idx = n * (C * spatial) + c * spatial + s;
                dst[idx] = (src[idx] - mean) * inv_std + bias;
            }
        }
    }
    return out;
}

Tensor* op_aten_max_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x = args[0];
    if (x->dtype != DataType::FLOAT32 || x->shape.size() != 4) return nullptr;
    
    auto get_pair = [](Tensor* t, int& v1, int& v2) {
        if (t && t->data) {
            if (t->dtype == DataType::INT32) {
                int32_t* p = (int32_t*)t->data;
                v1 = p[0]; v2 = (t->num_elements >= 2) ? p[1] : p[0];
            } else if (t->dtype == DataType::INT64) {
                int64_t* p = (int64_t*)t->data;
                v1 = (int)p[0]; v2 = (int)((t->num_elements >= 2) ? p[1] : p[0]);
            }
        }
    };

    int kH = 1, kW = 1; get_pair(args[1], kH, kW);
    int sH = kH, sW = kW; if (argc > 2) get_pair(args[2], sH, sW);
    int pH = 0, pW = 0; if (argc > 3) get_pair(args[3], pH, pW);
    int dH = 1, dW = 1; if (argc > 4) get_pair(args[4], dH, dW);
    
    int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int eff_kH = (kH - 1) * dH + 1;
    int eff_kW = (kW - 1) * dW + 1;
    int out_h = (H + 2 * pH - eff_kH) / sH + 1;
    int out_w = (W + 2 * pW - eff_kW) / sW + 1;
    if (out_h <= 0 || out_w <= 0) return nullptr;

    Tensor* out = create_result_tensor(ctx, {N, C, out_h, out_w}, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    float* src = (float*)x->data;
    float* dst = (float*)out->data;
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -1e30f;
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = oh * sH - pH + kh * dH;
                            int iw = ow * sW - pW + kw * dW;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float val = src[n * (C * H * W) + c * (H * W) + ih * W + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    dst[n * (C * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow] = max_val;
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_adaptive_avg_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* input = args[0];

    int H_out = 1, W_out = 1;
    if (args[1]->data) {
        if (args[1]->dtype == DataType::INT32) {
            int32_t* p = (int32_t*)args[1]->data;
            H_out = p[0]; W_out = (args[1]->num_elements >= 2) ? p[1] : p[0];
        } else if (args[1]->dtype == DataType::INT64) {
            int64_t* p = (int64_t*)args[1]->data;
            H_out = (int)p[0]; W_out = (int)((args[1]->num_elements >= 2) ? p[1] : p[0]);
        }
    }

    int N = 1, C = 1, H_in = 1, W_in = 1;
    if (input->shape.size() == 4) {
        N = input->shape[0]; C = input->shape[1]; H_in = input->shape[2]; W_in = input->shape[3];
    } else if (input->shape.size() == 3) {
        C = input->shape[0]; H_in = input->shape[1]; W_in = input->shape[2];
    } else {
        return nullptr;
    }
    
    std::vector<int> out_shape = input->shape;
    out_shape[out_shape.size() - 2] = H_out;
    out_shape[out_shape.size() - 1] = W_out;

    Tensor* result = create_result_tensor(ctx, out_shape, input->dtype);
    if (!result || !result->data) return result;

    float* in_data = (float*)input->data;
    float* out_data = (float*)result->data;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                int h_start = static_cast<int>(floorf((float)h_out * H_in / H_out));
                int h_end = static_cast<int>(ceilf((float)(h_out + 1) * H_in / H_out));
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int w_start = static_cast<int>(floorf((float)w_out * W_in / W_out));
                    int w_end = static_cast<int>(ceilf((float)(w_out + 1) * W_in / W_out));
                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) sum += in_data[n * C * H_in * W_in + c * H_in * W_in + h * W_in + w];
                    }
                    count = (h_end - h_start) * (w_end - w_start);
                    out_data[n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] = (count > 0) ? (sum / count) : 0.0f;
                }
            }
        }
    }
    return result;
}

Tensor* op_aten_tril_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    Tensor* x = (argc > 0) ? args[0] : nullptr;
    int diagonal = 0;
    if (argc > 1 && args[1] && args[1]->dtype == DataType::INT32) diagonal = *(int32_t*)args[1]->data;
    
    if (!x) {
        Tensor* out = create_result_tensor(ctx, {1, 1}, DataType::FLOAT32);
        if (out && out->data) ((float*)out->data)[0] = 1.0f;
        return out;
    }

    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    
    int rows = out->shape[out->shape.size() - 2];
    int cols = out->shape[out->shape.size() - 1];
    int outer = out->num_elements / (rows * cols);
    
    for (int b = 0; b < outer; ++b) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = b * (rows * cols) + r * cols + c;
                if (c - r <= diagonal) {
                    if (x->dtype == DataType::FLOAT32) ((float*)out->data)[idx] = ((float*)x->data)[idx];
                    else if (x->dtype == DataType::INT32) ((int32_t*)out->data)[idx] = ((int32_t*)x->data)[idx];
                } else {
                    if (x->dtype == DataType::FLOAT32) ((float*)out->data)[idx] = 0.0f;
                    else if (x->dtype == DataType::INT32) ((int32_t*)out->data)[idx] = 0;
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_split_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    
    Tensor* in = args[0];
    int dim = in->shape.size() - 1; 
    if (argc > 2 && args[2] && args[2]->data) {
        if (args[2]->dtype == DataType::INT64) dim = (int)((int64_t*)args[2]->data)[0];
        else dim = ((int32_t*)args[2]->data)[0];
    }
    if (dim < 0) dim += in->shape.size();
    
    int split_size = 1;
    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::INT64) split_size = (int)((int64_t*)args[1]->data)[0];
        else split_size = ((int32_t*)args[1]->data)[0];
    }
    
    if (split_size <= 0) {
        ESP_LOGE("KERNELS", "split_Tensor: invalid split_size %d", split_size);
        return nullptr;
    }
    
    Tensor* out = create_result_tensor(ctx, in->shape, in->dtype);
    if (!out || !out->data) return out;
    memcpy(out->data, in->data, out->size);
    
    out->quant_meta.block_size = split_size;
    out->quant_meta.axis = dim;
    
    return out;
}

Tensor* op_getitem(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    
    Tensor* in = args[0]; 
    int index = 0;
    if (args[1]->dtype == DataType::INT32) index = ((int32_t*)args[1]->data)[0];
    
    // -----------------------------------------------------------------
    // Tuple unpacking support (glued tensors, such as in lstm_cell)
    // -----------------------------------------------------------------
    if (in->quant_meta.axis == 127) { 
        std::vector<int> out_shape = in->shape;
        out_shape.erase(out_shape.begin()); // Remove the tuple dimension (dim 0)
        
        Tensor* out = create_result_tensor(ctx, out_shape, in->dtype);
        if (!out || !out->data) return out;
        
        size_t tuple_elem_size = out->size;
        size_t src_offset = index * tuple_elem_size;
        memcpy(out->data, (uint8_t*)in->data + src_offset, tuple_elem_size);
        return out;
    }
    
    // -----------------------------------------------------------------
    // Regular split logic
    // -----------------------------------------------------------------
    int split_size = in->quant_meta.block_size;
    int dim = in->quant_meta.axis;
    
    if (split_size == 0) return nullptr; 
    if (dim < 0) dim += in->shape.size();
    
    std::vector<int> out_shape = in->shape;
    int max_dim = out_shape[dim];
    
    int start_idx = index * split_size;
    int end_idx = start_idx + split_size;
    if (end_idx > max_dim) end_idx = max_dim;
    int actual_size = end_idx - start_idx;
    
    if (actual_size <= 0) return nullptr;
    out_shape[dim] = actual_size;
    
    Tensor* out = create_result_tensor(ctx, out_shape, in->dtype);
    if (!out || !out->data) return out;
    
    int outer = 1; for(int i=0; i<dim; ++i) outer *= in->shape[i];
    int inner = 1; for(size_t i=dim+1; i<in->shape.size(); ++i) inner *= in->shape[i];
    
    size_t elem_size = in->get_element_byte_size();
    uint8_t* src = (uint8_t*)in->data;
    uint8_t* dst = (uint8_t*)out->data;
    
    for (int o = 0; o < outer; ++o) {
        size_t src_offset = (o * max_dim * inner + start_idx * inner) * elem_size;
        size_t dst_offset = (o * actual_size * inner) * elem_size;
        size_t copy_bytes = actual_size * inner * elem_size;
        memcpy(dst + dst_offset, src + src_offset, copy_bytes);
    }
    return out;
}

#define UNARY_MATH_OP(name, func) \
Tensor* op_aten_##name##_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) { \
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr; \
    Tensor* x = args[0]; \
    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype); \
    if (!out || !out->data) return out; \
    if (x->dtype == DataType::FLOAT32) { \
        float* src = (float*)x->data; float* dst = (float*)out->data; \
        long long N = (long long)x->num_elements; \
        _Pragma("omp parallel for simd") \
        for (long long i = 0; i < N; ++i) dst[i] = func(src[i]); \
    } \
    return out; \
}
UNARY_MATH_OP(tanh, tanhf)
UNARY_MATH_OP(cos, cosf)
UNARY_MATH_OP(sin, sinf)
UNARY_MATH_OP(exp, expf)
UNARY_MATH_OP(log, [](float v){ return logf(std::max(v, 1e-12f)); })

Tensor* op_aten_pow_Tensor_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    float p = (args[1]->dtype == DataType::FLOAT32) ? ((float*)args[1]->data)[0] : (float)((int32_t*)args[1]->data)[0];
    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data; float* dst = (float*)out->data;
        long long N = (long long)x->num_elements;
        #pragma omp parallel for simd
        for (long long i = 0; i < N; ++i) dst[i] = powf(src[i], p);
    }
    return out;
}

Tensor* op_aten_min_other(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return broadcast_binary_op(ctx, args[0], args[1], [](float a, float b){ return std::min(a, b); });
}

Tensor* op_aten_zeros_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* out = create_result_tensor(ctx, args[0]->shape, args[0]->dtype);
    if (out && out->data) memset(out->data, 0, out->size);
    return out;
}

Tensor* op_aten_ones_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* out = create_result_tensor(ctx, args[0]->shape, args[0]->dtype);
    if (out && out->data) {
        if (out->dtype == DataType::FLOAT32) {
            float* p = (float*)out->data; for(size_t i=0; i<out->num_elements; ++i) p[i] = 1.0f;
        } else {
            int32_t* p = (int32_t*)out->data; for(size_t i=0; i<out->num_elements; ++i) p[i] = 1;
        }
    }
    return out;
}

Tensor* op_aten_full_like_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* out = create_result_tensor(ctx, args[0]->shape, args[0]->dtype);
    if (out && out->data) {
        if (out->dtype == DataType::FLOAT32) {
            float v = (args[1]->dtype == DataType::FLOAT32) ? ((float*)args[1]->data)[0] : (float)((int32_t*)args[1]->data)[0];
            float* p = (float*)out->data; for(size_t i=0; i<out->num_elements; ++i) p[i] = v;
        } else {
            int32_t v = (args[1]->dtype == DataType::INT32) ? ((int32_t*)args[1]->data)[0] : (int32_t)((float*)args[1]->data)[0];
            int32_t* p = (int32_t*)out->data; for(size_t i=0; i<out->num_elements; ++i) p[i] = v;
        }
    }
    return out;
}

Tensor* op_aten_mean_dim(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    std::vector<int> dims;
    if (args[1]->dtype == DataType::INT32) {
        int32_t* d = (int32_t*)args[1]->data;
        for (size_t i=0; i<args[1]->num_elements; ++i) dims.push_back(d[i] < 0 ? d[i] + x->shape.size() : d[i]);
    } else if (args[1]->dtype == DataType::INT64) {
        int64_t* d = (int64_t*)args[1]->data;
        for (size_t i=0; i<args[1]->num_elements; ++i) dims.push_back(d[i] < 0 ? d[i] + x->shape.size() : (int)d[i]);
    }
    bool keepdim = false;
    if (argc > 2 && args[2] && args[2]->data) {
        if (args[2]->dtype == DataType::BOOL) keepdim = ((uint8_t*)args[2]->data)[0] != 0;
        else if (args[2]->dtype == DataType::INT32) keepdim = ((int32_t*)args[2]->data)[0] != 0;
    }
    std::vector<bool> reduce_mask(x->shape.size(), false);
    for (int d : dims) reduce_mask[d] = true;
    std::vector<int> out_shape;
    int reduce_size = 1;
    for (size_t i=0; i<x->shape.size(); ++i) {
        if (reduce_mask[i]) {
            reduce_size *= x->shape[i];
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(x->shape[i]);
        }
    }
    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    memset(out->data, 0, out->size);
    float* dst = (float*)out->data;
    float* src = (float*)x->data; 
    int ndim = x->shape.size();
    std::vector<int> in_stride(ndim, 1);
    for (int i=ndim-2; i>=0; --i) in_stride[i] = in_stride[i+1] * x->shape[i+1];
    int ondim = out_shape.size();
    std::vector<int> os(ondim, 1);
    for (int i=ondim-2; i>=0; --i) os[i] = os[i+1] * out_shape[i+1];
    
    long long N = (long long)x->num_elements;
    
    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        size_t tmp = i, out_off = 0; int o_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = tmp / in_stride[d]; tmp %= in_stride[d];
            if (!reduce_mask[d]) out_off += coord * os[o_idx++];
            else if (keepdim) o_idx++;
        }
        
        // Atomic addition, since different threads can write to the same out_off
        #pragma omp atomic
        dst[out_off] += src[i];
    }
    
    long long OutN = (long long)out->num_elements;
    #pragma omp parallel for simd
    for (long long i = 0; i < OutN; ++i) dst[i] /= (float)reduce_size;
    return out;
}

Tensor* op_aten_permute_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    std::vector<int> dims;
    if (args[1]->dtype == DataType::INT32) {
        int32_t* d = (int32_t*)args[1]->data;
        for (size_t i=0; i<args[1]->num_elements; ++i) dims.push_back(d[i] < 0 ? d[i] + x->shape.size() : d[i]);
    } else if (args[1]->dtype == DataType::INT64) {
        int64_t* d = (int64_t*)args[1]->data;
        for (size_t i=0; i<args[1]->num_elements; ++i) dims.push_back(d[i] < 0 ? d[i] + x->shape.size() : (int)d[i]);
    }
    if (dims.size() != x->shape.size()) return nullptr;
    // 1. No-op (empty permutation) check
    bool is_noop = true;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != (int)i) { is_noop = false; break; }
    }
    if (is_noop) {
        Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
        if (out && out->data) memcpy(out->data, x->data, x->size);
        return out;
    }
    std::vector<int> out_shape(dims.size());
    for (size_t i=0; i<dims.size(); ++i) out_shape[i] = x->shape[dims[i]];
    Tensor* out = create_result_tensor(ctx, out_shape, x->dtype);
    if (!out || !out->data) return out;
    int ndim = x->shape.size();
    size_t elem_size = x->get_element_byte_size();
    const uint8_t* src = (const uint8_t*)x->data;
    uint8_t* dst = (uint8_t*)out->data;
    // Preparing for fast work with 32-bit tensors (float32 / int32)
    const uint32_t* src32 = (const uint32_t*)src;
    uint32_t* dst32 = (uint32_t*)dst;
    // =========================================================================
    // FAST PATHS (Extreme Acceleration for Frequent Patterns)
    // =========================================================================
    if (elem_size == 4) {
        // Pattern 1: Multi-Head Attention [0, 2, 1, 3] (B, S, H, D) -> (B, H, S, D)
        if (ndim == 4 && dims[0] == 0 && dims[1] == 2 && dims[2] == 1 && dims[3] == 3) {
            int B = x->shape[0], S = x->shape[1], H = x->shape[2], D = x->shape[3];
            int bytes_to_copy = D * 4;
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H; ++h) {
                    for (int s = 0; s < S; ++s) {
                        int src_idx = b*(S*H*D) + s*(H*D) + h*D;
                        int dst_idx = b*(H*S*D) + h*(S*D) + s*D;
                        memcpy(dst + dst_idx * 4, src + src_idx * 4, bytes_to_copy);
                    }
                }
            }
            return out;
        }
        // Pattern 2: Batch Matrix Transpose [0, 2, 1] (B, M, N) -> (B, N, M)
        if (ndim == 3 && dims[0] == 0 && dims[1] == 2 && dims[2] == 1) {
            int B = x->shape[0], M = x->shape[1], N = x->shape[2];
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int n = 0; n < N; ++n) {
                    #pragma omp simd
                    for (int m = 0; m < M; ++m) {
                        dst32[b*(N*M) + n*M + m] = src32[b*(M*N) + m*N + n];
                    }
                }
            }
            return out;
        }
        // Pattern 3: Simple Matrix [1, 0] (M, N) -> (N, M)
        if (ndim == 2 && dims[0] == 1 && dims[1] == 0) {
            int M = x->shape[0], N = x->shape[1];
            #pragma omp parallel for
            for (int n = 0; n < N; ++n) {
                #pragma omp simd
                for (int m = 0; m < M; ++m) {
                    dst32[n*M + m] = src32[m*N + n];
                }
            }
            return out;
        }
        // Pattern 4: Images NHWC -> NCHW [0, 3, 1, 2]
        if (ndim == 4 && dims[0] == 0 && dims[1] == 3 && dims[2] == 1 && dims[3] == 2) {
            int B = x->shape[0], H = x->shape[1], W = x->shape[2], C = x->shape[3];
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        #pragma omp simd
                        for (int w = 0; w < W; ++w) {
                            dst32[b*(C*H*W) + c*(H*W) + h*W + w] = src32[b*(H*W*C) + h*(W*C) + w*C + c];
                        }
                    }
                }
            }
            return out;
        }
        // Pattern 5: Images NCHW -> NHWC [0, 2, 3, 1]
        if (ndim == 4 && dims[0] == 0 && dims[1] == 2 && dims[2] == 3 && dims[3] == 1) {
            int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst32[b*(H*W*C) + h*(W*C) + w*C + c] = src32[b*(C*H*W) + c*(H*W) + h*W + w];
                        }
                    }
                }
            }
            return out;
        }
    }
    // =========================================================================
    // GENERAL FALLBACK (For non-standard dimensions and data types)
    // =========================================================================
    std::vector<int> in_stride(ndim, 1), out_stride(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_stride[i] = in_stride[i+1] * x->shape[i+1];
        out_stride[i] = out_stride[i+1] * out_shape[i+1];
    }
    if (elem_size == 4) {
        // Accelerated fallback without memcpy
        #pragma omp parallel for
        for (int i = 0; i < (int)out->num_elements; ++i) {
            int tmp = i, in_off = 0;
            for (int d = 0; d < ndim; ++d) {
                int coord = tmp / out_stride[d]; 
                tmp %= out_stride[d];
                in_off += coord * in_stride[dims[d]];
            }
            dst32[i] = src32[in_off];
        }
    } else {
        // Standard fallback for int8/int16/bool
        #pragma omp parallel for
        for (int i = 0; i < (int)out->num_elements; ++i) {
            int tmp = i, in_off = 0;
            for (int d = 0; d < ndim; ++d) {
                int coord = tmp / out_stride[d]; 
                tmp %= out_stride[d];
                in_off += coord * in_stride[dims[d]];
            }
            memcpy(dst + i * elem_size, src + in_off * elem_size, elem_size);
        }
    }
    return out;
}

Tensor* op_aten_triu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    Tensor* x = (argc > 0) ? args[0] : nullptr;
    int diagonal = 0;
    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::INT64) diagonal = (int)((int64_t*)args[1]->data)[0];
        else if (args[1]->dtype == DataType::INT32) diagonal = ((int32_t*)args[1]->data)[0];
    }
    
    if (!x) {
        Tensor* out = create_result_tensor(ctx, {1, 1}, DataType::FLOAT32);
        if (out && out->data) ((float*)out->data)[0] = 1.0f;
        return out;
    }

    Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
    if (!out || !out->data) return out;
    
    // Protection against 1D tensors
    int rows = out->shape.size() >= 2 ? out->shape[out->shape.size() - 2] : 1;
    int cols = out->shape.back();
    int outer = out->num_elements / (rows * cols);
    
    for (int b = 0; b < outer; ++b) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = b * (rows * cols) + r * cols + c;
                if (c >= r + diagonal) { 
                    if (x->dtype == DataType::FLOAT32) ((float*)out->data)[idx] = ((float*)x->data)[idx];
                    else if (x->dtype == DataType::INT32) ((int32_t*)out->data)[idx] = ((int32_t*)x->data)[idx];
                } else { 
                    if (x->dtype == DataType::FLOAT32) ((float*)out->data)[idx] = 0.0f;
                    else if (x->dtype == DataType::INT32) ((int32_t*)out->data)[idx] = 0;
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_cat_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2) return nullptr;
    int dim = 0;
    std::vector<Tensor*> valid_tensors;
    
    std::string perm = (ins.B < ctx->permutations.size()) ? ctx->permutations[ins.B] : "";
    if (!perm.empty()) {
        for (size_t i = 0; i < perm.size() && i < argc; ++i) {
            if (perm[i] == 'A' || perm[i] == 'i') {
                if (args[i] && args[i]->data) {
                    if (args[i]->dtype == DataType::INT32) dim = ((int32_t*)args[i]->data)[0];
                    else if (args[i]->dtype == DataType::INT64) dim = (int)((int64_t*)args[i]->data)[0];
                }
            } else if (args[i] && args[i]->data && args[i]->num_elements > 0) {
                valid_tensors.push_back(args[i]);
            }
        }
    } else {
        if (args[argc-1] && args[argc-1]->num_elements == 1 && (args[argc-1]->dtype == DataType::INT32 || args[argc-1]->dtype == DataType::INT64)) {
            if (args[argc-1]->dtype == DataType::INT32) dim = ((int32_t*)args[argc-1]->data)[0];
            else dim = (int)((int64_t*)args[argc-1]->data)[0];
            for (size_t i = 0; i < argc - 1; ++i) {
                if (args[i] && args[i]->data && args[i]->num_elements > 0) valid_tensors.push_back(args[i]);
            }
        } else {
            for (size_t i = 0; i < argc; ++i) {
                if (args[i] && args[i]->data && args[i]->num_elements > 0) valid_tensors.push_back(args[i]);
            }
        }
    }

    if (valid_tensors.empty()) return nullptr;

    // Protecting against the GPT-2 Streaming Bug (Mixing 2D and 4D Empty Cache Tensors)
    if (valid_tensors.size() == 2) {
        int ndim0 = valid_tensors[0]->shape.size();
        int ndim1 = valid_tensors[1]->shape.size();
        if ((ndim0 == 2 && ndim1 == 4) || (ndim0 == 4 && ndim1 == 2)) {
            Tensor* valid = (ndim0 == 4) ? valid_tensors[0] : valid_tensors[1];
            Tensor* out = create_result_tensor(ctx, valid->shape, valid->dtype);
            if (out && out->data) memcpy(out->data, valid->data, valid->size);
            return out;
        }
    }
    
    Tensor* ref_tensor = valid_tensors[0];
    int ndim = ref_tensor->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return nullptr;

    std::vector<int> out_shape = ref_tensor->shape;
    out_shape[dim] = 0;
    for (Tensor* t : valid_tensors) out_shape[dim] += t->shape[dim];

    Tensor* out = create_result_tensor(ctx, out_shape, ref_tensor->dtype);
    if (!out || !out->data) return out;

    int outer = 1; for(int i=0; i<dim; ++i) outer *= out_shape[i];
    int inner = 1; for(int i=dim+1; i<ndim; ++i) inner *= out_shape[i];

    size_t elem_size = out->get_element_byte_size();
    uint8_t* dst = (uint8_t*)out->data;

    for (int o = 0; o < outer; ++o) {
        int current_dim_offset = 0;
        for (Tensor* t : valid_tensors) {
            int t_dim_size = t->shape[dim];
            memcpy(
                dst + (o * out_shape[dim] + current_dim_offset) * inner * elem_size,
                (uint8_t*)t->data + (o * t_dim_size) * inner * elem_size,
                t_dim_size * inner * elem_size
            );
            current_dim_offset += t_dim_size;
        }
    }
    return out;
}

Tensor* op_aten_group_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 4 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    
    int num_groups = 1;
    if (args[1]->dtype == DataType::INT32) num_groups = ((int32_t*)args[1]->data)[0];
    else if (args[1]->dtype == DataType::INT64) num_groups = (int)((int64_t*)args[1]->data)[0];
    
    Tensor* w = args[2];
    Tensor* b = args[3];
    float eps = 1e-5f;
    if (argc > 4 && args[4] && args[4]->data) {
        if (args[4]->dtype == DataType::FLOAT32) eps = ((float*)args[4]->data)[0];
        else if (args[4]->dtype == DataType::FLOAT64) eps = (float)((double*)args[4]->data)[0];
    }

    Tensor* out = create_result_tensor(ctx, x->shape, DataType::FLOAT32);
    if (!out || !out->data) return out;

    if (x->shape.size() < 2) return out; // Protection against 1D tensors

    int N = x->shape[0];
    int C = x->shape[1];
    
    // SAFE spatial calculation (for 2D tensor it is equal to 1)
    int spatial = 1;
    for (size_t i=2; i<x->shape.size(); ++i) {
        if (x->shape[i] > 0) spatial *= x->shape[i];
    }

    int channels_per_group = C / num_groups;
    if (channels_per_group == 0) channels_per_group = 1;
    int group_size = channels_per_group * spatial;

    float* src = (float*)x->data;
    float* dst = (float*)out->data;
    float* w_ptr = (w && w->data) ? (float*)w->data : nullptr;
    float* b_ptr = (b && b->data) ? (float*)b->data : nullptr;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            float* g_src = src + n * (C * spatial) + g * group_size;
            float* g_dst = dst + n * (C * spatial) + g * group_size;

            double sum = 0.0;
            for (int i=0; i<group_size; ++i) sum += g_src[i];
            float mean = (float)(sum / group_size);

            double var_sum = 0.0;
            for (int i=0; i<group_size; ++i) {
                float diff = g_src[i] - mean;
                var_sum += diff * diff;
            }
            float inv_std = 1.0f / sqrtf((float)(var_sum / group_size) + eps);

            for (int c = 0; c < channels_per_group; ++c) {
                int global_c = g * channels_per_group + c;
                float weight = w_ptr ? w_ptr[global_c] : 1.0f;
                float bias = b_ptr ? b_ptr[global_c] : 0.0f;

                for (int s = 0; s < spatial; ++s) {
                    int idx = c * spatial + s;
                    g_dst[idx] = (g_src[idx] - mean) * inv_std * weight + bias;
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_upsample_nearest2d_vec(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* x = args[0];
    
    // If the tensor is not 4D, upsampling does not make sense, return a copy
    if (x->shape.size() != 4) {
        Tensor* out = create_result_tensor(ctx, x->shape, x->dtype);
        if (out && out->data) memcpy(out->data, x->data, x->size);
        return out;
    }

    int N = x->shape[0], C = x->shape[1], in_h = x->shape[2], in_w = x->shape[3];
    int out_h = in_h * 2, out_w = in_w * 2; // Default fallback (UNet)

    if (argc > 1 && args[1] && args[1]->data) {
        if (args[1]->dtype == DataType::INT32) {
            int32_t* p = (int32_t*)args[1]->data;
            out_h = p[0]; out_w = (args[1]->num_elements >= 2) ? p[1] : p[0];
        } else if (args[1]->dtype == DataType::INT64) {
            int64_t* p = (int64_t*)args[1]->data;
            out_h = (int)p[0]; out_w = (int)((args[1]->num_elements >= 2) ? p[1] : p[0]);
        }
    } else if (argc > 2 && args[2] && args[2]->data) {
        if (args[2]->dtype == DataType::FLOAT32) {
            float* p = (float*)args[2]->data;
            out_h = (int)(in_h * p[0]); 
            out_w = (int)(in_w * ((args[2]->num_elements >= 2) ? p[1] : p[0]));
        } else if (args[2]->dtype == DataType::FLOAT64) {
            double* p = (double*)args[2]->data;
            out_h = (int)(in_h * p[0]); 
            out_w = (int)(in_w * ((args[2]->num_elements >= 2) ? p[1] : p[0]));
        }
    }

    // Divide by 0 protection
    if (out_h <= 0) out_h = 1;
    if (out_w <= 0) out_w = 1;
    if (in_h <= 0 || in_w <= 0 || N <= 0 || C <= 0) {
        Tensor* out = create_result_tensor(ctx, {N, C, out_h, out_w}, DataType::FLOAT32);
        if (out && out->data) memset(out->data, 0, out->size);
        return out;
    }

    Tensor* out = create_result_tensor(ctx, {N, C, out_h, out_w}, DataType::FLOAT32);
    if (!out || !out->data) return out;

    float* src = (float*)x->data;
    float* dst = (float*)out->data;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                int ih = (int)floorf((float)oh * (float)in_h / (float)out_h);
                if (ih < 0) ih = 0;
                if (ih >= in_h) ih = in_h - 1;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    int iw = (int)floorf((float)ow * (float)in_w / (float)out_w);
                    if (iw < 0) iw = 0;
                    if (iw >= in_w) iw = in_w - 1;
                    
                    dst[n * C * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = 
                        src[n * C * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                }
            }
        }
    }
    return out;
}

Tensor* op_aten_slice_scatter_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* input = args[0];
    Tensor* src = args[1];

    int dim = 0, start = 0, end = input->shape[0], step = 1;
    if (argc > 2 && args[2] && args[2]->data) dim = ((int32_t*)args[2]->data)[0];
    if (argc > 3 && args[3] && args[3]->data) start = ((int32_t*)args[3]->data)[0];
    if (argc > 4 && args[4] && args[4]->data) end = ((int32_t*)args[4]->data)[0];
    if (argc > 5 && args[5] && args[5]->data) step = ((int32_t*)args[5]->data)[0];

    if (dim < 0) dim += input->shape.size();
    if (start < 0) start += input->shape[dim];
    if (end < 0) end += input->shape[dim];
    if (end > input->shape[dim]) end = input->shape[dim];

    Tensor* out = create_result_tensor(ctx, input->shape, input->dtype);
    if (!out || !out->data) return out;
    memcpy(out->data, input->data, input->size); 

    int inner = 1; for(int i=dim+1; i<(int)input->shape.size(); ++i) inner *= input->shape[i];
    int outer = input->num_elements / (input->shape[dim] * inner);
    size_t elem_size = input->get_element_byte_size();

    uint8_t* out_p = (uint8_t*)out->data;
    uint8_t* src_p = (uint8_t*)src->data;

    for (int o = 0; o < outer; ++o) {
        int src_idx = 0;
        for (int i = start; i < end; i += step) {
            if (src_idx >= src->shape[dim]) break; 
            memcpy(
                out_p + (o * input->shape[dim] + i) * inner * elem_size,
                src_p + (o * src->shape[dim] + src_idx) * inner * elem_size,
                inner * elem_size
            );
            src_idx++;
        }
    }
    return out;
}

Tensor* op_aten_sum_dim_IntList(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* x = args[0];
    
    bool keepdim = false;
    std::vector<int> dims;
    if (argc > 1 && args[1] && args[1]->data && args[1]->dtype == DataType::INT32) {
        int32_t* d = (int32_t*)args[1]->data;
        for (size_t i=0; i<args[1]->num_elements; ++i) dims.push_back(d[i] < 0 ? d[i] + x->shape.size() : d[i]);
    } else {
        for (size_t i=0; i<x->shape.size(); ++i) dims.push_back(i);
    }
    
    if (argc > 2 && args[2] && args[2]->data) {
        if (args[2]->dtype == DataType::BOOL) keepdim = ((uint8_t*)args[2]->data)[0] != 0;
        else if (args[2]->dtype == DataType::INT32) keepdim = ((int32_t*)args[2]->data)[0] != 0;
    }

    std::vector<bool> reduce_mask(x->shape.size(), false);
    for (int d : dims) reduce_mask[d] = true;

    std::vector<int> out_shape;
    for (size_t i=0; i<x->shape.size(); ++i) {
        if (reduce_mask[i]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(x->shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor* out = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!out || !out->data) return out;
    memset(out->data, 0, out->size);

    float* dst = (float*)out->data;
    int ndim = x->shape.size();
    std::vector<int> in_stride(ndim, 1);
    for (int i=ndim-2; i>=0; --i) in_stride[i] = in_stride[i+1] * x->shape[i+1];
    
    int ondim = out_shape.size();
    std::vector<int> os(ondim, 1);
    for (int i=ondim-2; i>=0; --i) os[i] = os[i+1] * out_shape[i+1];

    long long N = (long long)x->num_elements;

    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        size_t tmp = i, out_off = 0;
        int o_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = tmp / in_stride[d];
            tmp %= in_stride[d];
            if (!reduce_mask[d]) out_off += coord * os[o_idx++];
            else if (keepdim) o_idx++;
        }
        
        float val = (x->dtype == DataType::FLOAT32) ? ((float*)x->data)[i] : (float)((int32_t*)x->data)[i];
        #pragma omp atomic
        dst[out_off] += val;
    }
    return out;
}

Tensor* op_aten_scalar_tensor_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0] || !args[0]->data) return nullptr;
    Tensor* out = create_result_tensor(ctx, {1}, args[0]->dtype);
    if (!out || !out->data) return out;
    memcpy(out->data, args[0]->data, out->size);
    return out;
}

// Stub for TRNG functions
Tensor* op_backward_stub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return nullptr;
}

Tensor* op_aten_hardswish_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* grad_out = args[0];
    Tensor* x = args[1];
    Tensor* grad_in = create_result_tensor(ctx, x->shape, x->dtype);
    if (!grad_in || !grad_in->data) return grad_in;

    float* go = (float*)grad_out->data;
    float* xv = (float*)x->data;
    float* gi = (float*)grad_in->data;
    for (size_t i = 0; i < x->num_elements; ++i) {
        if (xv[i] < -3.0f) gi[i] = 0.0f;
        else if (xv[i] >= 3.0f) gi[i] = go[i];
        else gi[i] = go[i] * ((2.0f * xv[i] + 3.0f) / 6.0f);
    }
    return grad_in;
}

Tensor* op_aten_threshold_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* grad_out = args[0];
    Tensor* x = args[1];
    float threshold = (args[2]->dtype == DataType::FLOAT32) ? ((float*)args[2]->data)[0] : (float)((int32_t*)args[2]->data)[0];

    Tensor* grad_in = create_result_tensor(ctx, x->shape, x->dtype);
    if (!grad_in || !grad_in->data) return grad_in;

    float* go = (float*)grad_out->data;
    float* xv = (float*)x->data;
    float* gi = (float*)grad_in->data;
    for (size_t i=0; i<x->num_elements; ++i) {
        gi[i] = (xv[i] > threshold) ? go[i] : 0.0f;
    }
    return grad_in;
}

Tensor* op_aten_select_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 4 || !args[0] || !args[1] || !args[2] || !args[3]) return nullptr;
    Tensor* grad_out = args[0];
    
    std::vector<int> input_sizes;
    if (args[1]->dtype == DataType::INT32) {
        int32_t* p = (int32_t*)args[1]->data;
        for (size_t i = 0; i < args[1]->num_elements; ++i) input_sizes.push_back(p[i]);
    }
    int dim = ((int32_t*)args[2]->data)[0];
    int idx = ((int32_t*)args[3]->data)[0];

    if (dim < 0) dim += input_sizes.size();

    Tensor* grad_in = create_result_tensor(ctx, input_sizes, grad_out->dtype);
    if (!grad_in || !grad_in->data) return grad_in;
    memset(grad_in->data, 0, grad_in->size);

    int inner = 1; for (size_t i = dim + 1; i < input_sizes.size(); ++i) inner *= input_sizes[i];
    int outer = 1; for (int i = 0; i < dim; ++i) outer *= input_sizes[i];

    size_t elem_size = grad_out->get_element_byte_size();
    uint8_t* go_p = (uint8_t*)grad_out->data;
    uint8_t* gi_p = (uint8_t*)grad_in->data;

    #pragma omp parallel for
    for (int o = 0; o < outer; ++o) {
        memcpy(
            gi_p + (o * input_sizes[dim] + idx) * inner * elem_size,
            go_p + o * inner * elem_size,
            inner * elem_size
        );
    }
    return grad_in;
}

Tensor* op_aten_native_layer_norm_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 5 || !args[0] || !args[1] || !args[2] || !args[3] || !args[4]) return nullptr;
    Tensor* grad_out = args[0];
    Tensor* input = args[1];
    std::vector<int> norm_shape;
    if (args[2]->dtype == DataType::INT32) {
        int32_t* p = (int32_t*)args[2]->data;
        for(size_t i=0; i<args[2]->num_elements; ++i) norm_shape.push_back(p[i]);
    }
    Tensor* mean = args[3];
    Tensor* rstd = args[4];
    Tensor* weight = (argc > 5) ? args[5] : nullptr;
    
    Tensor* dX = create_result_tensor(ctx, input->shape, input->dtype);
    if (!dX || !dX->data) return dX;
    
    int inner_dim = 1; for(int d : norm_shape) inner_dim *= d;
    long long outer_dim = (long long)(input->num_elements / inner_dim);
    float* go = (float*)grad_out->data;
    float* in = (float*)input->data;
    float* m = (float*)mean->data;
    float* r = (float*)rstd->data;
    float* w = weight ? (float*)weight->data : nullptr;
    float* dx = (float*)dX->data;

    #pragma omp parallel for
    for (long long o = 0; o < outer_dim; ++o) {
        float sum_dx_norm = 0.0f;
        float sum_dx_norm_x = 0.0f;
        float mv = m[o];
        float rv = r[o];
        
        #pragma omp simd reduction(+:sum_dx_norm, sum_dx_norm_x)
        for (int i = 0; i < inner_dim; ++i) {
            float dx_norm = go[o * inner_dim + i] * (w ? w[i] : 1.0f);
            sum_dx_norm += dx_norm;
            sum_dx_norm_x += dx_norm * in[o * inner_dim + i];
        }
        
        #pragma omp simd
        for (int i = 0; i < inner_dim; ++i) {
            float dx_norm = go[o * inner_dim + i] * (w ? w[i] : 1.0f);
            float term1 = sum_dx_norm / inner_dim;
            float term2 = (in[o * inner_dim + i] - mv) * rv * rv * sum_dx_norm_x / inner_dim;
            dx[o * inner_dim + i] = (dx_norm - term1 - term2) * rv;
        }
    }
    return dX;
}

Tensor* op_aten_gelu_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* grad_out = args[0];
    Tensor* x = args[1];
    Tensor* grad_in = create_result_tensor(ctx, x->shape, x->dtype);
    if (!grad_in || !grad_in->data) return grad_in;

    float* go = (float*)grad_out->data;
    float* xv = (float*)x->data;
    float* gi = (float*)grad_in->data;
    const float sqrt_2 = 1.4142135623730951f;
    const float inv_sqrt_2pi = 0.3989422804014327f;

    for (size_t i=0; i<x->num_elements; ++i) {
        float cdf = 0.5f * (1.0f + std::erff(xv[i] / sqrt_2));
        float pdf = expf(-0.5f * xv[i] * xv[i]) * inv_sqrt_2pi;
        gi[i] = go[i] * (cdf + xv[i] * pdf);
    }
    return grad_in;
}

Tensor* op_aten_tanh_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* grad_out = args[0];
    Tensor* output = args[1];
    Tensor* grad_in = create_result_tensor(ctx, output->shape, output->dtype);
    if (!grad_in || !grad_in->data) return grad_in;
    float* go = (float*)grad_out->data;
    float* out = (float*)output->data;
    float* gi = (float*)grad_in->data;
    for (size_t i=0; i<output->num_elements; ++i) {
        gi[i] = go[i] * (1.0f - out[i] * out[i]);
    }
    return grad_in;
}

Tensor* op_aten__scaled_dot_product_flash_attention_for_cpu_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return nullptr; // FlashAttention CPU backward returns a tuple. Since this is a stub, return nullptr.
}

// ============================================================
// Logical scalar comparisons (eq, lt)
// ============================================================
template <typename Cmp>
Tensor* scalar_cmp_op(NacRuntimeContext* ctx, Tensor* x, Tensor* sc, Cmp cmp) {
    Tensor* out = create_result_tensor(ctx, x->shape, DataType::INT32);
    if (!out || !out->data) return out;
    int32_t* dst = (int32_t*)out->data;
    if (x->dtype == DataType::INT32) {
        int32_t sv = (sc->dtype == DataType::INT32) ? ((int32_t*)sc->data)[0] : (int32_t)((float*)sc->data)[0];
        int32_t* src = (int32_t*)x->data;
        for (size_t i=0; i<x->num_elements; ++i) dst[i] = cmp(src[i], sv) ? 1 : 0;
    } else {
        float sv = (sc->dtype == DataType::FLOAT32) ? ((float*)sc->data)[0] : (float)((int32_t*)sc->data)[0];
        float* src = (float*)x->data;
        for (size_t i=0; i<x->num_elements; ++i) dst[i] = cmp(src[i], sv) ? 1 : 0;
    }
    return out;
}

Tensor* op_aten_eq_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    return scalar_cmp_op(ctx, args[0], args[1], [](auto a, auto b){ return a == b; });
}

Tensor* op_aten_lt_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    return scalar_cmp_op(ctx, args[0], args[1], [](auto a, auto b){ return a < b; });
}

Tensor* op_aten_nll_loss_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* grad_output = args[0];
    Tensor* x = args[1];
    Tensor* target = args[2];
    Tensor* weight = (argc > 3) ? args[3] : nullptr;
    
    int reduction = 1; // mean
    if (argc > 4 && args[4] && args[4]->data) reduction = ((int32_t*)args[4]->data)[0];
    int ignore_index = -100;
    if (argc > 5 && args[5] && args[5]->data) ignore_index = ((int32_t*)args[5]->data)[0];

    Tensor* grad_input = create_result_tensor(ctx, x->shape, x->dtype);
    if (!grad_input || !grad_input->data) return grad_input;
    memset(grad_input->data, 0, grad_input->size);

    float go_val = (grad_output->dtype == DataType::FLOAT32) ? ((float*)grad_output->data)[0] : (float)((int32_t*)grad_output->data)[0];
    float* gi_data = (float*)grad_input->data;
    
    int batch_size = x->shape.size() > 1 ? x->shape[0] : 1;
    int classes = x->shape.size() > 1 ? x->shape[1] : x->shape[0];

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        int t = 0;
        if (target->dtype == DataType::INT32) t = ((int32_t*)target->data)[i];
        else if (target->dtype == DataType::INT64) t = (int)((int64_t*)target->data)[i];
        
        if (t != ignore_index && t >= 0 && t < classes) {
            float w = 1.0f;
            if (weight && weight->data && weight->dtype == DataType::FLOAT32) w = ((float*)weight->data)[t];
            float div = (reduction == 1) ? (float)batch_size : 1.0f;
            gi_data[i * classes + t] = -go_val * w / div;
        }
    }
    return grad_input;
}

Tensor* op_aten__log_softmax_backward_data_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* grad_output = args[0]; Tensor* output = args[1];
    
    if (output->shape.empty()) return nullptr; 

    int dim = ((int32_t*)args[2]->data)[0];
    if (dim < 0) dim += output->shape.size();
    if (dim < 0 || dim >= (int)output->shape.size()) return nullptr;
    
    Tensor* grad_input = create_result_tensor(ctx, output->shape, output->dtype);
    if (!grad_input || !grad_input->data) return grad_input;

    int outer = 1; for(int i=0; i<dim; ++i) outer *= output->shape[i];
    int dim_size = output->shape[dim];
    int inner = 1; for(size_t i=dim+1; i<output->shape.size(); ++i) inner *= output->shape[i];

    float* go_data = (float*)grad_output->data;
    float* out_data = (float*)output->data;
    float* gi_data = (float*)grad_input->data;
    
    bool go_is_scalar = grad_output->shape.empty() || grad_output->num_elements == 1;

    #pragma omp parallel for
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            float sum_val = 0.0f;
            for (int d = 0; d < dim_size; ++d) {
                int idx = o * dim_size * inner + d * inner + i;
                float go_val = go_is_scalar ? go_data[0] : go_data[idx];
                sum_val += go_val * out_data[idx];
            }
            for (int d = 0; d < dim_size; ++d) {
                int idx = o * dim_size * inner + d * inner + i;
                float go_val = go_is_scalar ? go_data[0] : go_data[idx];
                gi_data[idx] = go_val - expf(out_data[idx]) * sum_val;
            }
        }
    }
    return grad_input;
}

Tensor* op_aten_embedding_dense_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* grad_output = args[0]; // [B, S, D]
    Tensor* indices = args[1];     // [B, S]
    
    int num_weights = 1;
    if (args[2]->dtype == DataType::INT32) num_weights = ((int32_t*)args[2]->data)[0];
    else if (args[2]->dtype == DataType::INT64) num_weights = (int)((int64_t*)args[2]->data)[0];
    
    int padding_idx = -1;
    if (argc > 3 && args[3] && args[3]->data) {
        if (args[3]->dtype == DataType::INT32) padding_idx = ((int32_t*)args[3]->data)[0];
        else if (args[3]->dtype == DataType::INT64) padding_idx = (int)((int64_t*)args[3]->data)[0];
    }

    int hidden_size = grad_output->shape.empty() ? 1 : grad_output->shape.back();

    Tensor* grad_weight = create_result_tensor(ctx, {num_weights, hidden_size}, DataType::FLOAT32);
    if (!grad_weight || !grad_weight->data) return grad_weight;
    memset(grad_weight->data, 0, grad_weight->size);

    float* gw_data = (float*)grad_weight->data;
    float* go_data = (float*)grad_output->data;
    
    long long n_tokens = std::min((size_t)indices->num_elements, grad_output->num_elements / hidden_size);
    
    #pragma omp parallel for
    for (long long i = 0; i < n_tokens; ++i) {
        int token_id = 0;
        if (indices->dtype == DataType::INT32) token_id = ((int32_t*)indices->data)[i];
        else if (indices->dtype == DataType::INT64) token_id = (int)((int64_t*)indices->data)[i];
        
        if (token_id >= 0 && token_id < num_weights && token_id != padding_idx) {
            // Atomic addition, since different words in the text can have the same token_id
            for (int d = 0; d < hidden_size; ++d) {
                #pragma omp atomic
                gw_data[token_id * hidden_size + d] += go_data[i * hidden_size + d];
            }
        }
    }
    return grad_weight;
}

Tensor* op_aten_slice_backward_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 6 || !args[0] || !args[1] || !args[2] || !args[3] || !args[4] || !args[5]) return nullptr;
    Tensor* grad_output = args[0];
    
    std::vector<int> input_sizes;
    if (args[1]->dtype == DataType::INT32) {
        int32_t* p = (int32_t*)args[1]->data;
        for (size_t i = 0; i < args[1]->num_elements; ++i) input_sizes.push_back(p[i]);
    } else if (args[1]->dtype == DataType::INT64) {
        int64_t* p = (int64_t*)args[1]->data;
        for (size_t i = 0; i < args[1]->num_elements; ++i) input_sizes.push_back((int)p[i]);
    } else return nullptr;
    
    int dim = 0, start = 0, step = 1;
    if (args[2]->dtype == DataType::INT32) dim = ((int32_t*)args[2]->data)[0];
    else if (args[2]->dtype == DataType::INT64) dim = (int)((int64_t*)args[2]->data)[0];

    if (args[3]->dtype == DataType::INT32) start = ((int32_t*)args[3]->data)[0];
    else if (args[3]->dtype == DataType::INT64) start = (int)((int64_t*)args[3]->data)[0];

    if (args[5]->dtype == DataType::INT32) step = ((int32_t*)args[5]->data)[0];
    else if (args[5]->dtype == DataType::INT64) step = (int)((int64_t*)args[5]->data)[0];

    Tensor* grad_input = create_result_tensor(ctx, input_sizes, grad_output->dtype);
    if (!grad_input || !grad_input->data) return grad_input;
    memset(grad_input->data, 0, grad_input->size);

    if (dim < 0) dim += input_sizes.size();
    if (dim < 0 || dim >= (int)input_sizes.size()) return grad_input;

    int inner = 1; for (size_t i = dim + 1; i < input_sizes.size(); ++i) inner *= input_sizes[i];
    int outer = 1; for (int i = 0; i < dim; ++i) outer *= input_sizes[i];

    size_t elem_size = grad_output->get_element_byte_size();
    uint8_t* go_p = (uint8_t*)grad_output->data;
    uint8_t* gi_p = (uint8_t*)grad_input->data;

    if (grad_output->shape.empty()) {
        if (grad_input->size > 0 && grad_output->size > 0) memcpy(gi_p, go_p, std::min(grad_input->size, grad_output->size));
        return grad_input;
    }

    int s_max = std::min((int)grad_output->shape.back(), (int)grad_output->shape[dim]);
    if (s_max > (input_sizes[dim] - start + step - 1) / step) {
        s_max = (input_sizes[dim] - start + step - 1) / step;
    }

    #pragma omp parallel for
    for (int o = 0; o < outer; ++o) {
        for (int s = 0; s < s_max; ++s) {
            int src_idx = start + s * step;
            if (src_idx >= input_sizes[dim]) continue; 
            
            size_t dst_offset = (o * input_sizes[dim] + src_idx) * inner * elem_size;
            size_t src_offset = (o * grad_output->shape[dim] + s) * inner * elem_size;
            
            if (src_offset + inner * elem_size <= grad_output->size &&
                dst_offset + inner * elem_size <= grad_input->size) {
                memcpy(gi_p + dst_offset, go_p + src_offset, inner * elem_size);
            }
        }
    }
    return grad_input;
}

Tensor* op_aten_lstm_cell_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 5 || !args[0] || !args[1] || !args[2] || !args[3] || !args[4]) return nullptr;
    
    Tensor* x = args[0];
    Tensor* h = args[1];
    Tensor* c = args[2];
    Tensor* w_ih = args[3];
    Tensor* w_hh = args[4];
    Tensor* b_ih = (argc > 5) ? args[5] : nullptr;
    Tensor* b_hh = (argc > 6) ? args[6] : nullptr;

    int B = x->shape[0];
    int I = x->shape[1];
    int H = h->shape[1];

    // Pack H and C into one tensor [2, B, H]
    Tensor* out = create_result_tensor(ctx, {2, B, H}, DataType::FLOAT32);
    if (!out || !out->data) return out;
    
    // Special marker for the op_getitem function: it is a tuple
    out->quant_meta.axis = 127; 

    const float* px = (const float*)x->data;
    const float* ph = (const float*)h->data;
    const float* pc = (const float*)c->data;
    const float* pw_ih = (const float*)w_ih->data;
    const float* pw_hh = (const float*)w_hh->data;
    const float* pb_ih = b_ih && b_ih->data ? (const float*)b_ih->data : nullptr;
    const float* pb_hh = b_hh && b_hh->data ? (const float*)b_hh->data : nullptr;

    float* out_h = (float*)out->data;             // First half
    float* out_c = (float*)out->data + (B * H);   // The second half

    #pragma omp parallel for
    for (int b = 0; b < B; ++b) {
        const float* x_row = px + b * I;
        const float* h_row = ph + b * H;
        const float* c_row = pc + b * H;
        float* hout_row = out_h + b * H;
        float* cout_row = out_c + b * H;

        for (int j = 0; j < H; ++j) {
            float igate = pb_ih ? pb_ih[j] : 0.0f;
            float fgate = pb_ih ? pb_ih[H + j] : 0.0f;
            float ggate = pb_ih ? pb_ih[2*H + j] : 0.0f;
            float ogate = pb_ih ? pb_ih[3*H + j] : 0.0f;

            if (pb_hh) {
                igate += pb_hh[j];
                fgate += pb_hh[H + j];
                ggate += pb_hh[2*H + j];
                ogate += pb_hh[3*H + j];
            }

            // SIMD vectorization of Dot Products (matrix multiplication)
            #pragma omp simd
            for (int k = 0; k < I; ++k) {
                float xv = x_row[k];
                igate += xv * pw_ih[j * I + k];
                fgate += xv * pw_ih[(H + j) * I + k];
                ggate += xv * pw_ih[(2*H + j) * I + k];
                ogate += xv * pw_ih[(3*H + j) * I + k];
            }

            #pragma omp simd
            for (int k = 0; k < H; ++k) {
                float hv = h_row[k];
                igate += hv * pw_hh[j * H + k];
                fgate += hv * pw_hh[(H + j) * H + k];
                ggate += hv * pw_hh[(2*H + j) * H + k];
                ogate += hv * pw_hh[(3*H + j) * H + k];
            }

            // Activations
            igate = 1.0f / (1.0f + expf(-igate)); // Sigmoid
            fgate = 1.0f / (1.0f + expf(-fgate));
            ggate = tanhf(ggate);                 // Tanh
            ogate = 1.0f / (1.0f + expf(-ogate));

            float c_new = fgate * c_row[j] + igate * ggate;
            float h_new = ogate * tanhf(c_new);

            cout_row[j] = c_new;
            hout_row[j] = h_new;
        }
    }
    return out;
}

void register_kernels() {
    g_op_kernels[10] = &op_nac_pass;
    g_op_kernels[11] = &op_nac_add;
    g_op_kernels[12] = &op_nac_sub;
    g_op_kernels[13] = &op_nac_gt;
    g_op_kernels[14] = &op_nac_neg;
    g_op_kernels[15] = &op_nac_where;
    g_op_kernels[16] = &op_nac_clone;
    g_op_kernels[17] = &op_nac_view;
    g_op_kernels[18] = &op_nac_le;
    g_op_kernels[19] = &op_nac_arange;
    g_op_kernels[20] = &op_nac_mul;
    g_op_kernels[21] = &op_nac_div;
    g_op_kernels[22] = &op_aten_matmul_default;
    g_op_kernels[23] = &op_nac_transpose;
    g_op_kernels[24] = &op_nac_zeros;
    g_op_kernels[25] = &op_aten_zeros_like_default;
    g_op_kernels[26] = &op_nac_new_zeros;
    g_op_kernels[27] = &op_nac_ones;
    g_op_kernels[28] = &op_aten_ones_like_default;
    g_op_kernels[29] = &op_nac_new_ones;
    g_op_kernels[30] = &op_aten_full_default;
    g_op_kernels[31] = &op_aten_full_like_default;

    g_kernel_string_map["aten.add.Tensor"] = &op_nac_add;
    g_kernel_string_map["aten.sub.Tensor"] = &op_nac_sub;
    g_kernel_string_map["aten.bmm.default"] = &op_aten_matmul_default;
    g_kernel_string_map["aten.matmul.Tensor"] = &op_aten_matmul_default;
    g_kernel_string_map["aten.where.self"] = &op_nac_where;
    g_kernel_string_map["aten.arange.start"] = &op_nac_arange;
    g_kernel_string_map["aten.arange.default"] = &op_nac_arange;

    g_kernel_string_map["aten.linear.default"] = &op_aten_linear_default;
    g_kernel_string_map["aten.layer_norm.default"] = &op_aten_layer_norm_default;
    g_kernel_string_map["aten.silu.default"] = &op_aten_silu_default;
    g_kernel_string_map["aten.mul.default"] = &op_aten_mul_Tensor;
    g_kernel_string_map["aten.matmul.Tensor"] = &op_aten_mul_Tensor;
    g_kernel_string_map["aten.bmm.default"] = &op_aten_matmul_default;
    g_kernel_string_map["aten.addmm.default"] = &op_aten_addmm_default;
    g_kernel_string_map["aten.full.default"] = &op_aten_full_default;
    g_kernel_string_map["aten.softmax.int"] = &op_aten_softmax_int;
    g_kernel_string_map["aten.relu.default"] = &op_aten_relu_default;
    g_kernel_string_map["aten.gelu.default"] = &op_aten_gelu_default;
    g_kernel_string_map["aten.rsqrt.default"] = &op_aten_rsqrt_default;
    g_kernel_string_map["aten.embedding.default"] = &op_aten_embedding_default;
    g_kernel_string_map["aten.div.Tensor"] = &op_aten_div_Tensor;
    g_kernel_string_map["aten.transpose.int"] = &op_aten_transpose_int;
    g_kernel_string_map["aten.unsqueeze.default"] = &op_aten_unsqueeze_default;
    g_kernel_string_map["aten.sym_size.int"] = &op_aten_sym_size_int;
    g_kernel_string_map["aten.ne.Scalar"] = &op_aten_ne_Scalar;
    g_kernel_string_map["aten._to_copy.default"] = &op_aten__to_copy_default;
    g_kernel_string_map["aten.cumsum.default"] = &op_aten_cumsum_default;
    g_kernel_string_map["aten.type_as.default"] = &op_aten_type_as_default;
    g_kernel_string_map["aten.zeros.default"] = &op_aten_zeros_default;
    g_kernel_string_map["aten.slice.Tensor"] = &op_aten_slice_Tensor;
    g_kernel_string_map["aten.select.int"] = &op_aten_select_int;
    g_kernel_string_map["aten.scaled_dot_product_attention.default"] = &op_aten_scaled_dot_product_attention_default;
    g_kernel_string_map["aten.conv2d.default"] = &op_aten_conv2d_default; 
    g_kernel_string_map["aten._native_batch_norm_legit_no_training.default"] = &op_aten_native_batch_norm_default;
    g_kernel_string_map["aten.max_pool2d.default"] = &op_aten_max_pool2d_default;
    g_kernel_string_map["aten.adaptive_avg_pool2d.default"] = &op_aten_adaptive_avg_pool2d_default;
    g_kernel_string_map["aten.expand.default"] = &op_aten_expand_default;
    g_kernel_string_map["aten.tril.default"] = &op_aten_tril_default;
    g_kernel_string_map["aten.split.Tensor"] = &op_aten_split_Tensor;
    g_kernel_string_map["getitem"] = &op_getitem;
    g_kernel_string_map["aten.mean.dim"] = &op_aten_mean_dim;
    g_kernel_string_map["aten.pow.Tensor_Scalar"] = &op_aten_pow_Tensor_Scalar;
    g_kernel_string_map["aten.copy.default"] = &op_nac_clone;
    g_kernel_string_map["aten.eq.Scalar"] = &op_aten_eq_Scalar;
    g_kernel_string_map["aten.full_like.default"] = &op_aten_full_like_default;
    g_kernel_string_map["aten.log.default"] = &op_aten_log_default;
    g_kernel_string_map["aten.lt.Scalar"] = &op_aten_lt_Scalar;
    g_kernel_string_map["aten.min.other"] = &op_aten_min_other;
    g_kernel_string_map["aten.ones_like.default"] = &op_aten_ones_like_default;
    g_kernel_string_map["aten.permute.default"] = &op_aten_permute_default;
    g_kernel_string_map["aten.slice_scatter.default"] = &op_aten_slice_scatter_default;
    g_kernel_string_map["aten.tanh.default"] = &op_aten_tanh_default;
    g_kernel_string_map["aten.cat.default"] = &op_aten_cat_default;
    g_kernel_string_map["aten.cos.default"] = &op_aten_cos_default;
    g_kernel_string_map["aten.exp.default"] = &op_aten_exp_default;
    g_kernel_string_map["aten.group_norm.default"] = &op_aten_group_norm_default;
    g_kernel_string_map["aten.sin.default"] = &op_aten_sin_default;
    g_kernel_string_map["aten.upsample_nearest2d.vec"] = &op_aten_upsample_nearest2d_vec;
    g_kernel_string_map["aten.lt.Tensor"] = &op_aten_lt_Tensor;
    g_kernel_string_map["aten.sigmoid.default"] = &op_aten_sigmoid_default;
    g_kernel_string_map["aten.hardsigmoid.default"] = &op_aten_hardsigmoid_default;
    g_kernel_string_map["aten.hardswish.default"] = &op_aten_hardswish_default;
    g_kernel_string_map["aten.sum.dim_IntList"] = &op_aten_sum_dim_IntList;
    g_kernel_string_map["aten.scalar_tensor.default"] = &op_aten_scalar_tensor_default;

    // functions for TRNG:
    g_kernel_string_map["aten.nll_loss_backward.default"] = &op_aten_nll_loss_backward_default;
    g_kernel_string_map["aten._log_softmax_backward_data.default"] = &op_aten__log_softmax_backward_data_default;
    g_kernel_string_map["aten.embedding_dense_backward.default"] = &op_aten_embedding_dense_backward_default;
    g_kernel_string_map["aten.slice_backward.default"] = &op_aten_slice_backward_default;
    g_kernel_string_map["aten.nll_loss2d_backward.default"] = &op_aten_nll_loss_backward_default; // Aliasing
    g_kernel_string_map["aten._softmax_backward_data.default"] = &op_aten__log_softmax_backward_data_default; // Aliasing
    
    g_kernel_string_map["aten.hardswish_backward.default"] = &op_aten_hardswish_backward_default;
    g_kernel_string_map["aten.threshold_backward.default"] = &op_aten_threshold_backward_default;
    g_kernel_string_map["aten.select_backward.default"] = &op_aten_select_backward_default;
    g_kernel_string_map["aten.native_layer_norm_backward.default"] = &op_aten_native_layer_norm_backward_default;
    g_kernel_string_map["aten.gelu_backward.default"] = &op_aten_gelu_backward_default;
    g_kernel_string_map["aten._scaled_dot_product_flash_attention_for_cpu_backward.default"] = &op_aten__scaled_dot_product_flash_attention_for_cpu_backward_default;
    g_kernel_string_map["aten.tanh_backward.default"] = &op_aten_tanh_backward_default;

    g_kernel_string_map["aten.lstm_cell.default"] = &op_aten_lstm_cell_default;
    g_kernel_string_map["aten.max_pool2d_with_indices_backward.default"] = &op_backward_stub; // For backward pass of 1D CNN
}