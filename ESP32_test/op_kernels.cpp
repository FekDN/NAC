// op_kernels.cpp

#include "op_kernels.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits> // Для std::numeric_limits

// Глобальные карты для регистрации ядер
std::map<std::string, KernelFunc> g_kernel_string_map;
std::map<uint8_t, KernelFunc> g_op_kernels;

// Вспомогательная структура для парсинга 2D параметров (stride, padding и т.д.)
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

// Вспомогательная функция для создания тензора-результата
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
            ESP_LOGE("KERNELS", "OOM for result tensor! Shape[0]=%d, Size=%d", shape.empty() ? -1 : shape[0], result->size);
            ctx->tensor_pool.release(result);
            return nullptr;
        }
    } else {
        result->data = nullptr;
    }
    return result;
}

// --- ЯДРА ---

// ID 10: nac.pass
Tensor* op_nac_pass(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Эта функция НЕ ДОЛЖНА создавать копию.
    // Она просто "пробрасывает" первый входной тензор дальше по графу.
    // Система управления памятью (mmap) позаботится об освобождении.
    if (argc > 0 && args[0]) {
        return args[0]; // Просто возвращаем указатель на первый аргумент
    }
    return nullptr; // Если аргументов нет, возвращаем null
}

// ============================================================
// nac.neg
// ============================================================
Tensor* op_nac_neg(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];

    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;

    size_t N = x->num_elements;

    if (x->dtype == DataType::FLOAT32) {
        float* src = (float*)x->data;
        float* dst = (float*)result->data;
        for (size_t i = 0; i < N; ++i)
            dst[i] = -src[i];
    }
    else if (x->dtype == DataType::INT32) {
        int* src = (int*)x->data;
        int* dst = (int*)result->data;
        for (size_t i = 0; i < N; ++i)
            dst[i] = -src[i];
    }
    else if (x->dtype == DataType::INT8) {
        int8_t* src = (int8_t*)x->data;
        int8_t* dst = (int8_t*)result->data;
        for (size_t i = 0; i < N; ++i)
            dst[i] = -src[i];
    }

    return result;
}

// Вспомогательная шаблонная функция для сравнений, чтобы избежать дублирования кода
template <typename Op>
Tensor* op_nac_comparison(NacRuntimeContext* ctx, Tensor** args, size_t argc, Op op) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;

    Tensor* a = args[0];
    Tensor* b = args[1];

    Tensor* result = create_result_tensor(ctx, a->shape, DataType::INT32);
    if (!result) return nullptr;

    size_t N = result->num_elements;
    int* out = (int*)result->data;

    // Оптимизация: выносим проверку типов за пределы цикла
    if (a->dtype == DataType::FLOAT32 && b->dtype == DataType::FLOAT32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        for (size_t i = 0; i < N; ++i) {
            out[i] = op(a_data[i % a->num_elements], b_data[i % b->num_elements]);
        }
    } else if (a->dtype == DataType::INT32 && b->dtype == DataType::INT32) {
        int* a_data = (int*)a->data;
        int* b_data = (int*)b->data;
        for (size_t i = 0; i < N; ++i) {
            out[i] = op((float)a_data[i % a->num_elements], (float)b_data[i % b->num_elements]);
        }
    } else { // Общий, но более медленный случай с приведением типов
        for (size_t i = 0; i < N; ++i) {
            float av = 0.f, bv = 0.f;
            // Код извлечения значений оставлен для совместимости, но основные случаи покрыты выше
            if (a->dtype == DataType::FLOAT32) av = ((float*)a->data)[i % a->num_elements];
            else if (a->dtype == DataType::INT32) av = (float)((int*)a->data)[i % a->num_elements];
            else if (a->dtype == DataType::INT8) av = (float)((int8_t*)a->data)[i % a->num_elements];

            if (b->dtype == DataType::FLOAT32) bv = ((float*)b->data)[i % b->num_elements];
            else if (b->dtype == DataType::INT32) bv = (float)((int*)b->data)[i % b->num_elements];
            else if (b->dtype == DataType::INT8) bv = (float)((int8_t*)b->data)[i % b->num_elements];

            out[i] = op(av, bv);
        }
    }
    return result;
}

//ID 13: nac.gt 
Tensor* op_nac_gt(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_comparison(ctx, args, argc, [](float a, float b) { return a > b ? 1 : 0; });
}

//ID 18: nac.le
Tensor* op_nac_le(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return op_nac_comparison(ctx, args, argc, [](float a, float b) { return a <= b ? 1 : 0; });
}

// ID 17: nac.view
Tensor* op_nac_view(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* input = args[0];
    std::vector<int> new_shape;
    size_t c_idx = 1; 
    if (!ins.C.empty() && ins.C[0] > 0) {
        for (int i = 0; i < ins.C[0]; ++i) {
            if (c_idx >= ins.C.size()) break;
            uint16_t const_id = ins.C[c_idx++];
            Tensor* const_tensor = (const_id < ctx->constants.size()) ? ctx->constants[const_id].get() : nullptr;
            if (const_tensor && const_tensor->data) {
                new_shape.push_back(*(static_cast<int*>(const_tensor->data)));
            }
        }
    }
    
    long long num_elements = input->num_elements;
    long long product = 1;
    int minus_one_idx = -1;
    for(int i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) minus_one_idx = i;
        else if (new_shape[i] > 0) product *= new_shape[i];
    }
    if (minus_one_idx != -1 && product > 0) {
        new_shape[minus_one_idx] = num_elements / product;
    }

    Tensor* result = create_result_tensor(ctx, new_shape, input->dtype);
    if (result && result->num_elements == input->num_elements) {
        memcpy(result->data, input->data, result->size);
    } else if (result) {
        ESP_LOGE("op_nac_view", "Size mismatch after reshape!");
        ctx->tensor_pool.release(result);
        return nullptr;
    }
    return result;
}

// ============================================================
// nac.clone
// ============================================================
Tensor* op_nac_clone(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];

    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;

    if (x->size > 0 && x->data)
        memcpy(result->data, x->data, x->size);

    return result;
}

// ============================================================
// nac.where
// ============================================================
Tensor* op_nac_where(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;

    Tensor* cond = args[0];
    Tensor* x = args[1];
    Tensor* y = args[2];

    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;

    size_t N = result->num_elements;

    float* x_data = (float*)x->data;
    float* y_data = (float*)y->data;
    float* out = (float*)result->data;

    // condition может быть int / float
    for (size_t i = 0; i < N; ++i) {
        bool c;
        if (cond->dtype == DataType::INT32)
            c = ((int*)cond->data)[i % cond->num_elements] != 0;
        else
            c = ((float*)cond->data)[i % cond->num_elements] != 0.0f;

        out[i] = c ? x_data[i % x->num_elements]
                   : y_data[i % y->num_elements];
    }
    return result;
}

// ============================================================
// nac.arange
// ============================================================
Tensor* op_nac_arange(
    NacRuntimeContext* ctx,
    const ParsedInstruction& ins,
    Tensor** args,
    size_t argc
) {
    float start = 0.0f;
    float end   = 0.0f;
    float step  = 1.0f;

    // -------------------------------
    // Parse arguments
    // -------------------------------
    if (argc == 1 && args[0]) {
        if (args[0]->dtype == DataType::FLOAT32)
            end = ((float*)args[0]->data)[0];
        else if (args[0]->dtype == DataType::INT32)
            end = (float)((int*)args[0]->data)[0];
        else if (args[0]->dtype == DataType::INT8)
            end = (float)((int8_t*)args[0]->data)[0];
        else
            return nullptr;
    }
    else if (argc >= 2 && args[0] && args[1]) {
        if (args[0]->dtype == DataType::FLOAT32)
            start = ((float*)args[0]->data)[0];
        else if (args[0]->dtype == DataType::INT32)
            start = (float)((int*)args[0]->data)[0];
        else if (args[0]->dtype == DataType::INT8)
            start = (float)((int8_t*)args[0]->data)[0];

        if (args[1]->dtype == DataType::FLOAT32)
            end = ((float*)args[1]->data)[0];
        else if (args[1]->dtype == DataType::INT32)
            end = (float)((int*)args[1]->data)[0];
        else if (args[1]->dtype == DataType::INT8)
            end = (float)((int8_t*)args[1]->data)[0];

        if (argc >= 3 && args[2]) {
            if (args[2]->dtype == DataType::FLOAT32)
                step = ((float*)args[2]->data)[0];
            else if (args[2]->dtype == DataType::INT32)
                step = (float)((int*)args[2]->data)[0];
            else if (args[2]->dtype == DataType::INT8)
                step = (float)((int8_t*)args[2]->data)[0];
        }
    }
    else {
        return nullptr;
    }

    if (step == 0.0f) return nullptr;

    // -------------------------------
    // Compute length
    // -------------------------------
    int length = (int)((end - start) / step);
    if (length < 0) length = 0;

    // -------------------------------
    // Output dtype
    // -------------------------------
    DataType out_dtype =
        (start == (int)start && end == (int)end && step == (int)step)
        ? DataType::INT32
        : DataType::FLOAT32;

    // -------------------------------
    // Shape (1D)
    // -------------------------------
    std::vector<int> shape = { length };

    // -------------------------------
    // Create result tensor (CORRECT)
    // -------------------------------
    Tensor* result = create_result_tensor(
        ctx,
        shape,
        out_dtype
    );
    if (!result) return nullptr;

    // Fill tensor
    if (out_dtype == DataType::FLOAT32) {
        float* dst = (float*)result->data;
        for (int i = 0; i < length; ++i)
            dst[i] = start + step * i;
    }
    else {
        int* dst = (int*)result->data;
        for (int i = 0; i < length; ++i)
            dst[i] = (int)(start + step * i);
    }

    return result;
}

// aten.linear_default
Tensor* op_aten_linear_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x = args[0]; // Shape: (..., M, K)
    Tensor* w = args[1]; // Shape: (N, K)
    Tensor* b = (argc > 2 && args[2]) ? args[2] : nullptr;

    int M = x->shape.size() > 1 ? x->shape[x->shape.size() - 2] : 1;
    int K = x->shape.back();
    int N = w->shape[0];
    
    // Вычисляем префикс формы для батчей > 1
    std::vector<int> out_shape;
    if (x->shape.size() > 2) {
        out_shape.insert(out_shape.end(), x->shape.begin(), x->shape.end() - 2);
    }
    out_shape.push_back(M);
    out_shape.push_back(N);

    Tensor* result = create_result_tensor(ctx, out_shape, DataType::FLOAT32);
    if (!result) return nullptr;
    
    float* x_data = static_cast<float*>(x->data);
    float* w_data = static_cast<float*>(w->data);
    float* res_data = static_cast<float*>(result->data);
    
    int batch_size = x->num_elements / (M * K);

    // Оптимизация: Инициализируем выходной тензор с bias (или нулями)
    float* b_data = b ? static_cast<float*>(b->data) : nullptr;
    for(int batch = 0; batch < batch_size; ++batch) {
        for (int m = 0; m < M; ++m) {
            float* res_row = res_data + (batch * M + m) * N;
            if (b_data) {
                memcpy(res_row, b_data, N * sizeof(float));
            } else {
                memset(res_row, 0, N * sizeof(float));
            }
        }
    }

    // Оптимизация: Изменен порядок циклов на M, N, K для лучшей локальности кэша
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                const float* w_row = w_data + k; // Указатель на k-ый элемент в каждой строке весов
                const float x_val = x_data[(batch * M + m) * K + k];
                float* res_row = res_data + (batch * M + m) * N;
                // Умножаем x_val на всю строку весов (транспонированную)
                for (int n = 0; n < N; ++n) {
                     res_row[n] += x_val * w_data[n * K + k]; // w_data[n][k]
                }
            }
        }
    }
    
    return result;
}

// aten.layer_norm_default
Tensor* op_aten_layer_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;
    Tensor* x = args[0];
    Tensor* w = args[1];
    Tensor* b = args[2];
    float eps = 1e-5f; // Hardcoded, can be taken from args[3] if present
    int last_dim_size = x->shape.back();
    int outer_dims_size = x->num_elements / last_dim_size;
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* w_data = static_cast<float*>(w->data);
    float* b_data = static_cast<float*>(b->data);
    float* res_data = static_cast<float*>(result->data);
    for (int i = 0; i < outer_dims_size; ++i) {
        float* row_start = x_data + i * last_dim_size;
        float sum = 0.0f;
        for (int j = 0; j < last_dim_size; ++j) sum += row_start[j];
        float mean = sum / last_dim_size;
        float var_sum = 0.0f;
        for (int j = 0; j < last_dim_size; ++j) {
            float diff = row_start[j] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / sqrtf(var_sum / last_dim_size + eps);
        for (int j = 0; j < last_dim_size; ++j) {
            res_data[i * last_dim_size + j] = ((row_start[j] - mean) * inv_std) * w_data[j] + b_data[j];
        }
    }
    return result;
}

// aten.softmax_int
Tensor* op_aten_softmax_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    Tensor* dim_tensor = args[1];
    int dim = *(static_cast<int*>(dim_tensor->data));
    if (dim < 0) dim += x->shape.size();
    if (dim != x->shape.size() - 1) {
        ESP_LOGE("op_aten_softmax", "Softmax for non-last dimension is not implemented!");
        return nullptr;
    }
    int last_dim_size = x->shape.back();
    int outer_dims_size = x->num_elements / last_dim_size;
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for (int i = 0; i < outer_dims_size; ++i) {
        float* row_start = x_data + i * last_dim_size;
        float* res_row_start = res_data + i * last_dim_size;
        float max_val = row_start[0];
        for (int j = 1; j < last_dim_size; ++j) {
            if (row_start[j] > max_val) max_val = row_start[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < last_dim_size; ++j) {
            float val = expf(row_start[j] - max_val);
            res_row_start[j] = val;
            sum_exp += val;
        }
        for (int j = 0; j < last_dim_size; ++j) res_row_start[j] /= sum_exp;
    }
    return result;
}


template<typename Func>
Tensor* elementwise_binary_op(NacRuntimeContext* ctx, Tensor** args, size_t argc, Func op) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* a = args[0];
    Tensor* b = args[1];
    
    // TODO: Полная реализация broadcasting
    Tensor* result = create_result_tensor(ctx, a->shape, a->dtype); // Упрощенно
    if (!result) return nullptr;

    float* a_data = static_cast<float*>(a->data);
    float* b_data = static_cast<float*>(b->data);
    float* res_data = static_cast<float*>(result->data);

    if (a->num_elements > b->num_elements && b->num_elements > 0) {
        size_t inner_dim = b->num_elements;
        for (size_t i = 0; i < a->num_elements; ++i) res_data[i] = op(a_data[i], b_data[i % inner_dim]);
    } else if (b->num_elements > a->num_elements && a->num_elements > 0) {
        size_t inner_dim = a->num_elements;
        for (size_t i = 0; i < b->num_elements; ++i) res_data[i] = op(a_data[i % inner_dim], b_data[i]);
    } else {
        for (size_t i = 0; i < result->num_elements; ++i) res_data[i] = op(a_data[i], b_data[i]);
    }
    return result;
}

Tensor* op_nac_add(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return elementwise_binary_op(ctx, args, argc, [](float a, float b){ return a + b; });
}
Tensor* op_nac_sub(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return elementwise_binary_op(ctx, args, argc, [](float a, float b){ return a - b; });
}
Tensor* op_aten_mul_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return elementwise_binary_op(ctx, args, argc, [](float a, float b){ return a * b; });
}
Tensor* op_aten_div_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    return elementwise_binary_op(ctx, args, argc, [](float a, float b){ return a / b; });
}

Tensor* op_aten_relu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for(size_t i = 0; i < x->num_elements; ++i) {
        res_data[i] = std::max(0.0f, x_data[i]);
    }
    return result;
}

Tensor* op_aten_silu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for (size_t i = 0; i < x->num_elements; ++i) {
        float val = x_data[i];
        res_data[i] = val / (1.0f + expf(-val));
    }
    return result;
}

Tensor* op_aten_gelu_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    const float k0 = 0.79788456f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (size_t i = 0; i < x->num_elements; ++i) {
        float val = x_data[i];
        res_data[i] = 0.5f * val * (1.0f + tanhf(k0 * (val + k1 * val * val * val)));
    }
    return result;
}

Tensor* op_aten_rsqrt_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];
    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;
    float* x_data = static_cast<float*>(x->data);
    float* res_data = static_cast<float*>(result->data);
    for(size_t i=0; i<x->num_elements; ++i) {
        res_data[i] = 1.0f / sqrtf(x_data[i]);
    }
    return result;
}

// aten.transpose.int
Tensor* op_aten_transpose_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 3 || !args[0] || !args[0]->data || !args[1] || !args[1]->data || !args[2] || !args[2]->data) return nullptr;

    Tensor* x = args[0];
    int dim0 = *(int*)args[1]->data;
    int dim1 = *(int*)args[2]->data;

    // ... остальной код функции без изменений ...
    int ndim = x->shape.size();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 >= ndim || dim1 >= ndim) return nullptr;

    std::vector<int> out_shape = x->shape;
    std::swap(out_shape[dim0], out_shape[dim1]);

    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result) return nullptr;

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
        for (int d = 0; d < ndim; ++d)
            out_off += idx[d] * out_stride[d];

        dst[out_off] = src[i];
    }
    return result;
}

// aten.unsqueeze.default
Tensor* op_aten_unsqueeze_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    int dim = *(int*)args[1]->data;
    // ... остальной код функции без изменений ...
    int ndim = x->shape.size();
    if (dim < 0) dim += ndim + 1;

    std::vector<int> out_shape = x->shape;
    out_shape.insert(out_shape.begin() + dim, 1);

    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result) return nullptr;

    memcpy(result->data, x->data, x->get_byte_size());
    return result;
}

// aten.sym_size.int
Tensor* op_aten_sym_size_int(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 2 || !args[0] || !args[1] || !args[1]->data) return nullptr;
    Tensor* x = args[0];
    int dim = *(int*)args[1]->data;
    if (dim < 0) dim += x->shape.size();
    
    if (dim >= x->shape.size()) { // Дополнительная проверка на выход за границы
        ESP_LOGE("sym_size_int", "Dimension %d is out of bounds for shape size %d", dim, x->shape.size());
        return nullptr;
    }

    // Эта функция создает тензор-скаляр, поэтому она не зависит от x->data
    Tensor* result = create_result_tensor(ctx, {}, DataType::INT32);
    if (!result) return nullptr;
    
    // create_result_tensor для скаляра не выделяет data, нужно сделать это вручную
    if (!result->data) {
        result->data = alloc_fast(sizeof(int));
        if(!result->data) {
            ctx->tensor_pool.release(result);
            return nullptr;
        }
    }
    
    *(int*)result->data = x->shape[dim];
    result->num_elements = 1;
    result->size = sizeof(int);
    return result;
}

// aten.ne.Scalar
Tensor* op_aten_ne_Scalar(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;

    Tensor* x = args[0];
    float scalar = *(float*)args[1]->data;
    // ... остальной код функции без изменений ...
    Tensor* result = create_result_tensor(ctx, x->shape, DataType::INT32);
    if (!result) return nullptr;

    float* src = (float*)x->data;
    int* dst = (int*)result->data;

    for (int i = 0; i < x->num_elements; ++i)
        dst[i] = (src[i] != scalar);

    return result;
}

// aten._to_copy.default
Tensor* op_aten_to_copy_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 1 || !args[0]) return nullptr;
    Tensor* x = args[0];

    DataType target = x->dtype;
    if (argc > 1 && args[1])
        target = (DataType)*(int*)args[1]->data;

    Tensor* result = create_result_tensor(ctx, x->shape, target);
    if (!result) return nullptr;

    if (x->dtype == target) {
        memcpy(result->data, x->data, x->size);
        return result;
    }

    // float -> int
    if (x->dtype == DataType::FLOAT32 && target == DataType::INT32) {
        float* src = (float*)x->data;
        int* dst = (int*)result->data;
        for (int i = 0; i < x->num_elements; ++i) dst[i] = (int)src[i];
    }
    return result;
}

// aten.cumsum.default (1D or last-dim)
Tensor* op_aten_cumsum_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;

    Tensor* x = args[0];
    int dim = *(int*)args[1]->data;
    if (dim < 0) dim += x->shape.size();

    Tensor* result = create_result_tensor(ctx, x->shape, x->dtype);
    if (!result) return nullptr;

    int stride = 1;
    for (int i = dim + 1; i < x->shape.size(); ++i)
        stride *= x->shape[i];

    int block = x->shape[dim] * stride;
    int outer = x->num_elements / block;

    float* src = (float*)x->data;
    float* dst = (float*)result->data;

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
    return result;
}

// aten.type_as.default
Tensor* op_aten_type_as_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 2 || !args[0] || !args[1]) return nullptr;
    Tensor* x = args[0];
    Tensor* ref = args[1];

    Tensor* result = create_result_tensor(ctx, x->shape, ref->dtype);
    if (!result) return nullptr;

    memcpy(result->data, x->data, std::min(x->size, result->size));
    return result;
}

// aten.zeros.default
Tensor* op_aten_zeros_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    std::vector<int> shape;
    for (size_t i = 0; i < argc; ++i)
        shape.push_back(*(int*)args[i]->data);

    Tensor* result = create_result_tensor(ctx, shape, DataType::FLOAT32);
    if (!result) return nullptr;

    memset(result->data, 0, result->size);
    return result;
}

// aten.slice.Tensor (basic)
Tensor* op_aten_slice_Tensor(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 4 || !args[0]) return nullptr;
    Tensor* x = args[0];
    int dim = *(int*)args[1]->data;
    int start = *(int*)args[2]->data;
    int end = *(int*)args[3]->data;

    if (dim < 0) dim += x->shape.size();
    if (end < 0 || end > x->shape[dim]) end = x->shape[dim];

    std::vector<int> out_shape = x->shape;
    out_shape[dim] = end - start;

    Tensor* result = create_result_tensor(ctx, out_shape, x->dtype);
    if (!result) return nullptr;

    int inner = 1;
    for (int i = dim + 1; i < x->shape.size(); ++i)
        inner *= x->shape[i];

    int outer = x->num_elements / (x->shape[dim] * inner);

    float* src = (float*)x->data;
    float* dst = (float*)result->data;

    for (int o = 0; o < outer; ++o) {
        memcpy(
            dst + o * out_shape[dim] * inner,
            src + (o * x->shape[dim] + start) * inner,
            out_shape[dim] * inner * sizeof(float)
        );
    }
    return result;
}

// aten.expand.default (broadcast-only)
Tensor* op_aten_embedding_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Усиленная проверка
    if (argc < 2 || !args[0] || !args[0]->data || !args[1] || !args[1]->data) return nullptr;
    Tensor* weight = args[0];
    Tensor* indices = args[1];
    
    if (weight->shape.size() < 2 || indices->shape.empty()) return nullptr;

    int vocab_size = weight->shape[0];
    int hidden_size = weight->shape[1];
    int seq_len = indices->shape[0];
    // ... остальной код функции без изменений ...
    std::vector<int> out_shape = {seq_len, hidden_size};
    Tensor* result = create_result_tensor(ctx, out_shape, weight->dtype);
    if(!result) return nullptr;

    float* w_data = static_cast<float*>(weight->data);
    int* idx_data = static_cast<int*>(indices->data);
    float* res_data = static_cast<float*>(result->data);


    for(int i=0; i < seq_len; ++i) {
        int token_id = idx_data[i];
        if (token_id >= 0 && token_id < vocab_size) {
            memcpy(res_data + i * hidden_size, w_data + token_id * hidden_size, hidden_size * sizeof(float));
        } else {
            // Если ID вне словаря, лучше заполнить нулями, чем оставлять мусор
            memset(res_data + i * hidden_size, 0, hidden_size * sizeof(float));
        }
    }
    return result;
}

// aten.scaled_dot_product_attention.default
Tensor* op_aten_scaled_dot_product_attention_default(
    NacRuntimeContext* ctx,
    const ParsedInstruction& ins,
    Tensor** args,
    size_t argc
) {
    if (argc < 3 || !args[0] || !args[1] || !args[2]) return nullptr;

    Tensor* Q = args[0];  // (..., T, D)
    Tensor* K = args[1];  // (..., T_k, D)
    Tensor* V = args[2];  // (..., T_k, D_v)

    bool is_causal = (argc >= 6 && args[5] && (*(int*)args[5]->data) != 0);

    int ndim = Q->shape.size();
    if (ndim < 2) return nullptr;
    int D = Q->shape.back();
    int T = Q->shape[ndim - 2];
    int T_k = K->shape[ndim-2];
    int outer = Q->num_elements / (T * D);

    Tensor* output = create_result_tensor(ctx, Q->shape, Q->dtype);
    if (!output) return nullptr;

    float* q_data = (float*)Q->data;
    float* k_data = (float*)K->data;
    float* v_data = (float*)V->data;
    float* out_data = (float*)output->data;

    float scale = 1.0f / sqrtf((float)D);

    // Оптимизация: избегаем аллокации T x T_k матрицы.
    // Вместо этого используем "потоковый" онлайн-softmax.
    // Это радикально снижает потребление памяти и ускоряет вычисления.
    float* row_scores = (float*)alloc_fast(T_k * sizeof(float));
    if (!row_scores) {
        ESP_LOGE("SDPA", "OOM for row_scores buffer!");
        ctx->tensor_pool.release(output);
        return nullptr;
    }

    for (int o = 0; o < outer; ++o) {
        const float* q_batch = q_data + o * T * D;
        const float* k_batch = k_data + o * T_k * D;
        const float* v_batch = v_data + o * T_k * D;
        float* out_batch = out_data + o * T * D;

        for (int i = 0; i < T; ++i) { // Итерация по каждому вектору запроса q_i
            const float* q_i = q_batch + i * D;
            float* out_i = out_batch + i * D;

            // 1. Вычисляем QK^T для текущей строки i
            float max_score = -1e9f;
            for (int j = 0; j < T_k; ++j) {
                if (is_causal && j > i) {
                    row_scores[j] = -1e9f;
                } else {
                    const float* k_j = k_batch + j * D;
                    float score = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        score += q_i[d] * k_j[d];
                    }
                    row_scores[j] = score * scale;
                }
                if (row_scores[j] > max_score) {
                    max_score = row_scores[j];
                }
            }

            // 2. Стабильный Softmax для строки
            float sum_exp = 0.0f;
            for (int j = 0; j < T_k; ++j) {
                float val = expf(row_scores[j] - max_score);
                row_scores[j] = val; // Переиспользуем буфер для хранения exp-значений
                sum_exp += val;
            }
            float inv_sum_exp = 1.0f / sum_exp;
            for (int j = 0; j < T_k; ++j) {
                row_scores[j] *= inv_sum_exp; // Нормализованные веса внимания
            }
            
            // 3. Умножаем веса на V
            memset(out_i, 0, D * sizeof(float));
            for (int j = 0; j < T_k; ++j) {
                float weight = row_scores[j];
                const float* v_j = v_batch + j * D;
                for (int d = 0; d < D; ++d) {
                    out_i[d] += weight * v_j[d];
                }
            }
        }
    }

    free(row_scores);
    return output;
}

// aten.conv2d.default
Tensor* op_aten_conv2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    if (argc < 7 || !args[0] || !args[1] || !args[3] || !args[4] || !args[5] || !args[6]) return nullptr;

    Tensor* input = args[0];    // (N, C_in, H_in, W_in)
    Tensor* weight = args[1];   // (C_out, C_in/groups, kH, kW)
    Tensor* bias = args[2];     // (C_out)
    Param2D stride(args[3]);
    Param2D padding(args[4]);
    Param2D dilation(args[5]);
    int groups = *(int*)args[6]->data;

    int N = input->shape[0]; int C_in = input->shape[1]; int H_in = input->shape[2]; int W_in = input->shape[3];
    int C_out = weight->shape[0]; int kH = weight->shape[2]; int kW = weight->shape[3];

    int H_out = (H_in + 2 * padding.h - dilation.h * (kH - 1) - 1) / stride.h + 1;
    int W_out = (W_in + 2 * padding.w - dilation.w * (kW - 1) - 1) / stride.w + 1;

    std::vector<int> out_shape = {N, C_out, H_out, W_out};
    Tensor* result = create_result_tensor(ctx, out_shape, input->dtype);
    if (!result) return nullptr;

    float* in_data = (float*)input->data;
    float* w_data = (float*)weight->data;
    float* b_data = bias ? (float*)bias->data : nullptr;
    float* out_data = (float*)result->data;

    int C_in_per_group = C_in / groups;
    int C_out_per_group = C_out / groups;

    // Оптимизация: меняем порядок циклов для улучшения локальности кэша
    // N -> C_out -> H_out -> W_out ...
    // Это позволяет держать веса для одного выходного канала в кэше дольше.
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float* out_ptr_base = out_data + (n * C_out + c_out) * H_out * W_out;
            int g = c_out / C_out_per_group;
            int c_out_g = c_out % C_out_per_group;

            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    float sum = b_data ? b_data[c_out] : 0.0f;
                    for (int c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
                        int c_in = g * C_in_per_group + c_in_g;
                        const float* w_ptr = w_data + (c_out * C_in_per_group + c_in_g) * kH * kW;

                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int h_in = h_out * stride.h - padding.h + kh * dilation.h;
                                int w_in = w_out * stride.w - padding.w + kw * dilation.w;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    int in_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                                    sum += in_data[in_idx] * w_ptr[kh * kW + kw];
                                }
                            }
                        }
                    }
                    out_ptr_base[h_out * W_out + w_out] = sum;
                }
            }
        }
    }
    return result;
}

// aten._native_batch_norm_legit_no_training.default
Tensor* op_aten_native_batch_norm_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Аргументы: input, weight, bias, running_mean, running_var, training, momentum, eps
    // В режиме inference (no_training) training, momentum не используются
    if (argc < 8 || !args[0] || !args[3] || !args[4] || !args[7]) return nullptr;

    Tensor* input = args[0];        // (N, C, H, W)
    Tensor* weight = args[1];       // (gamma) (C) - может быть nullptr
    Tensor* bias = args[2];         // (beta) (C) - может быть nullptr
    Tensor* running_mean = args[3]; // (C)
    Tensor* running_var = args[4];  // (C)
    float eps = *(float*)args[7]->data;

    if (input->shape.size() < 2) return nullptr; // Должен быть хотя бы (N,C)

    int N = input->shape[0];
    int C = input->shape[1];
    int spatial_dim = input->num_elements / (N * C);

    Tensor* result = create_result_tensor(ctx, input->shape, input->dtype);
    if (!result) return nullptr;

    float* in_data = (float*)input->data;
    float* w_data = weight ? (float*)weight->data : nullptr;
    float* b_data = bias ? (float*)bias->data : nullptr;
    float* mean_data = (float*)running_mean->data;
    float* var_data = (float*)running_var->data;
    float* out_data = (float*)result->data;

    // Оптимизация: предвычисляем множители для каждого канала
    std::vector<float> a(C), b(C);
    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / sqrtf(var_data[c] + eps);
        float gamma = w_data ? w_data[c] : 1.0f;
        float beta = b_data ? b_data[c] : 0.0f;
        a[c] = gamma * inv_std;
        b[c] = beta - mean_data[c] * a[c];
    }
    
    // Применяем y = a*x + b
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float scale = a[c];
            float shift = b[c];
            int offset = (n * C + c) * spatial_dim;
            for (int i = 0; i < spatial_dim; ++i) {
                out_data[offset + i] = in_data[offset + i] * scale + shift;
            }
        }
    }
    return result;
}

// aten.max_pool2d.default
Tensor* op_aten_max_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Аргументы: input, kernel_size, stride, padding, dilation
    if (argc < 5 || !args[0] || !args[1] || !args[2] || !args[3] || !args[4]) return nullptr;

    Tensor* input = args[0];
    Param2D kernel_size(args[1]);
    Param2D stride(args[2]);
    Param2D padding(args[3]);
    Param2D dilation(args[4]);

    if (input->shape.size() != 4) return nullptr;

    int N = input->shape[0];
    int C = input->shape[1];
    int H_in = input->shape[2];
    int W_in = input->shape[3];

    int H_out = (H_in + 2 * padding.h - dilation.h * (kernel_size.h - 1) - 1) / stride.h + 1;
    int W_out = (W_in + 2 * padding.w - dilation.w * (kernel_size.w - 1) - 1) / stride.w + 1;

    std::vector<int> out_shape = {N, C, H_out, W_out};
    Tensor* result = create_result_tensor(ctx, out_shape, input->dtype);
    if (!result) return nullptr;

    float* in_data = (float*)input->data;
    float* out_data = (float*)result->data;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel_size.h; ++kh) {
                        for (int kw = 0; kw < kernel_size.w; ++kw) {
                            int h_in = h_out * stride.h - padding.h + kh * dilation.h;
                            int w_in = w_out * stride.w - padding.w + kw * dilation.w;

                            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                int in_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                                if (in_data[in_idx] > max_val) {
                                    max_val = in_data[in_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }
    return result;
}

// aten.adaptive_avg_pool2d.default
Tensor* op_aten_adaptive_avg_pool2d_default(NacRuntimeContext* ctx, const ParsedInstruction& ins, Tensor** args, size_t argc) {
    // Аргументы: input, output_size
    if (argc < 2 || !args[0] || !args[1]) return nullptr;

    Tensor* input = args[0];
    Param2D output_size(args[1]);

    if (input->shape.size() != 4) return nullptr;

    int N = input->shape[0];
    int C = input->shape[1];
    int H_in = input->shape[2];
    int W_in = input->shape[3];

    int H_out = output_size.h;
    int W_out = output_size.w;

    std::vector<int> out_shape = {N, C, H_out, W_out};
    Tensor* result = create_result_tensor(ctx, out_shape, input->dtype);
    if (!result) return nullptr;

    float* in_data = (float*)input->data;
    float* out_data = (float*)result->data;

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
                        for (int w = w_start; w < w_end; ++w) {
                            int in_idx = n * C * H_in * W_in + c * H_in * W_in + h * W_in + w;
                            sum += in_data[in_idx];
                        }
                    }
                    count = (h_end - h_start) * (w_end - w_start);
                    
                    int out_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
                    out_data[out_idx] = (count > 0) ? (sum / count) : 0.0f;
                }
            }
        }
    }
    return result;
}


void register_kernels() {
    // --- Стандартные 'nac.*' операции по фиксированным ID ---
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

    Serial.println("[KERNELS] Statically registered standard 'nac.*' kernels by ID.");

    // --- Кастомные 'aten.*' операции по ИМЕНАМ для динамической привязки ---
    
    // Ядра, которые у вас уже были реализованы (хотя бы частично)
    g_kernel_string_map["aten.linear.default"] = &op_aten_linear_default;
    g_kernel_string_map["aten.layer_norm.default"] = &op_aten_layer_norm_default;
    g_kernel_string_map["aten.silu.default"] = &op_aten_silu_default;
    g_kernel_string_map["aten.mul.Tensor"] = &op_aten_mul_Tensor;
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
    g_kernel_string_map["aten._to_copy.default"] = &op_aten_to_copy_default;
    g_kernel_string_map["aten.cumsum.default"] = &op_aten_cumsum_default;
    g_kernel_string_map["aten.type_as.default"] = &op_aten_type_as_default;
    g_kernel_string_map["aten.zeros.default"] = &op_aten_zeros_default;
    g_kernel_string_map["aten.slice.Tensor"] = &op_aten_slice_Tensor;
    g_kernel_string_map["aten.scaled_dot_product_attention.default"] = &op_aten_scaled_dot_product_attention_default;
    g_kernel_string_map["aten.conv2d.default"] = &op_aten_conv2d_default; 
    g_kernel_string_map["aten._native_batch_norm_legit_no_training.default"] = &op_aten_native_batch_norm_default;
    g_kernel_string_map["aten.max_pool2d.default"] = &op_aten_max_pool2d_default;
    g_kernel_string_map["aten.adaptive_avg_pool2d.default"] = &op_aten_adaptive_avg_pool2d_default;

    Serial.printf("[KERNELS] Registered %d custom kernels by name for dynamic mapping.\n", g_kernel_string_map.size());
}