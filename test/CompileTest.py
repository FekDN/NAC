# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

from NAC import generate_artifacts
from MEP_compiler import MEPCompiler
import os
import torch
import traceback
import numpy as np
import torchvision.models as models
from transformers import GPT2LMHeadModel, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Wrapper classes (unchanged from original)
# ============================================================

class Gpt2LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.lm_head = model.lm_head
        self.embeddings = model.transformer.wte

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embeddings(input_ids)
        float_mask = (1.0 - attention_mask.to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=float_mask)
        return self.lm_head(transformer_outputs[0])


# ============================================================
# process_resnet18
# ============================================================

def process_resnet18():
    print("\n" + "#"*20 + " PROCESSING RESNET-18 " + "#"*20)
    model_name = "resnet18"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    print("--- Компиляция MEP плана для ResNet-18 ---")
    c = MEPCompiler()

    # Загружаем модель
    c.res_load_model(MODEL_ID, MODEL_PATH)

    # Запрашиваем путь к изображению у пользователя.
    # res_load_dynamic(out_var, path_var, file_type) — динамическая загрузка файла:
    #   path_var  — MEP-переменная (результат src_user_prompt), не строковый литерал.
    #   file_type=2 → .npy, загрузка as-is
    #   file_type=3 → image + ImageNet preprocessing:
    #       PIL.Image.open → resize(256) → center_crop(224×224)
    #       → normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    #       → shape (1, 3, 224, 224) float32
    # Реализация препроцессинга: NacRuntime.preprocess_image_for_imagenet()
    # (вызывается MEP-интерпретатором и NACmodels_test.py напрямую).
    c.src_user_prompt("image_path", "Введите путь к изображению (jpg/png/…): ")
    c.res_load_dynamic("image_tensor", "image_path", file_type=3)

    # Запуск модели
    c.model_run_static(MODEL_ID, in_vars=["image_tensor"], out_vars=["logits"])

    # Постобработка: argmax + softmax
    c.math_aggregate('argmax', "predicted_idx_tensor", "logits")
    c.math_unary('softmax', "probabilities", "logits")
    c.tensor_info('to_py', 'predicted_idx', 'predicted_idx_tensor')

    # Вероятность предсказанного класса
    c.tensor_extract("confidence_tensor", "probabilities", "predicted_idx")
    c.tensor_info('to_py', 'confidence', 'confidence_tensor')

    # Форматированный вывод
    # NOTE: string_format принимает САМУ строку формата (compile-time константа),
    # не имя переменной. Инлайним строку напрямую.
    c.string_format("final_output",
        "\n--- Результат классификации ---\n"
        "  Класс (ImageNet ID): {}\n"
        "  Уверенность:         {:.2%}",
        ["predicted_idx", "confidence"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["predicted_idx", "confidence"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name=model_name,
        model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval(),
        dummy_args=(torch.randn(1, 3, 224, 224),),
        quantization_method='BLOCK_FP8',
        optimize=True,
        mep_program=mep_program
    )


# ============================================================
# process_distilbert_sentiment
# ============================================================

def process_distilbert_sentiment():
    print("\n" + "#"*20 + " PROCESSING DistilBERT FOR SENTIMENT (SAFE LOADING) " + "#"*20)
    repo = "distilbert-base-uncased-finetuned-sst-2-english"
    model_name = "distilbert-sst2-sentiment"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        tokenizer = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        print(f"!!!!! ERROR: Could not download DistilBERT model '{repo}'.\n{e}")
        return

    class DistilBertLogitsWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    dummy_args = (inputs['input_ids'], inputs['attention_mask'])
    max_len = model.config.max_position_embeddings
    seq_dim = torch.export.Dim("sequence", min=1, max=max_len - 1)
    dynamic_shapes = {"input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}}

    print("--- Компиляция MEP плана для DistilBERT ---")
    c = MEPCompiler()

    # Шаг 1: Загрузка ресурсов
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)  # res_type=0: компонент из NacRuntime

    # Шаг 2: Получение пользовательского ввода и препроцессинг
    c.src_user_prompt("user_text", "Введите текст для анализа: ")
    c.preproc_encode("tokenizer_runtime", "user_text", "input_ids_list")
    c.tensor_create(from_py={'out_var': "input_ids", 'in_var': "input_ids_list", 'dtype_code': 5})

    # Формируем attention_mask: те же размерности, что и у input_ids, заполненные 1
    c.tensor_info('shape', "input_shape", "input_ids")
    c.tensor_create(ones={'out_var': "attention_mask", 'shape_var': "input_shape", 'dtype_code': 5})

    # lifted_one — скаляр float32=1.0, первый вход графа (v0).
    # position_ids — arange(seq_len), четвёртый вход (v6).
    # Источник: NACmodels_test.py all_inputs = [lifted_one, input_ids, attention_mask, position_ids]
    c.src_constant("dim_index_1", 1)
    c.tensor_info('dim', "seq_len", "input_ids", dim_idx_var="dim_index_1")
    c.tensor_create(arange={'out_var': "position_ids", 'end_var': "seq_len", 'dtype_code': 5})
    c.src_constant("one_float", 1.0)
    c.tensor_create(from_py={'out_var': "lifted_one", 'in_var': "one_float", 'dtype_code': 0})

    # Шаг 3: Запуск модели → logits (1, 2)
    # ПРАВИЛЬНЫЙ ПОРЯДОК: [lifted_one, input_ids, attention_mask, position_ids]
    c.model_run_static(MODEL_ID,
                       in_vars=["lifted_one", "input_ids", "attention_mask", "position_ids"],
                       out_vars=["logits"])

    # Шаг 4: Постобработка — argmax и softmax
    c.math_aggregate('argmax', "prediction_idx_tensor", "logits")
    c.tensor_info('to_py', 'prediction_idx', 'prediction_idx_tensor')
    c.math_unary('softmax', "probabilities", "logits")

    # Ветвление для определения текстовой метки
    c.src_constant("zero", 0)
    c.src_constant("one", 1)
    c.src_constant("true_const", True)
    c.src_constant("label_negative", "NEGATIVE")
    c.src_constant("label_positive", "POSITIVE")

    c.logic_compare('eq', "is_negative_cond", "prediction_idx", "zero")
    c.flow_branch_if("is_negative_cond", jump_label="set_negative_label")
    c.sys_copy("prediction_label_text", "label_positive")
    c.flow_branch_if("true_const", jump_label="end_label_if")
    c.place_label("set_negative_label")
    c.sys_copy("prediction_label_text", "label_negative")
    c.place_label("end_label_if")

    # Извлекаем вероятности из тензора (1, 2)
    c.tensor_extract("prob_neg", "probabilities", "zero")
    c.tensor_extract("prob_pos", "probabilities", "one")

    # Шаг 5: Форматированный вывод
    c.string_format("final_output",
        "\n--- Результаты анализа ---\n"
        "  Метка:              {}\n"
        "  Уверенность (NEG):  {:.2%}\n"
        "  Уверенность (POS):  {:.2%}",
        ["prediction_label_text", "prob_neg", "prob_pos"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["prediction_label_text", "prob_neg", "prob_pos"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name=model_name,
        model=DistilBertLogitsWrapper(model),
        dummy_args=dummy_args,
        quantization_method='BLOCK_FP8',
        tokenizer_repo=repo,
        store_weights_internally=True,
        optimize=True,
        dynamic_shapes=dynamic_shapes,
        mep_program=mep_program
    )


# ============================================================
# process_roberta_base
# ============================================================

def process_roberta_base():
    print("\n" + "#"*20 + " PROCESSING ROBERTA-BASE " + "#"*20)
    repo = "roberta-base"
    model_name = "roberta-base-fill-mask"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    model = RobertaForMaskedLM.from_pretrained(repo).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer_input = "Hello I'm a <mask> model."
    inputs = tokenizer(tokenizer_input, return_tensors="pt")

    class RobertaCleanedWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.config.use_cache = False
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = "eager"
            class ZerosModule(torch.nn.Module):
                def __init__(self, ed=768): super().__init__(); self.ed = ed
                def forward(self, ids):
                    return torch.zeros(ids.shape[0], ids.shape[1], self.ed,
                                       device=ids.device, dtype=torch.float32)
            self.model.roberta.embeddings.token_type_embeddings = ZerosModule()
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    dummy_args = (inputs['input_ids'], inputs['attention_mask'])
    seq_dim = torch.export.Dim("sequence", min=1, max=model.config.max_position_embeddings - 2)
    dynamic_shapes = {"input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}}

    print("--- Компиляция MEP плана для RoBERTa fill-mask ---")
    c = MEPCompiler()

    # Шаг 1: Загрузка ресурсов
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)

    # Шаг 2: Ввод и препроцессинг
    c.src_user_prompt("user_text", "Введите предложение с токеном <mask>: ")
    c.preproc_encode("tokenizer_runtime", "user_text", "input_ids_list")
    c.tensor_create(from_py={'out_var': "input_ids", 'in_var': "input_ids_list", 'dtype_code': 5})

    # attention_mask — единицы, той же формы что и input_ids
    c.tensor_info('shape', "input_shape", "input_ids")
    c.tensor_create(ones={'out_var': "attention_mask", 'shape_var': "input_shape", 'dtype_code': 5})

    # lifted_one — скаляр float32=1.0, первый вход графа (v0).
    # NACmodels_test.py: all_inputs = [lifted_one, input_ids, attention_mask]
    c.src_constant("one_float", 1.0)
    c.tensor_create(from_py={'out_var': "lifted_one", 'in_var': "one_float", 'dtype_code': 0})

    # Vocab ID токена <mask> (для RoBERTa = 50264)
    c.preproc_get_id("tokenizer_runtime", "<mask>", "mask_token_id")

    # Шаг 3: Запуск модели → logits (1, seq_len, vocab_size)
    # ПРАВИЛЬНЫЙ ПОРЯДОК: [lifted_one, input_ids, attention_mask]
    c.model_run_static(MODEL_ID,
                       in_vars=["lifted_one", "input_ids", "attention_mask"],
                       out_vars=["logits"])

    # Шаг 4: Найти ПОЗИЦИЮ <mask> в input_ids, затем извлечь логиты.
    # mask_token_id — это vocab ID (~50264), его НЕЛЬЗЯ использовать как
    # индекс в seq-размерность logits. Нужно найти позицию в input_ids:
    #   mask_idx = np.where(input_ids[0] == mask_token_id)[0][0]
    c.logic_compare('eq', "is_mask_pos", "input_ids", "mask_token_id")  # bool (1, seq_len)
    c.math_aggregate('argmax', "mask_pos_idx", "is_mask_pos")           # первый True → позиция
    c.tensor_extract("mask_logits", "logits", "mask_pos_idx")
    c.math_unary('softmax', "mask_probs", "mask_logits")
    c.math_aggregate('argmax', "pred_id_tensor", "mask_logits")
    c.tensor_info('to_py', 'pred_id', 'pred_id_tensor')
    c.tensor_extract("pred_confidence_tensor", "mask_probs", "pred_id")
    c.tensor_info('to_py', 'pred_confidence', 'pred_confidence_tensor')
    c.preproc_decode("tokenizer_runtime", "pred_id", "predicted_word")

    # Шаг 5: Форматированный вывод
    c.string_format("final_output",
        "\n--- Результат предсказания маски ---\n"
        "  Предсказанное слово: {}\n"
        "  Уверенность:         {:.2%}",
        ["predicted_word", "pred_confidence"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["predicted_word", "pred_confidence"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name=model_name,
        model=RobertaCleanedWrapper(model),
        dummy_args=dummy_args,
        quantization_method='BLOCK_FP8',
        dynamic_shapes=dynamic_shapes,
        tokenizer_repo=repo,
        optimize=True,
        tokenizer_input=tokenizer_input,
        mep_program=mep_program
    )


# ============================================================
# process_gpt2
# ============================================================

def process_gpt2():
    print("\n" + "#"*20 + " PROCESSING GPT-2 " + "#"*20)
    model_name = "gpt2-text-generation"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"
    FIXED_SEQ_LEN = 64

    model = GPT2LMHeadModel.from_pretrained("gpt2", use_cache=False).eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    dummy_input_ids = torch.ones(1, FIXED_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.tril(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN)).bool()[None, None, :, :]

    # Сохраняем каузальную маску в .npy для загрузки MEP рантаймом
    causal_mask_np = np.tril(np.ones((FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=bool))[np.newaxis, np.newaxis, :, :]
    np.save("gpt2_causal_mask.npy", causal_mask_np)

    print("--- Компиляция MEP плана для GPT-2 ---")
    c = MEPCompiler()

    # Шаг 1: Загрузка ресурсов
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)
    c.res_load_datafile("const_causal_mask", "gpt2_causal_mask.npy", file_type=2)

    # Константы
    c.src_constant("max_tokens", 30)
    c.src_constant("model_len",  FIXED_SEQ_LEN)
    c.src_constant("const_one",  1)
    c.src_constant("const_zero", 0)
    c.src_constant("axis_one",   1)

    c.preproc_get_id("tokenizer_runtime", "<|endoftext|>", "eos_id")
    c.sys_copy("pad_id", "eos_id")

    # Шаг 2: Ввод и препроцессинг
    c.src_user_prompt("prompt", "Введите начало текста: ")
    c.io_write("prompt", dest_type=0, write_mode=0)
    c.preproc_encode("tokenizer_runtime", "prompt", "prompt_ids_list")
    c.tensor_create(from_py={'out_var': "generated_ids", 'in_var': "prompt_ids_list", 'dtype_code': 5})

    # Шаг 3: Цикл генерации
    c.flow_loop_start("max_tokens")

    # ПРАВЫЙ padding через concat, как в NACmodels_test.py:
    #   np.pad(ids, ((0,0),(0,PAD_LEN)), constant_values=pad_id)
    # Направление tensor_manipulate('pad') неопределено, поэтому строим явно.
    c.tensor_info('dim', "current_seq_len", "generated_ids", dim_idx_var="const_one")
    c.math_binary('sub', "pad_len", "model_len", "current_seq_len")
    # Создаём pad-тензор [pad_id * pad_len]: arange → *0 → +pad_id
    c.tensor_create(arange={'out_var': "pad_range", 'end_var': "pad_len", 'dtype_code': 5})
    c.math_binary('mul', "pad_zeros",  "pad_range", "const_zero")
    c.math_binary('add', "pad_tokens", "pad_zeros", "pad_id")
    # Конкатенируем [generated_ids | pad_tokens] → правый padding
    c.tensor_combine('concat', "input_ids_padded",
                     ["generated_ids", "pad_tokens"], axis_var="axis_one")

    # Запуск модели → logits (1, seq_len, vocab_size)
    c.model_run_static(MODEL_ID,
                       in_vars=["input_ids_padded", "const_causal_mask"],
                       out_vars=["logits"])

    # Логиты последнего реального токена (right padding → индекс = current_seq_len-1)
    c.math_binary('sub', "last_token_idx", "current_seq_len", "const_one")
    c.tensor_extract("next_token_logits", "logits", "last_token_idx")

    # Жадная выборка
    c.math_aggregate('argmax', "next_token_id_tensor", "next_token_logits")
    c.tensor_info('to_py', 'next_token_id', 'next_token_id_tensor')
    c.tensor_create(from_py={'out_var': "next_token_2d", 'in_var': "next_token_id", 'dtype_code': 5})

    # Конкатенация нового токена
    c.src_constant("axis_one", 1)
    c.tensor_combine('concat', "generated_ids", ["generated_ids", "next_token_2d"], axis_var="axis_one")

    # Стриминг токена в консоль
    c.preproc_decode("tokenizer_runtime", "next_token_id", "new_token_text")
    c.io_write("new_token_text", dest_type=0, write_mode=2)  # write_mode=2: без переноса строки

    # Выход по EOS
    c.logic_compare('eq', "is_eos", "next_token_id", "eos_id")
    c.flow_break_loop_if("is_eos")

    c.flow_loop_end()

    # Шаг 4: Завершение
    c.src_constant("done_msg", "\n--- Генерация завершена ---")
    c.io_write("done_msg", dest_type=0, write_mode=0)
    c.exec_return(["generated_ids"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name=model_name,
        model=Gpt2LogitsWrapper(model),
        dummy_args=(dummy_input_ids, dummy_mask),
        quantization_method='INT8_TENSOR',
        optimize=True,
        tokenizer_repo="gpt2",
        mep_program=mep_program
    )


# ============================================================
# process_gpt2_streaming
# ============================================================

def process_gpt2_streaming():
    print("\n" + "#"*20 + " PROCESSING GPT-2 (STATELESS FOR STREAMING) " + "#"*20)
    repo = "gpt2"
    model_name = "gpt2-streaming"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"
    FIXED_SEQ_LEN = 64

    model = GPT2LMHeadModel.from_pretrained(repo, use_cache=False).eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    dummy_input_ids = torch.ones(1, FIXED_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.tril(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN)).bool()[None, None, :, :]

    # Переиспользуем маску из process_gpt2 (или создаём заново)
    if not os.path.exists("gpt2_causal_mask.npy"):
        causal_mask_np = np.tril(np.ones((FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=bool))[np.newaxis, np.newaxis, :, :]
        np.save("gpt2_causal_mask.npy", causal_mask_np)

    print("--- Компиляция MEP плана для GPT-2 Streaming ---")
    c = MEPCompiler()

    # Шаг 1: Загрузка ресурсов
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)
    c.res_load_datafile("static_causal_mask", "gpt2_causal_mask.npy", file_type=2)

    # Константы
    c.src_constant("max_tokens", 50)
    c.src_constant("model_len",  FIXED_SEQ_LEN)
    c.src_constant("const_one",  1)
    c.src_constant("const_zero", 0)
    c.src_constant("axis_one",   1)

    c.preproc_get_id("tokenizer_runtime", "<|endoftext|>", "eos_id")
    c.sys_copy("pad_id", "eos_id")

    # Шаг 2: Ввод
    c.src_user_prompt("prompt", "Введите начало текста (streaming): ")
    c.io_write("prompt", dest_type=0, write_mode=0)
    c.preproc_encode("tokenizer_runtime", "prompt", "prompt_ids_list")
    c.tensor_create(from_py={'out_var': "generated_ids", 'in_var': "prompt_ids_list", 'dtype_code': 5})

    # Шаг 3: Цикл генерации
    c.flow_loop_start("max_tokens")

    # Используем правильное имя переменной: "generated_ids"
    c.tensor_info('dim', "current_seq_len", "generated_ids", dim_idx_var="const_one")
    
    c.math_binary('sub', "pad_len", "model_len", "current_seq_len")
    c.tensor_create(arange={'out_var': "pad_range", 'end_var': "pad_len", 'dtype_code': 5})
    c.math_binary('mul', "pad_zeros",  "pad_range", "const_zero")
    c.math_binary('add', "pad_tokens", "pad_zeros", "pad_id")
    c.tensor_combine('concat', "padded_input_ids",
                     ["generated_ids", "pad_tokens"], axis_var="axis_one")

    c.logic_compare('neq', "attention_mask", "padded_input_ids", "pad_id")
    c.math_binary('mul', "final_mask", "static_causal_mask", "attention_mask")

    c.model_run_static(MODEL_ID,
                       in_vars=["padded_input_ids", "final_mask"],
                       out_vars=["logits"])

    c.math_binary('sub', "last_token_idx", "current_seq_len", "const_one")
    c.tensor_extract("next_token_logits", "logits", "last_token_idx")

    c.math_aggregate('argmax', "next_token_id_tensor", "next_token_logits")
    c.tensor_info('to_py', 'next_token_id', 'next_token_id_tensor')
    c.tensor_create(from_py={'out_var': "next_token_2d", 'in_var': "next_token_id", 'dtype_code': 5})

    # Обновляем "generated_ids", а не создаем новую переменную "input_ids"
    c.tensor_combine('concat', "generated_ids", ["generated_ids", "next_token_2d"], axis_var="axis_one")

    c.preproc_decode("tokenizer_runtime", "next_token_id", "new_token_text")
    c.io_write("new_token_text", dest_type=0, write_mode=2)

    c.logic_compare('eq', "is_eos", "next_token_id", "eos_id")
    c.flow_break_loop_if("is_eos")

    c.flow_loop_end()

    c.src_constant("done_msg", "\n--- Стриминг завершён ---")
    c.io_write("done_msg", dest_type=0, write_mode=0)
    
    # Возвращаем "generated_ids"
    c.exec_return(["generated_ids"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name=model_name,
        model=Gpt2LogitsWrapper(model),
        dummy_args=(dummy_input_ids, dummy_mask),
        quantization_method='none',
        tokenizer_repo=repo,
        store_weights_internally=False,
        optimize=True,
        mep_program=mep_program
    )


# ============================================================
# process_t5_translation
# ============================================================

def process_t5_translation():
    import types
    from transformers import T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5Block

    print("\n#################### PROCESSING T5-SMALL ####################")
    repo = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(repo).eval()

    class ExportableT5EncoderBlock(torch.nn.Module):
        def __init__(self, original_block: T5Block):
            super().__init__()
            self.layer = original_block.layer
            if hasattr(self.layer[0].SelfAttention, "has_relative_attention_bias"):
                self.layer[0].SelfAttention.has_relative_attention_bias = False
        def forward(self, hidden_states, attention_mask=None, position_bias=None):
            self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask,
                                                   position_bias=position_bias)
            hidden_states = self_attention_outputs[0]
            hidden_states = self.layer[1](hidden_states)
            return (hidden_states,)

    class T5EncoderWrapper(torch.nn.Module):
        def __init__(self, original_encoder, model_instance):
            super().__init__()
            self.embed_tokens = original_encoder.embed_tokens
            self.final_layer_norm = original_encoder.final_layer_norm
            self.block = torch.nn.ModuleList(
                [ExportableT5EncoderBlock(block) for block in original_encoder.block]
            )
            self.get_extended_attention_mask = model_instance.get_extended_attention_mask
        def forward(self, input_ids, attention_mask, precomputed_bias):
            hidden_states = self.embed_tokens(input_ids)
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
            for layer_module in self.block:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask,
                                             position_bias=precomputed_bias)
                hidden_states = layer_outputs[0]
            hidden_states = self.final_layer_norm(hidden_states)
            return hidden_states

    seq_len = 64
    dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    original_compute_bias = model.encoder.block[0].layer[0].SelfAttention.compute_bias
    dummy_bias = original_compute_bias(seq_len, seq_len)

    # Сохраняем предвычисленное позиционное смещение T5 как .npy
    np.save("t5_bias.npy", dummy_bias.detach().numpy())

    seq_dim = torch.export.Dim("sequence", min=1, max=512)
    dynamic_shapes_encoder = {
        "input_ids": {1: seq_dim},
        "attention_mask": {1: seq_dim},
        "precomputed_bias": {2: seq_dim, 3: seq_dim}
    }
    clean_encoder_wrapper = T5EncoderWrapper(model.get_encoder(), model)

    print("\n--- Экспорт T5 Encoder (без MEP плана — вложенный компонент) ---")
    generate_artifacts(
        model_name="t5-encoder",
        model=clean_encoder_wrapper,
        dummy_args=(dummy_input_ids, dummy_attention_mask, dummy_bias),
        quantization_method='BLOCK_FP8',
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_encoder
    )

    class T5DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.get_decoder()
            self.lm_head = model.lm_head
        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            decoder_attention_mask = torch.ones_like(decoder_input_ids,
                                                     device=decoder_input_ids.device)
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False, return_dict=True
            )
            return self.lm_head(outputs.last_hidden_state)

    dummy_decoder_input_ids = torch.ones(1, 2, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(1, 64, model.config.d_model)
    dummy_encoder_attention_mask = torch.ones(1, 64, dtype=torch.long)
    dec_seq_dim = torch.export.Dim("decoder_sequence", min=1, max=512)
    dynamic_shapes_decoder = {
        "decoder_input_ids": {1: dec_seq_dim},
        "encoder_hidden_states": {1: seq_dim},
        "encoder_attention_mask": {1: seq_dim},
    }

    # ----------------------------------------------------------------
    # MEP план для ВСЕГО пайплайна T5 encoder → decoder
    # Хранится в артефакте decoder'а как точки входа
    # ----------------------------------------------------------------
    print("--- Компиляция MEP плана для T5 (encoder → decoder пайплайн) ---")
    c = MEPCompiler()
    ENCODER_ID = 0
    DECODER_ID = 1

    # Шаг 1: Загрузка обеих моделей и токенизатора
    c.res_load_model(ENCODER_ID, "t5-encoder.nac")
    c.res_load_model(DECODER_ID, "t5-decoder.nac")
    c.res_load_extern("tokenizer_runtime", 0, ENCODER_ID)  # токенизатор из encoder-артефакта

    # Предвычисленное позиционное смещение T5
    c.res_load_datafile("precomputed_bias", "t5_bias.npy", file_type=2)

    # Константы
    c.src_constant("max_decode_steps", 64)
    c.src_constant("const_one", 1)
    c.preproc_get_id("tokenizer_runtime", "</s>", "eos_id")
    c.preproc_get_id("tokenizer_runtime", "<pad>", "pad_id")

    # Шаг 2: Ввод и препроцессинг
    c.src_user_prompt("source_text",
        "Введите текст для перевода (например: 'translate English to German: Hello'): ")
    c.preproc_encode("tokenizer_runtime", "source_text", "enc_ids_list")
    c.tensor_create(from_py={'out_var': "encoder_input_ids", 'in_var': "enc_ids_list", 'dtype_code': 5})

    # Encoder attention mask (единицы)
    c.tensor_info('shape', "enc_shape", "encoder_input_ids")
    c.tensor_create(ones={'out_var': "encoder_attention_mask", 'shape_var': "enc_shape", 'dtype_code': 5})

    # Шаг 3: Запуск энкодера (один раз)
    c.model_run_static(ENCODER_ID,
                       in_vars=["encoder_input_ids", "encoder_attention_mask", "precomputed_bias"],
                       out_vars=["encoder_hidden_states"])

    # Инициализация декодера с <pad> токеном
    c.tensor_create(from_py={'out_var': "decoder_ids", 'in_var': "pad_id", 'dtype_code': 5})

    # Шаг 4: Авторегрессивный цикл декодирования
    c.flow_loop_start("max_decode_steps")

    c.model_run_static(DECODER_ID,
                       in_vars=["decoder_ids", "encoder_hidden_states", "encoder_attention_mask"],
                       out_vars=["decoder_logits"])

    # Жадная выборка на последней позиции
    c.tensor_info('dim', "dec_seq_len", "decoder_ids", dim_idx_var="const_one")
    c.math_binary('sub', "last_dec_idx", "dec_seq_len", "const_one")
    c.tensor_extract("last_logits", "decoder_logits", "last_dec_idx")
    c.math_aggregate('argmax', "next_id_tensor", "last_logits")
    c.tensor_info('to_py', 'next_id', 'next_id_tensor')

    # Присоединить к decoder_ids
    c.tensor_create(from_py={'out_var': "next_id_2d", 'in_var': "next_id", 'dtype_code': 5})
    c.src_constant("axis_one", 1)
    c.tensor_combine('concat', "decoder_ids", ["decoder_ids", "next_id_2d"], axis_var="axis_one")

    # Прерывание по EOS
    c.logic_compare('eq', "is_eos", "next_id", "eos_id")
    c.flow_break_loop_if("is_eos")

    c.flow_loop_end()

    # Шаг 5: Декодирование и вывод результата
    c.preproc_decode("tokenizer_runtime", "decoder_ids", "output_text")
    c.string_format("final_output", "\n--- Результат перевода ---\n  {}", ["output_text"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["output_text"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    print("\n--- Экспорт T5 Decoder + MEP пайплайн-оркестратор ---")
    generate_artifacts(
        model_name="t5-decoder",
        model=T5DecoderWrapper(model),
        dummy_args=(dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
        quantization_method='BLOCK_FP8',
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_decoder,
        mep_program=mep_program
    )


# ============================================================
# process_sd_unet_vae
# ============================================================

def process_sd_unet_vae():
    print("\n" + "#"*20 + " PROCESSING STABLE DIFFUSION (UNET & VAE) " + "#"*20)
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("Error: 'diffusers' required. pip install diffusers transformers accelerate")
        return

    repo = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float32)
    latent_res = 512 // 8
    NUM_INFERENCE_STEPS = 20

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet): super().__init__(); self.unet = unet
        def forward(self, s, t, e): return self.unet(s, t, e).sample

    dummy_latent = torch.randn(2, 4, latent_res, latent_res)
    dummy_text_embed = torch.randn(2, 77, 768)
    dummy_timestep = torch.tensor([999, 999], dtype=torch.float32)

    print("--- Экспорт SD UNet (без MEP плана — вложенный компонент) ---")
    generate_artifacts(
        model_name="sd-unet-256",
        model=UNetWrapper(pipe.unet.eval()),
        dummy_args=(dummy_latent, dummy_timestep, dummy_text_embed),
        quantization_method='BLOCK_FP8',
        optimize=True
    )

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latents): return self.vae.decode(latents, return_dict=False)[0]

    dummy_vae_latent = torch.randn(1, 4, latent_res, latent_res)

    # ----------------------------------------------------------------
    # MEP план для всего диффузионного пайплайна
    # Хранится в артефакте VAE как точки входа
    # Ожидаемые внешние файлы (подготовить заранее):
    #   sd_noise_latent.npy  — (2, 4, latent_res, latent_res) float32
    #   sd_text_embeds.npy   — (2, 77, 768) float32
    #   sd_timesteps.npy     — (NUM_INFERENCE_STEPS,) float32
    # ----------------------------------------------------------------
    print("--- Компиляция MEP плана для Stable Diffusion (UNet денойзинг + VAE декод) ---")
    c = MEPCompiler()
    UNET_ID = 0
    VAE_ID = 1

    # Шаг 1: Загрузка обеих моделей
    c.res_load_model(UNET_ID, "sd-unet-256.nac")
    c.res_load_model(VAE_ID, "sd-vae-decoder-256.nac")

    # Загрузка входных данных из файлов
    c.res_load_datafile("latents",         "sd_noise_latent.npy", file_type=2)
    c.res_load_datafile("text_embeds",     "sd_text_embeds.npy",  file_type=2)
    c.res_load_datafile("timesteps_array", "sd_timesteps.npy",    file_type=2)

    # Константы
    c.src_constant("num_steps",      NUM_INFERENCE_STEPS)
    c.src_constant("const_one",      1)
    c.src_constant("guidance_scale", 7.5)
    c.src_constant("step_size",      0.05)

    # Прогресс-заголовок
    c.src_constant("header_msg", f"Начало диффузии ({NUM_INFERENCE_STEPS} шагов)...")
    c.io_write("header_msg", dest_type=0, write_mode=0)

    # Шаг 2: Денойзинговый цикл
    c.flow_loop_start("num_steps")

    # Текущий шаг = num_steps - remaining
    c.tensor_info('dim', "remaining", "timesteps_array", dim_idx_var="const_one")
    c.math_binary('sub', "step_idx", "num_steps", "remaining")
    c.tensor_extract("current_timestep", "timesteps_array", "step_idx")

    # UNet: предсказание шума (classifier-free guidance: батч 2)
    c.model_run_static(UNET_ID,
                       in_vars=["latents", "current_timestep", "text_embeds"],
                       out_vars=["noise_pred"])

    # Применяем guidance scale
    c.math_binary('mul', "noise_pred_scaled", "noise_pred", "guidance_scale")

    # Упрощённый шаг планировщика: latents -= step_size * noise_pred_scaled
    c.math_binary('mul', "noise_delta", "noise_pred_scaled", "step_size")
    c.math_binary('sub', "latents", "latents", "noise_delta")

    c.flow_loop_end()

    # Шаг 3: Декодирование латентного пространства → пиксельное изображение
    c.model_run_static(VAE_ID, in_vars=["latents"], out_vars=["decoded_image"])

    # Шаг 4: Финальный вывод
    c.src_constant("done_format",
        "\n--- Диффузия завершена ---\n"
        "  Изображение доступно в контексте (decoded_image)\n"
        "  Для сохранения используйте serialize_object + io_write."
    )
    c.io_write("done_format", dest_type=0, write_mode=0)
    c.exec_return(["decoded_image"])

    mep_program = c.get_program()
    print("--- MEP план скомпилирован ---")

    generate_artifacts(
        model_name="sd-vae-decoder-256",
        model=VAEDecoderWrapper(pipe.vae.eval()),
        dummy_args=(dummy_vae_latent,),
        quantization_method='BLOCK_FP8',
        optimize=True,
        mep_program=mep_program
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    registry_filepath = os.path.join('./', 'registry.json')
    if os.path.exists(registry_filepath):
        try:
            if os.path.getsize(registry_filepath) == 0:
                os.remove(registry_filepath)
        except OSError as e:
            print(f"WARNING: Could not check/delete registry.json: {e}")

    try:
        process_resnet18()
        process_distilbert_sentiment()
        process_t5_translation()
        process_roberta_base()
        process_gpt2()
        process_sd_unet_vae()
        process_gpt2_streaming()
        print("\n" + "="*20 + " ALL MODELS PROCESSED SUCCESSFULLY " + "="*20)
    except Exception as e:
        print(f"\n!!!!!! ERROR DURING BATCH PROCESSING: {e}")
        traceback.print_exc()