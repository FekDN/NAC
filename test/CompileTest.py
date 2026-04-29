# --- START OF FILE CompileTest.py ---

# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import os

# ============================================================
# ENVIRONMENTAL CHECKS (STRICT VERSIONS FOR torch.export)
# ============================================================
TARGET_TORCH = "2.5.1"
TARGET_TVISION = "0.20.1"
TARGET_TRANSFORMERS = "4.57.3"

def verify_environment():
    try:
        import torch
        import torchvision
        import transformers
    except ImportError as e:
        print(f"\n[!] Missing required library: {e.name}")
        print_venv_instructions()
        sys.exit(1)

    mismatch = False
    
    # torch.__version__ часто содержит '+cu121', отрезаем это для проверки
    torch_ver = torch.__version__.split('+')[0]
    tvision_ver = torchvision.__version__.split('+')[0]
    trans_ver = transformers.__version__

    if torch_ver != TARGET_TORCH:
        print(f"[!] Version mismatch: torch (Installed: {torch_ver}, Required: {TARGET_TORCH})")
        mismatch = True
    if tvision_ver != TARGET_TVISION:
        print(f"[!] Version mismatch: torchvision (Installed: {tvision_ver}, Required: {TARGET_TVISION})")
        mismatch = True
    if trans_ver != TARGET_TRANSFORMERS:
        print(f"[!] Version mismatch: transformers (Installed: {trans_ver}, Required: {TARGET_TRANSFORMERS})")
        mismatch = True

    if mismatch:
        print_venv_instructions()
        sys.exit(1)

def print_venv_instructions():
    print("\n" + "!"*70)
    print("ENVIRONMENT SETUP REQUIRED".center(70))
    print("!"*70)
    print("The 'torch.export' compilation engine is highly dependent on exact")
    print("library versions. To prevent graph compilation errors, please create")
    print("a fresh virtual environment using the commands below:\n")
    print("For Windows:")
    print("  python -m venv nac_env")
    print("  nac_env\\Scripts\\activate")
    print(f"  pip install torch=={TARGET_TORCH} torchvision=={TARGET_TVISION} transformers=={TARGET_TRANSFORMERS} sympy==1.13.1")
    print("\nFor Linux / Mac:")
    print("  python3 -m venv nac_env")
    print("  source nac_env/bin/activate")
    print(f"  pip install torch=={TARGET_TORCH} torchvision=={TARGET_TVISION} transformers=={TARGET_TRANSFORMERS} sympy==1.13.1")
    print("!"*70 + "\n")

# Run the strict version check before loading heavy machinery
verify_environment()

# ============================================================
# NORMAL IMPORTS
# ============================================================
from NAC import generate_artifacts
from MEP_compiler import MEPCompiler
import torch
import traceback
import numpy as np
import torchvision.models as models
from transformers import GPT2LMHeadModel, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification

print(torch.__version__)

def unwrap_hf_forward(model):
    for m in model.modules():
        if hasattr(m.forward, "__wrapped__"):
            func = m.forward
            while hasattr(func, "__wrapped__"):
                func = func.__wrapped__
            m.forward = func.__get__(m, type(m))
    return model

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
    print("\n" + "#"*20 + " PROCESSING RESNET-18 (WITH TRNG & LOSS) " + "#"*20)
    model_name = "resnet18"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    print("--- Compiling MEP plan for ResNet-18 ---")
    c = MEPCompiler()

    c.res_load_model(MODEL_ID, MODEL_PATH)
    
    # Always load image
    c.src_user_prompt("image_path", "Enter path to image (jpg/png/...): ")
    c.res_load_dynamic("image_tensor", "image_path", file_type=3)

    # Execution Mode Branching
    c.src_exec_mode("exec_mode")
    c.src_constant("mode_train", "train")
    c.logic_compare("eq", "is_train_only", "exec_mode", "mode_train")
    c.flow_branch_if("is_train_only", "label_train_only")

    # --- 1. INFERENCE BLOCK ---
    c.model_run_static(MODEL_ID, in_vars=["image_tensor"], out_vars=["logits"])
    c.math_aggregate('argmax', "predicted_idx_tensor", "logits")
    c.math_unary('softmax', "probabilities", "logits")
    c.tensor_info('to_py', 'predicted_idx', 'predicted_idx_tensor')
    c.tensor_extract("confidence_tensor", "probabilities", "predicted_idx")
    c.tensor_info('to_py', 'confidence', 'confidence_tensor')

    c.string_format("final_output",
        "\n--- Classification Result ---\n"
        "  Class (ImageNet ID): {}\n"
        "  Confidence:          {:.2%}",
        ["predicted_idx", "confidence"])
    c.io_write("final_output", dest_type=0, write_mode=0)

    c.place_label("label_train_only")

    c.src_constant("mode_infer", "infer")
    c.logic_compare("eq", "is_infer_only", "exec_mode", "mode_infer")
    c.flow_branch_if("is_infer_only", "label_end")

    # --- 2. TRAINING BLOCK ---
    c.src_user_prompt("target_class", "Enter target class ID for training (e.g. 42): ", data_type=1)
    c.tensor_create(from_py={'out_var': "target_tensor", 'in_var': "target_class", 'dtype_code': 5}) 

    c.src_constant("num_epochs", 3)
    c.src_constant("learning_rate", 0.005) # Нормальный LR для FC слоя
    
    c.src_constant("train_msg", "\nStarting On-Device Fine-Tuning (3 Epochs)...")
    c.io_write("train_msg", dest_type=0, write_mode=0)

    c.flow_loop_start("num_epochs")
    c.model_zero_grad(MODEL_ID)
    c.model_train_step(
        model_id=MODEL_ID,
        loss_type=0,
        in_vars=["image_tensor"],      # ← было ["x"] — ошибка!
        target_vars=["target_tensor"], # ← было ["target"] — ошибка!
        out_loss_var="loss",
        lr_var="learning_rate",
        logits_var=None,
        head_weight_name="fc.weight",   # ← обязательно
        head_bias_name="fc.bias"
    )
    c.sys_debug_print("loss", "  Epoch Loss")
    c.flow_loop_end()

    c.src_constant("empty_str", "")
    c.model_save_weights(MODEL_ID, "empty_str", save_type=0)

    c.place_label("label_end")
    c.exec_return(["image_path"])

    mep_program = c.get_program()
    print("--- MEP plan compiled ---")

    dummy_args = (torch.randn(1, 3, 224, 224),)
    dummy_targets = (torch.tensor([42], dtype=torch.long),) 

    # ========================================================
    # ЗАМОРОЗКА БАЗОВЫХ СЛОЕВ (TRANSFER LEARNING)
    # ========================================================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')
        #param.requires_grad = name.startswith(('layer4', 'fc')) # Временно отключено
    model.eval()

    generate_artifacts(
        model_name=model_name,
        model=model,
        dummy_args=dummy_args,
        dummy_targets=dummy_targets,
        loss_type='cross_entropy',
        quantization_method='BLOCK_FP8', # Возвращаем на 'none' для точности
        optimize=True,
        mep_program=mep_program,
        generate_trng=True 
    )

# ============================================================
# process_distilbert_sentiment
# ============================================================

def process_distilbert_sentiment():
    print("\n" + "#"*20 + " PROCESSING DistilBERT FOR SENTIMENT " + "#"*20)
    repo = "distilbert-base-uncased-finetuned-sst-2-english"
    model_name = "distilbert-sst2-sentiment"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        model = unwrap_hf_forward(model)
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
    dummy_targets = (torch.tensor([1], dtype=torch.long),) # Class 1 (Positive)
    
    max_len = model.config.max_position_embeddings
    seq_dim = torch.export.Dim("sequence", min=1, max=max_len - 1)
    dynamic_shapes = {"input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}}

    c = MEPCompiler()
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)

    c.src_user_prompt("user_text", "Enter text for analysis: ")
    c.preproc_encode("tokenizer_runtime", "user_text", "input_ids_list")
    c.tensor_create(from_py={'out_var': "input_ids", 'in_var': "input_ids_list", 'dtype_code': 5})
    c.tensor_info('shape', "input_shape", "input_ids")
    c.tensor_create(ones={'out_var': "attention_mask", 'shape_var': "input_shape", 'dtype_code': 5})

    c.src_constant("dim_index_1", 1)
    c.tensor_info('dim', "seq_len", "input_ids", dim_idx_var="dim_index_1")
    c.tensor_create(arange={'out_var': "position_ids", 'end_var': "seq_len", 'dtype_code': 5})
    c.src_constant("one_float", 1.0)
    c.tensor_create(from_py={'out_var': "lifted_one", 'in_var': "one_float", 'dtype_code': 0})

    c.src_exec_mode("exec_mode")
    c.src_constant("mode_train", "train")
    c.logic_compare("eq", "is_train_only", "exec_mode", "mode_train")
    c.flow_branch_if("is_train_only", "label_train_only")

    # --- INFERENCE ---
    c.model_run_static(MODEL_ID,
                       in_vars=["lifted_one", "input_ids", "attention_mask", "position_ids"],
                       out_vars=["logits"])

    c.math_aggregate('argmax', "prediction_idx_tensor", "logits")
    c.tensor_info('to_py', 'prediction_idx', 'prediction_idx_tensor')
    c.math_unary('softmax', "probabilities", "logits")

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

    c.tensor_extract("prob_neg", "probabilities", "zero")
    c.tensor_extract("prob_pos", "probabilities", "one")

    c.string_format("final_output",
        "\n--- Analysis Results ---\n"
        "  Label:              {}\n"
        "  Confidence (NEG):   {:.2%}\n"
        "  Confidence (POS):   {:.2%}",
        ["prediction_label_text", "prob_neg", "prob_pos"])
    c.io_write("final_output", dest_type=0, write_mode=0)

    c.place_label("label_train_only")

    c.src_constant("mode_infer", "infer")
    c.logic_compare("eq", "is_infer_only", "exec_mode", "mode_infer")
    c.flow_branch_if("is_infer_only", "label_end")

    # --- TRAINING ---
    c.src_user_prompt("target_class", "Enter target sentiment (0=NEG, 1=POS): ", data_type=1)
    c.tensor_create(from_py={'out_var': "train_target_tensor", 'in_var': "target_class", 'dtype_code': 5})
    
    c.src_constant("learning_rate", 0.001)
    c.src_constant("num_train_epochs", 3)

    c.src_constant("train_start_msg", "\n--- Starting Fine-Tuning (3 epochs on current input) ---")
    c.io_write("train_start_msg", dest_type=0, write_mode=0)

    c.flow_loop_start("num_train_epochs")
    c.model_zero_grad(MODEL_ID)
    c.model_train_step(MODEL_ID, loss_type=0,
                       in_vars=["lifted_one", "input_ids", "attention_mask", "position_ids"],
                       target_vars=["train_target_tensor"],
                       out_loss_var="train_loss",
                       lr_var="learning_rate",
                       head_weight_name="model.classifier.weight",
                       head_bias_name="model.classifier.bias"
                      )
    c.sys_debug_print("train_loss", "  Epoch Loss")
    c.flow_loop_end()

    c.src_constant("empty_str", "")
    c.model_save_weights(MODEL_ID, "empty_str", save_type=0)

    c.place_label("label_end")
    c.exec_return(["user_text"])

    mep_program = c.get_program()

    generate_artifacts(
        model_name=model_name,
        model=DistilBertLogitsWrapper(model),
        dummy_args=dummy_args,
        dummy_targets=dummy_targets,
        loss_type='cross_entropy',
        quantization_method='BLOCK_FP8',
        tokenizer_repo=repo,
        store_weights_internally=True,
        optimize=True,
        dynamic_shapes=dynamic_shapes,
        mep_program=mep_program,
        generate_trng=True 
    )

# ============================================================
# process_gpt2
# ============================================================

def process_gpt2():
    print("\n" + "#"*20 + " PROCESSING GPT-2 " + "#"*20)
    model_name = "gpt2-text-generation"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"
    FIXED_SEQ_LEN = 128

    model = GPT2LMHeadModel.from_pretrained("gpt2", use_cache=False).eval()
    model = unwrap_hf_forward(model)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    dummy_input_ids = torch.ones(1, FIXED_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.tril(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN)).bool()[None, None, :, :]
    
    dummy_targets = (torch.cat([
        dummy_input_ids[:, 1:],
        torch.zeros(1, 1, dtype=torch.long)
    ], dim=1),) 

    causal_mask_np = np.tril(np.ones((FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=bool))[np.newaxis, np.newaxis, :, :]
    np.save("gpt2_causal_mask.npy", causal_mask_np)

    gpt2_train_target_np = dummy_targets[0].numpy().astype(np.int64)
    np.save("gpt2_train_target.npy", gpt2_train_target_np)

    c = MEPCompiler()
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)
    c.res_load_datafile("const_causal_mask", "gpt2_causal_mask.npy", file_type=2)

    c.src_constant("max_tokens",  30)
    c.src_constant("model_len",   FIXED_SEQ_LEN)
    c.src_constant("const_one",   1)
    c.src_constant("const_zero",  0)
    c.src_constant("axis_one",    1)
    c.src_constant("temperature", 0.7)
    c.src_constant("top_k",       50)

    c.preproc_get_id("tokenizer_runtime", "<|endoftext|>", "eos_id")
    c.sys_copy("pad_id", "eos_id")

    c.src_user_prompt("prompt", "Enter the beginning of the text: ")
    c.io_write("prompt", dest_type=0, write_mode=0)
    c.preproc_encode("tokenizer_runtime", "prompt", "prompt_ids_list")
    c.tensor_create(from_py={'out_var': "generated_ids", 'in_var': "prompt_ids_list", 'dtype_code': 5})

    c.src_exec_mode("exec_mode")
    c.src_constant("mode_train", "train")
    c.logic_compare("eq", "is_train_only", "exec_mode", "mode_train")
    c.flow_branch_if("is_train_only", "label_train_only")

    # --- INFERENCE ---
    c.flow_loop_start("max_tokens")
    c.tensor_info('dim', "current_seq_len", "generated_ids", dim_idx_var="const_one")
    c.math_binary('sub', "pad_len", "model_len", "current_seq_len")
    c.tensor_create(arange={'out_var': "pad_range", 'end_var': "pad_len", 'dtype_code': 5})
    c.math_binary('mul', "pad_zeros",  "pad_range", "const_zero")
    c.math_binary('add', "pad_tokens", "pad_zeros", "pad_id")
    c.tensor_combine('concat', "input_ids_padded", ["generated_ids", "pad_tokens"], axis_var="axis_one")

    c.logic_compare('neq', "attention_mask", "input_ids_padded", "pad_id")
    c.math_binary('mul', "final_mask", "const_causal_mask", "attention_mask")

    c.model_run_static(MODEL_ID, in_vars=["input_ids_padded", "final_mask"], out_vars=["logits"])

    c.math_binary('sub', "last_token_idx", "current_seq_len", "const_one")
    c.tensor_extract("next_token_logits", "logits", "last_token_idx")

    c.analysis_top_k("next_token_logits", "top_k", "topk_indices", "topk_vals")
    c.analysis_sample("next_token_logits", "temperature", "top_k", "next_token_id")

    c.tensor_create(from_py={'out_var': "next_token_2d", 'in_var': "next_token_id", 'dtype_code': 5})
    c.src_constant("axis_one", 1)
    c.tensor_combine('concat', "generated_ids", ["generated_ids", "next_token_2d"], axis_var="axis_one")

    c.preproc_decode("tokenizer_runtime", "next_token_id", "new_token_text")
    c.io_write("new_token_text", dest_type=0, write_mode=2)

    c.logic_compare('eq', "is_eos", "next_token_id", "eos_id")
    c.flow_break_loop_if("is_eos")

    c.flow_loop_end()

    c.src_constant("done_msg", "\n--- Generation complete ---\n")
    c.io_write("done_msg", dest_type=0, write_mode=0)

    c.place_label("label_train_only")

    c.src_constant("mode_infer", "infer")
    c.logic_compare("eq", "is_infer_only", "exec_mode", "mode_infer")
    c.flow_branch_if("is_infer_only", "label_end")

    # --- TRAINING ---
    c.tensor_info('dim', "current_seq_len", "generated_ids", dim_idx_var="const_one")
    c.math_binary('sub', "pad_len", "model_len", "current_seq_len")
    c.tensor_create(arange={'out_var': "pad_range", 'end_var': "pad_len", 'dtype_code': 5})
    c.math_binary('mul', "pad_zeros",  "pad_range", "const_zero")
    c.math_binary('add', "pad_tokens", "pad_zeros", "pad_id")
    c.tensor_combine('concat', "input_ids_padded", ["generated_ids", "pad_tokens"], axis_var="axis_one")

    c.logic_compare('neq', "attention_mask", "input_ids_padded", "pad_id")
    c.math_binary('mul', "final_mask", "const_causal_mask", "attention_mask")

    c.res_load_datafile("train_target_ids", "gpt2_train_target.npy", file_type=2)
    c.src_constant("learning_rate", 0.00001)
    c.src_constant("num_train_epochs", 3)

    c.src_constant("train_start_msg", "\n--- Starting Fine-Tuning (3 epochs) ---")
    c.io_write("train_start_msg", dest_type=0, write_mode=0)

    c.flow_loop_start("num_train_epochs")
    c.model_zero_grad(MODEL_ID)
    c.model_train_step(MODEL_ID, loss_type=0,
                       in_vars=["input_ids_padded", "final_mask"],
                       target_vars=["train_target_ids"],
                       out_loss_var="train_loss",
                       lr_var="learning_rate")
    c.sys_debug_print("train_loss", "  Epoch Loss")
    c.flow_loop_end()

    c.src_constant("empty_str", "")
    c.model_save_weights(MODEL_ID, "empty_str", save_type=0)

    c.place_label("label_end")
    c.exec_return(["generated_ids"])

    mep_program = c.get_program()

    generate_artifacts(
        model_name=model_name,
        model=Gpt2LogitsWrapper(model),
        dummy_args=(dummy_input_ids, dummy_mask),
        dummy_targets=dummy_targets,
        loss_type='cross_entropy',
        quantization_method='INT8_TENSOR',
        optimize=True,
        tokenizer_repo="gpt2",
        mep_program=mep_program,
        generate_trng=True 
    )

if __name__ == "__main__":
    registry_filepath = os.path.join('./', 'registry.json')
    if os.path.exists(registry_filepath):
        try:
            if os.path.getsize(registry_filepath) == 0:
                os.remove(registry_filepath)
        except OSError as e:
            pass

    try:
        process_resnet18()
        process_distilbert_sentiment()
        process_gpt2()
        print("\n" + "="*20 + " ALL MODELS PROCESSED SUCCESSFULLY " + "="*20)
    except Exception as e:
        print(f"\n!!!!!! ERROR DURING BATCH PROCESSING: {e}")
        traceback.print_exc()

# --- END OF FILE CompileTest.py ---