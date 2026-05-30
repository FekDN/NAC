# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import os

# ============================================================
# ENVIRONMENTAL CHECKS (STRICT VERSIONS FOR torch.export)
# ============================================================

TARGET_TORCH = "2.5.1"
TARGET_TVISION = "0.20.1"
TARGET_TRANSFORMERS = "4.57.3"
TARGET_SYMPY = "1.13.1"


def normalize_version(v: str) -> str:
    if v is None:
        return ""
    return v.split("+")[0].strip()


def verify_environment():
    # Python version check (strictly Python 3.11)
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print(
            f"\n[!] Python version mismatch "
            f"(Installed: {sys.version.split()[0]}, Required: 3.11.x)"
        )
        print_venv_instructions()
        sys.exit(1)

    try:
        import torch
        import torchvision
        import transformers
        import sympy
    except ImportError as e:
        print(f"\n[!] Missing required library: {e.name}")
        print_venv_instructions()
        sys.exit(1)

    mismatch = False

    torch_ver = normalize_version(torch.__version__)
    tvision_ver = normalize_version(torchvision.__version__)
    trans_ver = normalize_version(transformers.__version__)
    sympy_ver = normalize_version(sympy.__version__)

    if torch_ver != TARGET_TORCH:
        print(
            f"[!] Version mismatch: torch "
            f"(Installed: {torch_ver}, Required: {TARGET_TORCH})"
        )
        mismatch = True

    if tvision_ver != TARGET_TVISION:
        print(
            f"[!] Version mismatch: torchvision "
            f"(Installed: {tvision_ver}, Required: {TARGET_TVISION})"
        )
        mismatch = True

    if trans_ver != TARGET_TRANSFORMERS:
        print(
            f"[!] Version mismatch: transformers "
            f"(Installed: {trans_ver}, Required: {TARGET_TRANSFORMERS})"
        )
        mismatch = True

    if sympy_ver != TARGET_SYMPY:
        print(
            f"[!] Version mismatch: sympy "
            f"(Installed: {sympy_ver}, Required: {TARGET_SYMPY})"
        )
        mismatch = True

    if mismatch:
        print_venv_instructions()
        sys.exit(1)


def print_venv_instructions():
    print("\n" + "!" * 70)
    print("ENVIRONMENT SETUP REQUIRED".center(70))
    print("!" * 70)
    print("Strict dependency mode for torch.export compilation.\n")

    print("Required Python: 3.11.x\n")

    print("For Windows:")
    print("  py -3.11 -m venv nac_env")
    print("  nac_env\\Scripts\\activate")
    print(
        f"  pip install torch=={TARGET_TORCH} "
        f"torchvision=={TARGET_TVISION} "
        f"transformers=={TARGET_TRANSFORMERS} "
        f"sympy=={TARGET_SYMPY} "
        "numpy safetensors regex"
    )

    print("\nFor Linux / Mac:")
    print("  python3.11 -m venv nac_env")
    print("  source nac_env/bin/activate")
    print(
        f"  pip install torch=={TARGET_TORCH} "
        f"torchvision=={TARGET_TVISION} "
        f"transformers=={TARGET_TRANSFORMERS} "
        f"sympy=={TARGET_SYMPY} "
        "numpy safetensors regex"
    )

    print("!" * 70 + "\n")


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
    c.src_constant("learning_rate", 0.005) # Normal LR for FC layer
    
    c.src_constant("train_msg", "\nStarting On-Device Fine-Tuning (3 Epochs)...")
    c.io_write("train_msg", dest_type=0, write_mode=0)

    c.flow_loop_start("num_epochs")
    c.model_zero_grad(MODEL_ID)
    c.model_train_step(
        model_id=MODEL_ID,
        loss_type=0,
        in_vars=["image_tensor"],
        target_vars=["target_tensor"],
        out_loss_var="loss",
        lr_var="learning_rate",
        logits_var=None,
        head_weight_name="fc.weight",
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
    # FREEZING BASE LAYERS (TRANSFER LEARNING)
    # ========================================================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')
        #param.requires_grad = name.startswith(('layer4', 'fc')) # Temporarily disabled
    model.eval()

    generate_artifacts(
        model_name=model_name,
        model=model,
        dummy_args=dummy_args,
        dummy_targets=dummy_targets,
        loss_type='cross_entropy',
        quantization_method='BLOCK_FP8',
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
    gpt2_train_target_np = dummy_targets[0].numpy().astype(np.int64)
    mep_arrays = {
        "gpt2_causal_mask": causal_mask_np,
        "gpt2_train_target": gpt2_train_target_np
    }

    c = MEPCompiler()
    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)
    c.res_load_array("const_causal_mask", "gpt2_causal_mask")

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

    c.res_load_array("train_target_ids", "gpt2_train_target")
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
        generate_trng=True,
        mep_arrays=mep_arrays
    )


def process_mobilenet_v3():
    print("\n" + "#"*20 + " PROCESSING MOBILENET V3 (COMPUTER VISION) " + "#"*20)
    model_name = "mobilenet_v3_small"
    MODEL_ID = 0
    MODEL_PATH = f"{model_name}.nac"

    # 1. Loading the MobileNetV3 lightweight model from torchvision
    # Use weights=..., since pretrained=True is deprecated.
    print("Downloading MobileNetV3 from torchvision...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # For On-Device Training, freeze everything except the classifier.
    for name, param in model.named_parameters():
        #param.requires_grad = name.startswith('classifier')
        param.requires_grad = True

    model.eval()

    # 2. Preparing entries and targets
    dummy_args = (torch.randn(1, 3, 224, 224),)
    dummy_targets = (torch.tensor([42], dtype=torch.long),) 

    # 3. Drawing up an MEP plan
    c = MEPCompiler()
    c.res_load_model(MODEL_ID, MODEL_PATH)
    
    # Always request a picture
    c.src_user_prompt("image_path", "Enter path to image (jpg/png): ")
    c.res_load_dynamic("image_tensor", "image_path", file_type=3)

    # Selecting the operating mode (Inference / Training)
    c.src_exec_mode("exec_mode")
    c.src_constant("mode_train", "train")
    c.logic_compare("eq", "is_train_only", "exec_mode", "mode_train")
    c.flow_branch_if("is_train_only", "label_train_only")

    # --- INFERENCE BLOCK ---
    c.model_run_static(MODEL_ID, in_vars=["image_tensor"], out_vars=["logits"])
    
    # Obtain the probabilities and class index
    c.math_unary('softmax', "probabilities", "logits")
    c.math_aggregate('argmax', "predicted_idx_tensor", "logits")
    c.tensor_info('to_py', 'predicted_idx', 'predicted_idx_tensor')
    c.tensor_extract("confidence_tensor", "probabilities", "predicted_idx")
    c.tensor_info('to_py', 'confidence', 'confidence_tensor')

    c.string_format("final_output",
        "\n--- MobileNetV3 Classification ---\n"
        "  Class (ImageNet ID): {}\n"
        "  Confidence:          {:.2%}",
        ["predicted_idx", "confidence"])
    c.io_write("final_output", dest_type=0, write_mode=0)

    c.place_label("label_train_only")

    c.src_constant("mode_infer", "infer")
    c.logic_compare("eq", "is_infer_only", "exec_mode", "mode_infer")
    c.flow_branch_if("is_infer_only", "label_end")

    # --- TRAINING BLOCK ---
    c.src_user_prompt("target_class", "Enter target class ID for training (e.g. 42): ", data_type=1)
    c.tensor_create(from_py={'out_var': "target_tensor", 'in_var': "target_class", 'dtype_code': 5}) 

    c.src_constant("num_epochs", 3)
    c.src_constant("learning_rate", 0.005)
    
    c.src_constant("train_msg", "\nStarting MobileNet On-Device Fine-Tuning (3 Epochs)...")
    c.io_write("train_msg", dest_type=0, write_mode=0)

    c.flow_loop_start("num_epochs")
    c.model_zero_grad(MODEL_ID)
    
    # In MobileNetV3, the classifier is located at classifier.3.weight
    c.model_train_step(
        model_id=MODEL_ID,
        loss_type=0,
        in_vars=["image_tensor"],
        target_vars=["target_tensor"],
        out_loss_var="loss",
        lr_var="learning_rate",
        logits_var=None,
        head_weight_name="classifier.3.weight",  
        head_bias_name="classifier.3.bias"
    )
    c.sys_debug_print("loss", "  Epoch Loss")
    c.flow_loop_end()

    c.src_constant("empty_str", "")
    c.model_save_weights(MODEL_ID, "empty_str", save_type=0)

    c.place_label("label_end")
    c.exec_return(["image_path"])

    mep_program = c.get_program()
    print("--- MEP plan compiled for MobileNetV3 ---")

    # 4. Generating a .nac file
    generate_artifacts(
        model_name=model_name,
        model=model,
        dummy_args=dummy_args,
        dummy_targets=dummy_targets,
        loss_type='cross_entropy',
        quantization_method='none', # It is better not to quantize the weights of this model
        optimize=True,
        mep_program=mep_program,
        generate_trng=True 
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

    try:
        model = RobertaForMaskedLM.from_pretrained(repo).eval()
        model = unwrap_hf_forward(model)
        tokenizer = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        print(f"!!!!! ERROR: Could not download RoBERTa.\n{e}")
        return

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

    print("--- Compiling MEP plan for RoBERTa fill-mask ---")
    c = MEPCompiler()

    c.res_load_model(MODEL_ID, MODEL_PATH)
    c.res_load_extern("tokenizer_runtime", 0, MODEL_ID)

    c.src_user_prompt("user_text", "Enter a sentence with the <mask> token: ")
    c.preproc_encode("tokenizer_runtime", "user_text", "input_ids_list")
    c.tensor_create(from_py={'out_var': "input_ids", 'in_var': "input_ids_list", 'dtype_code': 5})

    c.tensor_info('shape', "input_shape", "input_ids")
    c.tensor_create(ones={'out_var': "attention_mask", 'shape_var': "input_shape", 'dtype_code': 5})

    c.src_constant("one_float", 1.0)
    c.tensor_create(from_py={'out_var': "lifted_one", 'in_var': "one_float", 'dtype_code': 0})

    c.preproc_get_id("tokenizer_runtime", "<mask>", "mask_token_id")

    c.model_run_static(MODEL_ID,
                       in_vars=["lifted_one", "input_ids", "attention_mask"],
                       out_vars=["logits"])

    c.logic_compare('eq', "is_mask_pos", "input_ids", "mask_token_id") 
    c.math_aggregate('argmax', "mask_pos_idx_tensor", "is_mask_pos") 
    c.tensor_info('to_py', 'mask_pos_idx', 'mask_pos_idx_tensor') # Converting to Python int
    c.tensor_extract("mask_logits", "logits", "mask_pos_idx")
    
    c.math_unary('softmax', "mask_probs", "mask_logits")
    c.math_aggregate('argmax', "pred_id_tensor", "mask_logits")
    c.tensor_info('to_py', 'pred_id', 'pred_id_tensor')
    c.tensor_extract("pred_confidence_tensor", "mask_probs", "pred_id")
    c.tensor_info('to_py', 'pred_confidence', 'pred_confidence_tensor')
    c.preproc_decode("tokenizer_runtime", "pred_id", "predicted_word")

    c.string_format("final_output",
        "\n--- Mask Prediction Result ---\n"
        "  Predicted word: {}\n"
        "  Confidence:     {:.2%}",
        ["predicted_word", "pred_confidence"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["predicted_word", "pred_confidence"])

    mep_program = c.get_program()
    print("--- MEP plan compiled ---")

    generate_artifacts(
        model_name=model_name,
        model=RobertaCleanedWrapper(model),
        dummy_args=dummy_args,
        loss_type='none',
        generate_trng=False, # Inference only
        quantization_method='INT8_TENSOR', # Speeding up BERT
        dynamic_shapes=dynamic_shapes,
        tokenizer_repo=repo,
        optimize=True,
        tokenizer_input=tokenizer_input,
        mep_program=mep_program
    )


# ============================================================
# process_t5_translation
# ============================================================

def process_t5_translation():
    import types
    try:
        from transformers import T5ForConditionalGeneration
        from transformers.models.t5.modeling_t5 import T5Block
    except ImportError:
        print("Transformers library required.")
        return

    print("\n" + "#"*20 + " PROCESSING T5-SMALL (ENCODER-DECODER) " + "#"*20)
    repo = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(repo).eval()
    model = unwrap_hf_forward(model)
    
    # -------------------------------------------------------------------------
    # 1. ENCODER EXPORT
    # -------------------------------------------------------------------------
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
    dummy_attention_mask[0, -10:] = 0 
    
    # SAVE a single BIAS function for both encoder and decoder
    compute_bias = model.encoder.block[0].layer[0].SelfAttention.compute_bias
    dummy_bias = compute_bias(seq_len, seq_len)
    
    mep_arrays = {
        "t5_bias": dummy_bias.squeeze(0).detach().numpy()
    }

    seq_dim = torch.export.Dim("sequence", min=1, max=512)
    dynamic_shapes_encoder = {
        "input_ids": {1: seq_dim},
        "attention_mask": {1: seq_dim},
        "precomputed_bias": {2: seq_dim, 3: seq_dim}
    }

    print("\n--- Exporting T5 Encoder ---")
    generate_artifacts(
        model_name="t5-encoder",
        model=T5EncoderWrapper(model.get_encoder(), model),
        dummy_args=(dummy_input_ids, dummy_attention_mask, dummy_bias),
        loss_type='none',
        generate_trng=False,
        quantization_method='none', 
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_encoder
    )

    # -------------------------------------------------------------------------
    # 2. DECODER EXPORT
    # -------------------------------------------------------------------------
    class ExportableT5DecoderBlock(torch.nn.Module):
        def __init__(self, original_block: T5Block):
            super().__init__()
            self.layer = original_block.layer
            # Disable bias generation within layers!
            if hasattr(self.layer[0].SelfAttention, "has_relative_attention_bias"):
                self.layer[0].SelfAttention.has_relative_attention_bias = False
            if hasattr(self.layer[1].EncDecAttention, "has_relative_attention_bias"):
                self.layer[1].EncDecAttention.has_relative_attention_bias = False
                
        def forward(self, hidden_states, attention_mask=None, position_bias=None, 
                    encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None):
            # Self Attention
            self_attn_out = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias)
            hidden_states = self_attn_out[0]
            # Cross Attention
            cross_attn_out = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, 
                                           attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias)
            hidden_states = cross_attn_out[0]
            # FFN
            hidden_states = self.layer[2](hidden_states)
            return (hidden_states,)

    class T5DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.get_decoder()
            self.lm_head = model.lm_head
            
        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            decoder_attention_mask = torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)
            # The original decoder will automatically create extended_mask, causal_mask and call all layers.
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False, 
                return_dict=True
            )
            return self.lm_head(outputs.last_hidden_state)

    # 1. Dummy variables for export
    dummy_decoder_input_ids = torch.ones(1, 3, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(1, 7, model.config.d_model)
    
    # To prevent torch.export from cutting the encoder mask, add zero.
    dummy_encoder_attention_mask = torch.ones(1, 7, dtype=torch.long)
    dummy_encoder_attention_mask[0, -1] = 0 
    
    # 2. Declare ALL dimensions as dynamic
    dec_seq_dim = torch.export.Dim("decoder_sequence", min=1, max=512)
    enc_seq_dim = torch.export.Dim("encoder_sequence", min=1, max=512)
    
    dynamic_shapes_decoder = {
        "decoder_input_ids": {1: dec_seq_dim},
        "encoder_hidden_states": {1: enc_seq_dim},
        "encoder_attention_mask": {1: enc_seq_dim},
    }

    # -------------------------------------------------------------------------
    # 3. MEP ORCHESTRATOR
    # -------------------------------------------------------------------------
    print("--- Compiling MEP plan for T5 (encoder → decoder pipeline) ---")
    c = MEPCompiler()
    ENCODER_ID = 0
    DECODER_ID = 1

    c.res_load_model(ENCODER_ID, "t5-encoder.nac")
    c.res_load_model(DECODER_ID, "t5-decoder.nac")
    c.res_load_extern("tokenizer_runtime", 0, ENCODER_ID) 
    
    # Load ONE large 8x64x64 matrix. We'll slice it right in memory!
    c.res_load_array("full_bias_matrix", "t5_bias")

    c.src_constant("max_decode_steps", 64)
    c.src_constant("model_len", 64)
    c.src_constant("const_one", 1)
    c.src_constant("const_zero", 0)
    c.src_constant("axis_one", 1)
    
    c.src_constant("axis_height", 1) # Matrix height axis
    c.src_constant("axis_width", 2)  # Matrix width axis
    
    c.preproc_get_id("tokenizer_runtime", "</s>", "eos_id")
    c.preproc_get_id("tokenizer_runtime", "<pad>", "pad_id")

    # Input
    c.src_user_prompt("source_text", "Enter text for translation (e.g.: 'translate English to German: Hello'): ")
    c.preproc_encode("tokenizer_runtime", "source_text", "enc_ids_list")
    c.tensor_create(from_py={'out_var': "encoder_input_ids_raw", 'in_var': "enc_ids_list", 'dtype_code': 5})
    
    # Adding </s>
    c.tensor_create(from_py={'out_var': "eos_tensor", 'in_var': "eos_id", 'dtype_code': 5})
    c.tensor_combine('concat', "encoder_input_ids_with_eos", ["encoder_input_ids_raw", "eos_tensor"], axis_var="axis_one")

    # Padding up to 64
    c.tensor_info('dim', "current_seq_len", "encoder_input_ids_with_eos", dim_idx_var="const_one")
    c.math_binary('sub', "pad_len", "model_len", "current_seq_len")
    c.tensor_create(arange={'out_var': "pad_range", 'end_var': "pad_len", 'dtype_code': 5})
    c.math_binary('mul', "pad_zeros", "pad_range", "const_zero")
    c.math_binary('add', "pad_tokens", "pad_zeros", "pad_id")
    c.tensor_combine('concat', "encoder_input_ids", ["encoder_input_ids_with_eos", "pad_tokens"], axis_var="axis_one")

    c.logic_compare('neq', "mask_bool", "encoder_input_ids", "pad_id")
    c.tensor_create(from_py={'out_var': "one_tensor", 'in_var': "const_one", 'dtype_code': 5})
    c.math_binary('mul', "encoder_attention_mask", "mask_bool", "one_tensor")

    # Add the batch dimension to the bias matrix so that it is (1, 8, 64, 64)
    c.src_constant("zero_dim", 0)
    c.tensor_manipulate("unsqueeze", "encoder_bias", "full_bias_matrix", dim_var="zero_dim")

    # ENCODER START
    c.model_run_static(ENCODER_ID,
                       in_vars=["encoder_input_ids", "encoder_attention_mask", "encoder_bias"],
                       out_vars=["encoder_hidden_states"])

    # Decoder Initialization
    c.src_constant("start_shape", [1, 1])
    c.tensor_create(zeros={'out_var': "decoder_ids", 'shape_var': "start_shape", 'dtype_code': 5})

    # Decoding cycle
    c.flow_loop_start("max_decode_steps")
    
    # 1. Get the current length of the decoder
    c.tensor_info('dim', "dec_seq_len", "decoder_ids", dim_idx_var="const_one")
    

    # START DECODER (WITH 5 ARGUMENTS!)
    c.model_run_static(DECODER_ID,
                       in_vars=["decoder_ids", "encoder_hidden_states", "encoder_attention_mask", "encoder_bias", "encoder_bias"],
                       out_vars=["decoder_logits"])

    c.math_binary('sub', "last_dec_idx", "dec_seq_len", "const_one")
    c.tensor_extract("last_logits", "decoder_logits", "last_dec_idx")
    c.math_aggregate('argmax', "next_id_tensor", "last_logits")
    c.tensor_info('to_py', 'next_id', 'next_id_tensor')

    c.tensor_create(from_py={'out_var': "next_id_2d", 'in_var': "next_id", 'dtype_code': 5})
    c.tensor_combine('concat', "decoder_ids", ["decoder_ids", "next_id_2d"], axis_var="axis_one")

    c.logic_compare('eq', "is_eos", "next_id", "eos_id")
    c.flow_break_loop_if("is_eos")
    c.flow_loop_end()

    c.preproc_decode("tokenizer_runtime", "decoder_ids", "output_text")
    c.string_format("final_output", "\n--- Translation Result ---\n  {}", ["output_text"])
    c.io_write("final_output", dest_type=0, write_mode=0)
    c.exec_return(["output_text"])

    mep_program = c.get_program()
    print("--- MEP plan compiled ---")

    print("\n--- Exporting T5 Decoder + MEP pipeline orchestrator ---")
    generate_artifacts(
        model_name="t5-decoder",
        model=T5DecoderWrapper(model),
        dummy_args=(dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
        loss_type='none',
        generate_trng=False,
        quantization_method='none', 
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_decoder,
        mep_program=mep_program,
        mep_arrays=mep_arrays
    )

# ============================================================
# process_sd_unet_vae
# ============================================================

def process_sd_unet_vae():
    print("\n" + "#"*20 + " PROCESSING STABLE DIFFUSION (FINAL DYNAMIC PIPELINE) " + "#"*20)
    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    except ImportError:
        print("Error: 'diffusers' required. pip install diffusers accelerate")
        return

    import struct
    import numpy as np
    import torch
    from MEP_compiler import MEPCompiler

    repo = "runwayml/stable-diffusion-v1-5"
    print(f"Loading pipeline from '{repo}'...")
    pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    
    latent_res = 512 // 8
    NUM_INFERENCE_STEPS = 20
    GUIDANCE_SCALE = 7.5

    # =====================================================================
    # 0. THE PERFECT UNIVERSAL MANIFESTO FOR CLIP
    # =====================================================================
    print("--- Constructing Manual TISA Manifest for CLIP ---")
    clip_manifest = [
        [0x01, {}], # 1. LOWERCASE
        [0x10, {'rules': [
            # IMPORTANT: CLIP ignores spaces when parsing!
            {'pattern': r'\s+', 'behavior': 'REMOVE', 'regex': True},
            # The original CLIP regular expression ([\p{N}] without the plus sign - each number separately)
            {'pattern': r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", 'regex': True},
            {'pattern': r"<\|startoftext\|>", 'protected': True, 'regex': True},
            {'pattern': r"<\|endoftext\|>", 'protected': True, 'regex': True}
        ]}],
        [0x15, {'pipeline': [{'type': 'BYTE_ENCODE'}]}], # 3. BYTE_ENCODE
        [0x20, {'suffix': '</w>'}],                      # 4. BPE_ENCODE with mandatory suffix
        [0x30, {'template': [('FIXED', 49406), ('SLOT', None), ('FIXED', 49407)]}] # 5. COMPOSE (BOS + text + EOS)
    ]

    # =====================================================================
    # 1. PRECALCULATION OF PLANNER ARRAYS FOR MEP
    # =====================================================================
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    timesteps = pipe.scheduler.timesteps.numpy().astype(np.float32)
    sigmas = pipe.scheduler.sigmas.numpy().astype(np.float32)
    
    scale_factors = 1.0 / np.sqrt(sigmas[:-1]**2 + 1.0)
    dts = np.diff(sigmas)
    
    np.random.seed(42) 
    init_noise = np.random.randn(1, 4, latent_res, latent_res).astype(np.float32) * sigmas[0]

    mep_arrays = {
        "sd_timesteps": np.expand_dims(timesteps, axis=(0, 2)),
        "sd_scales": np.expand_dims(scale_factors, axis=(0, 2)),
        "sd_dts": np.expand_dims(dts, axis=(0, 2)),
        "sd_init_noise": init_noise
    }

    # =====================================================================
    # 2-5. EXPORT OF MODELS
    # =====================================================================
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder): super().__init__(); self.encoder = encoder
        def forward(self, input_ids): 
            position_ids = torch.arange(77, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            return self.encoder(input_ids, position_ids=position_ids)[0]
            
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet): super().__init__(); self.unet = unet
        def forward(self, latents, timestep, encoder_hidden_states):
            return self.unet(latents, timestep.reshape(-1), encoder_hidden_states).sample

    class CFGEulerStep(torch.nn.Module):
        def __init__(self, scale): super().__init__(); self.scale = scale
        def forward(self, noise_pred_batch, latents, dt):
            uncond_pred = noise_pred_batch[0:1] 
            text_pred = noise_pred_batch[1:2]   
            noise_pred = uncond_pred + self.scale * (text_pred - uncond_pred)
            return latents + noise_pred * dt.reshape(-1, 1, 1, 1)

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latents): return self.vae.decode(latents, return_dict=False)[0]

    print("--- Exporting Models ---")
    
    generate_artifacts(
        "sd-text-encoder", 
        TextEncoderWrapper(pipe.text_encoder.eval()), 
        (torch.ones(1, 77, dtype=torch.long),), 
        loss_type='none', 
        quantization_method='none', 
        optimize=True, 
        tokenizer_repo="openai/clip-vit-large-patch14", 
        dynamic_shapes=None,
        tokenizer_manifest=clip_manifest
    )
    
    generate_artifacts("sd-unet", UNetWrapper(pipe.unet.eval()), (torch.randn(2, 4, latent_res, latent_res), torch.tensor([999.0]), torch.randn(2, 77, 768)), loss_type='none', quantization_method='none', optimize=True, dynamic_shapes=None)
    generate_artifacts("sd-euler-step", CFGEulerStep(scale=GUIDANCE_SCALE).eval(), (torch.randn(2, 4, latent_res, latent_res), torch.randn(1, 4, latent_res, latent_res), torch.tensor([0.1])), loss_type='none', quantization_method='none', optimize=True)

    # =====================================================================
    # 6. MEP ORCHESTRATOR COMPILATION
    # =====================================================================
    print("--- Compiling MEP Pipeline Orchestrator ---")
    c = MEPCompiler()
    TE_ID, UNET_ID, SCHED_ID, VAE_ID = 0, 1, 2, 3

    c.res_load_model(TE_ID, "sd-text-encoder.nac")
    c.res_load_model(UNET_ID, "sd-unet.nac")
    c.res_load_model(SCHED_ID, "sd-euler-step.nac")
    c.res_load_model(VAE_ID, "sd-vae-decoder.nac") 
    
    c.res_load_extern("tokenizer", 0, TE_ID)
    
    c.res_load_array("latents", "sd_init_noise")
    c.res_load_array("arr_timesteps", "sd_timesteps")
    c.res_load_array("arr_scales", "sd_scales")
    c.res_load_array("arr_dts", "sd_dts")

    c.src_constant("num_steps", NUM_INFERENCE_STEPS)
    c.src_constant("const_zero", 0)
    c.src_constant("const_one", 1)
    c.src_constant("const_77", 77)
    c.src_constant("eos_id", 49407)
    c.src_constant("axis_batch", 0)
    c.src_constant("axis_seq", 1)
    c.src_constant("step_idx", 0)

    # --- A1. Prepare Uncond (Empty Line) ---
    c.src_constant("empty_prompt", "")
    c.preproc_encode("tokenizer", "empty_prompt", "u_list")
    c.tensor_create(from_py={'out_var': "u_raw", 'in_var': "u_list", 'dtype_code': 5})
    
    # Complete the u_raw array with padding up to 77
    c.tensor_info('dim', "u_len", "u_raw", dim_idx_var="const_one")
    c.math_binary('sub', "u_pad_len", "const_77", "u_len")
    c.math_binary('max_elem', "u_pad_safe", "u_pad_len", "const_zero")
    c.tensor_create(arange={'out_var': "u_range", 'end_var': "u_pad_safe", 'dtype_code': 5})
    c.math_binary('mul', "u_zeros", "u_range", "const_zero")
    c.math_binary('add', "u_pads", "u_zeros", "eos_id")
    c.tensor_combine('concat', "uncond_ids_77", ["u_raw", "u_pads"], axis_var="axis_seq")
    c.model_run_static(TE_ID, in_vars=["uncond_ids_77"], out_vars=["uncond_embeds"])

    # --- A2. Preparing Text (User Input) ---
    c.src_user_prompt("prompt", "Enter image description: ")
    c.preproc_encode("tokenizer", "prompt", "t_list")
    c.tensor_create(from_py={'out_var': "t_raw", 'in_var': "t_list", 'dtype_code': 5})

    c.sys_debug_print("t_raw", ">>> Extracted Tokens from TISA (BOS + text + EOS)")
    
    # Pad the t_raw array to 77.
    c.tensor_info('dim', "t_len", "t_raw", dim_idx_var="const_one")
    c.math_binary('sub', "t_pad_len", "const_77", "t_len")
    c.math_binary('max_elem', "t_pad_safe", "t_pad_len", "const_zero")
    c.tensor_create(arange={'out_var': "t_range", 'end_var': "t_pad_safe", 'dtype_code': 5})
    c.math_binary('mul', "t_zeros", "t_range", "const_zero")
    c.math_binary('add', "t_pads", "t_zeros", "eos_id")
    c.tensor_combine('concat', "text_ids_77", ["t_raw", "t_pads"], axis_var="axis_seq")
    c.model_run_static(TE_ID, in_vars=["text_ids_77"], out_vars=["text_embeds"])

    # Building a batch (Uncond + Text)
    c.tensor_combine('concat', "batch_embeds", ["uncond_embeds", "text_embeds"], axis_var="axis_batch")

    c.src_constant("header_msg", f"\nGenerating image ({NUM_INFERENCE_STEPS} Euler steps, CFG={GUIDANCE_SCALE})...")
    c.io_write("header_msg", dest_type=0, write_mode=0)

    # Diffusion Cycle
    c.flow_loop_start("num_steps")

    c.tensor_extract("t", "arr_timesteps", "step_idx")
    c.tensor_extract("scale", "arr_scales", "step_idx")
    c.tensor_extract("dt", "arr_dts", "step_idx")

    c.math_binary('mul', "scaled_latent", "latents", "scale")
    c.tensor_combine('concat', "latent_batch", ["scaled_latent", "scaled_latent"], axis_var="axis_batch")

    c.model_run_static(UNET_ID, in_vars=["latent_batch", "t", "batch_embeds"], out_vars=["noise_pred_batch"])
    c.model_run_static(SCHED_ID, in_vars=["noise_pred_batch", "latents", "dt"], out_vars=["latents"])

    c.math_binary('add', "step_idx", "step_idx", "const_one")
    c.flow_loop_end()

    # Decoding VAE images
    c.src_constant("vae_scale", 5.489980785067252)
    c.math_binary('mul', "latents_for_vae", "latents", "vae_scale")
    c.model_run_static(VAE_ID, in_vars=["latents_for_vae"], out_vars=["decoded_image"])

    # Save
    c.serialize_object("png_bytes", "decoded_image", format_type=2)
    c.src_constant("output_filename", "sd_output.png")
    c.io_write("png_bytes", dest_type=2, dest_var="output_filename", write_mode=0)
    c.src_constant("done_msg", "\n--- Image saved to 'sd_output.png' ---\n")
    c.io_write("done_msg", dest_type=0, write_mode=0)
    c.exec_return(["output_filename"])

    print("--- Exporting VAE Decoder + MEP pipeline orchestrator ---")
    generate_artifacts(
        "sd-vae-decoder", 
        VAEDecoderWrapper(pipe.vae.eval()), 
        (torch.randn(1, 4, latent_res, latent_res),), 
        loss_type='none', generate_trng=False, quantization_method='none', optimize=True, 
        mep_program=c.get_program(), 
        mep_arrays=mep_arrays
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
        process_roberta_base()
        process_t5_translation()
        process_sd_unet_vae()
        process_mobilenet_v3()
        process_resnet18()
        process_distilbert_sentiment()
        process_gpt2()
        print("\n" + "="*20 + " ALL MODELS PROCESSED SUCCESSFULLY " + "="*20)
    except Exception as e:
        print(f"\n!!!!!! ERROR DURING BATCH PROCESSING: {e}")
        traceback.print_exc()
