# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import os
import sys
import numpy as np

from NAC_run import NacRuntime
from TISA_tokenizer import TISAVM
from NAC_kernels import softmax

def run_image_classification(nac, img):
    print("\n--- Running: Image Classification ---")
    os.environ['HF_HUB_OFFLINE'] = '1';
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1';
    os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    runtime = NacRuntime(nac)
    import json
    def preprocess_image(image_path: str) -> np.ndarray:
        try: from PIL import Image
        except ImportError: print("Error: Pillow required. pip install Pillow"); sys.exit(1)
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        left, top = (256 - 224) / 2, (256 - 224) / 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0)
    outputs = runtime.run([preprocess_image(img)])
    probs = softmax(outputs[0])
    top5 = np.argsort(probs[0])[::-1][:5]
    print("\n--- Top 5 Predictions ---")
    try:
        with open('imagenet_classes.json', 'r') as f: imagenet_classes = {int(k): v for k, v in json.load(f).items()}
        for i in top5: print(f"  - {imagenet_classes.get(i, f'ID:{i}')}: {probs[0,i]:.2%}")
    except FileNotFoundError:
        for i in top5: print(f"  - ID:{i}: {probs[0,i]:.2%}")

'''
# Example with the original tokenizer
def run_fill_mask(nac, txt):
    print("\n--- Running: Fill-Mask ---")
    os.environ['HF_HUB_OFFLINE'] = '1';
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1';
    os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    from transformers import AutoTokenizer
    import numpy as np
    if not hasattr(run_fill_mask, "_tok"): run_fill_mask._tok = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
    tok = run_fill_mask._tok
    if tok.mask_token not in txt: raise ValueError(f"Input must contain {tok.mask_token}")
    inputs = tok(txt, return_tensors="np")
    runtime = NacRuntime(nac)
    lifted_one = np.array(1.0, dtype=np.float32)
    all_inputs = [lifted_one, inputs["input_ids"], inputs["attention_mask"]]
    logits = runtime.run(all_inputs)[0]
    mask_idx = np.where(inputs["input_ids"][0] == tok.mask_token_id)[0][0]
    logits_vec = logits[0, mask_idx]
    K_out = 5
    topk_indices = np.argpartition(logits_vec, -K_out)[-K_out:]
    topk_sorted = topk_indices[np.argsort(logits_vec[topk_indices])[::-1]]
    top_logits = logits_vec[topk_sorted]
    probs = softmax(top_logits)
    print(f"\n--- Top {K_out} Predictions for '{tok.mask_token}' ---")
    for idx, p in zip(topk_sorted, probs): print(f"  - '{tok.decode(int(idx)).strip()}': {p:.2%}")
'''

# Example with built-in tokenizer
def run_fill_mask(nac, txt):
    print("\n--- Running: Fill-Mask (Autonomous) ---")
    # No dependency on transformers
    import numpy as np
    runtime = NacRuntime(nac)
    # Use the built-in tokenizer
    input_ids_list = runtime.encode(txt)
    input_ids = np.array([input_ids_list], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    # Find the token ID <mask> in the runtime dictionary
    mask_token_id = None
    if runtime.tokenizer and hasattr(runtime.tokenizer, 'res') and 'vocab' in runtime.tokenizer.res:
        vocab = runtime.tokenizer.res['vocab']
        # RoBERTa uses '<mask>', BERT '[MASK]'
        for token in ['<mask>', '[MASK]']:
            if token in vocab:
                mask_token_id = vocab[token]
                print(f"Identified mask token: '{token}' (ID: {mask_token_id})")
                break
    if mask_token_id is None:
        print("Warning: Could not identify mask token ID from vocab. Using default RoBERTa ID (50264).")
        mask_token_id = 50264 # Fallback for RoBERTa
    if mask_token_id not in input_ids_list:
        raise ValueError(f"Input text must contain the mask token (ID {mask_token_id}). Tokenized: {input_ids_list}")
    # Preparing inputs for the graph (for RoBERTa: [lifted_one, input_ids, attention_mask])
    lifted_one = np.array(1.0, dtype=np.float32)
    all_inputs = [lifted_one, input_ids, attention_mask]
    # Launching the model
    logits = runtime.run(all_inputs)[0]
    # Search for mask index
    mask_idx = np.where(input_ids[0] == mask_token_id)[0][0]
    logits_vec = logits[0, mask_idx]
    # Top 5 Predictions
    K_out = 5
    topk_indices = np.argpartition(logits_vec, -K_out)[-K_out:]
    topk_sorted = topk_indices[np.argsort(logits_vec[topk_indices])[::-1]]
    top_logits = logits_vec[topk_sorted]
    probs = softmax(top_logits)
    print(f"\n--- Top {K_out} Predictions ---")
    for idx, p in zip(topk_sorted, probs):
        # Decode the ID back into text using the built-in tokenizer
        token_str = runtime.decode([int(idx)], skip_special_tokens=False).strip()
        print(f"  - '{token_str}': {p:.2%}")

'''
# Example with the original tokenizer
def run_text_generation(nac_file, prompt, max_new_tokens=30, temperature=0.7, top_k=50):
    print("\n--- Running: Text Generation (Fixed-Length, with Sampling) ---")
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'; os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    import numpy as np
    from transformers import AutoTokenizer
    MODEL_FIXED_LEN = 64
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    runtime = NacRuntime(nac_file)
    causal_mask = np.tril(np.ones((MODEL_FIXED_LEN, MODEL_FIXED_LEN), dtype=bool))[np.newaxis, np.newaxis, :, :]
    generated_ids = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int64)
    print(f"\nOutput: {prompt}", end="", flush=True)
    for _ in range(max_new_tokens):
        context_ids = generated_ids[:, -MODEL_FIXED_LEN:]
        current_seq_len = context_ids.shape[1]
        input_ids_padded = np.pad(context_ids, ((0, 0), (0, MODEL_FIXED_LEN - current_seq_len)), 'constant', constant_values=tokenizer.pad_token_id)
        logits = runtime.run([input_ids_padded, causal_mask])[0]
        next_token_logits = logits[0, current_seq_len - 1, :]
        if temperature > 0:
            next_token_logits /= temperature
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                kth_logit = np.sort(next_token_logits, axis=-1)[-top_k]
                next_token_logits[next_token_logits < kth_logit] = -np.inf
            probs = softmax(next_token_logits)
            next_token_id = np.array([[np.random.choice(probs.shape[0], p=probs)]])
        else:
            next_token_id = np.array([[np.argmax(next_token_logits, axis=-1)]])
        generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)
        print(tokenizer.decode(next_token_id[0]), end="", flush=True)
        if next_token_id[0, 0] == tokenizer.eos_token_id: break
    print("\n--- Generation Complete ---")
'''

# Example with built-in tokenizer
def run_text_generation(nac_file, prompt, max_new_tokens=30, temperature=0.7, top_k=50):
    print("\n--- Running: Text Generation (Autonomous) ---")
    import numpy as np
    runtime = NacRuntime(nac_file)
    if not runtime.tokenizer:
        print("ERROR: TISAVM not initialized. Cannot run text generation.")
        return
    # Obtaining special token IDs from the dictionary
    vocab = runtime.tokenizer.res.get('vocab', {})
    eos_token_id = vocab.get('<|endoftext|>', 50256) # Standard EOS for GPT-2
    pad_token_id = eos_token_id # GPT-2 uses EOS as a PAD
    MODEL_FIXED_LEN = 64
    # Encoding the initial prompt
    generated_ids = np.array([runtime.encode(prompt)], dtype=np.int64)
    # The static causal mask with which the model was exported
    causal_mask = np.tril(np.ones((MODEL_FIXED_LEN, MODEL_FIXED_LEN), dtype=bool))[np.newaxis, np.newaxis, :, :]
    print(f"\nOutput: {prompt}", end="", flush=True)
    for _ in range(max_new_tokens):
        context_ids = generated_ids[:, -MODEL_FIXED_LEN:]
        current_seq_len = context_ids.shape[1]
        input_ids_padded = np.pad(
            context_ids,
            ((0, 0), (0, MODEL_FIXED_LEN - current_seq_len)),
            'constant',
            constant_values=pad_token_id
        )
        logits = runtime.run([input_ids_padded, causal_mask])[0]
        next_token_logits = logits[0, current_seq_len - 1, :]
        # Sampling logic
        if temperature > 0:
            next_token_logits /= temperature
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                kth_logit = np.sort(next_token_logits, axis=-1)[-top_k]
                next_token_logits[next_token_logits < kth_logit] = -np.inf
            probs = softmax(next_token_logits)
            next_token_id = np.array([[np.random.choice(probs.shape[0], p=probs)]])
        else:
            next_token_id = np.array([[np.argmax(next_token_logits, axis=-1)]])
        generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)
        # Decode and output only the new token.
        print(runtime.decode(next_token_id), end="", flush=True)
        if next_token_id[0, 0] == eos_token_id:
            break
    print("\n--- Generation Complete ---")


def run_streaming_generation(nac_path, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    print("\n--- Running: Text Generation (Stateless Emulated Streaming) ---")
    os.environ['HF_HUB_OFFLINE'] = '1';
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1';
    os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    from transformers import AutoTokenizer
    import time
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    runtime = NacRuntime(nac_path)
    MODEL_MAX_LEN = 64
    input_ids = tok(prompt, return_tensors="np")['input_ids'].astype(np.int64)
    print(f"Generating {max_new_tokens} new tokens...\n\nOutput: {prompt}", end="", flush=True)
    start_time = time.time()
    for _ in range(max_new_tokens):
        current_input_ids = input_ids[:, -MODEL_MAX_LEN:]
        seq_len = current_input_ids.shape[1]
        padding_needed = MODEL_MAX_LEN - seq_len
        padded_input_ids = np.pad(current_input_ids, ((0, 0), (0, padding_needed)), 'constant', constant_values=tok.pad_token_id) if padding_needed > 0 else current_input_ids
        attention_mask = (padded_input_ids != tok.pad_token_id).astype(bool)
        causal_mask = np.tril(np.ones((MODEL_MAX_LEN, MODEL_MAX_LEN), dtype=bool))
        final_mask = np.logical_and(attention_mask[:, np.newaxis, :], causal_mask[np.newaxis, :, :])[:, np.newaxis, :, :]
        logits = runtime.run([padded_input_ids, final_mask])[0]
        next_token_logits = logits[0, seq_len - 1, :]
        if temperature > 0:
            next_logits_processed = next_token_logits / temperature
            if top_k > 0 and top_k < next_logits_processed.shape[-1]:
                kth_logit = np.sort(next_logits_processed, axis=-1)[-top_k]
                next_logits_processed[next_logits_processed < kth_logit] = -np.inf
            probs = softmax(next_logits_processed)
            next_token_id = np.array([[np.random.choice(probs.shape[0], p=probs)]])
        else:
            next_token_id = np.array([[np.argmax(next_token_logits, axis=-1)]])
        input_ids = np.concatenate([input_ids, next_token_id], axis=1)
        if next_token_id[0, 0] == tok.eos_token_id: break
        print(tok.decode(next_token_id[0]), end="", flush=True)
    generation_duration = time.time() - start_time
    generated_tokens = input_ids.shape[1] - len(tok(prompt, return_tensors="np")['input_ids'][0])
    print("\n\n--- Performance ---")
    if generated_tokens > 0:
        print(f"Token generation ({generated_tokens} tokens): {generation_duration:.4f} seconds")
        print(f"Tokens per second (emulated streaming): {generated_tokens / generation_duration:.2f} tok/s")

def run_image_generation_sd(unet_nac_path: str, vae_nac_path: str, prompt: str):
    print("\n--- Running: Image Generation (Stable Diffusion) ---")
    os.environ['HF_HUB_OFFLINE'] = '1';
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1';
    os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    try:
        import torch; from transformers import CLIPTextModel, CLIPTokenizer; from diffusers import PNDMScheduler; from PIL import Image
    except ImportError: print("Error: 'diffusers', 'transformers', 'torch', 'Pillow' are required."); sys.exit(1)
    unet_runtime, vae_runtime = NacRuntime(unet_nac_path), NacRuntime(vae_nac_path)
    repo = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer"); text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").eval(); scheduler = PNDMScheduler.from_pretrained(repo, subfolder="scheduler")
    height, width, guidance_scale, num_inference_steps, seed = 512, 512, 7.5, 50, 42
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = torch.cat([text_encoder(uncond_input.input_ids)[0], text_encoder(text_input.input_ids)[0]]).cpu().numpy().astype(np.float32)
    print(f"Generating latents for '{prompt}' ({num_inference_steps} steps)...")
    generator = torch.manual_seed(seed)
    latents_torch = torch.randn((1, 4, height // 8, width // 8), generator=generator, dtype=torch.float32)
    scheduler.set_timesteps(num_inference_steps)
    latents_torch = latents_torch * scheduler.init_noise_sigma
    for i, t in enumerate(scheduler.timesteps):
        print(f"  Step {i+1}/{num_inference_steps} (Timestep {t.item()})")
        latent_model_input = torch.cat([latents_torch] * 2)
        scaled_latent_input = scheduler.scale_model_input(latent_model_input, t).numpy().astype(np.float32)
        timestep_input = np.array([t.item(), t.item()], dtype=np.float32)
        noise_pred = unet_runtime.run([scaled_latent_input, timestep_input, text_embeddings])[0]
        noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
        noise_pred = torch.from_numpy(noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond))
        latents_torch = scheduler.step(noise_pred, t, latents_torch).prev_sample
    print("\nDecoding latents into image...")
    image = vae_runtime.run([latents_torch.numpy() / 0.18215])[0]
    image = np.clip(image / 2 + 0.5, 0, 1)[0].transpose(1, 2, 0)
    pil_image = Image.fromarray((image * 255).round().astype(np.uint8))
    pil_image.save("sd_output.png")
    print(f"\n--- Image saved to sd_output.png ---")

'''
# Example with the original tokenizer
def run_sentiment_analysis(nac_file: str, text: str):
    print("\n--- Running: Sentiment Analysis ---")
    os.environ['HF_HUB_OFFLINE'] = '1';
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1';
    os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
    from transformers import AutoTokenizer
    import numpy as np
    if not hasattr(run_sentiment_analysis, "_tok"):
        repo = "distilbert-base-uncased-finetuned-sst-2-english"
        run_sentiment_analysis._tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    tok = run_sentiment_analysis._tok
    inputs = tok(text, return_tensors="np")
    all_inputs = [np.arange(0, inputs["input_ids"].shape[1], dtype=np.int64).reshape(1, -1), np.array(1.0, dtype=np.float32), inputs["input_ids"], inputs["attention_mask"]]
    runtime = NacRuntime(nac_file)
    logits = runtime.run(all_inputs)[0]
    prediction_idx = np.argmax(logits, axis=-1)[0]
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    probabilities = softmax(logits[0])
    print(f"\n--- Analysis Results for '{text}'---\n  Prediction: {label_map.get(prediction_idx, 'UNKNOWN')}\n  Confidence (NEGATIVE): {probabilities[0]:.2%}\n  Confidence (POSITIVE): {probabilities[1]:.2%}")
'''

# Example with built-in tokenizer
def run_sentiment_analysis(nac_file: str, text: str):
    print("\n--- Running: Sentiment Analysis (Autonomous) ---")
    import numpy as np
    
    runtime = NacRuntime(nac_file)
    if not runtime.tokenizer:
        print("ERROR: TISAVM not initialized. Cannot run sentiment analysis.")
        return

    # --- Шаг 1: Создаем все необходимые тензоры ---
    input_ids_list = runtime.encode(text)
    max_len = 512 
    if len(input_ids_list) > max_len:
        input_ids_list = input_ids_list[:max_len]
        
    input_ids = np.array([input_ids_list], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64) # Маска должна быть int64, как и input_ids
    seq_len = input_ids.shape[1]
    position_ids = np.arange(0, seq_len, dtype=np.int64).reshape(1, -1)
    lifted_one = np.array(1.0, dtype=np.float32)

    # --- Шаг 2: Собираем список входов в ТОЧНОМ ПОРЯДКЕ, который мы определили по реконструкции ---
    # v0 = unnamed_input_0  -> lifted_one
    # v1 = input_ids         -> input_ids
    # v2 = attention_mask    -> attention_mask
    # v6 = unnamed_input_6  -> position_ids
    #
    # Поскольку рантайм просто берет входы по порядку, мы должны их передать именно так.
    
    all_inputs = [
        lifted_one,      # для v0
        input_ids,       # для v1
        attention_mask,  # для v2
        position_ids     # для v6
    ]

    print("\n--- Final Model Inputs (in order) ---")
    print(f"  Input 0 (for v0): Lifted Constant {lifted_one.shape}")
    print(f"  Input 1 (for v1): input_ids {input_ids.shape}")
    print(f"  Input 2 (for v2): attention_mask {attention_mask.shape}")
    print(f"  Input 3 (for v6): position_ids {position_ids.shape}")

    # --- Шаг 3: Выполняем модель ---
    print("\nExecuting NAC runtime...")
    logits = runtime.run(all_inputs)[0]
    
    # --- Постобработка результата ---
    prediction_idx = np.argmax(logits, axis=-1)[0]
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    probabilities = softmax(logits[0])
    
    print(f"\n--- Analysis Results for '{text}'---")
    print(f"  Prediction: {label_map.get(prediction_idx, 'UNKNOWN')}")
    print(f"  Confidence (NEGATIVE): {probabilities[0]:.2%}")
    print(f"  Confidence (POSITIVE): {probabilities[1]:.2%}")

def run_translation_t5(encoder_nac_path: str, decoder_nac_path: str, text_to_translate: str, max_new_tokens: int = 50):
    import numpy as np; import torch; from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("\n--- Running: Translation with T5 (Encoder–Decoder) ---")
    os.environ["HF_HUB_OFFLINE"] = "1";
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder_runtime, decoder_runtime = NacRuntime(encoder_nac_path), NacRuntime(decoder_nac_path)
    repo = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(repo, local_files_only=True)
    pt_model = T5ForConditionalGeneration.from_pretrained(repo).eval()
    input_text = f"translate English to German: {text_to_translate}"
    inputs_pt = tokenizer(input_text, return_tensors="pt", truncation=True)
    seq_len = inputs_pt["input_ids"].shape[1]
    with torch.no_grad(): bias_tensor = pt_model.encoder.block[0].layer[0].SelfAttention.compute_bias(seq_len, seq_len)
    print(f"Running NAC Encoder...")
    encoder_hidden_states_np = encoder_runtime.run([inputs_pt["input_ids"].numpy(), inputs_pt["attention_mask"].numpy(), bias_tensor.numpy()])[0]
    print("Encoder finished. Starting Decoder generation loop...")
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    for _ in range(max_new_tokens):
        logits = decoder_runtime.run([decoder_input_ids, encoder_hidden_states_np, inputs_pt["attention_mask"].numpy()])[0]
        next_token_id = np.array([[np.argmax(logits[0, -1, :])]])
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=1)
        if next_token_id[0, 0] == tokenizer.eos_token_id: break
    translated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    print(f"\n--- NAC Translation Complete ---\n  Input:  '{text_to_translate}'\n  Output: '{translated_text}'")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:\n  Image Classify: python NACrun.py resnet18.nac <img.jpg>\n  Fill Mask:      python NACrun.py roberta...nac \"txt <mask> ...\"\n  Text Gen:       python NACrun.py gpt2...nac \"prompt...\"\n  Sentiment:      python NACrun.py distilbert...nac \"text...\"\n  Image Gen (SD): python NACrun.py sd-unet...nac \"prompt...\"\n  Translation:    python NACrun.py t5-encoder.nac t5-decoder.nac \"text...\"")
        sys.exit(1)
    
    nac_file1 = sys.argv[1]
    nac_file2 = sys.argv[2] if len(sys.argv) > 2 and ".nac" in sys.argv[2] else ""
    prompt = " ".join(sys.argv[3:]) if nac_file2 else " ".join(sys.argv[2:])

    try:
        if "resnet" in nac_file1: run_image_classification(nac_file1, prompt)
        elif "roberta" in nac_file1: run_fill_mask(nac_file1, prompt)
        elif "gpt2-streaming" in nac_file1: run_streaming_generation(nac_file1, prompt)
        elif "distilbert" in nac_file1: run_sentiment_analysis(nac_file1, prompt)
        elif "gpt2" in nac_file1: run_text_generation(nac_file1, prompt)
        elif "t5-" in nac_file1: run_translation_t5(nac_file1, nac_file2, prompt)
        elif "sd-unet" in nac_file1: 
            vae_path = nac_file1.replace("unet", "vae-decoder")
            run_image_generation_sd(nac_file1, vae_path, prompt)
        else: print(f"Warning: Cannot determine task from filename '{nac_file1}'.")
    except Exception as e:
        print(f"\n!!!!!! ERROR: {e}"); import traceback; traceback.print_exc()