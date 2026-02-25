# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
#
# NACmodels_test.py — Universal NAC Model Runner
#
# Every .nac file compiled with MEP support carries a self-describing execution
# plan (ORCH section) that completely specifies:
#   - which model(s) to load
#   - what user inputs to request
#   - how to preprocess data
#   - how to run inference
#   - how to postprocess and display results
#
# run_nac_model() reads that plan and delegates everything to the MEP interpreter.
# No model-specific code is required here.
#
# Usage (CLI):
#   python NACmodels_test.py resnet18.nac
#   python NACmodels_test.py distilbert-sst2-sentiment.nac
#   python NACmodels_test.py gpt2-text-generation.nac
#   python NACmodels_test.py gpt2-streaming.nac
#   python NACmodels_test.py roberta-base-fill-mask.nac
#   python NACmodels_test.py t5-decoder.nac          # MEP plan loads t5-encoder.nac internally
#   python NACmodels_test.py sd-vae-decoder-256.nac  # MEP plan loads sd-unet-256.nac internally

import os
import sys
import numpy as np

from NAC_run import NacRuntime
from NAC_kernels import softmax


# =============================================================================
# Universal runner — the only function needed for MEP-enabled .nac files
# =============================================================================

def run_nac_model(nac_path: str, cli_args: list = None):
    """
    Universal NAC model runner.

    Loads the ORCH section from *nac_path* via NacRuntime.get_mep_plan(),
    creates a MEPInterpreter and executes the embedded plan. The plan handles
    everything autonomously:

        src_user_prompt   → interactive input (или CLI-аргумент из cli_args)
        res_load_dynamic  → dynamic file loading / preprocessing
                            (e.g. ImageNet pipeline для изображений, file_type=3)
        preproc_encode    → tokenization via built-in TISAVM
        model_run_static  → inference through NacRuntime
        io_write          → streaming / formatted output to stdout
        exec_return       → final result value(s)

    Parameters
    ----------
    nac_path  : путь к .nac файлу
    cli_args  : опциональный список CLI-аргументов для автоответа на src_user_prompt
                в порядке поступления. Например:
                    run_nac_model("resnet18.nac", ["hen.jpg"])
                Если список пуст или None — используется интерактивный input().

    Returns the value(s) from exec_return, or None on exec_halt / completion.

    Raises
    ------
    FileNotFoundError : если nac_path не существует
    RuntimeError      : если .nac файл не содержит MEP плана (пустой ORCH)
    """
    if not os.path.isfile(nac_path):
        raise FileNotFoundError(f"NAC file not found: '{nac_path}'")

    # Загружаем .nac чтобы получить ORCH секцию (байткод + пул констант).
    # NacRuntime создаётся MEP планом сам через res_load_model — здесь только читаем ORCH.
    probe = NacRuntime(nac_path)
    bytecode, constants = probe.get_mep_plan()

    if bytecode is None:
        raise RuntimeError(
            f"'{nac_path}' has no MEP execution plan (empty ORCH section).\n"
            "Recompile with CompileTest.py to embed a plan, or use the\n"
            "legacy reference functions below."
        )

    print(f"--- MEP: {len(bytecode)} bytes bytecode, {len(constants)} constants ---")
    from MEP_interpreter import MEPInterpreter
    interpreter = MEPInterpreter(bytecode, constants, pre_answers=cli_args)
    return interpreter.run()


# =============================================================================
# Legacy reference implementations
# =============================================================================
# Retained for:
#   * Testing .nac files compiled WITHOUT a MEP plan
#   * Debugging / validating MEP output against a known-good baseline
#   * Pipelines where preprocessing falls outside MEP ISA v1.0
#     (T5 positional bias, Stable Diffusion CFG guidance, PNDM scheduler)
#
# For MEP-enabled .nac files prefer run_nac_model() above.
# =============================================================================

def run_image_classification(nac: str, img: str = None):
    """ResNet-18 image classification — legacy reference."""
    print("\n--- Running: Image Classification (Legacy) ---")
    import json
    runtime = NacRuntime(nac)
    if not img:
        img = input("Введите путь к изображению (jpg/png/...): ").strip()
    if not img:
        print("ERROR: Image path is required."); return

    # Canonical preprocessing shared with MEP res_load_dynamic(file_type=3)
    image_tensor = NacRuntime.preprocess_image_for_imagenet(img)
    outputs = runtime.run([image_tensor])
    probs   = softmax(outputs[0])
    top5    = np.argsort(probs[0])[::-1][:5]
    print("\n--- Top 5 Predictions ---")
    try:
        with open('imagenet_classes.json', 'r') as f:
            cls = {int(k): v for k, v in json.load(f).items()}
        for i in top5: print(f"  - {cls.get(i, f'ID:{i}')}: {probs[0, i]:.2%}")
    except FileNotFoundError:
        for i in top5: print(f"  - ID:{i}: {probs[0, i]:.2%}")


def run_fill_mask(nac: str, txt: str = None):
    """RoBERTa fill-mask — legacy reference."""
    print("\n--- Running: Fill-Mask (Legacy) ---")
    runtime = NacRuntime(nac)
    if not txt:
        txt = input("Введите предложение с токеном <mask>: ").strip()

    input_ids_list = runtime.encode(txt)
    input_ids      = np.array([input_ids_list], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)

    mask_token_id = 50264   # RoBERTa default
    if runtime.tokenizer and hasattr(runtime.tokenizer, 'res'):
        vocab = runtime.tokenizer.res.get('vocab', {})
        for t in ['<mask>', '[MASK]']:
            if t in vocab:
                mask_token_id = vocab[t]; break

    lifted_one = np.array(1.0, dtype=np.float32)
    logits     = runtime.run([lifted_one, input_ids, attention_mask])[0]
    mask_idx   = np.where(input_ids[0] == mask_token_id)[0][0]
    logits_vec = logits[0, mask_idx]
    K = 5
    topk = np.argpartition(logits_vec, -K)[-K:]
    topk = topk[np.argsort(logits_vec[topk])[::-1]]
    probs = softmax(logits_vec[topk])
    print(f"\n--- Top {K} Predictions ---")
    for idx, p in zip(topk, probs):
        print(f"  - '{runtime.decode([int(idx)], skip_special_tokens=False).strip()}': {p:.2%}")


def run_text_generation(nac_file: str, prompt: str = None, max_new_tokens: int = 30,
                        temperature: float = 0.7, top_k: int = 50):
    """GPT-2 text generation — legacy reference."""
    print("\n--- Running: Text Generation (Legacy) ---")
    runtime = NacRuntime(nac_file)
    if not runtime.tokenizer: print("ERROR: TISAVM not initialized."); return
    if not prompt:
        prompt = input("Введите начало текста: ").strip()

    vocab         = runtime.tokenizer.res.get('vocab', {})
    eos_id        = vocab.get('<|endoftext|>', 50256)
    MODEL_LEN     = 64
    causal_mask   = np.tril(np.ones((MODEL_LEN, MODEL_LEN), dtype=bool))[np.newaxis, np.newaxis]
    generated_ids = np.array([runtime.encode(prompt)], dtype=np.int64)

    print(f"\nOutput: {prompt}", end="", flush=True)
    for _ in range(max_new_tokens):
        ctx        = generated_ids[:, -MODEL_LEN:]
        seq_len    = ctx.shape[1]
        padded     = np.pad(ctx, ((0, 0), (0, MODEL_LEN - seq_len)),
                            'constant', constant_values=eos_id)
        logits     = runtime.run([padded, causal_mask])[0]
        nlt        = logits[0, seq_len - 1, :]
        if temperature > 0:
            nlt /= temperature
            if top_k > 0 and top_k < nlt.shape[-1]:
                nlt[nlt < np.sort(nlt)[-top_k]] = -np.inf
            probs = softmax(nlt)
            next_id = np.array([[np.random.choice(probs.shape[0], p=probs)]])
        else:
            next_id = np.array([[np.argmax(nlt)]])
        generated_ids = np.concatenate([generated_ids, next_id], axis=1)
        print(runtime.decode(next_id), end="", flush=True)
        if next_id[0, 0] == eos_id: break
    print("\n--- Generation Complete ---")


def run_streaming_generation(nac_path: str, prompt: str = None, max_new_tokens: int = 50,
                              temperature: float = 0.7, top_k: int = 50):
    """GPT-2 stateless streaming — legacy reference (no transformers dependency)."""
    print("\n--- Running: Streaming Generation (Legacy) ---")
    import time
    runtime = NacRuntime(nac_path)
    if not runtime.tokenizer: print("ERROR: TISAVM not initialized."); return
    if not prompt:
        prompt = input("Введите начало текста (streaming): ").strip()

    vocab     = runtime.tokenizer.res.get('vocab', {})
    eos_id    = vocab.get('<|endoftext|>', 50256)
    MODEL_LEN = 64
    input_ids = np.array([runtime.encode(prompt)], dtype=np.int64)

    print(f"Output: {prompt}", end="", flush=True)
    t0 = time.time()
    for _ in range(max_new_tokens):
        cur    = input_ids[:, -MODEL_LEN:]
        slen   = cur.shape[1]
        pad_n  = MODEL_LEN - slen
        padded = np.pad(cur, ((0, 0), (0, pad_n)), 'constant', constant_values=eos_id) if pad_n > 0 else cur
        # Dynamic mask: causal AND attention (correct for right-padded inputs)
        attn   = (padded != eos_id)
        causal = np.tril(np.ones((MODEL_LEN, MODEL_LEN), dtype=bool))
        mask   = np.logical_and(attn[:, np.newaxis, :], causal[np.newaxis])[:, np.newaxis]
        logits = runtime.run([padded, mask])[0]
        nlt    = logits[0, slen - 1, :]
        if temperature > 0:
            nlt = nlt / temperature
            if top_k > 0 and top_k < nlt.shape[-1]:
                nlt[nlt < np.sort(nlt)[-top_k]] = -np.inf
            probs  = softmax(nlt)
            next_id = np.array([[np.random.choice(probs.shape[0], p=probs)]])
        else:
            next_id = np.array([[np.argmax(nlt)]])
        input_ids = np.concatenate([input_ids, next_id], axis=1)
        if next_id[0, 0] == eos_id: break
        print(runtime.decode(next_id), end="", flush=True)
    dur = time.time() - t0
    gen = input_ids.shape[1] - len(runtime.encode(prompt))
    print(f"\n\n--- {gen} tokens in {dur:.3f}s ({gen/dur:.1f} tok/s) ---")


def run_sentiment_analysis(nac_file: str, text: str = None):
    """DistilBERT sentiment analysis — legacy reference."""
    print("\n--- Running: Sentiment Analysis (Legacy) ---")
    runtime = NacRuntime(nac_file)
    if not runtime.tokenizer: print("ERROR: TISAVM not initialized."); return
    if not text:
        text = input("Введите текст для анализа: ").strip()

    ids    = np.array([runtime.encode(text)[:512]], dtype=np.int64)
    attn   = np.ones_like(ids, dtype=np.int64)
    pos    = np.arange(ids.shape[1], dtype=np.int64).reshape(1, -1)
    one    = np.array(1.0, dtype=np.float32)
    logits = runtime.run([one, ids, attn, pos])[0]
    pred   = int(np.argmax(logits, axis=-1)[0])
    label  = {0: "NEGATIVE", 1: "POSITIVE"}.get(pred, "UNKNOWN")
    probs  = softmax(logits[0])
    print(f"\n--- Sentiment Analysis ---\n"
          f"  Prediction: {label}\n"
          f"  Confidence (NEG): {probs[0]:.2%}\n"
          f"  Confidence (POS): {probs[1]:.2%}")


def run_translation_t5(encoder_nac: str, decoder_nac: str, text: str = None,
                       max_new_tokens: int = 50):
    """T5 encoder-decoder translation — legacy reference (requires torch+transformers)."""
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("\n--- Running: T5 Translation (Legacy) ---")
    if not text:
        text = input("Введите текст для перевода: ").strip()

    enc = NacRuntime(encoder_nac)
    dec = NacRuntime(decoder_nac)
    tok = T5Tokenizer.from_pretrained("t5-small", local_files_only=True)
    mdl = T5ForConditionalGeneration.from_pretrained("t5-small").eval()
    inp = tok(f"translate English to German: {text}", return_tensors="pt", truncation=True)
    slen = inp["input_ids"].shape[1]
    with torch.no_grad():
        bias = mdl.encoder.block[0].layer[0].SelfAttention.compute_bias(slen, slen)
    enc_h = enc.run([inp["input_ids"].numpy(), inp["attention_mask"].numpy(), bias.numpy()])[0]
    dec_ids = np.array([[tok.pad_token_id]], dtype=np.int64)
    for _ in range(max_new_tokens):
        logits  = dec.run([dec_ids, enc_h, inp["attention_mask"].numpy()])[0]
        next_id = np.array([[np.argmax(logits[0, -1, :])]])
        dec_ids = np.concatenate([dec_ids, next_id], axis=1)
        if next_id[0, 0] == tok.eos_token_id: break
    result = tok.decode(dec_ids[0], skip_special_tokens=True)
    print(f"\n--- Translation ---\n  Input:  '{text}'\n  Output: '{result}'")


def run_image_generation_sd(unet_nac: str, vae_nac: str, prompt: str = None):
    """Stable Diffusion — legacy reference (requires torch+diffusers+transformers+Pillow)."""
    print("\n--- Running: Stable Diffusion (Legacy) ---")
    try:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import PNDMScheduler
        from PIL import Image
    except ImportError:
        print("Requires: torch, diffusers, transformers, Pillow"); sys.exit(1)
    if not prompt:
        prompt = input("Введите промпт: ").strip()

    unet = NacRuntime(unet_nac)
    vae  = NacRuntime(vae_nac)
    repo = "runwayml/stable-diffusion-v1-5"
    tok  = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    enc  = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").eval()
    sch  = PNDMScheduler.from_pretrained(repo, subfolder="scheduler")
    h, w, gs, ns, seed = 512, 512, 7.5, 50, 42
    ti = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
    ui = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt")
    with torch.no_grad():
        emb = torch.cat([enc(ui.input_ids)[0], enc(ti.input_ids)[0]]).cpu().numpy().astype(np.float32)
    lat = torch.randn((1, 4, h//8, w//8), generator=torch.manual_seed(seed)) * sch.init_noise_sigma
    sch.set_timesteps(ns)
    for i, t in enumerate(sch.timesteps):
        print(f"  Step {i+1}/{ns}")
        inp = sch.scale_model_input(torch.cat([lat]*2), t).numpy().astype(np.float32)
        ts  = np.array([t.item(), t.item()], dtype=np.float32)
        pred = unet.run([inp, ts, emb])[0]
        pu, pt = np.split(pred, 2)
        lat = sch.step(torch.from_numpy(pu + gs*(pt-pu)), t, lat).prev_sample
    img = np.clip(vae.run([lat.numpy()/0.18215])[0]/2+0.5, 0, 1)[0].transpose(1,2,0)
    Image.fromarray((img*255).round().astype(np.uint8)).save("sd_output.png")
    print("\n--- Image saved to sd_output.png ---")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Universal NAC Model Runner\n"
            "\nPreferred usage (MEP-enabled .nac — fully autonomous):\n"
            "  python NACmodels_test.py <model.nac>\n"
            "\nThe embedded MEP plan handles all user prompts and preprocessing.\n"
            "\nLegacy usage (for .nac files without MEP plan):\n"
            "  python NACmodels_test.py resnet18.nac            [img.jpg]\n"
            "  python NACmodels_test.py roberta...nac           \"text <mask> ...\"\n"
            "  python NACmodels_test.py gpt2...nac              \"prompt\"\n"
            "  python NACmodels_test.py gpt2-streaming...nac\n"
            "  python NACmodels_test.py distilbert...nac        \"text\"\n"
            "  python NACmodels_test.py sd-unet...nac           \"prompt\"\n"
            "  python NACmodels_test.py t5-encoder.nac t5-decoder.nac \"text\""
        )
        sys.exit(1)

    nac_file1 = sys.argv[1]
    nac_file2 = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".nac") else ""
    rest      = sys.argv[(3 if nac_file2 else 2):]
    arg       = " ".join(rest) if rest else None

    try:
        # ── Try universal MEP runner first ──────────────────────────────────
        # CLI args after the .nac filename(s) are passed as pre_answers for
        # src_user_prompt instructions, so the plan runs non-interactively:
        #   python NACmodels_test.py resnet18.nac hen.jpg
        #   python NACmodels_test.py distilbert.nac "This movie is great"
        run_nac_model(nac_file1, cli_args=rest or None)

    except RuntimeError as mep_err:
        # ── Fallback: legacy runner for .nac files without a MEP plan ───────
        print(f"[INFO] MEP runner: {mep_err}\n[INFO] Falling back to legacy runner.")
        try:
            if   "resnet"    in nac_file1: run_image_classification(nac_file1, arg)
            elif "roberta"   in nac_file1: run_fill_mask(nac_file1, arg)
            elif "streaming" in nac_file1: run_streaming_generation(nac_file1, arg)
            elif "distilbert"in nac_file1: run_sentiment_analysis(nac_file1, arg)
            elif "gpt2"      in nac_file1: run_text_generation(nac_file1, arg)
            elif "t5-"       in nac_file1: run_translation_t5(nac_file1, nac_file2, arg)
            elif "sd-"       in nac_file1:
                run_image_generation_sd(nac_file1, nac_file1.replace("unet","vae-decoder"), arg)
            else:
                print(f"Cannot determine task from '{nac_file1}'.")
        except Exception as e:
            print(f"\n!!!!!! ERROR: {e}")
            import traceback; traceback.print_exc()

    except Exception as e:
        print(f"\n!!!!!! ERROR: {e}")
        import traceback; traceback.print_exc()