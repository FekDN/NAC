# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

from NAC import generate_artifacts
import os
import torch
import traceback
import torchvision.models as models
from transformers import GPT2LMHeadModel, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification

def process_gpt2_streaming():
    print("\n" + "#"*20 + " PROCESSING GPT-2 (STATELESS FOR STREAMING) " + "#"*20)
    repo = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(repo, use_cache=False).eval()
    if hasattr(model.config, "_attn_implementation"): model.config._attn_implementation = "eager"
    FIXED_SEQ_LEN = 64
    dummy_input_ids = torch.ones(1, FIXED_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.tril(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN)).bool()[None, None, :, :]
    generate_artifacts(
        model_name="gpt2-streaming", 
        model=Gpt2LogitsWrapper(model), 
        dummy_args=(dummy_input_ids, dummy_mask), 
        quantization_method='none', 
        tokenizer_repo=repo, 
        store_weights_internally=False, 
        optimize=True
    )

def process_sd_unet_vae():
    print("\n" + "#"*20 + " PROCESSING STABLE DIFFUSION (UNET & VAE) " + "#"*20)
    try: from diffusers import StableDiffusionPipeline
    except ImportError: print("Error: 'diffusers' required. pip install diffusers transformers accelerate"); return
    repo = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float32)
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet): super().__init__(); self.unet = unet
        def forward(self, s, t, e): return self.unet(s, t, e).sample
    latent_res = 512//8
    dummy_latent = torch.randn(2, 4, latent_res, latent_res) 
    dummy_text_embed = torch.randn(2, 77, 768)
    dummy_timestep = torch.tensor([999, 999], dtype=torch.float32)
    generate_artifacts(model_name="sd-unet-256", model=UNetWrapper(pipe.unet.eval()), dummy_args=(dummy_latent, dummy_timestep, dummy_text_embed), quantization_method='INT8_TENSOR', optimize=True)
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, latents): return self.vae.decode(latents, return_dict=False)[0]
    dummy_vae_latent = torch.randn(1, 4, latent_res, latent_res)
    generate_artifacts(model_name="sd-vae-decoder-256", model=VAEDecoderWrapper(pipe.vae.eval()), dummy_args=(dummy_vae_latent,), quantization_method='INT8_TENSOR', optimize=True)

def process_roberta_base():
    print("\n" + "#"*20 + " PROCESSING ROBERTA-BASE " + "#"*20)
    repo = "roberta-base"
    model = RobertaForMaskedLM.from_pretrained(repo).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer_input = "Hello I'm a <mask> model."
    inputs = tokenizer(tokenizer_input, return_tensors="pt")
    class RobertaCleanedWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
            self.model.config.use_cache = False
            if hasattr(self.model.config, "_attn_implementation"): self.model.config._attn_implementation = "eager"
            class ZerosModule(torch.nn.Module):
                def __init__(self, ed=768): super().__init__(); self.ed = ed
                def forward(self, ids): return torch.zeros(ids.shape[0], ids.shape[1], self.ed, device=ids.device, dtype=torch.float32)
            self.model.roberta.embeddings.token_type_embeddings = ZerosModule()
        def forward(self, input_ids, attention_mask): return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    dummy_args = (inputs['input_ids'], inputs['attention_mask'])
    seq_dim = torch.export.Dim("sequence", min=1, max=model.config.max_position_embeddings - 2)
    dynamic_shapes = {"input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}}
    generate_artifacts(model_name="roberta-base-fill-mask", model=RobertaCleanedWrapper(model), dummy_args=dummy_args, quantization_method='INT8_TENSOR', dynamic_shapes=dynamic_shapes, tokenizer_repo=repo, optimize=True, tokenizer_input=tokenizer_input)

class Gpt2LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.transformer = model.transformer; self.lm_head = model.lm_head; self.embeddings = model.transformer.wte
    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embeddings(input_ids)
        float_mask = (1.0 - attention_mask.to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=float_mask)
        return self.lm_head(transformer_outputs[0])

def process_gpt2():
    print("\n" + "#"*20 + " PROCESSING GPT-2 " + "#"*20)
    model = GPT2LMHeadModel.from_pretrained("gpt2", use_cache=False).eval()
    if hasattr(model.config, "_attn_implementation"): model.config._attn_implementation = "eager"
    FIXED_SEQ_LEN = 64
    dummy_input_ids = torch.ones(1, FIXED_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.tril(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN)).bool()[None, None, :, :]
    generate_artifacts(
        model_name="gpt2-text-generation",
        model=Gpt2LogitsWrapper(model),
        dummy_args=(dummy_input_ids, dummy_mask),
        quantization_method='INT8_TENSOR',
        optimize=True,
        tokenizer_repo="gpt2"
    )

def process_distilbert_sentiment():
    print("\n" + "#"*20 + " PROCESSING DistilBERT FOR SENTIMENT (SAFE LOADING) " + "#"*20)
    repo = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        tokenizer = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        print(f"!!!!! ERROR: Could not download DistilBERT model '{repo}'.\n{e}")
        return

    # Возвращаемся к простому и правильному wrapper'у
    class DistilBertLogitsWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Используем простой dummy_args, как было изначально.
    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    dummy_args = (inputs['input_ids'], inputs['attention_mask'])
    
    # --- ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ: Изменяем max в Dim ---
    # Задаем максимальную длину на 1 меньше, чем абсолютный максимум модели.
    # Это избегает граничного условия, на котором спотыкается torch.export.
    max_len = model.config.max_position_embeddings
    seq_dim = torch.export.Dim("sequence", min=1, max=max_len - 1)
    # ---------------------------------------------------------
    
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "attention_mask": {1: seq_dim}
    }
    
    generate_artifacts(
        model_name="distilbert-sst2-sentiment",
        model=DistilBertLogitsWrapper(model),
        dummy_args=dummy_args,
        quantization_method='INT8_TENSOR',
        tokenizer_repo=repo,
        store_weights_internally=True,
        optimize=True,
        dynamic_shapes=dynamic_shapes
    )

def process_t5_translation():
    import torch
    import types
    from transformers import T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5Block
    print("\n#################### PROCESSING T5-SMALL (Final Stable Version) ####################")
    repo = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(repo).eval()
    class ExportableT5EncoderBlock(torch.nn.Module):
        def __init__(self, original_block: T5Block):
            super().__init__()
            self.layer = original_block.layer
            if hasattr(self.layer[0].SelfAttention, "has_relative_attention_bias"):
                self.layer[0].SelfAttention.has_relative_attention_bias = False
        def forward(self, hidden_states, attention_mask=None, position_bias=None):
            self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias)
            hidden_states = self_attention_outputs[0]
            hidden_states = self.layer[1](hidden_states)
            return (hidden_states,)

    class T5EncoderWrapper(torch.nn.Module):
        def __init__(self, original_encoder, model_instance):
            super().__init__()
            self.embed_tokens = original_encoder.embed_tokens
            self.final_layer_norm = original_encoder.final_layer_norm
            self.block = torch.nn.ModuleList([ExportableT5EncoderBlock(block) for block in original_encoder.block])
            self.get_extended_attention_mask = model_instance.get_extended_attention_mask
        def forward(self, input_ids, attention_mask, precomputed_bias):
            hidden_states = self.embed_tokens(input_ids)
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
            for layer_module in self.block:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=precomputed_bias)
                hidden_states = layer_outputs[0]
            hidden_states = self.final_layer_norm(hidden_states)
            return hidden_states

    seq_len = 64
    dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    original_compute_bias = model.encoder.block[0].layer[0].SelfAttention.compute_bias
    dummy_bias = original_compute_bias(seq_len, seq_len)
    seq_dim = torch.export.Dim("sequence", min=1, max=512)
    dynamic_shapes_encoder = { "input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}, "precomputed_bias": {2: seq_dim, 3: seq_dim} }
    clean_encoder_wrapper = T5EncoderWrapper(model.get_encoder(), model)

    print("\n--- Exporting T5 Encoder (Final Stable Version) ---")
    generate_artifacts(
        model_name="t5-encoder",
        model=clean_encoder_wrapper,
        dummy_args=(dummy_input_ids, dummy_attention_mask, dummy_bias),
        quantization_method='INT8_TENSOR',
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_encoder
    )

    class T5DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.decoder = model.get_decoder(); self.lm_head = model.lm_head
        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            decoder_attention_mask = torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)
            outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=False, return_dict=True)
            return self.lm_head(outputs.last_hidden_state)

    dummy_decoder_input_ids = torch.ones(1, 2, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(1, 64, model.config.d_model)
    dummy_encoder_attention_mask = torch.ones(1, 64, dtype=torch.long)
    dec_seq_dim = torch.export.Dim("decoder_sequence", min=1, max=512)
    dynamic_shapes_decoder = { "decoder_input_ids": {1: dec_seq_dim}, "encoder_hidden_states": {1: seq_dim}, "encoder_attention_mask": {1: seq_dim}, }

    print("\n--- Exporting T5 Decoder ---")
    generate_artifacts(
        model_name="t5-decoder",
        model=T5DecoderWrapper(model),
        dummy_args=(dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
        quantization_method='INT8_TENSOR',
        tokenizer_repo=repo,
        optimize=True,
        dynamic_shapes=dynamic_shapes_decoder
    )

def process_resnet18():
    print("\n" + "#"*20 + " PROCESSING RESNET-18 " + "#"*20)
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    generate_artifacts(
        model_name="resnet18",
        model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval(),
        dummy_args=(torch.randn(1, 3, 224, 224),),
        quantization_method='INT8_TENSOR',
        optimize = True
    )

if __name__ == "__main__":
    registry_filepath = os.path.join('./', 'registry.json')
    if os.path.exists(registry_filepath):
        try:
            if os.path.getsize(registry_filepath) == 0: os.remove(registry_filepath)
        except OSError as e: print(f"WARNING: Could not check/delete registry.json: {e}")
    try:
        process_resnet18()
        process_distilbert_sentiment()
        process_t5_translation()
        process_roberta_base()
        process_gpt2()
        process_sd_unet_vae()
        process_gpt2_streaming()
        print("\n" + "="*20 + " ALL MODELS PROCESSED SUCCESSFULLY " + "="*20)
    except Exception as e: print(f"\n!!!!!! ERROR DURING BATCH PROCESSING: {e}"); traceback.print_exc()
