# NAC-Core Verilog

NAC-Core is a synthesizable soft-core shell for NAC v1.8 execution. It follows
the decoupled access/execute model used by the reference runtime:

- ORCH/dispatcher stream decodes NAC ABCD instructions from `OPS`.
- Fixed numeric operation IDs are used only for `NAC_STD_*`; ATen/CMAP
  operation IDs are loaded per model as `A -> hardware recipe` entries.
- Optional `cmd_load_model` / `cmd_run_inference` split caches decoded
  microcode per context, so repeated inference can run without reparsing `OPS`.
- MMAP engine executes scheduled `PRELOAD`, `FREE`, `SAVE_RESULT`, and `FORWARD`
  commands on instruction ticks, with optional static/pinned entries.
- A programmable vector DSP pipeline exposes one physical MAC array, one SFU
  lane array, and reduction hardware that can be time-multiplexed by kernels
  such as matmul, Conv2D, LayerNorm, and Softmax.
- Cached execution can evaluate standard `<CONTROL_FLOW>` entries through an
  external predicate response and redirect the instruction cache PC.
- Reliability blocks include progress-based watchdog reset and SECDED ECC for
  scratchpad/result-table storage.
- Compression/decode primitives cover BFP microscaling, structured N:M
  sparsity, palette weight decode, zero-mask compaction, and zero-run RLE. The
  `nac_codec_dsp_pipeline` wrapper wires registered decode stages into the DSP
  input path and RLE encode sideband.
- TISA packetization validates manifest framing, and `nac_tisa_tokenizer`
  wires the packetizer into a safe local text frontend. The local frontend
  executes exact ASCII `LOWERCASE` and GPT-2/TISA `BYTE_ENCODE`; Unicode,
  regex/partition, BPE, WordPiece, Unigram, and compose stages are flagged as
  requiring an external tokenizer engine instead of being approximated.

Important source files:

- `rtl/nac_core.v` - top-level orchestration shell.
- `rtl/nac_abcd_decoder.v` - NAC ABCD instruction decoder.
- `rtl/nac_descriptor.vh` - descriptor flag layout (`is_bfp`,
  `is_sparse_2_4`, `is_rle`, palette/static flags).
- `rtl/nac_mmap_engine.v` - independent memory schedule engine.
- `rtl/nac_dsp_pipeline.v` - reusable vector DSP datapath.
- `rtl/nac_codec_dsp_pipeline.v` - registered memory-format decode wrapper
  feeding `nac_dsp_pipeline`.
- `rtl/nac_tensor_codec_pipeline.v` - pipelined BFP/palette/sparsity decode.
- `rtl/nac_kernel_sequencer.v` - multi-pass LayerNorm/Softmax mode sequencer.
- `rtl/nac_scratchpad.v` - dual-port banked BRAM scratchpad.
- `rtl/nac_instruction_cache.v` - decoded ABCD microcode cache per context.
- `rtl/nac_csr_bank.v` - per-context configuration/status registers.
- `rtl/nac_ecc_secded.v` and `rtl/nac_watchdog.v` - reliability primitives.
- `rtl/nac_bfp_microscale_decode.v`, `rtl/nac_structured_sparsity_decode.v`,
  `rtl/nac_palette_decode.v`, `rtl/nac_activation_rle_codec.v` - optional
  compression/format decode primitives.
- `rtl/nac_mep_packetizer.v` - MEP instruction boundary packetizer.
- `rtl/nac_mep_vm.v` - hardware MEP master VM for context, scalar logic,
  TISA/model/storage dispatch, and flow-control redirects.
- `rtl/nac_scalar_alu.v` - sequential scalar ALU used by MEP; 64-bit
  multiply/divide are iterative, not single-cycle combinational paths.
- `rtl/nac_tisa_packetizer.v` - TISA opcode/payload packetizer.
- `rtl/nac_tisa_tokenizer.v` - TISA manifest loader plus safe local text
  frontend.
- `rtl/nac_tisa_tokenizer_frontend.v` and `rtl/nac_tisa_byte_map.v` -
  exact local TISA byte-level primitives.
- `rtl/nac_dma_manager.v` - MMAP-to-DMA/bank manager for streaming tensor
  loads, saves, forwards, frees, and static pinning.
- `rtl/nac_soc_top.v` - explicit SoC integration shell wiring MEP, TISA,
  NAC core, DMA manager, bank allocator, scratchpad, and codec/DSP.
- `docs/NAC-Core_ARCH.md` - architecture notes and standards mapping.
- `docs/NAC-Core_ARCH_RU.md` - Russian architecture summary.
- `docs/COMPLIANCE.md` - checked standard-compliance invariants.
- `tb/*_tb.v` - smoke and memory-safety tests for decoder, MMAP, DSP,
  address generation, bank allocation, and top-level MMAP backpressure.

Run smoke tests when Icarus Verilog is installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_dsp_pipeline_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_abcd_decoder_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_mmap_engine_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_kernel_sequencer_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_mep_packetizer_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_mep_vm_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_scalar_alu_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_tisa_packetizer_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_tisa_tokenizer_frontend_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_tisa_tokenizer_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_op_table_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_op_dispatch_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_core_mmap_backpressure_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_addr_gen_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_bank_allocator_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_ecc_secded_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_watchdog_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_instruction_cache_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_core_load_run_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_core_control_flow_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_compression_primitives_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_codec_dsp_pipeline_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_scratchpad_ecc_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_result_descriptor_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_dma_manager_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_model_lifecycle_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_soc_top_tb
```

Current scope limits are explicit: `CONVERGENCE/MoE`, full TRNG backpropagation
and optimizer sequencing for `MODEL_TRAIN_STEP`, and full resource-backed TISA
VM execution are not complete in this RTL. `NAC_DSP_WEIGHT_UPDATE` is a
lane-wise SGD update primitive, not a complete training engine. Unsupported
execution paths report errors or `requires_external_engine` instead of falling
back to model-specific behavior.
