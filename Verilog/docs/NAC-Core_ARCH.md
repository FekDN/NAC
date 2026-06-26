# NAC-Core architecture notes

## Mapping to NAC v1.8

The core follows the file semantics documented by NAC v1.8:

- The 100-byte NAC header is handled by `nac_header_parser`.
- `OPS` is a byte stream of ABCD instructions. `A` and `B` are decoded directly,
  `C` is read only when the preloaded PERM entry says constants are present, and
  `D` length is the preloaded PERM arity.
- System ops are decoded with the special NAC rules: `<INPUT>` (`A=2`),
  `<OUTPUT>` (`A=3`), and reserved `<CONTROL_FLOW>` (`A=6`) do not depend on
  PERM. `A=6` is parsed as `C=[3,true_branch_len,false_branch_len]` and
  `D=[predicate_offset]`; branch execution is intentionally not claimed until
  predicate descriptor evaluation is wired into the dispatcher.
- `MMAP` is represented as a tick-indexed schedule. The engine is independent of
  the dispatcher and wakes on each committed instruction tick.
- `CMAP` string names are converted during section loading into an `A -> recipe`
  operation table. Only `NAC_STD_*` kernels have fixed numeric IDs; ATen names
  may use different `A` values in different `.nac` files. Dispatch still uses
  only the ABCD `A` field plus the loaded recipe at execution time; unknown
  `A` values raise an error instead of falling back.

This keeps the HDL datapath hardware-oriented while preserving the binary
standard's behavior.

## Dual-stream orchestration

`nac_core` contains two logical streams:

- `nac_orch_fsm`: accepts decoded ABCD instructions, handles `<INPUT>` and
  `<OUTPUT>`, and issues kernel commands for regular operations (`A >= 10`).
  The kernel sideband includes `A/B/C/D`, so the tensor datapath can gather
  dependencies using NAC's relative `D` offsets and constant IDs. `nac_core`
  also exposes two descriptor-table read ports for that gather path.
- `nac_mmap_engine`: receives the committed instruction index as a tick and
  emits memory-side pulses for `PRELOAD`, `FREE`, `SAVE_RESULT`, and `FORWARD`.
  `SAVE_RESULT` and `FORWARD` also copy descriptors inside the result table;
  accepted `FREE` commands invalidate descriptors, matching the reference
  runtime's cache logic. Ready/valid backpressure is honored before descriptor
  table state changes.

The dispatcher never performs tensor math. It only builds routing and kernel
configuration commands. The memory stream is not blocked by arithmetic kernels;
it is only synchronized by commit ticks.

## Load/run lifecycle

The top level keeps legacy `start` for single-shot streams, and also exposes a
production lifecycle:

- `cmd_load_model` parses `OPS` once, writes decoded ABCD commands into
  `nac_instruction_cache`, and leaves per-context CSR state intact.
- `cmd_run_inference` replays cached commands from BRAM-like instruction memory
  for the selected context without consuming `OPS` bytes.
- `cmd_clear_model` clears the selected context and static MMAP pin state.

CSR banks hold per-context configuration such as quantization flags, I/O counts,
`d_model`, and section offsets. This lets a host switch between configured
models by changing `context_id` instead of reparsing the model.

Cached execution also supports standard `<CONTROL_FLOW>` branch entry. ORCH
requests the predicate descriptor from an external predicate evaluator, receives
one boolean response, redirects the instruction-cache PC to the selected block,
and skips the false block after a taken true block. This preserves the current
ABCD encoding; it does not introduce private branch opcodes.

## Programmable DSP pipeline

`nac_dsp_pipeline` is the reusable physical compute fabric:

- `nac_mac_array` performs lane-wise products and a dot-product reduction.
- `nac_reduction_tree` provides sum/max reductions for atomics-free aggregation.
- `nac_sfu` provides exact fixed-point ReLU/hardsigmoid/hardswish lanes.
- `EXP` and `RSQRT` modes use the explicit exact-SFU request/response port;
  the DSP pipeline does not substitute approximations.
- `nac_kernel_sequencer` expands high-level kernel classes into the DSP modes
  consumed by the physical pipeline.
- `nac_systolic_array` is available as an optional 2D MAC fabric for skewed
  matrix/conv streams, while the 1D MAC array remains available for elementwise
  and reduction-heavy kernels.
- `NAC_DSP_WEIGHT_UPDATE` performs the lane-wise SGD primitive
  `W_new = W_old - LR * Gradient`. It is intentionally a primitive; full TRNG
  graph execution still needs a training sequencer.

## Tensor format path

`nac_codec_dsp_pipeline` is the registered integration wrapper for memory-format
decode:

1. Memory/scratchpad data enters `nac_tensor_codec_pipeline`.
2. Optional BFP or palette decode expands compressed weights.
3. Optional structured 2:4 sparsity decode expands compact nonzero values into
   a dense vector.
4. A registered dense vector feeds `nac_dsp_pipeline`.
5. Optional RLE sideband is produced from DSP/SFU output for activation writes.

The wrapper parameters default from `HW_CFG`: compression, sparsity, and
non-fixed data-type features are generated only when enabled by the selected
FPGA profile.

High-level kernels are decomposed into physical DSP modes:

- MatMul/Linear/Conv2D: repeated `MAC` tiles driven by address generators.
- Elementwise NAC ops: `ADD`, `SUB`, `MUL`, comparisons, negation.
- LayerNorm: reduction passes for mean/variance, followed by scale/SFU passes.
- Softmax: max reduction, subtract, exact EXP, sum reduction, then scale.

The same MAC and SFU hardware is reused over time; no layer owns a private DSP
block.

## Reliability and compression

- `nac_watchdog` observes protocol progress, not just raw elapsed time. A
  timeout is raised only while the core is busy and no handshakes/commits occur.
- `nac_ecc_secded` supplies SECDED protection for scratchpad and result-table
  storage.
- Safe-FSM defaults route illegal encoded states to error handling.
- Optional memory-format primitives include zero-mask compaction, structured
  N:M sparsity decode, BFP microscale decode, palette weight decode, and
  lossless zero-run RLE for activation streams.

## TISA and MEP

The repository includes byte-accurate packetizers:

- `nac_tisa_packetizer` validates `TISA` magic/version and emits exact
  opcode/payload spans with payload bytes preserved.
- `nac_mep_packetizer` emits exact instruction byte boundaries for fixed and
  count-dependent MEP instructions used by the compiler/interpreters.

These blocks do not introduce model-specific instructions or private cases.

They also do not execute the full high-level languages by themselves:
`MODEL_TRAIN_STEP` is packetized but no backprop/optimizer datapath is claimed.

TISA now has a safe local frontend:

- `nac_tisa_tokenizer` connects the binary manifest packetizer to
  `nac_tisa_tokenizer_frontend`.
- The local frontend can execute exact ASCII `LOWERCASE` and exact TISA/GPT-2
  `BYTE_ENCODE` streaming transforms.
- Manifests containing Unicode normalization, replacements, category filters,
  prepend, regex/partition, BPE, WordPiece, Unigram, or compose stages are
  flagged with `requires_external_engine`. Local execution is rejected instead
  of approximating those stages.

Full TISAVM parity with the Python/C++ implementations requires additional
hardware engines or host-assisted engines for UCD tables, partition matching,
vocabulary lookup, BPE merge ranks, WordPiece search, Unigram Viterbi scores,
and compose templates.

## Integration contract

The intended FPGA integration is:

1. A loader reads NAC sections and configures PERM arity/constant flags, CMAP
   operation recipes, MMAP schedule records, parameter descriptors, and input
   descriptors.
2. `nac_core` consumes `OPS` bytes and emits kernel/input/MMAP commands.
3. A tensor datapath uses descriptors, address generators, scratchpad banks, and
   `nac_dsp_pipeline` to execute physical tiles.
4. MMAP `PRELOAD` fills scratchpad banks ahead of the dispatcher, while `FREE`
   returns banks to the allocator after last use.

Kernel-specific microcode must consume the emitted ABCD fields and descriptor
ports; it must not select operands from any side channel when those operands are
already present in ABCD `C` or `D`.

For large tensors, address generators calculate candidate addresses in an
extended signed range and raise `error` before an address wraps. Bank allocation
also reports full, invalid-free, and double-free conditions explicitly.
