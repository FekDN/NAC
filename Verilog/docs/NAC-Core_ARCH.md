# NAC-Core architecture notes

## Mapping to NAC v1.8

The core follows the file semantics documented by NAC v1.8:

- The 100-byte NAC header is handled by `nac_header_parser`.
- `OPS` is a byte stream of ABCD instructions. `A` and `B` are decoded directly,
  `C` is read only when the preloaded PERM entry says constants are present, and
  `D` length is the preloaded PERM arity.
- System ops are decoded with the special NAC rules: `<INPUT>` (`A=2`) and
  `<OUTPUT>` (`A=3`) do not depend on PERM.
- `MMAP` is represented as a tick-indexed schedule. The engine is independent of
  the dispatcher and wakes on each committed instruction tick.
- `CMAP` string names are converted during section loading into an `A -> class`
  operation table. Dispatch still uses only the ABCD `A` field at execution
  time; unknown `A` values raise an error instead of falling back.

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
  `FREE` invalidates descriptors, matching the reference runtime's cache logic.

The dispatcher never performs tensor math. It only builds routing and kernel
configuration commands. The memory stream is not blocked by arithmetic kernels;
it is only synchronized by commit ticks.

## Programmable DSP pipeline

`nac_dsp_pipeline` is the reusable physical compute fabric:

- `nac_mac_array` performs lane-wise products and a dot-product reduction.
- `nac_reduction_tree` provides sum/max reductions for atomics-free aggregation.
- `nac_sfu` provides exact fixed-point ReLU/hardsigmoid/hardswish lanes.
- `EXP` and `RSQRT` modes use the explicit exact-SFU request/response port;
  the DSP pipeline does not substitute approximations.
- `nac_kernel_sequencer` expands high-level kernel classes into the DSP modes
  consumed by the physical pipeline.

High-level kernels are decomposed into physical DSP modes:

- MatMul/Linear/Conv2D: repeated `MAC` tiles driven by address generators.
- Elementwise NAC ops: `ADD`, `SUB`, `MUL`, comparisons, negation.
- LayerNorm: reduction passes for mean/variance, followed by scale/SFU passes.
- Softmax: max reduction, subtract, exact EXP, sum reduction, then scale.

The same MAC and SFU hardware is reused over time; no layer owns a private DSP
block.

## TISA and MEP

The repository includes byte-accurate packetizers:

- `nac_tisa_packetizer` validates `TISA` magic/version and emits exact
  opcode/payload spans with payload bytes preserved.
- `nac_mep_packetizer` emits exact instruction byte boundaries for fixed and
  count-dependent MEP instructions used by the compiler/interpreters.

These blocks do not introduce model-specific instructions or private cases.

## Integration contract

The intended FPGA integration is:

1. A loader reads NAC sections and configures PERM arity/constant flags, CMAP
   kernel classes, MMAP schedule records, parameter descriptors, and input
   descriptors.
2. `nac_core` consumes `OPS` bytes and emits kernel/input/MMAP commands.
3. A tensor datapath uses descriptors, address generators, scratchpad banks, and
   `nac_dsp_pipeline` to execute physical tiles.
4. MMAP `PRELOAD` fills scratchpad banks ahead of the dispatcher, while `FREE`
   returns banks to the allocator after last use.

Kernel-specific microcode must consume the emitted ABCD fields and descriptor
ports; it must not select operands from any side channel when those operands are
already present in ABCD `C` or `D`.
