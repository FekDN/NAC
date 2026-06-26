# NAC-Core compliance notes

## Hard rules

- Runtime dispatch uses the ABCD `A` field. Custom operation metadata may only
  be loaded from standard `CMAP` into the operation recipe table before
  execution.
- Only fixed NAC standard kernels (`NAC_STD_*`) have fixed numeric IDs. ATen
  names stored in `CMAP` may use different `A` IDs in different `.nac` files;
  hardware dispatch must therefore use the loaded `A -> recipe` entry
  (`kernel_class`, `dsp_mode`, and dispatch flags), not hardcoded ATen IDs.
- Operand selection is driven by ABCD `C` and `D`. The kernel sideband carries
  `A/B/C/D` unchanged, and the result table read ports are addressed by
  `current_instruction + signed(D[i])`.
- Unknown `A >= 10` operations are errors. They are not converted to pass,
  clone, or any other fallback behavior.
- `MMAP` actions use standard action codes only:
  `SAVE_RESULT=10`, `FREE=20`, `FORWARD=30`, `PRELOAD=40`.
- `SAVE_RESULT` copies from the MMAP active tick, not from a later dispatcher
  tick.
- MMAP commands are committed to the descriptor table only when the matching
  ready signal accepts the command. Backpressure cannot free or forward a
  descriptor early.
- Nonlinear DSP modes `EXP` and `RSQRT` require exact SFU response over the
  explicit request/response interface. The DSP pipeline does not approximate
  them internally.

## NAC ABCD

Implemented parser behavior:

- Regular operations `A >= 10` read `C` only when the loaded PERM metadata says
  the signature contains constant-bearing codes.
- `D` length is exactly the loaded PERM arity.
- `<INPUT>` and `<OUTPUT>` use the special `A < 10` structures used by the
  current NAC compiler/runtime.
- `<CONTROL_FLOW>` (`A=6`) is parsed with the reserved standard shape
  `C=[3,true_branch_len,false_branch_len]`, `D=[predicate_offset]`. In cached
  execution, `nac_orch_fsm` requests the predicate value from an external
  predicate evaluator and redirects `nac_instruction_cache` to the true/false
  branch. The true path skips the false block after `true_branch_len`
  instructions. Legacy one-pass byte streaming still cannot random-seek.
- Unsupported reserved system operations are not silently executed.
- `<CONVERGENCE>` (`A=7`) remains experimental and is not executed.

## MEP

`nac_mep_packetizer` preserves byte-accurate instruction boundaries for fixed
and count-dependent MEP instructions. It supports the current compiler/runtime
length rules, including `MODEL_RUN_STATIC`, `MODEL_RUN_DYNAMIC`,
`MODEL_TRAIN_STEP`, extern calls, loops, branches, return, and halt. The RTL
does not implement backpropagation or SGD/Adam update datapaths for
`MODEL_TRAIN_STEP`; it preserves the packet boundary. The DSP has a
`NAC_DSP_WEIGHT_UPDATE` primitive for `W_new = W_old - LR * Gradient`, but the
full TRNG scheduler/backprop graph and optimizer sequencing are not complete.

## TISA

`nac_tisa_packetizer` validates `TISA` magic/version framing and emits exact
`opcode`, `payload_len`, and payload bytes. The payload is not rewritten,
special-cased, or expanded.

`nac_tisa_tokenizer` wires the packetizer into `nac_tisa_tokenizer_frontend`.
The local frontend executes only primitives whose byte-level behavior is exact
without model resources:

- `LOWERCASE` (`0x01`) for ASCII input bytes. If a non-ASCII byte is observed
  while this local path is active, `run_error` is asserted instead of emitting
  incorrect Unicode lowercase output.
- `BYTE_ENCODE` (`0x15`) using the reference Python byte map
  `33..126,161..172,174..255 -> same codepoint`, missing bytes ->
  `U+0100..`, emitted as UTF-8 bytes.

All resource-dependent or Unicode/regex-dependent TISA opcodes
(`UNICODE_NORM`, `REPLACE`, `FILTER_CATEGORY`, `PREPEND`, `PARTITION_RULES`,
`BPE_ENCODE`, `WORDPIECE_ENCODE`, `UNIGRAM_ENCODE`, and `COMPOSE`) are accepted
as known TISA opcodes but set `requires_external_engine`. A local run is then
rejected with `run_error`. Unknown opcodes and malformed local payload lengths
set `load_error`.

## Configuration and precision

- `nac_config.vh` is the single packed hardware-configuration source used by
  Icarus-compatible Verilog builds. It replaces a typed SystemVerilog struct
  while preserving one profile word through the module hierarchy.
- FP8/BF16 settings are configuration metadata unless connected to dedicated
  floating-point datapaths. The current MAC/SFU arithmetic remains the
  fixed/integer datapath. `nac_codec_dsp_pipeline` wires registered BFP,
  palette, and structured-sparsity decode stages into the DSP input path, and
  exposes RLE output sideband for activation writes.
- Descriptor flags start above the legacy 64-bit descriptor payload:
  `is_bfp`, `is_sparse_2_4`, `is_rle`, `is_palette`, and `static`.
- ARRS offsets are parsed by the header path, but ARRS-to-scratchpad projection
  is not claimed as complete until the memory loader path maps named arrays
  into bank descriptors.

## Model universality

The RTL contains no checks for specific model names, tensor names, layer names,
or architecture families. Model behavior is supplied by NAC sections and by the
standard ABCD/MEP/TISA bytecode streams.

## Memory safety

- `nac_mmap_engine` drains skipped ticks in order and reports invalid MMAP
  action codes instead of executing them.
- Static MMAP entries can pin already-loaded model data across repeated
  `cmd_run_inference` calls and are cleared only by explicit load/clear
  commands.
- `nac_instruction_cache` stores decoded ABCD commands per context during
  `cmd_load_model`; `cmd_run_inference` replays cached instructions without
  consuming `OPS` bytes.
- MMAP runtime progress state is reset for each `cmd_run_inference` while the
  loaded schedule and static pin-completion bits are preserved.
- `nac_watchdog` resets the core only after busy cycles make no protocol
  progress for the configured limit.
- `nac_scratchpad` and `nac_result_table` can store SECDED ECC next to data and
  report/correct single-bit read errors while flagging double-bit errors.
- `nac_addr_gen` computes the next address in an extended signed range and
  raises `error` before an address can wrap outside the configured address
  width.
- `nac_bank_allocator` reports allocation failure, invalid free, and
  double-free cases without indexing outside the configured bank set.
