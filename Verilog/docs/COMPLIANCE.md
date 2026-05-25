# NAC-Core compliance notes

## Hard rules

- Runtime dispatch uses the ABCD `A` field. Custom operation metadata may only
  be loaded from standard `CMAP` into the `A -> kernel_class` table before
  execution.
- Operand selection is driven by ABCD `C` and `D`. The kernel sideband carries
  `A/B/C/D` unchanged, and the result table read ports are addressed by
  `current_instruction + signed(D[i])`.
- Unknown `A >= 10` operations are errors. They are not converted to pass,
  clone, or any other fallback behavior.
- `MMAP` actions use standard action codes only:
  `SAVE_RESULT=10`, `FREE=20`, `FORWARD=30`, `PRELOAD=40`.
- `SAVE_RESULT` copies from the MMAP active tick, not from a later dispatcher
  tick.
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
- Unsupported reserved system operations are not silently executed.

## MEP

`nac_mep_packetizer` preserves byte-accurate instruction boundaries for fixed
and count-dependent MEP instructions. It supports the current compiler/runtime
length rules, including `MODEL_RUN_STATIC`, `MODEL_RUN_DYNAMIC`,
`MODEL_TRAIN_STEP`, extern calls, loops, branches, return, and halt.

## TISA

`nac_tisa_packetizer` validates `TISA` magic/version framing and emits exact
`opcode`, `payload_len`, and payload bytes. The payload is not rewritten,
special-cased, or expanded.

## Model universality

The RTL contains no checks for specific model names, tensor names, layer names,
or architecture families. Model behavior is supplied by NAC sections and by the
standard ABCD/MEP/TISA bytecode streams.
