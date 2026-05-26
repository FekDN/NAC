# NAC-Core Verilog

NAC-Core is a synthesizable soft-core shell for NAC v1.8 execution. It follows
the decoupled access/execute model used by the reference runtime:

- ORCH/dispatcher stream decodes NAC ABCD instructions from `OPS`.
- MMAP engine executes scheduled `PRELOAD`, `FREE`, `SAVE_RESULT`, and `FORWARD`
  commands on instruction ticks.
- A programmable vector DSP pipeline exposes one physical MAC array, one SFU
  lane array, and reduction hardware that can be time-multiplexed by kernels
  such as matmul, Conv2D, LayerNorm, and Softmax.
- TISA and MEP packetizers preserve bytecode boundaries and payload bytes
  according to the standard encodings.

Important source files:

- `rtl/nac_core.v` - top-level orchestration shell.
- `rtl/nac_abcd_decoder.v` - NAC ABCD instruction decoder.
- `rtl/nac_mmap_engine.v` - independent memory schedule engine.
- `rtl/nac_dsp_pipeline.v` - reusable vector DSP datapath.
- `rtl/nac_kernel_sequencer.v` - multi-pass LayerNorm/Softmax mode sequencer.
- `rtl/nac_scratchpad.v` - dual-port banked BRAM scratchpad.
- `rtl/nac_mep_packetizer.v` - MEP instruction boundary packetizer.
- `rtl/nac_tisa_packetizer.v` - TISA opcode/payload packetizer.
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
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_tisa_packetizer_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_op_dispatch_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_core_mmap_backpressure_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_addr_gen_tb
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -Top nac_bank_allocator_tb
```
