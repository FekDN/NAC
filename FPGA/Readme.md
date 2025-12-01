# NAC Hardware Processor (FPGA Implementation)

> **Status:** Advanced Proof of Concept (PoC)  
> **Target Platform:** Xilinx Artix-7 (XC7A100T / XC7A200T) or AMD Zynq UltraScale+ MPSoC (e.g., ZU2CG)  
> **Architecture:** Memory-Centric Domain Specific Architecture (DSA) for NAC ISA

## Overview

This repository contains the Verilog HDL implementation of a hardware processor for the **Neural Architecture Code (NAC)** Instruction Set Architecture (ISA).

Unlike traditional FPGA AI accelerators that "bake" a model's architecture into the silicon logic (requiring re-synthesis for every new model), this processor acts as a **true Domain-Specific Processor**. It fetches and executes NAC binary instructions directly from external DDR memory, treating a neural network as a program rather than a fixed circuit.

This design paradigm allows for dynamic model switching (e.g., from ResNet to a Transformer) in milliseconds by simply loading new "software" (NAC code + weights) into RAM, all without reconfiguring the FPGA bitstream.

## Principles of Operation

The core philosophy is **"Instruction Stream Processing"** coupled with **"Weight Streaming"**. The FPGA does not store the entire model; it executes it on the fly.

### 1. The "Game Console" Analogy
The FPGA is programmed once with a universal "NAC Processor" bitstream.
- **The "Cartridge":** The Neural Network definition (NAC code, weights, and metadata) resides in external DDR memory, loaded from an SD card (Artix) or by a host processor (Zynq).
- **The "Player":** The FPGA core reads instructions, fetches weights on-demand, manipulates tensor metadata, performs computations, and writes results back to memory.

### 2. Architecture Modules
*   **`NAC_Zynq_Wrapper` / `NAC_Artix_Wrapper`:** Top-level modules that integrate the core into specific hardware environments. The Zynq version uses AXI4-Lite for control and AXI4-HP for data, while the Artix version includes a self-booting `NAC_SD_Loader`.
*   **`NAC_Processor_Core` (The Controller):** A complex Finite State Machine (FSM) that serves as the main control unit. It parses the NAC binary format (`A-B-C-D[]`) and manages the entire execution flow. It includes:
    *   **Tensor Metadata Engine:** Manages the `shape` and `strides` of all intermediate tensors, enabling instantaneous, zero-cost execution of metadata operations like `view`, `transpose`, and `permute`.
    *   **Hardware Stack:** Implements hardware recursion for `OP_PATTERN` calls (subroutines), allowing for highly compressed, nested graph execution.
    *   **RLE Engine:** Natively executes `OP_COPY` (Run-Length Encoding) loops, saving instruction fetch bandwidth.
*   **`NAC_Byte_Fetcher`:** An intelligent pre-fetcher that bridges the 32-bit AXI bus and the variable-length 8-bit NAC instruction stream, handling alignment and buffering transparently.
*   **`NAC_Stream_Prefetcher`:** A dedicated AXI burst-read engine for streaming weights and parameters from DDR directly to the compute units.
*   **`NAC_ALU_Core` (The Compute Engine):** A powerful, pipelined vector processor designed to be mapped efficiently to FPGA DSP and BRAM slices.
    *   **Multi-Format Arithmetic:** Natively supports **Q16.16** fixed-point for scalar operations and activations, and **INT8 SIMD** (`DP4A`) for high-throughput matrix multiplication (`LINEAR`, `MATMUL`, `CONV2D`).
    *   **Stall-Capable Pipeline:** The entire datapath supports automatic pipeline stalling (backpressure) for data hazards, multi-cycle operations, or memory latency, ensuring data integrity.

### 3. Weight Streaming (Zero-BRAM-Weight Design)
A critical feature of this implementation is that **model weights are never stored in FPGA Block RAM (BRAM)**.
*   FPGA BRAM is used *exclusively* for holding activations (the intermediate tensors being computed).
*   Weights are streamed on-demand directly from external DDR memory into the DSP units via a dedicated, high-priority AXI channel managed by the `NAC_Stream_Prefetcher`.
*   This "Zero-BRAM-Weight" approach allows the processor to execute models of virtually unlimited size, constrained only by external memory capacity, not by the limited on-chip resources of the FPGA.

### 4. Memory Arbitration
The system implements a strict, multi-channel priority arbiter for the single AXI master interface to DDR memory:
1.  **Weight Stream** (Highest priority during computation).
2.  **Instruction Fetch** (Prefetches upcoming instructions).
3.  **FSM/DMA Control** (Handles I/O, configuration reads, and writing results).

## 🛠️ Current Implementation Status

This implementation is an **advanced, feature-complete proof of concept**. It contains the full control logic, data paths, and arithmetic units required for a production-grade system, ready for synthesis and deployment.

*   ✅ **Full NAC v1.1 Parsing:** Complete decoding of Header (`A`, `B`, `C`) and variable-length Payload (`D[]`).
*   ✅ **Tensor Metadata Support:** Full hardware support for instantaneous `view`, `transpose`, and `permute` operations via direct manipulation of tensor descriptors.
*   ✅ **Pattern & Compression Support:** Hardware stack for nested patterns and native RLE (`COPY`) execution.
*   ✅ **Multi-Precision Arithmetic:** Implemented **Q16.16 Fixed-Point** math with saturation and **INT8 SIMD** dot-product engine.
*   ✅ **Complex OpCode Support:**
    *   **Division:** Implemented with a multi-cycle iterative divider FSM.
    *   **Activations:** Implemented via dedicated, high-speed LUTs for `GELU`, `TANH`, and `ERF` (with Tanh-based fallback).
    *   **Normalization:** Full two-pass FSM implementation for `LayerNorm` and `BatchNorm`, including a fast inverse square root approximation.
    *   **Pooling:** FSM implementation for `MaxPool2D`.
*   ✅ **AXI4 Integration:** The core is fully wrapped with AXI4 Master and AXI4-Lite Slave interfaces, making it IP-ready for AMD/Xilinx Vivado IPI.
*   ✅ **Streaming Architecture:** Implemented direct DDR-to-DSP weight path with flow control.

## Roadmap (Next Steps)

1.  **Full OpCode Hardening:**
    *   Implement a deeply pipelined, high-performance `Conv2D` engine, potentially using `im2col` techniques.
    *   Add a CORDIC or LUT-based hardware block for `exp()` to enable a fully accurate `Softmax` implementation.
2.  **Precision & Performance Enhancements:**
    *   Transition arithmetic units to **BF16 (Bfloat16)** or **FP16**, which can be efficiently implemented by leveraging the DSP48 pre-adders and SIMD modes to process two 16-bit operations per clock cycle.
3.  **Multi-Core Scaling:**
    *   Develop a "Cluster" wrapper around 4–8 `NAC_Processor_Core` instances.
    *   Implement a high-level scheduler and DMA engine to distribute tensor tiles across the cores, enabling true data parallelism.
4.  **Control Flow Support:**
    *   Expand the NAC ISA and the `NAC_Processor_Core` FSM to support conditional (`IF`) and unconditional (`JMP`) branch instructions, enabling execution of dynamic graphs (e.g., Mixture of Experts, RNNs).

## Hardware Requirements

*   **FPGA:**
    *   **Standalone:** Xilinx Artix-7 series (XC7A100T or larger recommended for sufficient BRAM/DSP headroom).
    *   **SoC:** AMD Zynq UltraScale+ MPSoC (e.g., ZU2CG on an Ultra96-V2 board).
*   **Memory:** External DDR3/DDR4 (minimum 256 MB).
*   **Storage:** SD Card (for Artix) or any storage accessible by the host OS (for Zynq).

---

*   feklindn@gmail.com

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
