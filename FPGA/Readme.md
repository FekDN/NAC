# NAC Hardware Processor (FPGA Implementation)

> **Status:** Proof of Concept (PoC)  
> **Target Platform:** Xilinx Artix-7 (XC7A100T / XC7A200T)  
> **Architecture:** Memory-Centric Domain Specific Architecture (DSA)

## Overview

This repository contains the Verilog HDL implementation of a hardware interpreter for **Neural Architecture Code (NAC)**. 

Unlike traditional FPGA AI accelerators that "bake" the model architecture into the silicon logic (requiring re-synthesis for every new model), this processor acts as a **Domain Specific Processor**. It executes NAC binary streams (`.b64`) directly from external memory. 

This allows changing the neural network model (e.g., swapping ResNet for BERT) in milliseconds by simply updating the data in RAM, without reconfiguring the FPGA bitstream.

## Principles of Operation

The core philosophy of this design is **"Instruction Stream Processing"** coupled with **"Weight Streaming"**. The FPGA does not store the model; it executes it.

### 1. The "Game Console" Model
The FPGA is configured once with a generic "NAC Processor" bitstream. 
- **The "Cartridge":** The Neural Network definition (NAC code + Weights) resides in external DDR memory.
- **The "Player":** The FPGA reads instructions, fetches weights on-demand, and writes results back to memory.

### 2. Architecture Modules
*   **`NAC_Byte_Fetcher`:** An intelligent pre-fetcher that bridges the gap between the 32-bit/64-bit DDR interface and the variable-length 8-bit NAC instruction stream. It handles alignment and buffering transparently.
*   **`NAC_Processor_Top` (The Controller):** A complex FSM that parses the NAC Header (`A-B-C-D`) format. It manages:
    *   **Hardware Recursion:** Uses a hardware stack to handle `OP_PATTERN` calls (subroutines), allowing for highly compressed nested graph execution.
    *   **RLE Engine:** Natively executes `OP_COPY` (Run-Length Encoding) loops without fetching new instructions, saving memory bandwidth.
*   **`NAC_ALU_Core` (The Math Engine):** A pipelined vector processor mapped to DSP48E1 slices. It uses a Ring Buffer in BRAM for activation storage.

### 3. Weight Streaming (Zero-BRAM-Weight)
A critical feature of this implementation is that **weights are never stored in FPGA BRAM**. 
*   FPGA BRAM (Block RAM) is used *only* for Activations (intermediate tensors).
*   Weights are streamed directly from DDR memory into the DSP units via a dedicated bus arbitration channel.
*   **Pipeline Backpressure:** If the memory bandwidth saturates, the Arithmetic Pipeline automatically stalls (pauses) to wait for data, ensuring data integrity without complex synchronization software.

### 4. Memory Arbitration
The system implements a strict priority arbiter for the single DDR bus:
1.  **Weight Stream** (Highest Priority during compute).
2.  **Instruction Fetch** (Prefetching next ops).
3.  **DMA/Control** (Loading configurations or I/O).

## 🛠️ Current Implementation Status

This code is a **high-fidelity concept**. It implements the full control logic and data flow required for a production chip, but uses simplified arithmetic for clarity.

*   ✅ **Full NAC v1.1 Parsing:** Header, Payload, Variable D-field.
*   ✅ **Pattern Support:** Hardware stack for nested patterns.
*   ✅ **Compression:** Native RLE (`COPY`) execution.
*   ✅ **Streaming:** Direct DDR-to-DSP weight path.
*   ✅ **Protocol:** Control Block memory map for model switching.
*   ⚠️ **Arithmetic:** Currently configured for INT32 (for logic verification).

## Roadmap (TODO)

To make this a viable competitor to commercial NPUs, the following features need to be implemented:

1.  **Full OpCode Support:**
    *   Expand `NAC_ALU_Core` to support all 255 fundamental operations (currently supports basic ADD/MUL/RELU/LINEAR).
    *   Implement `Conv2d` tiling logic.
2.  **Precision Upgrade (FP16 / BF16):**
    *   Move from INT32 to **BF16 (Bfloat16)** or **FP16**.
    *   **DSP Trick:** Utilize the Artix-7 DSP48E1 pre-adders and SIMD modes to process two 16-bit operations per clock cycle per DSP slice.
3.  **Multi-Core Scaling:**
    *   Implement a "Cluster" architecture with 4–8 `NAC_ALU_Core` instances working in parallel on different tensor tiles.
4.  **Bus Integration:**
    *   Replace the generic `mem_req/grant` interface with a robust **AXI4 Master** wrapper.
    *   Integrate a Xilinx **MIG (Memory Interface Generator)** controller for physical DDR3 access on Nexys/Arty boards.
5.  **Control Flow Support:**
    *   Add hardware support for `IF`, `WHILE`, and `JMP` instructions to support dynamic graphs (e.g., Mixture of Experts).
6.  **Auto-Registry Loading:**
    *   Harden the `S_LOAD_REGISTRY` state machine to robustly parse variable-length Registry tables directly from storage.

## Hardware Requirements

*   **FPGA:** Xilinx Artix-7 series (XC7A100T recommended for reasonable BRAM/DSP headroom).
*   **Memory:** External DDR3 (minimum 256 MB) accessible via MIG.
*   **Storage:** SD Card or Quad-SPI Flash (for storing model binaries).

---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
