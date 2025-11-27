`ifndef NAC_DEFINES_VH
`define NAC_DEFINES_VH

// ============================================================================
// 1. NAC v1.1 System Control Codes (Fixed Standards)
// ============================================================================
// These OpCodes (0-9) are reserved by the specification and handled directly
// by the FSM (NAC_Processor_Top), bypassing the Translation LUT.

`define OP_RETURN       8'd0   // End of Pattern (Virtual Return)
`define OP_PAD          8'd1   // Reserved for alignment
`define OP_DATA_INPUT   8'd2   // <DATA_INPUT>: DMA Load (DDR -> BRAM)
`define OP_PARAM_INPUT  8'd3   // <PARAM_INPUT>: Legacy (usually unused)
`define OP_OUTPUT       8'd4   // <OUTPUT>: DMA Store (BRAM -> DDR)
`define OP_CONST_REF    8'd5   // <CONST_REF>: Parameter reference
`define OP_PATTERN      8'd6   // <PATTERN_PREFIX>: Call Subroutine
`define OP_COPY         8'd7   // <COPY_PREFIX>: RLE Loop

// ============================================================================
// 2. Control Block Memory Map (DDR Base Address 0x0000_0000)
// ============================================================================
// The Host (PC/MCU) writes configuration data to these addresses before
// starting the FPGA.

// --- Command / Status Registers ---
`define REG_CMD         32'h00  // Write: 1 = Load Config, 2 = Run Inference
`define REG_STATUS      32'h04  // Read: 0 = Idle, 1 = Busy, 2 = Done, 0xFF = Error

// --- Memory Pointers (32-bit Addresses) ---
// The FSM reads these sequentially starting from 0x08.
`define REG_REGISTRY    32'h08  // Pointer to Pattern Registry (Jump Table)
`define REG_CODE        32'h0C  // Pointer to .b64 Instruction Stream
`define REG_WEIGHTS     32'h10  // Base Pointer to Weights Blob
`define REG_INPUT       32'h14  // Pointer to Input Tensor Buffer
`define REG_OUTPUT      32'h18  // Pointer to Output Tensor Buffer
`define REG_OPMAP       32'h1C  // Pointer to OpCode Translation LUT
`define REG_VARMAP      32'h20  // Pointer to Variation Translation LUT

// ============================================================================
// 3. Hardware Constraints & Parameters
// ============================================================================
// These define the physical size of buffers and ALUs.

// Vector Length (Granularity of computation)
// 512 elements * 32-bit = 2 KB per vector.
`define VECTOR_LEN      512

// Tensor Memory Size (Ring Buffer)
// 32 slots * 2 KB = 64 KB Total BRAM usage.
// Compatible with Artix-7 XC7A100T (which has ~4.8Mb BRAM).
`define TENSOR_SLOTS    32

// Modulo Mask for Ring Buffer Addressing
// Must be (TENSOR_SLOTS - 1).
`define SLOT_MASK       5'h1F

// Maximum inputs per node (defines size of dependency decoding buffer)
`define MAX_INPUTS      16

// Maximum Recursion Depth for Patterns
`define STACK_DEPTH     16

// Fixed Point Math Constant (Q16.16)
// 1.0 represented as integer.
`define FIXED_ONE       32'd65536 

`endif