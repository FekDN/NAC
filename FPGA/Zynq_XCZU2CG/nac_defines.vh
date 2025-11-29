`ifndef NAC_DEFINES_VH
`define NAC_DEFINES_VH

// ============================================================================
// 1. NAC v1.1 System Control Codes (Dynamic OpCodes)
// ============================================================================
// These are the raw byte values found in the .b64 binary stream.
// Codes 0-9 are Reserved/Fixed by specification and handled by the Top FSM directly.
// Codes 10-255 are translated via LUT to Hardware OpCodes.

`define OP_RETURN       8'd0   // End of Pattern (Virtual Return / NOP)
`define OP_PAD          8'd1   // Padding (Reserved for alignment)
`define OP_DATA_INPUT   8'd2   // <DATA_INPUT>: DMA Load (DDR -> BRAM)
`define OP_PARAM_INPUT  8'd3   // <PARAM_INPUT>: Legacy (usually unused)
`define OP_OUTPUT       8'd4   // <OUTPUT>: DMA Store (BRAM -> DDR)
`define OP_CONST_REF    8'd5   // <CONST_REF>: Parameter reference
`define OP_PATTERN      8'd6   // <PATTERN_PREFIX>: Call Subroutine (Push Stack)
`define OP_COPY         8'd7   // <COPY_PREFIX>: RLE Loop

// ============================================================================
// 2. Control Block Memory Map (DDR Base Address 0x0000_0000)
// ============================================================================
// The Host (Zynq PS / Cortex-A53) writes configuration data here before starting PL.

// --- Command / Status Registers ---
`define REG_CMD         32'h00  // Write: 1 = Load Config (Read Pointers), 2 = Run Inference
`define REG_STATUS      32'h04  // Read: 0 = Idle, 1 = Busy, 2 = Done, 0xFF = Error

// --- Memory Pointers (32-bit Addresses) ---
// The FSM reads these sequentially starting from 0x08.
`define REG_REGISTRY    32'h08  // Pointer to Pattern Registry (Jump Table)
`define REG_CODE        32'h0C  // Pointer to .b64 Instruction Stream
`define REG_WEIGHTS     32'h10  // Base Pointer to Weights Blob (Quantized INT8)
`define REG_INPUT       32'h14  // Pointer to Input Tensor Buffer
`define REG_OUTPUT      32'h18  // Pointer to Output Tensor Buffer
`define REG_OPMAP       32'h1C  // Pointer to OpCode Translation LUT (256 bytes)
`define REG_VARMAP      32'h20  // Pointer to Variation Translation LUT (256 bytes)

// ============================================================================
// 3. Hardware Constraints & Parameters (XCZU2CG Optimized)
// ============================================================================

// Vector Length (Granularity of computation)
// 512 elements * 4 bytes (32-bit) = 2 KB per vector slot.
// In INT8 Mode, one 32-bit word holds 4 elements, so this is 2048 INT8 values effectively.
`define VECTOR_LEN      512

// Tensor Memory Size (Ring Buffer)
// 32 slots * 2 KB = 64 KB Total BRAM usage.
// Easily fits in Zynq UltraScale+ BRAM/URAM.
`define TENSOR_SLOTS    32

// Modulo Mask for Ring Buffer Addressing
// Must be (TENSOR_SLOTS - 1). Used for wrapping IDs.
`define SLOT_MASK       5'h1F

// Maximum inputs per node (defines size of dependency decoding buffer in FSM)
`define MAX_INPUTS      16

// Maximum Recursion Depth for Patterns (Hardware Stack Size)
`define STACK_DEPTH     16

// Fixed Point Math Constant (Q16.16)
// 1.0 represented as integer. Used for Logic/Mask operations.
`define FIXED_ONE       32'd65536 

`endif