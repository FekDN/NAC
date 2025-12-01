`ifndef NAC_DEFINES_VH
`define NAC_DEFINES_VH

// ============================================================================
// 1. NAC v1.1 System Control Codes (Dynamic OpCodes)
// ============================================================================
// These values correspond to bytes in the NAC binary stream.
// Codes 0-9 are reserved and are handled directly by the main FSM.

`define OP_RETURN       8'd0   // End of a pattern (return from subroutine)
`define OP_PAD          8'd1   // Reserved for alignment
`define OP_DATA_INPUT   8'd2   // Load input tensor from DDR to BRAM
`define OP_PARAM_INPUT  8'd3   // Deprecated, not used
`define OP_OUTPUT       8'd4   // Unload output tensor from BRAM to DDR
`define OP_CONST_REF    8'd5   // Reference to a parameter (weights)
`define OP_PATTERN      8'd6   // Call a subroutine (pattern)
`define OP_COPY         8'd7   // RLE compression command (loop)

/// ============================================================================
// 2. Configuration Pointers: Offsets and Register Map
// ============================================================================
// This block defines how the NAC core finds its data in memory.
//
// TWO USAGE MODES:
//
// 1. Zynq (SoC): The values below are offsets in the AXI-Lite register file.
//    The processor (PS) writes full 32-bit addresses to these registers.
//
// 2. Artix (Standalone): The values below are FIXED OFFSETS (in bytes)
//    relative to the base DDR address (`DDR_BASE_ADDR`). The bootloader from the SD card
//    must place the binary files exactly at these offsets.

// --- Register Offsets (Zynq) ---
`define REG_CMD_OFFSET         32'h00  // Write: 1=Start
`define REG_STATUS_OFFSET      32'h04  // Read: Status

// --- Pointer Offsets (BOTH MODES) ---
`define PTR_REGISTRY_OFFSET    32'h08
`define PTR_CODE_OFFSET        32'h0C
`define PTR_WEIGHTS_OFFSET     32'h10
`define PTR_INPUT_OFFSET       32'h14
`define PTR_OUTPUT_OFFSET      32'h18
`define PTR_OPMAP_OFFSET       32'h1C
`define PTR_VARMAP_OFFSET      32'h20

// ============================================================================
// 3. Hardware Constraints & Parameters
// ============================================================================

// Maximum dimensionality of tensors (e.g., 4 for [N, C, H, W])
// This parameter defines the size of buffers for storing metadata.
`define MAX_DIMS      4

// Vector length (the base data block size for element-wise operations)
// 512 elements * 4 bytes (32-bit) = 2 KB per slot.
`define VECTOR_LEN      512

// Number of slots for storing intermediate tensors in BRAM.
// 32 slots * 2 KB = 64 KB of total BRAM size.
`define TENSOR_SLOTS    32

// Mask for the circular buffer addressing of slots (TENSOR_SLOTS - 1).
`define SLOT_MASK       5'h1F

// Maximum number of inputs (D[]) for a single graph node.
`define MAX_INPUTS      16

// Maximum recursion depth for pattern calls (hardware stack size).
`define STACK_DEPTH     16

// Constant for representing 1.0 in Q16.16 fixed-point format.
`define FIXED_ONE       32'd65536 

`endif // NAC_DEFINES_VH