`ifndef NAC_DEFINES_VH
`define NAC_DEFINES_VH

// ============================================================================
// NAC v1.1 Operation Codes (Must match registry.json)
// ============================================================================
// Control Flow & Structure
`define OP_RETURN       8'd0   // End of Pattern
`define OP_DATA_INPUT   8'd2   // Load data from DDR to Internal Memory
`define OP_OUTPUT       8'd4   // Store result from Internal Memory to DDR
`define OP_PATTERN      8'd6   // Call Subroutine (Pattern)
`define OP_COPY         8'd7   // RLE Compression (Repeat instruction)

// Fundamental Math Operations
`define OP_ADD          8'd10  // Element-wise Add
`define OP_MUL          8'd11  // Element-wise Mul
`define OP_RELU         8'd20  // Activation (ReLU)
`define OP_LINEAR       8'd50  // Linear Layer (Requires Weight Streaming)

// ============================================================================
// Memory Map Configuration (Control Block at DDR Address 0x0000_0000)
// ============================================================================
`define REG_CMD         32'h00  // Write: 1=Load Config/Registry, 2=Run Inference
`define REG_STATUS      32'h04  // Read: 0=Idle, 1=Busy, 2=Done, 0xFF=Error
`define REG_REGISTRY    32'h08  // Pointer to Pattern Registry Table in DDR
`define REG_CODE        32'h0C  // Pointer to .b64 Bytecode in DDR
`define REG_WEIGHTS     32'h10  // Base Pointer to Weights Blob in DDR
`define REG_INPUT       32'h14  // Pointer to Input Tensor
`define REG_OUTPUT      32'h18  // Pointer to Output Tensor

// ============================================================================
// Hardware Parameters
// ============================================================================
`define VECTOR_LEN      512     // Processing block size (INT32 words)
`define TENSOR_SLOTS    32      // Number of slots in Ring Buffer
`define SLOT_MASK       5'h1F   // Mask for modulo arithmetic (32-1)
`define STACK_DEPTH     16      // Maximum Pattern recursion depth
`define MAX_INPUTS      16      // Maximum inputs per NAC node

`endif
