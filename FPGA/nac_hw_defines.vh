`ifndef NAC_HW_DEFINES_VH
`define NAC_HW_DEFINES_VH

// ============================================================================
// INTERNAL HARDWARE OPCODES (5-bit)
// ============================================================================
// These constants map 1-to-1 with the 'case(hw_opcode)' in nac_alu_core.v

// --- 0. Control ---
`define H_NOP           5'd0

// --- 1. Basic Math ---
`define H_ADD           5'd1
`define H_SUB           5'd2
`define H_MUL           5'd3
`define H_DIV           5'd4
`define H_NEG           5'd5
`define H_POW           5'd6

// --- 2. Trigonometry & Math Functions ---
`define H_SIN           5'd7
`define H_COS           5'd8
`define H_ERF           5'd9

// --- 3. Activations ---
`define H_RELU          5'd10
`define H_GELU          5'd11
`define H_TANH          5'd12
`define H_SOFTMAX       5'd13  // Placeholder/Simple approx

// --- 4. Matrix & Accumulators ---
`define H_LINEAR        5'd14  // Dot Product / MAC
`define H_MATMUL        5'd15  // Same as Linear in this architecture
`define H_SUM           5'd16  // Sum Reduction
`define H_MEAN          5'd17  // Mean Reduction
`define H_OUTER         5'd18  // Outer Product (Vector * Vector)

// --- 5. Logic & Comparisons (Masks) ---
`define H_EQ            5'd19  // Equal
`define H_NE            5'd20  // Not Equal
`define H_GT            5'd21  // Greater Than
`define H_LT            5'd22  // Less Than
`define H_MASKED_FILL   5'd23  // Selection logic

// --- 6. Generators ---
`define H_ARANGE        5'd24  // Index generator
`define H_FULL          5'd25  // Constant generator
`define H_TRIU          5'd26  // Upper Triangle Mask

// --- 7. Data Movement ---
`define H_COPY          5'd27  // Copy / Clone / Slice
`define H_EMBED         5'd28  // Embedding Lookup

// --- Reserved for future ---
`define H_CONV2D        5'd29 

`endif