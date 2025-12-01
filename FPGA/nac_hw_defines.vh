`ifndef NAC_HW_DEFINES_VH
`define NAC_HW_DEFINES_VH

// ============================================================================
// INTERNAL HARDWARE OPCODES (6-bit)
// ============================================================================
// These constants have a 1-to-1 mapping with the `case(hw_opcode)` in nac_alu_core.v.
// The width is extended to 6 bits to support all necessary operations.

// --- 0. Control ---
`define H_NOP           6'd0   // No Operation

// --- 1. Basic Math (Q16.16 Scalar) ---
`define H_ADD           6'd1   // Addition with saturation
`define H_SUB           6'd2   // Subtraction with saturation
`define H_MUL           6'd3   // Fixed-point multiplication (with result shifting)
`define H_DIV           6'd4   // Division (multi-cycle)
`define H_NEG           6'd5   // Unary minus
`define H_POW           6'd6   // Power (implemented as square x*x)

// --- 2. Math Functions (Q16.16 Scalar, LUT/Approximation) ---
`define H_SIN           6'd7   // Sine approximation
`define H_COS           6'd8   // Cosine approximation
`define H_ERF           6'd9   // Error function approximation

// --- 3. Activations ---
`define H_RELU          6'd10  // ReLU (supports SIMD for INT8)
`define H_GELU          6'd11  // GELU (via LUT)
`define H_TANH          6'd12  // Tanh (via LUT)
`define H_SOFTMAX       6'd13  // Softmax (multi-pass operation)

// --- 4. Matrix, Reductions & Normalization (INT8 SIMD / Multi-pass) ---
`define H_LINEAR        6'd14  // Linear layer (SIMD Dot Product)
`define H_MATMUL        6'd15  // Matrix multiplication (similar to H_LINEAR)
`define H_SUM           6'd16  // Summation over a vector (Reduction)
`define H_MEAN          6'd17  // Mean over a vector (Reduction)
`define H_LAYER_NORM    6'd18  // Layer Normalization (two-pass operation)
`define H_BATCH_NORM    6'd30  // Batch Normalization (similar to LayerNorm, but with streamed parameters)

// --- 5. Logic & Comparisons (Q16.16 Scalar) ---
`define H_EQ            6'd19  // Equal
`define H_NE            6'd20  // Not Equal
`define H_GT            6'd21  // Greater Than
`define H_LT            6'd22  // Less Than
`define H_MASKED_FILL   6'd23  // Masked Fill

// --- 6. Generators ---
`define H_ARANGE        6'd24  // Sequence generator (0, 1, 2...)
`define H_FULL          6'd25  // Fill with a constant
`define H_TRIU          6'd26  // Upper triangular mask

// --- 7. Data Movement & Specialized Compute ---
`define H_COPY          6'd27  // Copy / Dropout in inference mode
`define H_EMBED         6'd28  // Embedding (simple copy from the weights stream)
`define H_CONV2D        6'd29  // 2D Convolution (complex multi-pass operation)
`define H_MAX_POOL2D    6'd31  // 2D Max Pooling
`define H_ADAPTIVE_AVG_POOL2D 6'd32 // 2D Adaptive Average Pooling (similar to MEAN, but with different parameters)

// --- 8. Metadata Pseudo-Opcodes (Handled by Top FSM) ---
// These opcodes do not invoke the ALU but modify tensor descriptors.
`define H_VIEW          6'd40  // Change shape (Reshape)
`define H_TRANSPOSE     6'd41  // Transpose (swap 2 axes)
`define H_PERMUTE       6'd42  // Permute all axes
`define H_UNSQUEEZE     6'd43  // Add a dimension of size 1
`define H_CAT           6'd44  // Concatenation (reserved)
`define H_SPLIT         6'd45  // Split (reserved)

// --- Reserved for future use ---
// `define H_OUTER         6'd18 // Reserved

`endif // NAC_HW_DEFINES_VH