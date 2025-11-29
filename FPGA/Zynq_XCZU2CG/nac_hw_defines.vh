`ifndef NAC_HW_DEFINES_VH
`define NAC_HW_DEFINES_VH

// ============================================================================
// INTERNAL HARDWARE OPCODES (5-bit)
// ============================================================================
// These constants map 1-to-1 with the 'case(hw_opcode)' in nac_alu_core.v.
//
// MODES:
// - SIMD: Operates on 4x INT8 values packed in a 32-bit word.
// - SCALAR: Operates on 1x 32-bit value (Q16.16 Fixed Point).

// --- 0. Control ---
`define H_NOP           5'd0   // No Operation

// --- 1. Basic Math (Q16.16 Scalar) ---
`define H_ADD           5'd1   // Add (Saturated)
`define H_SUB           5'd2   // Subtract (Saturated)
`define H_MUL           5'd3   // Multiply (Result >> 16)
`define H_DIV           5'd4   // Division (Multi-cycle Stall FSM)
`define H_NEG           5'd5   // Negate
`define H_POW           5'd6   // Square (x*x)

// --- 2. Trigonometry & Math Functions (Q16.16 Scalar) ---
`define H_SIN           5'd7   // Sine Approx
`define H_COS           5'd8   // Cosine Approx
`define H_ERF           5'd9   // Error Function (Tanh Approx)

// --- 3. Activations ---
`define H_RELU          5'd10  // [SIMD Supported] ReLU on 4 bytes
`define H_GELU          5'd11  // Q16.16 Scalar GELU Approx
`define H_TANH          5'd12  // Q16.16 Scalar Tanh Approx
`define H_SOFTMAX       5'd13  // Placeholder (Scalar)

// --- 4. Matrix & Accumulators (INT8 SIMD Optimized) ---
// These operations use the 4-way Dot Product engine.
// Input: 4x INT8 Activations (A) and 4x INT8 Weights (B).
// Output: Accumulate into 64-bit sum.
`define H_LINEAR        5'd14  // SIMD DP4A (Dot Product 4-Accumulate)
`define H_MATMUL        5'd15  // Same as Linear (Dense Matrix Mult)

// --- Reductions (Q16.16 Scalar) ---
`define H_SUM           5'd16  // Sum Reduction (Accumulate 32-bit)
`define H_MEAN          5'd17  // Mean Reduction (Accumulate + Shift)
`define H_OUTER         5'd18  // Outer Product (Reserved)

// --- 5. Logic & Comparisons (Q16.16 Scalar) ---
// Returns 1.0 (Fixed Point) or 0.
`define H_EQ            5'd19  // Equal
`define H_NE            5'd20  // Not Equal
`define H_GT            5'd21  // Greater Than
`define H_LT            5'd22  // Less Than
`define H_MASKED_FILL   5'd23  // Selection Logic / Mux

// --- 6. Generators (Q16.16 Scalar) ---
`define H_ARANGE        5'd24  // Index generator (0, 1, 2...)
`define H_FULL          5'd25  // Constant generator (Fill with B)
`define H_TRIU          5'd26  // Upper Triangle Mask

// --- 7. Data Movement ---
`define H_COPY          5'd27  // Copy / Clone / Slice (Passthrough)
`define H_EMBED         5'd28  // Embedding Lookup (Passthrough from Stream)

// --- Reserved ---
`define H_CONV2D        5'd29  // Convolution (Future INT8 feature)

`endif