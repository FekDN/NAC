`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_ALU_Core #(
    parameter RAM_STYLE = "block" // "block" for BRAM, "distributed" for LUTRAM (small sizes)
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Command Interface
    // ========================================================================
    input  wire        start,          // Pulse: Start execution of the vector/matrix op
    input  wire [4:0]  hw_opcode,      // Hardware OpCode
    input  wire [15:0] src1_id,        // Slot ID for Input A
    input  wire [15:0] src2_id_bram,   // Slot ID for Input B (if not streaming)
    input  wire [15:0] dst_id,         // Slot ID for Output
    output reg         done,           // Pulse: Execution Finished

    // ========================================================================
    // Weight Streaming Interface (DDR -> ALU bypass)
    // ========================================================================
    input  wire        use_stream,     // 1 = Operand B comes from AXI Stream
    input  wire [31:0] stream_data,    // Raw data (Q16.16 or 4xINT8 packed)
    input  wire        stream_valid,   // Data Valid
    output wire        stream_ready,   // Backpressure signal

    // ========================================================================
    // DMA / Memory Access Interface
    // ========================================================================
    // Write Port (DDR -> Internal Mem)
    input  wire        dma_wr_en,
    input  wire [31:0] dma_wr_data,
    input  wire [15:0] dma_wr_offset,  // 0..VECTOR_LEN-1
    // Read Port (Internal Mem -> DDR)
    input  wire        dma_rd_req,
    output wire [31:0] dma_rd_data
);

    // ========================================================================
    // Constants
    // ========================================================================
    localparam signed [31:0] MAX_POS = 32'h7FFFFFFF;
    localparam signed [31:0] MIN_NEG = 32'h80000000;
    localparam signed [31:0] FP_ONE  = `FIXED_ONE; // 65536
    localparam SHIFT_MEAN = $clog2(`VECTOR_LEN);

    // ========================================================================
    // 1. Tensor Memory (Dual Port)
    // ========================================================================
    // Stores activations. Configured as BRAM (Block RAM) on UltraScale+.
    (* ram_style = RAM_STYLE *) 
    reg signed [31:0] tensor_mem [0:(`TENSOR_SLOTS * `VECTOR_LEN) - 1];

    // ========================================================================
    // 2. Pipeline Registers
    // ========================================================================
    reg [15:0] base_src1, base_src2, base_dst;
    reg [15:0] vec_cnt;
    reg        active;

    // Pipeline Stages
    reg [31:0] op_a;        // Src1 (Register A)
    reg [31:0] op_b;        // Src2 (Register B) - From Mem or DMA
    reg [31:0] wb_data;     // Result to write back
    reg        wb_enable;   // Write enable for result
    reg [15:0] wb_addr;     // Address for write back

    // Accumulator (64-bit to prevent overflow during Reductions/DotProducts)
    reg signed [63:0] accumulator;
    reg is_accum_op;

    // ========================================================================
    // 3. Divider State Machine (Radix-16 Optimization)
    // ========================================================================
    reg        div_busy;
    reg [4:0]  div_counter; // Counts 8 cycles (4 bits per cycle)
    reg        div_sign;
    reg [63:0] div_dividend;
    reg [31:0] div_divisor;
    reg [31:0] div_result;

    // ========================================================================
    // 4. Flow Control
    // ========================================================================
    wire stall_div = (hw_opcode == `H_DIV) && div_busy;
    // Stall pipeline if:
    // 1. Streaming is enabled but data is missing.
    // 2. Divider is busy calculating.
    wire stall = (active && use_stream && !stream_valid) || stall_div;

    // Signal to Stream Prefetcher that we are ready to consume data
    assign stream_ready = (active && use_stream && !div_busy);

    // ========================================================================
    // 5. Math Functions
    // ========================================================================

    // --- Saturation Logic (Clamp 64-bit -> 32-bit) ---
    function signed [31:0] saturate;
        input signed [63:0] val;
        begin
            if (val > MAX_POS) saturate = MAX_POS;
            else if (val < MIN_NEG) saturate = MIN_NEG;
            else saturate = val[31:0];
        end
    endfunction

    // --- INT8 SIMD Dot Product (DSP48E2 Optimized) ---
    // Computes: a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    function signed [31:0] simd_dot_product;
        input [31:0] packed_a;
        input [31:0] packed_b;
        reg signed [7:0] a0, a1, a2, a3;
        reg signed [7:0] b0, b1, b2, b3;
        reg signed [16:0] p0, p1, p2, p3; 
        begin
            // Explicit sign extension for correct INT8 math
            a0 = $signed(packed_a[7:0]);   b0 = $signed(packed_b[7:0]);
            a1 = $signed(packed_a[15:8]);  b1 = $signed(packed_b[15:8]);
            a2 = $signed(packed_a[23:16]); b2 = $signed(packed_b[23:16]);
            a3 = $signed(packed_a[31:24]); b3 = $signed(packed_b[31:24]);
            
            p0 = a0 * b0;
            p1 = a1 * b1;
            p2 = a2 * b2;
            p3 = a3 * b3;
            
            simd_dot_product = p0 + p1 + p2 + p3;
        end
    endfunction

    // --- INT8 SIMD ReLU ---
    function [31:0] simd_relu;
        input [31:0] packed_val;
        reg signed [7:0] b0, b1, b2, b3;
        begin
            b0 = $signed(packed_val[7:0]);
            b1 = $signed(packed_val[15:8]);
            b2 = $signed(packed_val[23:16]);
            b3 = $signed(packed_val[31:24]);
            
            simd_relu = {
                (b3 > 0 ? b3 : 8'd0),
                (b2 > 0 ? b2 : 8'd0),
                (b1 > 0 ? b1 : 8'd0),
                (b0 > 0 ? b0 : 8'd0)
            };
        end
    endfunction

    // --- Q16.16 Non-Linear Approximations ---
    function signed [31:0] fx_gelu;
        input signed [31:0] x;
        begin
            // Simple Piecewise Linear Approximation
            if (x > 32'd262144) fx_gelu = x; // > 4.0
            else if (x < -32'd262144) fx_gelu = 0; // < -4.0
            else if (x > 0) fx_gelu = (x >>> 1) + (x >>> 2); // 0.75x
            else fx_gelu = 0; // Negative regime approx
        end
    endfunction

    function signed [31:0] fx_tanh;
        input signed [31:0] x;
        begin
            if (x > 32'd131072) fx_tanh = FP_ONE; // > 2.0
            else if (x < -32'd131072) fx_tanh = -FP_ONE; // < -2.0
            else fx_tanh = x; // Linear regime
        end
    endfunction

    // ========================================================================
    // 6. Memory Port A (Write Logic)
    // ========================================================================
    always @(posedge clk) begin
        // Priority 1: External DMA Write
        if (dma_wr_en) begin
            tensor_mem[base_dst + dma_wr_offset] <= dma_wr_data;
        end 
        // Priority 2: Reduction Finalize (Write Accumulator at end of vector)
        else if (is_accum_op && done) begin
            if (hw_opcode == `H_MEAN)
                tensor_mem[base_dst] <= saturate(accumulator >>> SHIFT_MEAN);
            else
                // SUM, LINEAR, MATMUL results
                tensor_mem[base_dst] <= saturate(accumulator);
        end 
        // Priority 3: Pipeline Result Writeback
        else if (wb_enable && !stall && !is_accum_op) begin
            tensor_mem[wb_addr] <= wb_data;
        end

        // Read Operand A (Always Active)
        if (!stall) begin
            op_a <= tensor_mem[base_src1 + vec_cnt];
        end
    end

    // ========================================================================
    // 7. Memory Port B (Read Logic)
    // ========================================================================
    assign dma_rd_data = op_b;

    always @(posedge clk) begin
        if (!stall) begin
            if (dma_rd_req) begin
                // DMA Read Mode: Use offset provided by DMA engine
                op_b <= tensor_mem[base_src1 + dma_wr_offset]; 
            end else begin
                // Compute Mode: Standard Vector sequence
                op_b <= tensor_mem[base_src2 + vec_cnt];
            end
        end
    end

    // ========================================================================
    // 8. Execution Pipeline
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0;
            vec_cnt <= 0;
            wb_enable <= 0;
            wb_addr <= 0;
            wb_data <= 0;
            done <= 0;
            accumulator <= 0;
            is_accum_op <= 0;
            div_busy <= 0;
            div_counter <= 0;
            base_src1 <= 0; base_src2 <= 0; base_dst <= 0;
        end else begin
            done <= 0; // Reset Done Pulse

            if (start) begin
                // --- INITIALIZE NEW OPERATION ---
                active <= 1;
                vec_cnt <= 0;
                wb_enable <= 0;
                accumulator <= 0;
                div_busy <= 0;
                
                // Identify Reduction Ops
                is_accum_op <= (hw_opcode == `H_SUM || hw_opcode == `H_MEAN || 
                                hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL);

                // Calculate Base Addresses (Ring Buffer Logic)
                base_src1 <= (src1_id & `SLOT_MASK) * `VECTOR_LEN;
                base_src2 <= (src2_id_bram & `SLOT_MASK) * `VECTOR_LEN;
                base_dst  <= (dst_id  & `SLOT_MASK) * `VECTOR_LEN;
            end 
            else if (active) begin
                
                // Select Operand B source
                reg [31:0] final_b;
                final_b = use_stream ? stream_data : op_b;
                
                reg signed [63:0] wide_res; 

                // ------------------------------------------------------------
                // DIVIDER: Radix-16 Optimization (4 bits per clock)
                // ------------------------------------------------------------
                if (hw_opcode == `H_DIV) begin
                    if (!div_busy) begin
                        // Init Divider
                        if (final_b == 0) begin
                            wb_data <= 0; // Protect against DivByZero
                        end else begin
                            // Calculate Sign
                            div_sign <= (op_a[31] ^ final_b[31]);
                            // Absolute Value Setup + Pre-shift for Q16.16
                            div_dividend <= {32'd0, ($signed(op_a) < 0 ? -$signed(op_a) : $signed(op_a))} << 16; 
                            div_divisor  <= ($signed(final_b) < 0 ? -$signed(final_b) : $signed(final_b));
                            div_result   <= 0;
                            
                            // Set steps: 32 bits / 4 bits_per_clk = 8 cycles
                            div_counter  <= 8; 
                            div_busy     <= 1; // Pipeline STALL
                            wb_enable    <= 0;
                        end
                    end 
                    else begin
                        // --- UNROLLED DIVISION LOOP (Combinational) ---
                        // Variables for the chain
                        reg [63:0] d_curr;
                        reg [31:0] q_curr;
                        reg [63:0] d_next;
                        reg [31:0] q_next;
                        reg [32:0] diff; // 33-bit for borrow check
                        integer i;

                        d_curr = div_dividend;
                        q_curr = div_result;

                        // Execute 4 steps per clock
                        for (i = 0; i < 4; i = i + 1) begin
                            d_next = d_curr << 1;
                            q_next = q_curr << 1;
                            // Subtraction test
                            diff = {1'b0, d_next[63:32]} - {1'b0, div_divisor};
                            
                            if (diff[32] == 0) begin // If result >= 0
                                d_next[63:32] = diff[31:0]; // Update Remainder
                                q_next[0] = 1'b1;           // Set Quotient bit
                            end
                            d_curr = d_next;
                            q_curr = q_next;
                        end

                        // Register Update
                        div_dividend <= d_curr;
                        div_result   <= q_curr;

                        if (div_counter == 1) begin
                            // Finished
                            wb_data <= div_sign ? -div_result : div_result;
                            div_busy <= 0; // Release STALL
                            wb_enable <= 1;
                            wb_addr   <= base_dst + vec_cnt;
                            
                            // Manual Increment (since stalled)
                            if (vec_cnt == `VECTOR_LEN - 1) begin
                                active <= 0; done <= 1; wb_enable <= 0;
                            end else begin
                                vec_cnt <= vec_cnt + 1;
                            end
                        end else begin
                            div_counter <= div_counter - 1;
                        end
                    end
                end 
                
                // ------------------------------------------------------------
                // STANDARD OPERATIONS (Single Cycle)
                // ------------------------------------------------------------
                if (!stall) begin
                    if (hw_opcode != `H_DIV) begin
                        case (hw_opcode)
                            // === INT8 SIMD Ops ===
                            `H_LINEAR, `H_MATMUL: begin
                                // 4x MAC ops per clock (Uses DSP48E2)
                                accumulator <= accumulator + simd_dot_product(op_a, final_b);
                                wb_data <= 0; 
                            end
                            `H_RELU: begin
                                wb_data <= simd_relu(op_a);
                            end

                            // === Q16.16 Basic Math ===
                            `H_ADD: begin wide_res = $signed(op_a) + $signed(final_b); wb_data <= saturate(wide_res); end
                            `H_SUB: begin wide_res = $signed(op_a) - $signed(final_b); wb_data <= saturate(wide_res); end
                            `H_MUL: begin wide_res = $signed(op_a) * $signed(final_b); wb_data <= saturate(wide_res >>> 16); end
                            `H_NEG: wb_data <= -$signed(op_a);
                            `H_POW: begin wide_res = $signed(op_a) * $signed(op_a); wb_data <= saturate(wide_res >>> 16); end

                            // === Q16.16 Activations & Funcs ===
                            `H_GELU: wb_data <= fx_gelu($signed(op_a));
                            `H_TANH, `H_ERF: wb_data <= fx_tanh($signed(op_a));
                            `H_SIN, `H_COS: wb_data <= 0; // TODO: Implement CORDIC or LUT if needed

                            // === Logic & Masks ===
                            `H_EQ: wb_data <= (op_a == final_b) ? FP_ONE : 0;
                            `H_NE: wb_data <= (op_a != final_b) ? FP_ONE : 0;
                            `H_GT: wb_data <= ($signed(op_a) > $signed(final_b)) ? FP_ONE : 0;
                            `H_LT: wb_data <= ($signed(op_a) < $signed(final_b)) ? FP_ONE : 0;
                            `H_MASKED_FILL: wb_data <= (op_a != 0) ? final_b : 0;
                            
                            // === Generators ===
                            `H_ARANGE: wb_data <= saturate(vec_cnt << 16);
                            `H_FULL, `H_EMBED: wb_data <= final_b;
                            
                            // === Memory Ops ===
                            `H_COPY: wb_data <= op_a;

                            // === Reductions (Scalar Accumulate) ===
                            `H_SUM, `H_MEAN: begin
                                accumulator <= accumulator + $signed(op_a);
                                wb_data <= 0;
                            end
                            
                            default: wb_data <= op_a; // NOP / Passthrough
                        endcase

                        // Writeback Pipeline
                        wb_addr   <= base_dst + vec_cnt;
                        wb_enable <= 1;

                        // Loop Counter
                        if (vec_cnt == `VECTOR_LEN - 1) begin
                            active <= 0;    
                            done <= 1;      
                            wb_enable <= 0; 
                        end else begin
                            vec_cnt <= vec_cnt + 1;
                        end
                    end 
                end 
            end 
            else begin
                // Idle State
                wb_enable <= 0;
            end
        end
    end

endmodule
