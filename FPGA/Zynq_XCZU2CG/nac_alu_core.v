`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_ALU_Core #(
    parameter RAM_STYLE = "block" // Can be "block" (BRAM) or "distributed" (LUTRAM)
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Command Interface (From Main FSM)
    // ========================================================================
    input  wire        start,          // Pulse: Start execution of the vector
    input  wire [4:0]  hw_opcode,      // Static Hardware OpCode (Translated)
    input  wire [15:0] src1_id,        // Input A Tensor ID (Ring Buffer Slot)
    input  wire [15:0] src2_id_bram,   // Input B Tensor ID (Ring Buffer Slot)
    input  wire [15:0] dst_id,         // Output Tensor ID (Ring Buffer Slot)
    output reg         done,           // Pulse: Vector execution finished

    // ========================================================================
    // Weight Streaming Interface (DDR -> ALU)
    // ========================================================================
    input  wire        use_stream,      // 1 = Input B comes from Stream
    input  wire [31:0] stream_data,     // The weight data payload (32-bit word)
    input  wire        stream_valid,    // Handshake: Data is valid
    output wire        stream_ready,    // Handshake: Core ready (Backpressure)

    // ========================================================================
    // DMA Interfaces (Priority Access)
    // ========================================================================
    // Write (DDR -> BRAM)
    input  wire        dma_wr_en,
    input  wire [31:0] dma_wr_data,
    input  wire [15:0] dma_wr_offset,   // Offset 0..VECTOR_LEN-1
    // Read (BRAM -> DDR)
    input  wire        dma_rd_req,
    output wire [31:0] dma_rd_data
);

    // ========================================================================
    // Constants & Configuration
    // ========================================================================
    // Q16.16 Fixed Point Saturation Limits
    localparam signed [31:0] MAX_POS = 32'h7FFFFFFF;
    localparam signed [31:0] MIN_NEG = 32'h80000000;
    localparam signed [31:0] FP_ONE  = `FIXED_ONE;
    localparam SHIFT_MEAN = $clog2(`VECTOR_LEN);

    // ========================================================================
    // 1. Internal Memory (Tensor Activations)
    // ========================================================================
    // True Dual-Port RAM. Stores 32-bit words.
    // In INT8 Mode, each word holds 4 packed values [31:24][23:16][15:8][7:0].
    (* ram_style = RAM_STYLE *) 
    reg signed [31:0] tensor_mem [0:(`TENSOR_SLOTS * `VECTOR_LEN) - 1];

    // ========================================================================
    // 2. Pipeline State Registers
    // ========================================================================
    reg [15:0] base_src1, base_src2, base_dst;
    reg [15:0] vec_cnt;       // Current Element Index (0..511)
    reg        active;        // 1 = Pipeline Running
    
    // Pipeline Operands
    reg [31:0] op_a;          // Operand A (From BRAM)
    reg [31:0] op_b;          // Operand B (From BRAM)
    reg [31:0] wb_data;       // Writeback Data
    reg        wb_enable;     // Write Enable Flag
    reg [15:0] wb_addr;       // Write Address
    
    // Accumulator for Reductions
    // 64-bit to support both Q16.16 (32+16) and INT8 sums (8+8+logN)
    reg signed [63:0] accumulator;
    reg        is_accum_op;

    // ========================================================================
    // 3. Division Logic (FSM for Q16.16)
    // ========================================================================
    // Division is slow, so we implement it as a multi-cycle stall.
    reg        div_busy;
    reg [5:0]  div_counter;   // 32 iterations
    reg        div_sign;
    reg [63:0] div_dividend;
    reg [31:0] div_divisor;
    reg [31:0] div_result;

    // ========================================================================
    // 4. Flow Control Logic
    // ========================================================================
    // Stall if:
    // 1. We need stream data but it's not valid.
    // 2. We are performing a multi-cycle division.
    wire stall_div = (hw_opcode == `H_DIV) && div_busy;
    wire stall = (active && use_stream && !stream_valid) || stall_div;
    
    // Ready to accept stream if active, needing stream, and not stalled by div
    assign stream_ready = (active && use_stream && !div_busy);

    // ========================================================================
    // 5. Math Helper Functions
    // ========================================================================
    
    // --- Saturation (Clamp 64-bit to 32-bit) ---
    function signed [31:0] saturate;
        input signed [63:0] val;
        begin
            if (val > MAX_POS) saturate = MAX_POS;
            else if (val < MIN_NEG) saturate = MIN_NEG;
            else saturate = val[31:0];
        end
    endfunction

    // --- INT8 SIMD: Dot Product 4 (DP4) ---
    // Multiplies 4 pairs of 8-bit integers and sums them up.
    // Result is 32-bit signed integer.
    function signed [31:0] simd_dot_product;
        input [31:0] packed_a;
        input [31:0] packed_b;
        reg signed [7:0] a0, a1, a2, a3;
        reg signed [7:0] b0, b1, b2, b3;
        reg signed [16:0] p0, p1, p2, p3; 
        begin
            // Unpack
            a0 = packed_a[7:0];   b0 = packed_b[7:0];
            a1 = packed_a[15:8];  b1 = packed_b[15:8];
            a2 = packed_a[23:16]; b2 = packed_b[23:16];
            a3 = packed_a[31:24]; b3 = packed_b[31:24];
            
            // Multiply (Mapped to DSP48E2 pre-adders or logic)
            p0 = a0 * b0;
            p1 = a1 * b1;
            p2 = a2 * b2;
            p3 = a3 * b3;
            
            // Sum
            simd_dot_product = p0 + p1 + p2 + p3;
        end
    endfunction

    // --- INT8 SIMD: ReLU ---
    // Applies ReLU to each byte independently.
    function [31:0] simd_relu;
        input [31:0] packed_val;
        reg signed [7:0] b0, b1, b2, b3;
        begin
            b0 = packed_val[7:0];
            b1 = packed_val[15:8];
            b2 = packed_val[23:16];
            b3 = packed_val[31:24];
            
            simd_relu = {
                (b3 > 0 ? b3 : 8'd0),
                (b2 > 0 ? b2 : 8'd0),
                (b1 > 0 ? b1 : 8'd0),
                (b0 > 0 ? b0 : 8'd0)
            };
        end
    endfunction

    // --- Q16.16 Functions ---
    function signed [31:0] fx_gelu;
        input signed [31:0] x;
        begin
            if (x > 32'd262144) fx_gelu = x;
            else if (x < -32'd262144) fx_gelu = 0;
            else if (x > 0) fx_gelu = (x >>> 1) + (x >>> 2);
            else fx_gelu = 0; 
        end
    endfunction

    function signed [31:0] fx_tanh;
        input signed [31:0] x;
        begin
            if (x > 32'd131072) fx_tanh = FP_ONE;
            else if (x < -32'd131072) fx_tanh = -FP_ONE;
            else fx_tanh = x;
        end
    endfunction

    // ========================================================================
    // 6. Memory Port A (Write Priority & Read A)
    // ========================================================================
    always @(posedge clk) begin
        // Priority 1: DMA Write
        if (dma_wr_en) begin
            tensor_mem[base_dst + dma_wr_offset] <= dma_wr_data;
        end 
        // Priority 2: Reduction Finalize (End of vector)
        else if (is_accum_op && done) begin
            if (hw_opcode == `H_SUM || hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL)
                // Note: For INT8 Linear, result is usually 32-bit. 
                // In a real quantization flow, we'd scale/shift here to back to INT8.
                // Here we saturate to 32-bit for flexibility.
                tensor_mem[base_dst] <= saturate(accumulator);
            else if (hw_opcode == `H_MEAN)
                // Q16.16 Mean
                tensor_mem[base_dst] <= saturate(accumulator >>> SHIFT_MEAN);
        end 
        // Priority 3: Pipeline Writeback
        else if (wb_enable && !stall && !is_accum_op) begin
            tensor_mem[wb_addr] <= wb_data;
        end

        // Read Operand A
        if (!stall) begin
            op_a <= tensor_mem[base_src1 + vec_cnt];
        end
    end

    // ========================================================================
    // 7. Memory Port B (Read B / DMA Read)
    // ========================================================================
    assign dma_rd_data = op_b;

    always @(posedge clk) begin
        if (!stall) begin
            if (dma_rd_req) begin
                // DMA Read uses specific offset
                op_b <= tensor_mem[base_src1 + dma_wr_offset];
            end else begin
                // Compute uses vector counter
                op_b <= tensor_mem[base_src2 + vec_cnt];
            end
        end
    end

    // ========================================================================
    // 8. Main Execution Pipeline
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
            done <= 0; // Reset Pulse

            // --- START COMMAND ---
            if (start) begin
                active <= 1;
                vec_cnt <= 0;
                wb_enable <= 0;
                accumulator <= 0;
                div_busy <= 0; // Reset Div FSM
                
                is_accum_op <= (hw_opcode == `H_SUM || hw_opcode == `H_MEAN || 
                                hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL);

                // Slot addressing
                base_src1 <= (src1_id & `SLOT_MASK) * `VECTOR_LEN;
                base_src2 <= (src2_id_bram & `SLOT_MASK) * `VECTOR_LEN;
                base_dst  <= (dst_id  & `SLOT_MASK) * `VECTOR_LEN;
            end 
            // --- EXECUTION LOOP ---
            else if (active) begin
                
                // Operand Selection (Stream vs BRAM)
                reg [31:0] final_b;
                final_b = use_stream ? stream_data : op_b;
                
                // Temp vars
                reg signed [63:0] wide_res; 

                // ------------------------------------------------------------
                // SPECIAL CASE: DIVISION (Q16.16 Long Division FSM)
                // ------------------------------------------------------------
                if (hw_opcode == `H_DIV) begin
                    if (!div_busy) begin
                        // Init
                        if (final_b == 0) begin
                            wb_data <= 0; // Div by zero protection
                        end else begin
                            // Prepare Q16.16 Div: (A << 16) / B
                            div_sign <= (op_a[31] ^ final_b[31]);
                            div_dividend <= {32'd0, ($signed(op_a) < 0 ? -$signed(op_a) : $signed(op_a))} << 16; 
                            div_divisor  <= ($signed(final_b) < 0 ? -$signed(final_b) : $signed(final_b));
                            div_result   <= 0;
                            div_counter  <= 32;
                            div_busy     <= 1; // Assert STALL
                            wb_enable    <= 0;
                        end
                    end 
                    else begin
                        // Iterate
                        reg [63:0] next_dividend;
                        next_dividend = div_dividend << 1;
                        
                        if (next_dividend[63:32] >= div_divisor) begin
                            next_dividend[63:32] = next_dividend[63:32] - div_divisor;
                            div_result = (div_result << 1) | 1'b1;
                        end else begin
                            div_result = (div_result << 1);
                        end
                        div_dividend <= next_dividend;

                        if (div_counter == 1) begin
                            // Finish
                            wb_data <= div_sign ? -div_result : div_result;
                            div_busy <= 0; // Release STALL
                            wb_enable <= 1;
                            wb_addr   <= base_dst + vec_cnt;
                            
                            // Manual Increment (since main logic is blocked by stall)
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
                // STANDARD OPERATIONS (If not stalled)
                // ------------------------------------------------------------
                if (!stall) begin
                    if (hw_opcode != `H_DIV) begin
                        case (hw_opcode)
                            // === INT8 SIMD OPERATIONS ===
                            `H_LINEAR, `H_MATMUL: begin
                                // 4x Multiply-Accumulate per clock
                                accumulator <= accumulator + simd_dot_product(op_a, final_b);
                                wb_data <= 0; 
                            end
                            `H_RELU: begin
                                // 4x ReLU per clock
                                wb_data <= simd_relu(op_a);
                            end

                            // === Q16.16 BASIC MATH ===
                            `H_ADD: begin wide_res = $signed(op_a) + $signed(final_b); wb_data <= saturate(wide_res); end
                            `H_SUB: begin wide_res = $signed(op_a) - $signed(final_b); wb_data <= saturate(wide_res); end
                            `H_MUL: begin wide_res = $signed(op_a) * $signed(final_b); wb_data <= saturate(wide_res >>> 16); end
                            `H_NEG: wb_data <= -$signed(op_a);
                            `H_POW: begin wide_res = $signed(op_a) * $signed(op_a); wb_data <= saturate(wide_res >>> 16); end

                            // === Q16.16 ACTIVATIONS ===
                            `H_GELU: wb_data <= fx_gelu($signed(op_a));
                            `H_TANH, `H_ERF: wb_data <= fx_tanh($signed(op_a));
                            
                            // === LOGIC & MASKS ===
                            `H_EQ: wb_data <= (op_a == final_b) ? FP_ONE : 0;
                            `H_NE: wb_data <= (op_a != final_b) ? FP_ONE : 0;
                            `H_GT: wb_data <= ($signed(op_a) > $signed(final_b)) ? FP_ONE : 0;
                            `H_LT: wb_data <= ($signed(op_a) < $signed(final_b)) ? FP_ONE : 0;
                            `H_MASKED_FILL: wb_data <= (op_a != 0) ? final_b : 0;
                            
                            // === GENERATORS ===
                            `H_ARANGE: wb_data <= saturate(vec_cnt << 16);
                            `H_FULL, `H_EMBED: wb_data <= final_b;
                            
                            // === DATA MOVEMENT ===
                            `H_COPY: wb_data <= op_a;

                            // === Q16.16 REDUCTIONS ===
                            `H_SUM, `H_MEAN: begin
                                accumulator <= accumulator + $signed(op_a);
                                wb_data <= 0;
                            end
                            
                            default: wb_data <= op_a;
                        endcase

                        // Writeback & Loop Logic
                        wb_addr   <= base_dst + vec_cnt;
                        wb_enable <= 1;

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
                // Idle
                wb_enable <= 0;
            end
        end
    end

endmodule