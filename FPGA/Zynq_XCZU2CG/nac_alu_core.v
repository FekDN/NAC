`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_ALU_Core #(
    parameter RAM_STYLE      = "block", // "block" (BRAM) or "distributed" (LUTRAM)
    parameter USE_DSP        = "yes",   // Force usage of DSP48 slices for Math
    parameter USE_XILINX_DIV = 0,       // 0 = FSM (Slow/Small), 1 = Xilinx IP (Fast/Large)
    
    // NEW OPTION: SIMD Implementation Strategy
    // 0 = Registers inside always (1 cycle latency, compact code, lower Fmax)
    // 1 = Manual Pipeline (3 cycle latency, best for Fmax > 100MHz, pure RTL)
    // 2 = Xilinx IP Macro (Instantiation of xbip_dsp48_macro, requires .xci)
    parameter SIMD_MODE      = 1        
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Command Interface (From Main FSM)
    // ========================================================================
    input  wire        start,          
    input  wire [4:0]  hw_opcode,      
    input  wire [15:0] src1_id,        
    input  wire [15:0] src2_id_bram,   
    input  wire [15:0] dst_id,         
    output reg         done,           

    // ========================================================================
    // Weight Streaming Interface (DDR -> ALU)
    // ========================================================================
    input  wire        use_stream,      
    input  wire [31:0] stream_data,     
    input  wire        stream_valid,    
    output wire        stream_ready,    

    // ========================================================================
    // DMA Interfaces
    // ========================================================================
    input  wire        dma_wr_en,
    input  wire [31:0] dma_wr_data,
    input  wire [15:0] dma_wr_offset,   
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

    // Latency determination based on SIMD_MODE
    localparam integer SIMD_LATENCY = (SIMD_MODE == 2) ? 4 : // IP usually has 3-4 cycles
                                      (SIMD_MODE == 1) ? 3 : // Manual pipeline
                                      1;                     // Registered always block

    // ========================================================================
    // 1. Internal Memory
    // ========================================================================
    (* ram_style = RAM_STYLE *) 
    reg signed [31:0] tensor_mem [0:(`TENSOR_SLOTS * `VECTOR_LEN) - 1];

    // ========================================================================
    // 2. Activation LUT
    // ========================================================================
    (* ram_style = "block" *) reg [31:0] activation_lut [0:2047]; // Changed to block per previous recommendation
    initial $readmemh("nac_activation_lut.mem", activation_lut); 

    // ========================================================================
    // 3. Pipeline State Registers
    // ========================================================================
    reg [15:0] base_src1, base_src2, base_dst;
    reg [15:0] vec_cnt;       
    reg        active;        
    reg signed [31:0] op_a;   
    reg signed [31:0] op_b;   
    
    // Writeback Stage
    reg [31:0] wb_data;       
    reg        wb_enable;     
    reg [15:0] wb_addr;       
    
    reg signed [63:0] accumulator;
    reg        is_accum_op;
    
    // Determine Operand B
    reg signed [31:0] final_b;
    always @(*) final_b = use_stream ? $signed(stream_data) : op_b;

    // ========================================================================
    // 4. Division Logic (Kept as is)
    // ========================================================================
    reg        div_busy_fsm;
    reg [3:0]  div_counter;   
    reg        div_sign;
    reg [63:0] div_dividend;
    reg [31:0] div_divisor;
    reg [31:0] div_result_fsm;

    wire        div_ip_in_ready;
    wire        div_ip_out_valid;
    wire [47:0] div_ip_out_data;
    reg  [15:0] div_out_cnt;     

    wire stall_div_fsm = (hw_opcode == `H_DIV) && div_busy_fsm && (USE_XILINX_DIV == 0);
    wire stall_div_ip  = (hw_opcode == `H_DIV) && !div_ip_in_ready && (USE_XILINX_DIV == 1);
    wire stall = (active && use_stream && !stream_valid) || stall_div_fsm || stall_div_ip;
    assign stream_ready = (active && use_stream && !stall_div_fsm && !stall_div_ip);

    // IP Instantiation (Placeholder from original)
    generate
        if (USE_XILINX_DIV == 1) begin : gen_div_ip
             div_gen_0 u_div_ip (
                .aclk(clk),
                .s_axis_divisor_tvalid(active && (hw_opcode == `H_DIV) && !stall),
                .s_axis_divisor_tready(div_ip_in_ready),
                .s_axis_divisor_tdata(final_b),
                .s_axis_dividend_tvalid(active && (hw_opcode == `H_DIV) && !stall),
                .s_axis_dividend_tready(),
                .s_axis_dividend_tdata({op_a, 16'd0}),
                .m_axis_dout_tvalid(div_ip_out_valid),
                .m_axis_dout_tdata(div_ip_out_data)
            );
        end else begin : gen_no_div_ip
            assign div_ip_in_ready  = 1'b0;
            assign div_ip_out_valid = 1'b0;
            assign div_ip_out_data  = 48'd0;
        end
    endgenerate

    // ========================================================================
    // 5. SIMD / Dot Product Logic (Configurable Implementation)
    // ========================================================================
    wire signed [31:0] simd_result_wire;
    wire               simd_result_valid;
    
    // We only enable the SIMD pipeline when inputs are valid and we are processing SIMD opcodes
    wire simd_input_valid = active && !stall && (hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL);

    generate
        if (SIMD_MODE == 1) begin : gen_simd_manual_pipeline
            // --- Option 1: Manual RTL Pipeline (3 Cycles) ---
            // Cycle 1: Multiply 4 pairs
            // Cycle 2: Add pairs (Tree adder)
            // Cycle 3: Final Sum available
            
            reg signed [16:0] p0, p1, p2, p3;
            reg signed [31:0] sum_stage1;
            reg signed [31:0] sum_stage2;
            reg [2:0] pipe_valid_sr;

            always @(posedge clk) begin
                // Stage 1: Multiplication
                if (simd_input_valid) begin
                    p0 <= $signed(op_a[7:0])   * $signed(final_b[7:0]);
                    p1 <= $signed(op_a[15:8])  * $signed(final_b[15:8]);
                    p2 <= $signed(op_a[23:16]) * $signed(final_b[23:16]);
                    p3 <= $signed(op_a[31:24]) * $signed(final_b[31:24]);
                end else if (!stall) begin
                    p0 <= 0; p1 <= 0; p2 <= 0; p3 <= 0;
                end

                // Stage 2: Tree Adder
                sum_stage1 <= p0 + p1 + p2 + p3;
                
                // Stage 3: Output Register
                sum_stage2 <= sum_stage1;
                
                // Control Pipeline
                if (!rst_n) pipe_valid_sr <= 0;
                else if (!stall) pipe_valid_sr <= {pipe_valid_sr[1:0], simd_input_valid};
            end
            
            assign simd_result_wire = sum_stage2;
            assign simd_result_valid = pipe_valid_sr[2];

        end else if (SIMD_MODE == 2) begin : gen_simd_xilinx_ip
            // --- Option 2: Xilinx DSP Macro IP ---
            // Instantiation template. Assuming latency 4.
            // Requires 'xbip_dsp48_macro_0' configured for A*B+C accumulation or dot product.
            // Since standard DSP48 handles 18x27, 4x INT8 needs logic or special IP mode.
            // Ideally, this uses the "Dot Product" IP from Xilinx.
            
            wire [31:0] p_out;
            wire p_valid;
            
            // Placeholder: Replace with actual IP instantiation
            // xbip_dsp48_macro_0 u_dsp (
            //    .CLK(clk),
            //    .A(op_a), .B(final_b),
            //    .P(p_out) ...
            // );
            
            // For code correctness without actual .xci, falling back to behavior
            // wrapped in registers to emulate IP latency.
            reg signed [31:0] ip_emu_pipe [0:3];
            reg [3:0] ip_valid_sr;
            integer i;
            
            function signed [31:0] func_dot;
                input [31:0] a, b;
                begin
                    func_dot = ($signed(a[7:0]) * $signed(b[7:0])) + 
                               ($signed(a[15:8]) * $signed(b[15:8])) +
                               ($signed(a[23:16]) * $signed(b[23:16])) +
                               ($signed(a[31:24]) * $signed(b[31:24]));
                end
            endfunction

            always @(posedge clk) begin
                if (!stall) begin
                    ip_emu_pipe[0] <= func_dot(op_a, final_b);
                    for (i=1; i<4; i=i+1) ip_emu_pipe[i] <= ip_emu_pipe[i-1];
                    ip_valid_sr <= {ip_valid_sr[2:0], simd_input_valid};
                end
            end
            
            assign simd_result_wire = ip_emu_pipe[3];
            assign simd_result_valid = ip_valid_sr[3];

        end else begin : gen_simd_basic
            // --- Option 0: Registers inside always (1 cycle) ---
            // Helps timing slightly compared to pure combinational, 
            // but keeps logic compact.
            
            reg signed [31:0] r_dot_res;
            reg r_valid;
            
            always @(posedge clk) begin
                if (simd_input_valid) begin
                    // DSP Inference hints
                    (* use_dsp = USE_DSP *) 
                    r_dot_res <= ($signed(op_a[7:0])   * $signed(final_b[7:0])) +
                                 ($signed(op_a[15:8])  * $signed(final_b[15:8])) +
                                 ($signed(op_a[23:16]) * $signed(final_b[23:16])) +
                                 ($signed(op_a[31:24]) * $signed(final_b[31:24]));
                    r_valid <= 1;
                end else if (!stall) begin
                    r_valid <= 0;
                    r_dot_res <= 0;
                end
            end
            
            assign simd_result_wire = r_dot_res;
            assign simd_result_valid = r_valid;
        end
    endgenerate

    // ========================================================================
    // 6. Helpers
    // ========================================================================
    function signed [31:0] saturate;
        input signed [63:0] val;
        begin
            if (val > MAX_POS) saturate = MAX_POS;
            else if (val < MIN_NEG) saturate = MIN_NEG;
            else saturate = val[31:0];
        end
    endfunction

    function [31:0] simd_relu;
        input [31:0] packed_val;
        begin
            simd_relu = {
                (packed_val[31:24] > 0 ? packed_val[31:24] : 8'd0),
                (packed_val[23:16] > 0 ? packed_val[23:16] : 8'd0),
                (packed_val[15:8]  > 0 ? packed_val[15:8]  : 8'd0),
                (packed_val[7:0]   > 0 ? packed_val[7:0]   : 8'd0)
            };
        end
    endfunction

    // ========================================================================
    // 7. Memory Interface Logic
    // ========================================================================
    assign dma_rd_data = op_b;

    always @(posedge clk) begin
        // Read Port A
        if (!stall) op_a <= tensor_mem[base_src1 + vec_cnt];
        // Read Port B / DMA
        if (!stall) begin
            if (dma_rd_req) op_b <= tensor_mem[base_src1 + dma_wr_offset];
            else op_b <= tensor_mem[base_src2 + vec_cnt];
        end

        // Write Port (Prioritized)
        if (dma_wr_en) begin
            tensor_mem[base_dst + dma_wr_offset] <= dma_wr_data;
        end 
        else if (is_accum_op && done) begin
            // Finalize Reduction
            if (hw_opcode == `H_MEAN)
                tensor_mem[base_dst] <= saturate(accumulator >>> SHIFT_MEAN);
            else
                tensor_mem[base_dst] <= saturate(accumulator);
        end 
        else if (wb_enable && !stall && !is_accum_op) begin
            tensor_mem[wb_addr] <= wb_data;
        end
        else if ((USE_XILINX_DIV == 1) && div_ip_out_valid) begin
            tensor_mem[base_dst + div_out_cnt] <= div_ip_out_data[31:0]; 
        end
    end

    // ========================================================================
    // 8. Main Execution FSM
    // ========================================================================
    
    // LUT Address Calculation (Logic outside always block)
    wire [10:0] lut_idx = op_a[19:9] + 11'd1024;
    reg [2:0] pipeline_flush_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0; vec_cnt <= 0;
            wb_enable <= 0; wb_addr <= 0; wb_data <= 0;
            done <= 0;
            accumulator <= 0; is_accum_op <= 0;
            div_busy_fsm <= 0; div_counter <= 0;
            base_src1 <= 0; base_src2 <= 0; base_dst <= 0;
            div_out_cnt <= 0;
            pipeline_flush_cnt <= 0;
        end else begin
            done <= 0; 
            wb_enable <= 0; // Default off

            // --- START ---
            if (start) begin
                active <= 1;
                vec_cnt <= 0;
                accumulator <= 0;
                div_busy_fsm <= 0;
                div_out_cnt <= 0;
                pipeline_flush_cnt <= 0;
                
                is_accum_op <= (hw_opcode == `H_SUM || hw_opcode == `H_MEAN || 
                                hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL);

                base_src1 <= (src1_id & `SLOT_MASK) * `VECTOR_LEN;
                base_src2 <= (src2_id_bram & `SLOT_MASK) * `VECTOR_LEN;
                base_dst  <= (dst_id  & `SLOT_MASK) * `VECTOR_LEN;
            end 
            
            // --- ACTIVE EXECUTION ---
            else if (active || (pipeline_flush_cnt > 0) || (USE_XILINX_DIV == 1 && hw_opcode == `H_DIV && div_out_cnt < `VECTOR_LEN)) begin
                
                // SIMD ACCUMULATOR UPDATE (Decoupled from main loop)
                if (simd_result_valid) begin
                    accumulator <= accumulator + simd_result_wire;
                end

                // INPUT FEEDING LOGIC
                if (active) begin
                    // Wait for stall (DDR stream or FSM Div)
                    if (!stall) begin
                        // Is this the last input?
                        if (vec_cnt == `VECTOR_LEN) begin
                            active <= 0; // Stop feeding inputs
                            
                            // If we are doing SIMD, we need to wait for the pipeline to flush
                            if (hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL) begin
                                pipeline_flush_cnt <= SIMD_LATENCY;
                            end else begin
                                // Standard Op: Done immediately (latency 0 effectively for logic below)
                                if (USE_XILINX_DIV == 0 || hw_opcode != `H_DIV) done <= 1;
                            end
                        end

                        // Execute Scalar Ops (Latency 0 / Immediate)
                        if (hw_opcode != `H_DIV && hw_opcode != `H_LINEAR && hw_opcode != `H_MATMUL) begin
                            reg signed [63:0] wide_res;
                            
                            // Delayed WB address (since BRAM read took 1 cycle, vec_cnt is ahead)
                            // Ideally, vec_cnt represents address currently being READ.
                            // WB address matches the data currently in op_a/op_b.
                            wb_addr <= base_dst + vec_cnt; 

                            case (hw_opcode)
                                `H_RELU: wb_data <= simd_relu(op_a);
                                `H_ADD: begin wide_res = op_a + final_b; wb_data <= saturate(wide_res); end
                                `H_SUB: begin wide_res = op_a - final_b; wb_data <= saturate(wide_res); end
                                `H_MUL: begin wide_res = op_a * final_b; wb_data <= saturate(wide_res >>> 16); end
                                `H_NEG: wb_data <= -op_a;
                                `H_POW: begin wide_res = op_a * op_a; wb_data <= saturate(wide_res >>> 16); end
                                `H_GELU, `H_TANH, `H_ERF: wb_data <= activation_lut[lut_idx];
                                `H_EQ: wb_data <= (op_a == final_b) ? FP_ONE : 0;
                                `H_NE: wb_data <= (op_a != final_b) ? FP_ONE : 0;
                                `H_GT: wb_data <= (op_a > final_b)  ? FP_ONE : 0;
                                `H_LT: wb_data <= (op_a < final_b)  ? FP_ONE : 0;
                                `H_MASKED_FILL: wb_data <= (op_a != 0) ? final_b : 0;
                                `H_ARANGE: wb_data <= saturate(vec_cnt << 16);
                                `H_FULL, `H_EMBED: wb_data <= final_b;
                                `H_COPY: wb_data <= op_a;
                                `H_SUM, `H_MEAN: begin
                                    accumulator <= accumulator + op_a;
                                    wb_data <= 0; // No WB per cycle
                                    wb_enable <= 0;
                                end
                                default: begin wb_data <= op_a; wb_enable <= 1; end
                            endcase
                            
                            if (hw_opcode != `H_SUM && hw_opcode != `H_MEAN) wb_enable <= 1;
                        end

                        // Division Logic (FSM Mode)
                        if (hw_opcode == `H_DIV && USE_XILINX_DIV == 0) begin
                             // ... (Same FSM Logic as original code) ...
                             if (!div_busy_fsm) begin
                                if (final_b == 0) begin
                                    wb_data <= MAX_POS; wb_enable <= 1; wb_addr <= base_dst + vec_cnt;
                                end else begin
                                    div_sign <= (op_a[31] ^ final_b[31]);
                                    div_dividend <= {32'd0, (op_a < 0 ? -op_a : op_a)} << 16; 
                                    div_divisor  <= (final_b < 0 ? -final_b : final_b);
                                    div_result_fsm <= 0; div_counter <= 8; 
                                    div_busy_fsm <= 1; wb_enable <= 0;
                                end
                             end
                        end
                        
                        // Increment Input Counter
                        if (!div_busy_fsm) vec_cnt <= vec_cnt + 1;
                    end
                end 
                else if (pipeline_flush_cnt > 0) begin
                    // FLUSH STATE: Waiting for SIMD Pipeline to empty
                    pipeline_flush_cnt <= pipeline_flush_cnt - 1;
                    if (pipeline_flush_cnt == 1) begin
                        done <= 1; // Accumulation finished
                    end
                end

                // Division FSM Execution (Parallel to main stall)
                if (hw_opcode == `H_DIV && div_busy_fsm && USE_XILINX_DIV == 0) begin
                    reg [63:0] d_curr, d_next;
                    reg [31:0] q_curr, q_next;
                    reg [32:0] diff; 
                    integer k;
                    d_curr = div_dividend; q_curr = div_result_fsm;
                    for (k = 0; k < 4; k = k + 1) begin
                        d_next = d_curr << 1; q_next = q_curr << 1;
                        diff = {1'b0, d_next[63:32]} - {1'b0, div_divisor};
                        if (diff[32] == 0) begin d_next[63:32] = diff[31:0]; q_next[0] = 1'b1; end
                        d_curr = d_next; q_curr = q_next;
                    end
                    div_dividend <= d_curr; div_result_fsm <= q_curr;

                    if (div_counter == 1) begin
                        wb_data <= div_sign ? -div_result_fsm : div_result_fsm;
                        div_busy_fsm <= 0; wb_enable <= 1; wb_addr <= base_dst + vec_cnt;
                        vec_cnt <= vec_cnt + 1; // Resume vector counter
                    end else begin
                        div_counter <= div_counter - 1;
                    end
                end
                
                // IP Division Result Tracking
                if (USE_XILINX_DIV == 1 && hw_opcode == `H_DIV && div_ip_out_valid) begin
                    div_out_cnt <= div_out_cnt + 1;
                    if (div_out_cnt == `VECTOR_LEN - 1) done <= 1;
                end
            end
        end
    end

endmodule