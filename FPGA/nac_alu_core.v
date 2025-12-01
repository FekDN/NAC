`include "nac_defines.vh"
`include "nac_hw_defines.vh"

// ============================================================================
// USE_ERF_LUT directive
// ============================================================================
// By default, a dedicated ERF LUT file is used for higher accuracy.
// To disable this and fall back to a Tanh-based approximation (saving one BRAM),
// comment out the following line or manage it via synthesis tool properties.
`define USE_ERF_LUT

module NAC_ALU_Core #(
    parameter RAM_STYLE      = "block", // "block" (BRAM) or "distributed" (LUTRAM)
    parameter USE_DSP        = "yes",   // Force usage of DSP48 slices for Math
    parameter USE_XILINX_DIV = 0,       // 0 = FSM (Slow/Small), 1 = Xilinx IP (Fast/Large)
    
    // NEW OPTION: SIMD Implementation Strategy
    // 0 = Registers inside always (1 cycle latency, compact code, lower Fmax)
    // 1 = Manual Pipeline (3 cycle latency, best for Fmax > 100MHz, pure RTL)
    // 2 = Xilinx IP Macro (Instantiation of xbip_dsp48_macro, requires .xci)
    parameter SIMD_MODE      = 1,
    parameter MAX_PARAMS     = 8        // Max parameters for metadata operations like VIEW, PERMUTE
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Command Interface (From Main FSM)
    // ========================================================================
    input  wire        start,          
    input  wire [5:0]  hw_opcode,      // Expanded to 6 bits for new opcodes
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
    output wire [31:0] dma_rd_data,

    // ========================================================================
    // Tensor Metadata Interface (NEW)
    // ========================================================================
    input wire [31:0] src1_shape [0:`MAX_DIMS-1],
    input wire [31:0] src1_strides [0:`MAX_DIMS-1],
    input wire [3:0]  src1_dims,
    input wire [31:0] src2_shape [0:`MAX_DIMS-1],
    input wire [31:0] src2_strides [0:`MAX_DIMS-1],
    input wire [3:0]  src2_dims,
    input wire [31:0] meta_param_buf [0:MAX_PARAMS-1], // Buffer for metadata op parameters
    output reg [31:0] dst_shape_out [0:`MAX_DIMS-1],
    output reg [31:0] dst_strides_out [0:`MAX_DIMS-1],
    output reg [3:0]  dst_dims_out
);

    // ========================================================================
    // 1. Constants, Memory, and Helper Functions
    // ========================================================================
    localparam signed [31:0] MAX_POS = 32'h7FFFFFFF;
    localparam signed [31:0] MIN_NEG = 32'h80000000;
    localparam signed [31:0] FP_ONE  = `FIXED_ONE; // 65536
    localparam SHIFT_MEAN = $clog2(`VECTOR_LEN);

    // Latency determination based on SIMD_MODE
    localparam integer SIMD_LATENCY = (SIMD_MODE == 2) ? 4 : // IP usually has 3-4 cycles
                                      (SIMD_MODE == 1) ? 3 : // Manual pipeline
                                      1;                     // Registered always block

    // BRAM for tensor data storage.
    // Note: BRAM content is undefined on power-up. Zeroing out (clearing)
    // must be handled by an H_FULL operation if required by the compute graph.
    (* ram_style = RAM_STYLE *) 
    reg signed [31:0] tensor_mem [0:(`TENSOR_SLOTS * `VECTOR_LEN) - 1];

    // Saturation function to clamp values within 32-bit signed range
    function signed [31:0] saturate;
        input signed [63:0] val;
        begin
            if (val > MAX_POS) saturate = MAX_POS;
            else if (val < MIN_NEG) saturate = MIN_NEG;
            else saturate = val[31:0];
        end
    endfunction

    // SIMD ReLU function for packed 8-bit integers
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
    // 2. Activation LUTs
    // ========================================================================
    // Separate LUTs for each activation function. 
    // This allows for better resource management and potential for power savings.
    (* ram_style = "block" *) reg [31:0] gelu_lut [0:2047];
    (* ram_style = "block" *) reg [31:0] tanh_lut [0:2047];

    `ifdef USE_ERF_LUT
        // If USE_ERF_LUT is defined, instantiate a dedicated BRAM for ERF.
        (* ram_style = "block" *) reg [31:0] erf_lut [0:2047];
    `endif

    initial begin
        // Load memory files into the respective LUTs.
        $readmemh("nac_gelu_lut.mem", gelu_lut);
        $readmemh("nac_tanh_lut.mem", tanh_lut);
        
        // Conditionally load the ERF LUT file.
        `ifdef USE_ERF_LUT
            $readmemh("nac_erf_lut.mem", erf_lut);
        `endif
    end

    // ========================================================================
    // 3. State Registers, Pipeline, and Stall Logic
    // ========================================================================
    reg [15:0] base_src1, base_src2, base_dst;
    reg [15:0] vec_cnt;       
    reg        active;        
    reg signed [31:0] op_a, op_b;   
    
    // Writeback Stage
    reg [31:0] wb_data;       
    reg        wb_enable;     
    reg [15:0] wb_addr;       
    
    // Accumulators
    reg signed [63:0] accumulator;
    reg signed [63:0] sum_sq_accumulator; // For variance calculation
    reg        is_accum_op;
    
    // FSM for complex multi-pass operations
    reg [3:0] compute_state;
    localparam CS_IDLE             = 4'd0; // State for idle or simple element-wise ops
    localparam CS_BUSY             = 4'd1; // Generic start state for a complex op
    // LayerNorm States
    localparam CS_LN_PASS1_STATS   = 4'd2; // Pass 1: Calculate sum and sum of squares
    localparam CS_LN_CALC_INV_STD  = 4'd3; // Calculate 1 / sqrt(variance + epsilon)
    localparam CS_LN_PASS2_NORM    = 4'd4; // Pass 2: Normalize the input vector
    // Softmax States (placeholder)
    localparam CS_SM_FIND_MAX      = 4'd5;
    localparam CS_SM_SUM_EXP       = 4'd6;
    localparam CS_SM_CALC_INV      = 4'd7;
    localparam CS_SM_DIVIDE        = 4'd8;
    // Pooling and Conv states
    localparam CS_POOL_BUSY        = 4'd9;
    localparam CS_CONV_BUSY        = 4'd10;
    
    // Temporary registers for multi-step calculations
    reg signed [31:0] temp_reg1; // Used for mean / max_val
    reg signed [31:0] temp_reg2; // Used for inv_stddev / inv_sum_exp

    // Stall Logic
    reg stall_complex_op;
    always @(*) begin
        stall_complex_op = (compute_state inside {CS_LN_CALC_INV_STD, CS_SM_CALC_INV});
    end
    
    wire stall_div_fsm = (hw_opcode == `H_DIV) && div_busy_fsm && (USE_XILINX_DIV == 0);
    wire stall_div_ip  = (hw_opcode == `H_DIV) && !div_ip_in_ready && (USE_XILINX_DIV == 1);
    wire stall = (active && use_stream && !stream_valid) || stall_div_fsm || stall_div_ip || stall_complex_op;
    assign stream_ready = active && use_stream && !stall;

    // Operand B source selection
    reg signed [31:0] final_b;
    always @(*) begin
        final_b = use_stream ? $signed(stream_data) : op_b;
    end
    
    // --- LUT Result Selection Logic with Fallback ---
    // Calculate the index based on the input operand 'a'.
    // op_a is Q16.16, we use the upper bits for indexing.
    wire [10:0] lut_idx = op_a[19:9] + 11'd1024;
    reg signed [31:0] lut_result;

    always @(*) begin
        // Select the result from the appropriate LUT based on the current operation.
        case (hw_opcode)
            `H_GELU: lut_result = gelu_lut[lut_idx];
            `H_TANH: lut_result = tanh_lut[lut_idx];
            
            `H_ERF: begin
                // Conditional logic for ERF.
                `ifdef USE_ERF_LUT
                    // Use the dedicated, high-accuracy ERF LUT if available.
                    lut_result = erf_lut[lut_idx];
                `else
                    // Fallback: Approximate ERF using the Tanh LUT.
                    // The approximation is: erf(x) ≈ tanh(1.128 * x).
                    // This requires scaling the input index before the Tanh LUT lookup.
                    // For Q16.16 fixed-point, 1.128 is approximately 74000.
                    // We calculate a new index for the Tanh LUT: new_idx = (x * 1.128)
                    reg signed [31:0] scaled_op_a;
                    reg [10:0] erf_approx_idx;
                    
                    // Scale the input operand by the approximation constant.
                    scaled_op_a = (op_a * 32'd74000) >>> 16;
                    // Calculate the new index from the scaled value.
                    erf_approx_idx = scaled_op_a[19:9] + 11'd1024;
                    lut_result = tanh_lut[erf_approx_idx];
                `endif
            end
            
            default: lut_result = 32'd0; // Default to zero for non-LUT operations.
        endcase
    end

    // ========================================================================
    // 4. Hardware Sub-modules (Division, SIMD, InvSqrt, Exp)
    // ========================================================================
    // --- Division Logic ---
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
    
    generate
        if (USE_XILINX_DIV == 1) begin : gen_div_ip
             div_gen_0 u_div_ip (
                .aclk(clk),
                .s_axis_divisor_tvalid(active && (hw_opcode == `H_DIV) && !stall && compute_state == CS_IDLE),
                .s_axis_divisor_tready(div_ip_in_ready),
                .s_axis_divisor_tdata(final_b),
                .s_axis_dividend_tvalid(active && (hw_opcode == `H_DIV) && !stall && compute_state == CS_IDLE),
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

    // --- SIMD / Dot Product Logic ---
    wire signed [31:0] simd_result_wire;
    wire               simd_result_valid;
    wire simd_input_valid = active && !stall && (hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL) && compute_state == CS_IDLE;
    
    generate
        // Manual 3-stage pipeline for best performance in pure RTL.
        if (SIMD_MODE == 1) begin : gen_simd_manual_pipeline
            reg signed [16:0] p0, p1, p2, p3;    // Multiplier results
            reg signed [31:0] sum_stage1;        // First adder stage
            reg signed [31:0] sum_stage2;        // Second adder stage (output)
            reg [2:0] pipe_valid_sr;             // Validity shift register

            always @(posedge clk) begin
                if (simd_input_valid) begin
                    // Stage 1: Parallel multiplications
                    p0 <= $signed(op_a[7:0])   * $signed(final_b[7:0]);
                    p1 <= $signed(op_a[15:8])  * $signed(final_b[15:8]);
                    p2 <= $signed(op_a[23:16]) * $signed(final_b[23:16]);
                    p3 <= $signed(op_a[31:24]) * $signed(final_b[31:24]);
                end else if (!stall) begin
                    p0 <= 0; p1 <= 0; p2 <= 0; p3 <= 0;
                end
                // Stage 2: Adder tree part 1
                sum_stage1 <= p0 + p1 + p2 + p3;
                // Stage 3: Final register stage
                sum_stage2 <= sum_stage1;

                if (!rst_n) pipe_valid_sr <= 0;
                else if (!stall) pipe_valid_sr <= {pipe_valid_sr[1:0], simd_input_valid};
            end
            assign simd_result_wire = sum_stage2;
            assign simd_result_valid = pipe_valid_sr[2];
            
        // Xilinx IP instantiation (requires a .xci file for xbip_dsp48_macro)
        end else if (SIMD_MODE == 2) begin : gen_simd_xilinx_ip
            // For synthesis, this would be an instantiation of the DSP48 macro.
            // For simulation, we emulate its behavior and latency.
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
            
        // Basic single-cycle implementation.
        end else begin : gen_simd_basic
            reg signed [31:0] r_dot_res;
            reg r_valid;
            always @(posedge clk) begin
                if (simd_input_valid) begin
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

    // --- Fast Inverse Square Root (for LayerNorm) ---
    reg [3:0] inv_sqrt_cnt;
    always @(posedge clk) begin
        if (!rst_n) begin
            inv_sqrt_cnt <= 0;
        end else begin
            if (compute_state == CS_LN_CALC_INV_STD) begin
                if (inv_sqrt_cnt == 0) begin
                    // First approximation using the "magic number" trick for floating point numbers,
                    // adapted for fixed-point representation.
                    temp_reg2 <= 32'h5f3759df - (temp_reg1 >>> 1); 
                end else if (inv_sqrt_cnt < 4) begin // 2 Newton-Raphson iterations for refinement
                    reg signed [63:0] term1, term2;
                    // Formula: y_new = y * (1.5 - (0.5 * x * y^2))
                    // Where x is the input variance (temp_reg1) and y is the current approximation (temp_reg2).
                    // All calculations are done in Q16.16 fixed-point arithmetic.
                    term1 = $signed(temp_reg2) * $signed(temp_reg2); // y^2 (Q32.32)
                    term2 = $signed(temp_reg1) * term1;              // x*y^2 (Q48.48)
                    // (1.5 - (term2 >>> 33)) = (0x18000 - (0.5*x*y^2)). Result is Q16.16
                    temp_reg2 <= ($signed(temp_reg2) * (32'h18000 - (term2 >>> 33))) >>> 16;
                end
                inv_sqrt_cnt <= inv_sqrt_cnt + 1;
            end else begin
                inv_sqrt_cnt <= 0;
            end
        end
    end
    
    // --- Exp Approximation (for Softmax) ---
    // Using a simple polynomial for the range [-10, 0] in Q16.16
    function signed [31:0] exp_approx;
        input signed [31:0] x; // x in Q16.16 format, assumed to be <= 0
        reg signed [63:0] x2, x3;
        begin
            // Taylor series approximation: exp(x) ≈ 1 + x + x^2/2! + x^3/6!
            x2 = (x * x) >>> 16;
            x3 = (x2 * x) >>> 16;
            exp_approx = FP_ONE + x + (x2 >>> 1) + (x3 / 6);
        end
    endfunction

    // ========================================================================
    // 5. Memory Interface & AGU (Address Generation Unit)
    // ========================================================================
    assign dma_rd_data = op_b;
    
    // Read address registers. These are assigned conditionally in the main FSM.
    reg  [15:0] rd_addr_a;
    wire [15:0] rd_addr_b = base_src2 + vec_cnt;
    
    // Iterators for multi-dimensional operations
    reg [15:0] iter_n, iter_c, iter_h, iter_w;
    reg [15:0] iter_kh, iter_kw;
    
    // Combinatorial Address Generation for MaxPool2D
    // Calculates the source read address based on the current position in the NCHW loops
    wire [15:0] pool_read_addr;
    assign pool_read_addr = base_src1 + 
                         iter_n * src1_strides[0] +
                         iter_c * src1_strides[1] +
                         (iter_h * 2 + iter_kh) * src1_strides[2] + // iter_h*2 = vertical stride
                         (iter_w * 2 + iter_kw) * src1_strides[3];  // iter_w*2 = horizontal stride

    always @(posedge clk) begin
        // Read data for the pipeline
        if (!stall) begin
            // rd_addr_a is assigned in the main FSM based on the operation
            op_a <= tensor_mem[rd_addr_a];
            if (dma_rd_req) begin
                op_b <= tensor_mem[base_dst + dma_wr_offset];
            end else begin
                op_b <= tensor_mem[rd_addr_b];
            end
        end

        // Prioritized write port
        if (dma_wr_en) begin
            tensor_mem[base_dst + dma_wr_offset] <= dma_wr_data;
        end 
        else if (is_accum_op && done) begin
            // Finalize reduction operations on 'done' signal
            if (hw_opcode == `H_MEAN || hw_opcode == `H_ADAPTIVE_AVG_POOL2D) begin
                tensor_mem[base_dst] <= saturate(accumulator >>> SHIFT_MEAN);
            end else begin
                tensor_mem[base_dst] <= saturate(accumulator);
            end
        end 
        else if (wb_enable && !stall) begin
            tensor_mem[wb_addr] <= wb_data;
        end
        else if ((USE_XILINX_DIV == 1) && div_ip_out_valid) begin
            tensor_mem[base_dst + div_out_cnt] <= div_ip_out_data[31:0]; 
        end
    end

    // ========================================================================
    // 6. Main Execution FSM
    // ========================================================================
    
    reg [2:0] pipeline_flush_cnt;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0;
            vec_cnt <= 0;
            wb_enable <= 0;
            wb_addr <= 0;
            wb_data <= 0;
            done <= 0;
            accumulator <= 0;
            sum_sq_accumulator <= 0;
            is_accum_op <= 0;
            div_busy_fsm <= 0;
            div_counter <= 0;
            base_src1 <= 0;
            base_src2 <= 0;
            base_dst <= 0;
            div_out_cnt <= 0;
            pipeline_flush_cnt <= 0;
            compute_state <= CS_IDLE;
            temp_reg1 <= 0;
            temp_reg2 <= 0;
            iter_n <= 0; iter_c <= 0; iter_h <= 0; iter_w <= 0;
            iter_kh <= 0; iter_kw <= 0;
            dst_dims_out <= 0;
            rd_addr_a <= 0;
            for (integer i=0; i<`MAX_DIMS; i=i+1) begin
                dst_shape_out[i] <= 0;
                dst_strides_out[i] <= 0;
            end
        end else begin
            done <= 0; 
            wb_enable <= 0; // Default off, enabled explicitly

            // --- FSM START ---
            if (start) begin
                active <= 1;
                vec_cnt <= 0;
                accumulator <= 0;
                sum_sq_accumulator <= 0;
                div_busy_fsm <= 0;
                div_out_cnt <= 0;
                pipeline_flush_cnt <= 0;
                
                is_accum_op <= (hw_opcode inside {`H_SUM, `H_MEAN, `H_LINEAR, `H_MATMUL, `H_ADAPTIVE_AVG_POOL2D});

                // Calculate base addresses from tensor slot IDs
                base_src1 <= (src1_id & `SLOT_MASK) * `VECTOR_LEN;
                base_src2 <= (src2_id_bram & `SLOT_MASK) * `VECTOR_LEN;
                base_dst  <= (dst_id  & `SLOT_MASK) * `VECTOR_LEN;

                // --- IMMEDIATE EXECUTION FOR METADATA OPS ---
                if (hw_opcode >= `H_VIEW && hw_opcode <= `H_UNSQUEEZE) begin
                    active <= 0; // These ops don't process data
                    done <= 1;   // They are done in one cycle
                    integer i;
                    
                    // Default behavior: copy metadata from source to destination.
                    // Specific operations will override parts of this.
                    dst_dims_out <= src1_dims;
                    dst_shape_out <= src1_shape;
                    dst_strides_out <= src1_strides;

                    case (hw_opcode)
                        `H_VIEW: begin
                            reg [3:0] new_dims;
                            reg [3:0] inferred_dim_idx;
                            reg [31:0] total_elements, product_of_new_dims;
                            reg [31:0] temp_shape_out [0:`MAX_DIMS-1];
                            
                            // 1. Calculate the total number of elements in the source tensor.
                            total_elements = 1;
                            for (i = 0; i < src1_dims; i = i + 1) begin
                                total_elements = total_elements * src1_shape[i];
                            end
                            
                            // 2. Read the new shape from the parameter buffer and find the inferred dimension (-1).
                            new_dims = 0;
                            product_of_new_dims = 1;
                            inferred_dim_idx = `MAX_DIMS; // Use an invalid index as a flag
                            for (i = 0; i < `MAX_DIMS; i = i + 1) begin
                                if (meta_param_buf[i] == 32'hFFFFFFFF) begin // -1 signifies inferred dimension
                                    inferred_dim_idx = i;
                                    temp_shape_out[i] = 0; // Placeholder
                                end else if (meta_param_buf[i] != 0) begin // 0 marks the end of shape list
                                    temp_shape_out[i] = meta_param_buf[i];
                                    product_of_new_dims = product_of_new_dims * meta_param_buf[i];
                                    new_dims = new_dims + 1;
                                end else begin
                                    temp_shape_out[i] = 0; // Zero out unused dimension slots
                                end
                            end
                            
                            // 3. Calculate the value for the inferred dimension.
                            if (inferred_dim_idx != `MAX_DIMS) begin
                                temp_shape_out[inferred_dim_idx] = total_elements / product_of_new_dims;
                                new_dims = new_dims + 1;
                            end
                            dst_shape_out <= temp_shape_out;
                            dst_dims_out <= new_dims;
                            
                            // 4. Recalculate strides for the new contiguous tensor shape.
                            dst_strides_out[new_dims-1] <= 1;
                            for (i = new_dims - 2; i >= 0; i = i - 1) begin
                                dst_strides_out[i] <= dst_strides_out[i+1] * temp_shape_out[i+1];
                            end
                        end
                        `H_TRANSPOSE: begin
                            // Parameters dim0, dim1 are read from meta_param_buf
                            reg [31:0] dim0, dim1, temp_s;
                            dim0 = meta_param_buf[0];
                            dim1 = meta_param_buf[1];
                            
                            if (dim0 < src1_dims && dim1 < src1_dims) begin
                                // Swap shapes
                                temp_s = dst_shape_out[dim0];
                                dst_shape_out[dim0] <= dst_shape_out[dim1];
                                dst_shape_out[dim1] <= temp_s;
                                
                                // Swap strides
                                temp_s = dst_strides_out[dim0];
                                dst_strides_out[dim0] <= dst_strides_out[dim1];
                                dst_strides_out[dim1] <= temp_s;
                            end
                        end
                        `H_PERMUTE: begin
                            // Parameter is a list of new dimension order, e.g., (0, 2, 1)
                            reg [31:0] new_shape [0:`MAX_DIMS-1];
                            reg [31:0] new_strides [0:`MAX_DIMS-1];
                            
                            // Create temporary new shape and strides based on the order in meta_param_buf
                            for (i = 0; i < src1_dims; i = i + 1) begin
                                new_shape[i] = src1_shape[meta_param_buf[i]];
                                new_strides[i] = src1_strides[meta_param_buf[i]];
                            end
                            
                            // Write back the reordered metadata
                            dst_shape_out <= new_shape;
                            dst_strides_out <= new_strides;
                        end
                        `H_UNSQUEEZE: begin
                            // Parameter 'dim' (where to insert the new axis) comes from meta_param_buf[0]
                            reg [31:0] dim_to_unsqueeze;
                            dim_to_unsqueeze = meta_param_buf[0];
                            
                            // 1. Increment the number of dimensions
                            dst_dims_out <= src1_dims + 1;
                            
                            // 2. Shift all shapes and strides from 'dim' onwards to the right
                            for (i = src1_dims; i > dim_to_unsqueeze; i = i - 1) begin
                                dst_shape_out[i] <= src1_shape[i-1];
                                dst_strides_out[i] <= src1_strides[i-1];
                            end

                            // 3. Copy shape/strides before the insertion point
                            for (i=0; i < dim_to_unsqueeze; i=i+1) begin
                                 dst_shape_out[i] <= src1_shape[i];
                                 dst_strides_out[i] <= src1_strides[i];
                            end
                            
                            // 4. Insert the new axis of size 1
                            dst_shape_out[dim_to_unsqueeze] <= 1;
                            
                            // 5. Calculate the new stride for this axis
                            if (dim_to_unsqueeze == src1_dims) begin
                                // If adding to the end, stride is 1 (for contiguous)
                                dst_strides_out[dim_to_unsqueeze] <= 1;
                            end else begin
                                // New stride is old stride at that position multiplied by the shape
                                // to make space for the new dimension.
                                dst_strides_out[dim_to_unsqueeze] <= src1_strides[dim_to_unsqueeze-1];
                            end
                        end
                    endcase
                end
                // --- INITIALIZE FSM FOR COMPLEX COMPUTE OPS ---
                else if (hw_opcode inside {`H_LAYER_NORM, `H_BATCH_NORM, `H_SOFTMAX, `H_MAX_POOL2D, `H_CONV2D}) begin
                    compute_state <= CS_BUSY;
                end 
                // --- DEFAULT TO IDLE FOR SIMPLE OPS ---
                else begin
                    compute_state <= CS_IDLE;
                end
            end 
            
            // --- ACTIVE EXECUTION STATE ---
            else if (active) begin
                
                if (!stall) begin
                    // ----------------------------------------------------
                    // --- FSM for Complex Multi-Pass Operations ---
                    // ----------------------------------------------------
                    if (compute_state != CS_IDLE) begin
                        wb_enable <= 0; // Writes are explicitly controlled inside states
                        case(hw_opcode)
                            `H_LAYER_NORM, `H_BATCH_NORM: begin
                                // Assign read address for both passes
                                rd_addr_a <= base_src1 + vec_cnt;
                                case (compute_state)
                                    CS_BUSY: begin 
                                        vec_cnt <= 0; 
                                        accumulator <= 0; 
                                        sum_sq_accumulator <= 0; 
                                        compute_state <= CS_LN_PASS1_STATS; 
                                    end
                                    CS_LN_PASS1_STATS: begin
                                        // Pass 1: Calculate sum and sum of squares
                                        accumulator <= accumulator + op_a;
                                        // op_a is Q16.16, so op_a*op_a is Q32.32. Shift to get back to Q47.16
                                        sum_sq_accumulator <= sum_sq_accumulator + ((op_a * op_a) >>> 16);
                                        
                                        if (vec_cnt == `VECTOR_LEN - 1) begin
                                            compute_state <= CS_LN_CALC_INV_STD;
                                        end
                                        vec_cnt <= vec_cnt + 1;
                                    end
                                    CS_LN_CALC_INV_STD: begin
                                        // This state calculates mean and 1/stddev. It stalls the pipeline.
                                        reg signed [63:0] var;
                                        // E[x^2] - (E[x])^2
                                        temp_reg1 <= saturate(accumulator >>> SHIFT_MEAN); // mean = E[x]
                                        var = (sum_sq_accumulator >>> SHIFT_MEAN) - ((temp_reg1*temp_reg1) >>> 16);
                                        temp_reg1 <= var + 1; // variance + epsilon (epsilon=1 in Q16.16 is tiny)
                                        // Wait for InvSqrt to finish
                                        if(inv_sqrt_cnt == 4) begin 
                                            vec_cnt <= 0; 
                                            compute_state <= CS_LN_PASS2_NORM; 
                                        end
                                    end
                                    CS_LN_PASS2_NORM: begin
                                        // Pass 2: (x - mean) * inv_stddev * gamma + beta
                                        // stream_data provides gamma and beta packed or sequentially
                                        // Assuming stream_data is gamma for now, beta is next
                                        reg signed [63:0] norm_val_wide;
                                        norm_val_wide = ($signed(op_a) - $signed(temp_reg1)) * $signed(temp_reg2);
                                        wb_data <= saturate((norm_val_wide >>> 16)); // Apply gamma/beta in a subsequent op if needed
                                        wb_addr <= base_dst + vec_cnt; 
                                        wb_enable <= 1;
                                        
                                        if (vec_cnt == `VECTOR_LEN - 1) begin 
                                            active <= 0; 
                                            done <= 1; 
                                            compute_state <= CS_IDLE; 
                                        end
                                        vec_cnt <= vec_cnt + 1;
                                    end
                                endcase
                            end
                            `H_MAX_POOL2D: begin
                                // --- 2D Max Pooling FSM (Kernel=2x2, Stride=2) ---
                                // Assumes NCHW format: src1_shape[0]=N, [1]=C, [2]=H, [3]=W
                                case(compute_state)
                                    CS_BUSY: begin // Initialization
                                        iter_n <= 0; iter_c <= 0; iter_h <= 0; iter_w <= 0;
                                        iter_kh <= 0; iter_kw <= 0;
                                        temp_reg1 <= MIN_NEG; // temp_reg1 holds the max value in the current window
                                        wb_enable <= 0;
                                        compute_state <= CS_POOL_BUSY;
                                    end
                                    
                                    CS_POOL_BUSY: begin
                                        // Set the read address for the current element in the 2x2 window
                                        rd_addr_a <= pool_read_addr;
                                        
                                        // Compare the incoming data with the current maximum
                                        if (op_a > temp_reg1) begin
                                            temp_reg1 <= op_a;
                                        end
                                        
                                        // --- Loop Control for the 2x2 Kernel ---
                                        if (iter_kw == 1) begin
                                            iter_kw <= 0;
                                            if (iter_kh == 1) begin
                                                // End of a 2x2 window. Write the max value.
                                                iter_kh <= 0;
                                                wb_data <= temp_reg1;
                                                // Calculate write address for the output tensor
                                                wb_addr <= base_dst + 
                                                           iter_n * (src1_shape[1]*src1_shape[2]*src1_shape[3]/4) +
                                                           iter_c * (src1_shape[2]*src1_shape[3]/4) +
                                                           iter_h * (src1_shape[3]/2) +
                                                           iter_w;
                                                wb_enable <= 1;
                                                temp_reg1 <= MIN_NEG; // Reset max for next window
                                                
                                                // --- Loop Control for the Output Tensor Dimensions ---
                                                if (iter_w == (src1_shape[3] / 2) - 1) begin
                                                    iter_w <= 0;
                                                    if (iter_h == (src1_shape[2] / 2) - 1) begin
                                                        iter_h <= 0;
                                                        if (iter_c == src1_shape[1] - 1) begin
                                                            iter_c <= 0;
                                                            if (iter_n == src1_shape[0] - 1) begin
                                                                // --- ENTIRE OPERATION COMPLETE ---
                                                                active <= 0;
                                                                done <= 1;
                                                                compute_state <= CS_IDLE;
                                                            end else iter_n <= iter_n + 1;
                                                        end else iter_c <= iter_c + 1;
                                                    end else iter_h <= iter_h + 1;
                                                end else iter_w <= iter_w + 1;
                                                
                                            end else iter_kh <= iter_kh + 1;
                                        end else iter_kw <= iter_kw + 1;
                                    end
                                endcase
                            end
                            // ... other complex FSMs would go here
                        endcase
                    end
                    // ----------------------------------------------------
                    // --- Standard Element-wise and Simple Operations ---
                    // ----------------------------------------------------
                    else if (compute_state == CS_IDLE) begin
                        // This block contains the original FSM logic for simple ops
                        reg signed [63:0] wide_res;
                        
                        // Set standard linear read address for operand A
                        rd_addr_a <= base_src1 + vec_cnt;
                        
                        // Increment vector counter
                        vec_cnt <= vec_cnt + 1;

                        // Check for end of vector processing
                        if (vec_cnt == `VECTOR_LEN - 1) begin
                             // For standard ops, we are done at the end of the vector.
                             // For ops with pipeline latency (SIMD), we need to flush.
                            if (hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL) begin
                                active <= 0; // Stop feeding inputs
                                pipeline_flush_cnt <= SIMD_LATENCY;
                            end else begin
                                active <= 0;
                                if (USE_XILINX_DIV == 0 || hw_opcode != `H_DIV) done <= 1;
                            end
                        end

                        // Latched write address for next cycle
                        wb_addr <= base_dst + vec_cnt; 
                        
                        // Set default enable, can be overridden below
                        wb_enable <= 1;

                        case (hw_opcode)
                            `H_RELU: wb_data <= simd_relu(op_a);
                            `H_ADD: begin wide_res = op_a + final_b; wb_data <= saturate(wide_res); end
                            `H_SUB: begin wide_res = op_a - final_b; wb_data <= saturate(wide_res); end
                            `H_MUL: begin wide_res = op_a * final_b; wb_data <= saturate(wide_res >>> 16); end
                            `H_NEG: wb_data <= -op_a;
                            `H_POW: begin wide_res = op_a * op_a; wb_data <= saturate(wide_res >>> 16); end
                            `H_GELU, `H_TANH, `H_ERF: wb_data <= lut_result;
                            `H_EQ: wb_data <= (op_a == final_b) ? FP_ONE : 0;
                            `H_NE: wb_data <= (op_a != final_b) ? FP_ONE : 0;
                            `H_GT: wb_data <= (op_a > final_b)  ? FP_ONE : 0;
                            `H_LT: wb_data <= (op_a < final_b)  ? FP_ONE : 0;
                            `H_MASKED_FILL: wb_data <= (op_a != 0) ? final_b : 0;
                            `H_ARANGE: wb_data <= saturate({vec_cnt, 16'd0});
                            `H_FULL, `H_EMBED: wb_data <= final_b;
                            `H_COPY: wb_data <= op_a;
                            `H_SUM, `H_MEAN: begin
                                accumulator <= accumulator + op_a;
                                wb_enable <= 0; // No writeback per cycle
                            end
                            `H_LINEAR, `H_MATMUL: begin
                                if(simd_result_valid) accumulator <= accumulator + simd_result_wire;
                                wb_enable <= 0; // No writeback per cycle for SIMD ops.
                            end
                            `H_DIV: begin
                                if (USE_XILINX_DIV == 0) begin
                                    // FSM based division
                                    if (!div_busy_fsm) begin
                                       if (final_b == 0) begin
                                           wb_data <= MAX_POS; 
                                       end else begin
                                           div_sign <= (op_a[31] ^ final_b[31]);
                                           div_dividend <= {32'd0, (op_a < 0 ? -op_a : op_a)} << 16; 
                                           div_divisor  <= (final_b < 0 ? -final_b : final_b);
                                           div_result_fsm <= 0; div_counter <= 8; 
                                           div_busy_fsm <= 1; wb_enable <= 0;
                                       end
                                    end
                                end else begin
                                    wb_enable <= 0; // IP handles writes
                                end
                            end
                            default: wb_data <= op_a;
                        endcase
                    end
                end // !stall
            end // active
            else if (pipeline_flush_cnt > 0) begin
                // FLUSH STATE: Waiting for SIMD Pipeline to empty
                pipeline_flush_cnt <= pipeline_flush_cnt - 1;
                if (simd_result_valid) accumulator <= accumulator + simd_result_wire;
                if (pipeline_flush_cnt == 1) begin
                    done <= 1; // Accumulation finished, result is ready.
                end
            end

            // --- FSM Division Execution (runs in parallel to main logic when busy) ---
            if (hw_opcode == `H_DIV && div_busy_fsm && USE_XILINX_DIV == 0) begin
                // Non-restoring division algorithm implementation.
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
            
            // --- IP Division Result Tracking ---
            if (USE_XILINX_DIV == 1 && hw_opcode == `H_DIV && div_ip_out_valid) begin
                div_out_cnt <= div_out_cnt + 1;
                if (div_out_cnt == `VECTOR_LEN - 1) done <= 1;
            end
        end // else: !rst_n
    end

endmodule