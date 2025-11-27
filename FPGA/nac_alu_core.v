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
    input  wire        use_stream,      // 1 = Input B comes from DDR Stream
    input  wire [31:0] stream_data,     // The weight/constant data payload
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
    
    // Q16.16 One
    localparam signed [31:0] FP_ONE  = `FIXED_ONE;

    // Automatic bit-shift calculation for Mean (Avg) operation
    // e.g., if VECTOR_LEN is 512, SHIFT_MEAN = 9.
    localparam SHIFT_MEAN = $clog2(`VECTOR_LEN);

    // ========================================================================
    // 1. Internal Memory (Tensor Activations)
    // ========================================================================
    // True Dual-Port RAM.
    (* ram_style = RAM_STYLE *) 
    reg signed [31:0] tensor_mem [0:(`TENSOR_SLOTS * `VECTOR_LEN) - 1];

    // ========================================================================
    // 2. Pipeline State Registers
    // ========================================================================
    // Base addresses calculated from Ring Buffer Slots
    reg [15:0] base_src1; 
    reg [15:0] base_src2;
    reg [15:0] base_dst;
    
    // Execution Control
    reg [15:0] vec_cnt;       // Current Element Index (0..511)
    reg        active;        // 1 = Pipeline Running
    
    // Pipeline Stage Registers
    reg signed [31:0] op_a;   // Operand A
    reg signed [31:0] op_b;   // Operand B
    reg signed [31:0] wb_data; // Result to Write Back
    reg        wb_enable;     // Write Enable Flag
    reg [15:0] wb_addr;       // Write Address
    
    // Accumulator for Reductions (64-bit to prevent overflow)
    reg signed [63:0] accumulator;
    reg        is_accum_op;   // 1 = Operation is reduction (Sum/Mean/Linear)

    // ========================================================================
    // 2.5 Division State Machine Registers
    // ========================================================================
    reg        div_busy;      // Flag: Division FSM is running
    reg [5:0]  div_counter;   // Iteration counter (32 downto 0)
    reg        div_sign;      // Result sign bit
    reg [63:0] div_dividend;  // Working register for dividend/remainder
    reg [31:0] div_divisor;   // Absolute value of divisor
    reg [31:0] div_result;    // Result quotient

    // ========================================================================
    // 3. Flow Control Logic
    // ========================================================================
    
    // Detect if we need to stall for division
    // Note: We only check `div_busy` here. The logic to START division happens in the main FSM.
    wire stall_div = (hw_opcode == `H_DIV) && div_busy;

    // STALL CONDITION: 
    // 1. Active AND Streaming AND Data invalid
    // 2. Active AND Division is busy
    wire stall;
    assign stall = (active && use_stream && !stream_valid) || stall_div;
    
    // STREAM READY:
    // We are ready if active, need stream, AND NOT busy with division logic
    assign stream_ready = (active && use_stream && !div_busy);

    // ========================================================================
    // 4. Math Helper Functions (Behavioral)
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

    // --- GELU Approximation (Q16.16) ---
    function signed [31:0] fx_gelu;
        input signed [31:0] x;
        begin
            if (x > 32'd262144) fx_gelu = x; // x > 4.0
            else if (x < -32'd262144) fx_gelu = 0; // x < -4.0
            else begin
                // Simple Piecewise Linear Approx
                if (x > 0) fx_gelu = (x >>> 1) + (x >>> 2); // ~0.75x
                else fx_gelu = 0; 
            end
        end
    endfunction

    // --- Tanh Approximation (Q16.16) ---
    function signed [31:0] fx_tanh;
        input signed [31:0] x;
        begin
            if (x > 32'd131072) fx_tanh = FP_ONE; // > 2.0
            else if (x < -32'd131072) fx_tanh = -FP_ONE; // < -2.0
            else fx_tanh = x; // Linear in middle
        end
    endfunction

    // --- Sin/Cos Approximations (Q16.16) ---
    function signed [31:0] fx_sin;
        input signed [31:0] x;
        begin
            fx_sin = x; // Placeholder linear approx
        end
    endfunction

    function signed [31:0] fx_cos;
        input signed [31:0] x;
        begin
            fx_cos = FP_ONE - ((x * x) >>> 17); // 1 - x^2/2
        end
    endfunction

    // ========================================================================
    // 5. Memory Port A (Write Priority & Read A)
    // ========================================================================
    always @(posedge clk) begin
        // --- Write Mux (Priority: DMA > Accumulator > Pipeline) ---
        
        if (dma_wr_en) begin
            // 1. DMA Input Load
            tensor_mem[base_dst + dma_wr_offset] <= dma_wr_data;
        end 
        else if (is_accum_op && done) begin
            // 2. Reduction Finalize
            if (hw_opcode == `H_SUM)
                tensor_mem[base_dst] <= saturate(accumulator);
            else if (hw_opcode == `H_MEAN)
                tensor_mem[base_dst] <= saturate(accumulator >>> SHIFT_MEAN);
            else 
                tensor_mem[base_dst] <= saturate(accumulator >>> 16);
        end 
        else if (wb_enable && !stall && !is_accum_op) begin
            // 3. Pipeline Result (Element-wise)
            // Note: Even if division just finished, 'stall' goes low same cycle 'wb_enable' goes high.
            tensor_mem[wb_addr] <= wb_data;
        end

        // --- Read Operation (Operand A) ---
        if (!stall) begin
            op_a <= tensor_mem[base_src1 + vec_cnt];
        end
    end

    // ========================================================================
    // 6. Memory Port B (Read B / DMA Read)
    // ========================================================================
    assign dma_rd_data = op_b;

    always @(posedge clk) begin
        if (!stall) begin
            if (dma_rd_req) begin
                op_b <= tensor_mem[base_src1 + vec_cnt];
            end else begin
                op_b <= tensor_mem[base_src2 + vec_cnt];
            end
        end
    end

    // ========================================================================
    // 7. Execution Pipeline Stage
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
            base_src1 <= 0; base_src2 <= 0; base_dst <= 0;
            // Reset Div State
            div_busy <= 0;
            div_counter <= 0;
            div_dividend <= 0;
            div_divisor <= 0;
            div_result <= 0;
            div_sign <= 0;
        end else begin
            done <= 0; // Reset Pulse

            if (start) begin
                // --- SETUP STATE ---
                active <= 1;
                vec_cnt <= 0;
                wb_enable <= 0;
                accumulator <= 0;
                div_busy <= 0; // Ensure div logic is reset
                
                is_accum_op <= (hw_opcode == `H_SUM || hw_opcode == `H_MEAN || 
                                hw_opcode == `H_LINEAR || hw_opcode == `H_MATMUL);

                base_src1 <= (src1_id & `SLOT_MASK) * `VECTOR_LEN;
                base_src2 <= (src2_id_bram & `SLOT_MASK) * `VECTOR_LEN;
                base_dst  <= (dst_id  & `SLOT_MASK) * `VECTOR_LEN;
            end 
            else if (active) begin
                
                // --- Prepare Operands ---
                reg signed [31:0] final_b;
                final_b = use_stream ? $signed(stream_data) : op_b;
                reg signed [63:0] wide_res; 

                // ------------------------------------------------------------
                // DIVISION FSM
                // ------------------------------------------------------------
                if (hw_opcode == `H_DIV) begin
                    if (!div_busy) begin
                        // START DIVISION (Cycle 0)
                        if (final_b == 0) begin
                            // Handle Divide by Zero (Return Max/Min or 0)
                            wb_data <= 0; 
                            // Treat as single cycle op, let standard logic handle wb below
                        end else begin
                            // Initialize Long Division
                            // Abs(OpA)
                            if (op_a[31]) div_dividend <= {32'd0, -op_a};
                            else          div_dividend <= {32'd0, op_a};
                            
                            // Shift Dividend left by 16 for Q16.16 result precision
                            // Effectively: (A * 2^16) / B
                            div_dividend <= div_dividend << 16; 

                            // Abs(OpB)
                            if (final_b[31]) div_divisor <= -final_b;
                            else             div_divisor <= final_b;

                            // Calculate Result Sign (XOR)
                            div_sign <= op_a[31] ^ final_b[31];
                            
                            div_result <= 0;
                            div_counter <= 32; // 32 iterations for 32-bit integer division
                            div_busy <= 1;     // STALL PIPELINE
                            wb_enable <= 0;    // Disable writeback while calculating
                        end
                    end 
                    else begin
                        // ITERATE DIVISION (Cycles 1-32)
                        // Standard Shift-Subtract non-restoring algorithm
                        
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
                            // FINISH DIVISION
                            wb_data <= div_sign ? -div_result : div_result;
                            div_busy <= 0; // RELEASE STALL
                            
                            // Setup Writeback for this result
                            wb_addr   <= base_dst + vec_cnt;
                            wb_enable <= 1;

                            // Manually Increment Loop Counter here (since main logic is stalled)
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
                // STANDARD OPERATIONS (No Stall)
                // ------------------------------------------------------------
                if (!stall) begin
                    // Only execute if NOT stalling (Stream valid AND Div not busy)
                    // Note: If Division just finished (div_busy went 0), we skip this block 
                    // for the current cycle because we handled WB/Increment inside the Div block logic.
                    
                    if (hw_opcode != `H_DIV) begin
                        case (hw_opcode)
                            // --- Basic Math (Saturated) ---
                            `H_ADD: begin 
                                wide_res = op_a + final_b; 
                                wb_data <= saturate(wide_res); 
                            end
                            `H_SUB: begin 
                                wide_res = op_a - final_b; 
                                wb_data <= saturate(wide_res); 
                            end
                            `H_MUL: begin 
                                wide_res = op_a * final_b; 
                                wb_data <= saturate(wide_res >>> 16); 
                            end
                            `H_NEG: wb_data <= -op_a;
                            `H_POW: begin
                                wide_res = op_a * op_a;
                                wb_data <= saturate(wide_res >>> 16);
                            end

                            // --- Activations ---
                            `H_RELU: wb_data <= (op_a > 0) ? op_a : 0;
                            `H_GELU: wb_data <= fx_gelu(op_a);
                            `H_TANH: wb_data <= fx_tanh(op_a);
                            
                            // --- Math Funcs ---
                            `H_SIN:  wb_data <= fx_sin(op_a);
                            `H_COS:  wb_data <= fx_cos(op_a);
                            `H_ERF:  wb_data <= fx_tanh(op_a); 

                            // --- Reductions ---
                            `H_LINEAR, `H_MATMUL: begin
                                wide_res = op_a * final_b;
                                accumulator <= accumulator + wide_res;
                                wb_data <= 0; 
                            end
                            `H_SUM, `H_MEAN: begin
                                accumulator <= accumulator + op_a;
                                wb_data <= 0;
                            end

                            // --- Logic / Masks ---
                            `H_EQ: wb_data <= (op_a == final_b) ? FP_ONE : 0;
                            `H_NE: wb_data <= (op_a != final_b) ? FP_ONE : 0;
                            `H_GT: wb_data <= (op_a > final_b)  ? FP_ONE : 0;
                            `H_LT: wb_data <= (op_a < final_b)  ? FP_ONE : 0;
                            
                            `H_MASKED_FILL: begin
                                wb_data <= (op_a != 0) ? final_b : 0; 
                            end

                            // --- Generators ---
                            `H_ARANGE: wb_data <= saturate(vec_cnt << 16);
                            `H_FULL:   wb_data <= final_b;

                            // --- Movement ---
                            `H_COPY:   wb_data <= op_a;
                            `H_EMBED:  wb_data <= final_b; 

                            default:   wb_data <= op_a;
                        endcase

                        // Writeback Config
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
                    // else: hw_opcode == H_DIV but div_busy is 0 (first cycle)
                    // The division init block handles this case above.
                end 
                // else: STALL state. 
            end 
            else begin
                // Idle State
                wb_enable <= 0;
            end
        end
    end

endmodule