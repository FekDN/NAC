`include "nac_defines.vh"

module NAC_Processor_Top #(
    parameter CONTROL_BASE = 32'h0000_0000 // Base address for Control Block
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // External Memory Interface (DDR3)
    // Generic Request/Grant Protocol (Compatible with AXI Wrappers)
    // ========================================================================
    output reg  [31:0] mem_addr,
    output reg         mem_req,     // Read Request Strobe
    output reg         mem_we,      // Write Enable Strobe
    output reg  [31:0] mem_wdata,
    input  wire        mem_grant,   // Bus Arbiter accepted request
    input  wire        mem_valid,   // Read Data Valid
    input  wire [31:0] mem_rdata,   // Read Data Payload

    // ========================================================================
    // Status & Debug
    // ========================================================================
    output reg  [3:0]  led,
    output reg         irq_done,    // Interrupt: Inference Complete
    output reg         busy         // System Busy
);

    // ========================================================================
    // Internal Registers & Configuration
    // ========================================================================
    
    // 1. Memory Pointers (Loaded from Control Block)
    reg [31:0] ptr_registry;
    reg [31:0] ptr_code;
    reg [31:0] ptr_weights;
    reg [31:0] ptr_input;
    reg [31:0] ptr_output;

    // 2. Pattern Registry (Pattern ID -> Memory Address)
    // 256 entries * 32-bit address
    (* ram_style = "block" *)
    reg [31:0] pattern_lut [0:255]; 
    reg [7:0]  lut_load_idx;

    // 3. Execution Context (Stack & PC)
    reg [31:0] pc;
    reg [31:0] call_stack [0:`STACK_DEPTH-1];
    reg [4:0]  sp;

    // 4. Decoded Instruction Registers
    reg [7:0]  dec_A;         // Opcode
    reg [7:0]  dec_B;         // Variation
    reg [15:0] dec_C;         // Constant/Group ID
    reg [7:0]  dec_D_len;     // Num Inputs
    reg [15:0] dec_inputs [0:`MAX_INPUTS-1];
    
    // Decoding Helpers
    reg [2:0]  header_byte_idx;
    reg [7:0]  payload_byte_idx;
    reg [7:0]  temp_hi_byte;  // For Big Endian assembly

    // 5. RLE (Copy) State
    reg        rle_active;
    reg [15:0] rle_count;
    reg [7:0]  rle_template_A;

    // 6. DMA & Stream Counters
    reg [15:0] dma_vec_cnt;
    
    // 7. Weight Streaming Logic
    reg [31:0] stream_mem_addr;
    
    // ========================================================================
    // Submodule Interfaces
    // ========================================================================

    // --- Fetcher Signals ---
    reg        fetch_flush;
    reg        fetch_byte_req;
    wire [7:0] fetch_byte_data;
    wire       fetch_byte_valid;
    wire       fetch_busy;
    wire [31:0] fetch_mem_addr;
    wire        fetch_mem_req;

    // --- ALU Signals ---
    reg        alu_start;
    reg        alu_stream_en;
    reg        alu_dma_wr_en;
    reg        alu_dma_rd_req;
    wire       alu_done;
    wire       alu_stream_ready;
    wire [31:0] alu_dma_rd_data;
    
    // To handle RLE masquerading
    wire [7:0] effective_opcode = rle_active ? rle_template_A : dec_A;

    // ========================================================================
    // Module Instantiations
    // ========================================================================

    NAC_Byte_Fetcher fetcher (
        .clk(clk), 
        .rst_n(rst_n),
        .flush(fetch_flush),
        .start_addr(pc),
        .busy(fetch_busy),
        .byte_req(fetch_byte_req),
        .byte_data(fetch_byte_data),
        .byte_valid(fetch_byte_valid),
        // Memory Interface (Connected via Arbiter)
        .mem_addr(fetch_mem_addr),
        .mem_req(fetch_mem_req),
        .mem_grant(mem_grant && (arbiter_sel == 0)), // Grant only if Fetcher selected
        .mem_valid(mem_valid),
        .mem_rdata(mem_rdata)
    );

    NAC_ALU_Core alu (
        .clk(clk),
        .rst_n(rst_n),
        .start(alu_start),
        .opcode(effective_opcode),
        .src1_id(dec_inputs[0]),
        .src2_id_bram(dec_inputs[1]),
        .dst_id(dec_inputs[0]), // In-place operation
        .done(alu_done),
        // Streaming Interface
        .use_stream(alu_stream_en),
        .stream_data(mem_rdata),           // Direct DDR feed
        .stream_valid(mem_valid && (arbiter_sel == 2)), // Valid only if Streamer selected
        .stream_ready(alu_stream_ready),
        // DMA Interface
        .dma_wr_en(alu_dma_wr_en),
        .dma_wr_data(mem_rdata), // Direct DDR feed for Input DMA
        .dma_wr_offset(dma_vec_cnt),
        .dma_rd_req(alu_dma_rd_req),
        .dma_rd_data(alu_dma_rd_data)
    );

    // ========================================================================
    // Memory Arbiter
    // ========================================================================
    // 0 = Fetcher (Code)
    // 1 = FSM (Config / DMA / Registry)
    // 2 = Streamer (Weights)
    reg [1:0] arbiter_sel;

    // FSM Internal Memory Signals
    reg [31:0] fsm_mem_addr;
    reg        fsm_mem_req;
    reg        fsm_mem_we;
    reg [31:0] fsm_mem_wdata;

    // Arbiter Mux
    always @(*) begin
        case (arbiter_sel)
            2'd0: begin // Fetcher
                mem_addr  = fetch_mem_addr;
                mem_req   = fetch_mem_req;
                mem_we    = 1'b0;
                mem_wdata = 32'd0;
            end
            2'd1: begin // FSM
                mem_addr  = fsm_mem_addr;
                mem_req   = fsm_mem_req;
                mem_we    = fsm_mem_we;
                mem_wdata = fsm_mem_wdata;
            end
            2'd2: begin // Streamer
                mem_addr  = stream_mem_addr;
                mem_req   = 1'b1; // Streamer always wants data if active
                mem_we    = 1'b0;
                mem_wdata = 32'd0;
                
                // Backpressure logic: If ALU is stalled/not ready, we shouldn't req
                // In generic DDR, holding REQ high is usually "Read Next".
                // Simple logic: Only Request if ALU is ready to accept.
                if (!alu_stream_ready) mem_req = 1'b0; 
            end
            default: begin
                mem_addr = 0; mem_req = 0; mem_we = 0; mem_wdata = 0;
            end
        endcase
    end

    // ========================================================================
    // Main Finite State Machine
    // ========================================================================
    localparam S_BOOT           = 0;
    localparam S_READ_CMD       = 1;
    localparam S_CFG_SEQ        = 2; // Read 5 config pointers
    localparam S_LOAD_REGISTRY  = 3;
    localparam S_IDLE           = 4; // Ready to run
    localparam S_FETCH_HDR      = 5;
    localparam S_DECODE_OP      = 6; // Analyze Header
    localparam S_FETCH_PAY      = 7;
    localparam S_EXEC_DECISION  = 8;
    localparam S_DMA_IN         = 9;
    localparam S_DMA_OUT        = 10;
    localparam S_PREP_MATH      = 11;
    localparam S_RUN_MATH       = 12;
    localparam S_DONE           = 13;

    reg [3:0] state;
    reg [2:0] cfg_seq_cnt; // For reading 5 config words

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_BOOT;
            arbiter_sel <= 1;
            busy <= 0;
            irq_done <= 0;
            led <= 0;
            
            pc <= 0; sp <= 0;
            fetch_flush <= 1;
            rle_active <= 0;
            
            fsm_mem_req <= 0; fsm_mem_we <= 0;
            fetch_byte_req <= 0;
            alu_start <= 0; alu_dma_wr_en <= 0; alu_dma_rd_req <= 0;
        end else begin
            // ----------------------------------------------------------------
            // Default Pulse Resets
            // ----------------------------------------------------------------
            fsm_mem_req <= 0;
            fsm_mem_we <= 0;
            fetch_byte_req <= 0;
            fetch_flush <= 0;
            alu_start <= 0;
            alu_dma_wr_en <= 0;
            alu_dma_rd_req <= 0;
            irq_done <= 0;

            case (state)
                // ------------------------------------------------------------
                // 1. BOOT & CONFIGURATION
                // ------------------------------------------------------------
                S_BOOT: begin
                    busy <= 0;
                    arbiter_sel <= 1; // FSM owns bus
                    fsm_mem_addr <= CONTROL_BASE + `REG_CMD;
                    fsm_mem_req <= 1;
                    
                    if (mem_valid && mem_grant) begin
                        if (mem_rdata == 1) begin // Load Config
                            state <= S_CFG_SEQ;
                            cfg_seq_cnt <= 0;
                            busy <= 1;
                        end
                        else if (mem_rdata == 2) begin // Run Inference
                            state <= S_IDLE;
                            busy <= 1;
                        end
                    end
                end

                S_CFG_SEQ: begin
                    // Read 5 pointers sequentially: REGISTRY, CODE, WEIGHTS, INPUT, OUTPUT
                    // Base offset 0x08, 0x0C, 0x10, 0x14, 0x18
                    fsm_mem_addr <= CONTROL_BASE + `REG_REGISTRY + (cfg_seq_cnt * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        case (cfg_seq_cnt)
                            0: ptr_registry <= mem_rdata;
                            1: ptr_code     <= mem_rdata;
                            2: ptr_weights  <= mem_rdata;
                            3: ptr_input    <= mem_rdata;
                            4: ptr_output   <= mem_rdata;
                        endcase
                        
                        if (cfg_seq_cnt == 4) begin
                            state <= S_LOAD_REGISTRY;
                            lut_load_idx <= 0;
                        end else begin
                            cfg_seq_cnt <= cfg_seq_cnt + 1;
                        end
                    end
                end

                S_LOAD_REGISTRY: begin
                    fsm_mem_addr <= ptr_registry + (lut_load_idx * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        pattern_lut[lut_load_idx] <= mem_rdata;
                        lut_load_idx <= lut_load_idx + 1;
                        if (lut_load_idx == 255) begin
                            state <= S_BOOT; // Done loading, wait for RUN cmd
                            led <= 4'b0001;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 2. RUN SETUP
                // ------------------------------------------------------------
                S_IDLE: begin
                    pc <= ptr_code;
                    fetch_flush <= 1; // Initialize Fetcher
                    sp <= 0;
                    rle_active <= 0;
                    header_byte_idx <= 0;
                    state <= S_FETCH_HDR;
                    led <= 4'b0010;
                end

                // ------------------------------------------------------------
                // 3. FETCH & DECODE
                // ------------------------------------------------------------
                S_FETCH_HDR: begin
                    arbiter_sel <= 0; // Give bus to Fetcher
                    fetch_byte_req <= 1;
                    
                    if (fetch_byte_valid) begin
                        // Assemble Header (5 Bytes)
                        case (header_byte_idx)
                            0: dec_A <= fetch_byte_data;
                            1: dec_B <= fetch_byte_data;
                            2: dec_C[15:8] <= fetch_byte_data; // Big Endian
                            3: dec_C[7:0] <= fetch_byte_data;
                            4: dec_D_len <= fetch_byte_data;
                        endcase
                        
                        // PC update is implicit in Fetcher flush logic, 
                        // but logic keeps track for Call/Return
                        pc <= pc + 1; 

                        if (header_byte_idx == 4) begin
                            state <= S_DECODE_OP;
                        end else begin
                            header_byte_idx <= header_byte_idx + 1;
                        end
                    end
                end

                S_DECODE_OP: begin
                    if (dec_D_len == 0) begin
                        state <= S_EXEC_DECISION;
                    end else begin
                        payload_byte_idx <= 0;
                        state <= S_FETCH_PAY;
                    end
                end

                S_FETCH_PAY: begin
                    fetch_byte_req <= 1;
                    if (fetch_byte_valid) begin
                        pc <= pc + 1;
                        
                        if (payload_byte_idx[0] == 0) begin
                            temp_hi_byte <= fetch_byte_data; // Store MSB
                        end else begin
                            // Combine MSB+LSB -> 16-bit Input ID
                            dec_inputs[payload_byte_idx >> 1] <= {temp_hi_byte, fetch_byte_data};
                        end
                        
                        if (payload_byte_idx == (dec_D_len * 2) - 1) begin
                            state <= S_EXEC_DECISION;
                        end else begin
                            payload_byte_idx <= payload_byte_idx + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 4. EXECUTION DISPATCH
                // ------------------------------------------------------------
                S_EXEC_DECISION: begin
                    case (dec_A)
                        `OP_PATTERN: begin
                            call_stack[sp] <= pc;
                            sp <= sp + 1;
                            pc <= pattern_lut[dec_C[7:0]]; // Jump to Pattern Addr
                            fetch_flush <= 1; // Invalidate fetcher buffer
                            header_byte_idx <= 0;
                            state <= S_FETCH_HDR;
                        end
                        
                        `OP_RETURN: begin
                            sp <= sp - 1;
                            pc <= call_stack[sp-1]; // Return
                            fetch_flush <= 1;
                            header_byte_idx <= 0;
                            state <= S_FETCH_HDR;
                        end
                        
                        `OP_COPY: begin
                            rle_active <= 1;
                            rle_count <= dec_C;
                            rle_template_A <= dec_inputs[0][7:0]; // Template is 1st input
                            state <= S_PREP_MATH;
                        end

                        `OP_DATA_INPUT: begin
                            dma_vec_cnt <= 0;
                            arbiter_sel <= 1; // FSM needs bus for DMA
                            state <= S_DMA_IN;
                        end

                        `OP_OUTPUT: begin
                            dma_vec_cnt <= 0;
                            arbiter_sel <= 1;
                            state <= S_DMA_OUT;
                        end

                        default: begin // Standard Math
                            state <= S_PREP_MATH;
                        end
                    endcase
                end

                // ------------------------------------------------------------
                // 5. DMA OPERATIONS
                // ------------------------------------------------------------
                S_DMA_IN: begin
                    // Read External DDR -> Write Internal ALU BRAM
                    fsm_mem_addr <= ptr_input + (dma_vec_cnt * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        alu_dma_wr_en <= 1; // Push to ALU
                        // dma_wr_data connected directly to mem_rdata via wire
                        
                        if (dma_vec_cnt == `VECTOR_LEN - 1) begin
                            header_byte_idx <= 0;
                            state <= S_FETCH_HDR;
                        end else begin
                            dma_vec_cnt <= dma_vec_cnt + 1;
                        end
                    end
                end

                S_DMA_OUT: begin
                    // Read Internal ALU BRAM -> Write External DDR
                    
                    // Cycle 1: Request data from ALU
                    alu_dma_rd_req <= 1;
                    
                    // Cycle 2: Write to Memory (Assuming 1 cycle RAM latency)
                    // Note: In strict sync logic, we'd wait a state. 
                    // Here we pipeline: Request N+1, Write N.
                    // For robustness in this code, we rely on the ALU's combinatorial/registered read.
                    
                    fsm_mem_addr <= ptr_output + (dma_vec_cnt * 4);
                    fsm_mem_wdata <= alu_dma_rd_data;
                    fsm_mem_we <= 1;
                    fsm_mem_req <= 1;
                    
                    if (mem_grant) begin
                        if (dma_vec_cnt == `VECTOR_LEN - 1) begin
                             irq_done <= 1;
                             state <= S_DONE;
                        end else begin
                            dma_vec_cnt <= dma_vec_cnt + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 6. MATH EXECUTION & STREAMING
                // ------------------------------------------------------------
                S_PREP_MATH: begin
                    // Check if Weight Streaming is needed
                    // Condition: Fundamental Math Op + Constant ID > 0 (implies weights)
                    if (dec_C > 0 && effective_opcode >= 10) begin
                        arbiter_sel <= 2; // Streamer owns bus
                        alu_stream_en <= 1;
                        stream_mem_addr <= ptr_weights + (dec_C * `VECTOR_LEN * 4);
                    end else begin
                        arbiter_sel <= 0; // Bus idle (or back to Fetcher)
                        alu_stream_en <= 0;
                    end
                    
                    alu_start <= 1;
                    state <= S_RUN_MATH;
                end

                S_RUN_MATH: begin
                    // While running, logic for streaming is handled by Arbiter Mux 
                    // (Slot 2 sends REQ if ALU Ready) and ALU (Consumes Valid)
                    
                    if (alu_stream_en && mem_grant && (arbiter_sel == 2)) begin
                        // If grant received, advance address for next burst/word
                        stream_mem_addr <= stream_mem_addr + 4;
                    end

                    if (alu_done) begin
                        if (rle_active) begin
                            if (rle_count == 1) begin
                                rle_active <= 0;
                                header_byte_idx <= 0;
                                state <= S_FETCH_HDR;
                            end else begin
                                rle_count <= rle_count - 1;
                                state <= S_PREP_MATH; // Repeat
                            end
                        end else begin
                            header_byte_idx <= 0;
                            state <= S_FETCH_HDR;
                        end
                    end
                end

                S_DONE: begin
                    led <= 4'b1111;
                    busy <= 0;
                    // Trap until Reset
                end

            endcase
        end
    end

endmodule
