`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_Processor_Top #(
    parameter CONTROL_BASE = 32'h0000_0000 // Base address of the Control Block in DDR
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // External Memory Interface (DDR3/AXI Wrapper)
    // ========================================================================
    // Generic Request/Grant interface. Maps to AXI4-Full/Stream physically.
    output reg  [31:0] mem_addr,
    output reg         mem_req,     // Read Request Strobe
    output reg         mem_we,      // Write Enable Strobe
    output reg  [31:0] mem_wdata,   // Write Data
    input  wire        mem_grant,   // Bus Arbiter accepted the request
    input  wire        mem_valid,   // Read Data Valid pulse
    input  wire [31:0] mem_rdata,   // Read Data payload

    // ========================================================================
    // Status & Debug Interface
    // ========================================================================
    output reg  [3:0]  led,         // Debug LEDs
    output reg         irq_done,    // High when inference finishes
    output reg         busy         // High when FSM is active
);

    // ========================================================================
    // 1. Internal Memory & Look-Up Tables (LUTs)
    // ========================================================================

    // --- OpCode Translation LUT ---
    // Maps Dynamic NAC ID (8-bit) -> {StreamFlag, HW_OpCode}
    // Loaded from DDR. 256 entries.
    // Format: [7:6]=Rsrv, [5]=StreamEn, [4:0]=HW_ID
    (* ram_style = "distributed" *)
    reg [7:0] opcode_lut [0:255]; 

    // --- Variation Translation LUT ---
    // Maps Dynamic Variation ID (8-bit) -> Input Routing Mask
    // Loaded from DDR. 256 entries.
    // Format: [7:4]=Src2_Index, [3:0]=Src1_Index
    (* ram_style = "distributed" *)
    reg [7:0] variation_lut [0:255];

    // --- Pattern Registry ---
    // Maps Pattern ID (8-bit) -> Memory Address (32-bit)
    // Loaded from DDR. 256 entries.
    (* ram_style = "block" *)
    reg [31:0] pattern_lut [0:255]; 

    // ========================================================================
    // 2. Configuration Registers
    // ========================================================================
    reg [31:0] ptr_registry; // Pointer to Pattern Table
    reg [31:0] ptr_code;     // Pointer to Instruction Stream
    reg [31:0] ptr_weights;  // Pointer to Weights Blob
    reg [31:0] ptr_input;    // Pointer to Input Tensor
    reg [31:0] ptr_output;   // Pointer to Output Tensor
    reg [31:0] ptr_opmap;    // Pointer to OpCode LUT
    reg [31:0] ptr_varmap;   // Pointer to Variation LUT

    // ========================================================================
    // 3. Execution Context (PC & Stack)
    // ========================================================================
    reg [31:0] pc;                          // Program Counter (Byte Address)
    reg [31:0] call_stack [0:`STACK_DEPTH-1]; // Hardware Recursion Stack
    reg [4:0]  sp;                          // Stack Pointer

    // ========================================================================
    // 4. Decoding Registers
    // ========================================================================
    // Raw fields extracted from the byte stream
    reg [7:0]  raw_A;       // OpCode
    reg [7:0]  raw_B;       // Variation
    reg [15:0] raw_C;       // Constant/Count
    reg [7:0]  raw_D_len;   // Number of Inputs
    reg [15:0] raw_inputs [0:`MAX_INPUTS-1]; // Input Dependency IDs

    // Decoding State Helpers
    reg [2:0]  hdr_byte_idx; // 0..4 for Header
    reg [7:0]  pay_byte_idx; // 0..N for Payload
    reg [7:0]  temp_hi_byte; // Buffer for Big Endian assembly

    // ========================================================================
    // 5. RLE (Copy) State Machine Registers
    // ========================================================================
    reg        rle_active;      // 1 = Executing a COPY loop
    reg [15:0] rle_counter;     // Remaining iterations
    reg [7:0]  rle_template_A;  // The OpCode being copied
    reg [7:0]  rle_template_B;  // The Variation being copied
    reg [15:0] rle_template_C;  // The Constant being copied

    // ========================================================================
    // 6. ALU Interface Signals
    // ========================================================================
    // Signals driven by FSM after Translation
    reg [4:0]  alu_hw_opcode;   // Static Silicon ID
    reg        alu_stream_en;   // 1 = Enable weight streaming
    reg [3:0]  routed_src1_idx; // Routed Index for Port A
    reg [3:0]  routed_src2_idx; // Routed Index for Port B
    
    // Control Signals
    reg        alu_start;
    wire       alu_done;
    wire       alu_stream_ready;
    
    // DMA Signals (FSM acting as DMA controller)
    reg [15:0] dma_cnt;
    reg        dma_wr_en;
    reg        dma_rd_req;
    wire [31:0] dma_rd_data;

    // Streaming Logic
    reg [31:0] stream_mem_addr;

    // ========================================================================
    // 7. FSM States & Counters
    // ========================================================================
    localparam S_BOOT           = 5'd0;
    localparam S_READ_CFG_PTRS  = 5'd1;
    localparam S_UNPACK_OPMAP   = 5'd2; // Loading OpCode LUT
    localparam S_UNPACK_VARMAP  = 5'd3; // Loading Variation LUT
    localparam S_LOAD_REGISTRY  = 5'd4; // Loading Pattern LUT
    localparam S_IDLE           = 5'd5;
    localparam S_FETCH_HEADER   = 5'd6;
    localparam S_FETCH_PAYLOAD  = 5'd7;
    localparam S_TRANSLATE      = 5'd8; // Dynamic -> Static Translation
    localparam S_EXEC_DISPATCH  = 5'd9;
    localparam S_DMA_INPUT      = 5'd10;
    localparam S_DMA_OUTPUT     = 5'd11;
    localparam S_PREP_MATH      = 5'd12;
    localparam S_RUN_MATH       = 5'd13;
    localparam S_DONE           = 5'd14;

    reg [4:0] state;
    
    // Config Loading Counters
    reg [2:0] cfg_seq_cnt; // 0..6
    reg [8:0] lut_load_idx; // 0..255
    reg [2:0] word_unpack_cnt; // For unpacking 32-bit words into bytes

    // ========================================================================
    // 8. Submodule Instantiations
    // ========================================================================

    // --- Memory Arbiter Selection ---
    // 0: Fetcher (Instruction Stream)
    // 1: FSM (Config / LUT Load / DMA)
    // 2: Streamer (Weight Stream)
    reg [1:0] arbiter_sel;

    // --- Byte Fetcher ---
    reg        fetch_flush;
    reg        fetch_byte_req;
    wire [7:0] fetch_byte_data;
    wire       fetch_byte_valid;
    wire       fetch_busy;
    wire [31:0] fetch_mem_addr;
    wire        fetch_mem_req;

    NAC_Byte_Fetcher fetcher (
        .clk(clk), 
        .rst_n(rst_n),
        .flush(fetch_flush),
        .start_addr(pc),
        .busy(fetch_busy),
        .byte_req(fetch_byte_req),
        .byte_data(fetch_byte_data),
        .byte_valid(fetch_byte_valid),
        // Memory Interface (Connected to Arbiter input 0)
        .mem_addr(fetch_mem_addr),
        .mem_req(fetch_mem_req),
        .mem_grant(mem_grant && (arbiter_sel == 2'd0)),
        .mem_valid(mem_valid),
        .mem_rdata(mem_rdata)
    );

    // --- ALU Core ---
    NAC_ALU_Core alu (
        .clk(clk),
        .rst_n(rst_n),
        .start(alu_start),
        .hw_opcode(alu_hw_opcode), // Using Translated HW ID
        .src1_id(raw_inputs[routed_src1_idx]), // Using Routed Input
        .src2_id_bram(raw_inputs[routed_src2_idx]), // Using Routed Input
        .dst_id(raw_inputs[routed_src1_idx]), // In-place writeback
        .done(alu_done),
        
        // Weight Streaming
        .use_stream(alu_stream_en),
        .stream_data(mem_rdata), // Direct feed from DDR
        .stream_valid(mem_valid && (arbiter_sel == 2'd2)),
        .stream_ready(alu_stream_ready),
        
        // DMA Channels
        .dma_wr_en(dma_wr_en),
        .dma_wr_data(mem_rdata), // Direct feed from DDR
        .dma_wr_offset(dma_cnt),
        .dma_rd_req(dma_rd_req),
        .dma_rd_data(dma_rd_data)
    );

    // ========================================================================
    // 9. Memory Arbiter (3-Way Priority Mux)
    // ========================================================================
    // Internal FSM Memory Signals
    reg [31:0] fsm_mem_addr;
    reg        fsm_mem_req;
    reg        fsm_mem_we;
    reg [31:0] fsm_mem_wdata;

    always @(*) begin
        case (arbiter_sel)
            2'd0: begin // Fetcher
                mem_addr  = fetch_mem_addr;
                mem_req   = fetch_mem_req;
                mem_we    = 1'b0;
                mem_wdata = 32'd0;
            end
            2'd1: begin // FSM Controller
                mem_addr  = fsm_mem_addr;
                mem_req   = fsm_mem_req;
                mem_we    = fsm_mem_we;
                mem_wdata = fsm_mem_wdata;
            end
            2'd2: begin // Weight Streamer
                mem_addr  = stream_mem_addr;
                // Backpressure: Only request if ALU is ready to consume
                mem_req   = alu_stream_ready ? 1'b1 : 1'b0; 
                mem_we    = 1'b0;
                mem_wdata = 32'd0;
            end
            default: begin
                mem_addr = 0; mem_req = 0; mem_we = 0; mem_wdata = 0;
            end
        endcase
    end

    // ========================================================================
    // 10. Main Finite State Machine
    // ========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_BOOT;
            pc <= 0;
            sp <= 0;
            busy <= 0;
            irq_done <= 0;
            led <= 0;
            arbiter_sel <= 1; // Default to FSM
            
            // Submodule Resets
            fetch_flush <= 1;
            fetch_byte_req <= 0;
            alu_start <= 0;
            dma_wr_en <= 0;
            dma_rd_req <= 0;
            
            // Registers
            rle_active <= 0;
            fsm_mem_req <= 0;
            fsm_mem_we <= 0;
        end else begin
            // ----------------------------------------------------------------
            // Pulse Resets (Cleared every cycle)
            // ----------------------------------------------------------------
            fsm_mem_req <= 0;
            fsm_mem_we <= 0;
            fetch_byte_req <= 0;
            fetch_flush <= 0;
            alu_start <= 0;
            dma_wr_en <= 0;
            dma_rd_req <= 0;

            case (state)
                // ------------------------------------------------------------
                // 1. BOOT SEQUENCE: Poll Command Register
                // ------------------------------------------------------------
                S_BOOT: begin
                    arbiter_sel <= 1; // FSM Control
                    busy <= 0;
                    // Poll Command Register at 0x00
                    fsm_mem_addr <= CONTROL_BASE + `REG_CMD;
                    fsm_mem_req <= 1;
                    
                    if (mem_valid && mem_grant) begin
                        if (mem_rdata == 1) begin // CMD: Load Config
                            state <= S_READ_CFG_PTRS;
                            cfg_seq_cnt <= 0;
                            busy <= 1;
                        end else if (mem_rdata == 2) begin // CMD: Run
                            state <= S_IDLE;
                            busy <= 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 2. LOAD CONFIGURATION POINTERS
                // ------------------------------------------------------------
                S_READ_CFG_PTRS: begin
                    // Read pointers sequentially (Offsets 0x08 to 0x20)
                    fsm_mem_addr <= CONTROL_BASE + `REG_REGISTRY + (cfg_seq_cnt * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        case (cfg_seq_cnt)
                            0: ptr_registry <= mem_rdata;
                            1: ptr_code     <= mem_rdata;
                            2: ptr_weights  <= mem_rdata;
                            3: ptr_input    <= mem_rdata;
                            4: ptr_output   <= mem_rdata;
                            5: ptr_opmap    <= mem_rdata;
                            6: ptr_varmap   <= mem_rdata;
                        endcase
                        
                        if (cfg_seq_cnt == 6) begin
                            state <= S_UNPACK_OPMAP;
                            lut_load_idx <= 0;
                            word_unpack_cnt <= 0;
                        end else begin
                            cfg_seq_cnt <= cfg_seq_cnt + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 3. LOAD OPCODE LUT (Unpack 32-bit words -> 4x 8-bit entries)
                // ------------------------------------------------------------
                S_UNPACK_OPMAP: begin
                    // We read 1 word (4 bytes) at a time
                    if (word_unpack_cnt == 0) begin
                        fsm_mem_addr <= ptr_opmap + (lut_load_idx); // Byte addressed
                        fsm_mem_req <= 1;
                    end

                    // Fill LUT from valid memory data
                    if (mem_valid) begin
                        // Assuming Host writes Little Endian, memory returns it packed.
                        // NAC tools pack [Idx3, Idx2, Idx1, Idx0].
                        opcode_lut[lut_load_idx]   <= mem_rdata[7:0];
                        opcode_lut[lut_load_idx+1] <= mem_rdata[15:8];
                        opcode_lut[lut_load_idx+2] <= mem_rdata[23:16];
                        opcode_lut[lut_load_idx+3] <= mem_rdata[31:24];
                        
                        lut_load_idx <= lut_load_idx + 4;
                        if (lut_load_idx >= 252) begin
                            state <= S_UNPACK_VARMAP;
                            lut_load_idx <= 0;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 4. LOAD VARIATION LUT
                // ------------------------------------------------------------
                S_UNPACK_VARMAP: begin
                    if (word_unpack_cnt == 0) begin
                        fsm_mem_addr <= ptr_varmap + (lut_load_idx);
                        fsm_mem_req <= 1;
                    end

                    if (mem_valid) begin
                        variation_lut[lut_load_idx]   <= mem_rdata[7:0];
                        variation_lut[lut_load_idx+1] <= mem_rdata[15:8];
                        variation_lut[lut_load_idx+2] <= mem_rdata[23:16];
                        variation_lut[lut_load_idx+3] <= mem_rdata[31:24];
                        
                        lut_load_idx <= lut_load_idx + 4;
                        if (lut_load_idx >= 252) begin
                            state <= S_LOAD_REGISTRY;
                            lut_load_idx <= 0;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 5. LOAD PATTERN REGISTRY
                // ------------------------------------------------------------
                S_LOAD_REGISTRY: begin
                    // Read 32-bit addresses directly
                    fsm_mem_addr <= ptr_registry + (lut_load_idx * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        pattern_lut[lut_load_idx] <= mem_rdata;
                        lut_load_idx <= lut_load_idx + 1;
                        if (lut_load_idx == 255) begin
                            state <= S_BOOT; // Configuration Done
                            led <= 4'b0001; // Indication
                        end
                    end
                end

                // ------------------------------------------------------------
                // 6. RUN SETUP
                // ------------------------------------------------------------
                S_IDLE: begin
                    pc <= ptr_code;
                    fetch_flush <= 1; // Reset Fetcher Buffer
                    sp <= 0;
                    rle_active <= 0;
                    
                    state <= S_FETCH_HEADER;
                    hdr_byte_idx <= 0;
                    
                    arbiter_sel <= 0; // Hand over bus to Fetcher
                    led <= 4'b0010;   // Run LED
                end

                // ------------------------------------------------------------
                // 7. FETCH HEADER (5 Bytes: A, B, C_hi, C_lo, D_len)
                // ------------------------------------------------------------
                S_FETCH_HEADER: begin
                    arbiter_sel <= 0; // Fetcher Mode
                    fetch_byte_req <= 1;
                    
                    if (fetch_byte_valid) begin
                        case (hdr_byte_idx)
                            0: raw_A <= fetch_byte_data;
                            1: raw_B <= fetch_byte_data;
                            2: raw_C[15:8] <= fetch_byte_data; // MSB (Big Endian)
                            3: raw_C[7:0] <= fetch_byte_data;  // LSB
                            4: raw_D_len <= fetch_byte_data;
                        endcase
                        
                        pc <= pc + 1; // Track logical PC
                        
                        if (hdr_byte_idx == 4) begin
                            if (raw_D_len == 0) begin
                                state <= S_TRANSLATE; // No inputs, skip payload
                            end else begin
                                state <= S_FETCH_PAYLOAD;
                                pay_byte_idx <= 0;
                            end
                        end else begin
                            hdr_byte_idx <= hdr_byte_idx + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 8. FETCH PAYLOAD (2 Bytes per Input)
                // ------------------------------------------------------------
                S_FETCH_PAYLOAD: begin
                    fetch_byte_req <= 1;
                    if (fetch_byte_valid) begin
                        pc <= pc + 1;
                        
                        // Assemble 16-bit Input IDs (Big Endian Stream)
                        if (pay_byte_idx[0] == 0) begin
                            temp_hi_byte <= fetch_byte_data;
                        end else begin
                            raw_inputs[pay_byte_idx >> 1] <= {temp_hi_byte, fetch_byte_data};
                        end
                        
                        pay_byte_idx <= pay_byte_idx + 1;
                        
                        // Check if we read (2 * D_len) bytes
                        if (pay_byte_idx == (raw_D_len * 2) - 1) begin
                            state <= S_TRANSLATE;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 9. DYNAMIC TRANSLATION LAYER
                // ------------------------------------------------------------
                S_TRANSLATE: begin
                    // Determine effective instruction (Standard or RLE Template)
                    reg [7:0] eff_A = rle_active ? rle_template_A : raw_A;
                    reg [7:0] eff_B = rle_active ? rle_template_B : raw_B;
                    // Note: 'raw_inputs' are already loaded or persistent from RLE start

                    if (eff_A < 10) begin
                        // System Codes (0-9) do not need translation
                        state <= S_EXEC_DISPATCH;
                    end else begin
                        // 1. Lookup Hardware OpCode Control Word
                        // Bit 5 = Stream Enable, Bits 4:0 = HW_ID
                        reg [7:0] ctrl_word = opcode_lut[eff_A];
                        hw_opcode <= ctrl_word[4:0];
                        alu_stream_en <= ctrl_word[5];

                        // 2. Lookup Input Routing Mask
                        // Bits 7:4 = Src2 Index, Bits 3:0 = Src1 Index
                        reg [7:0] routing = variation_lut[eff_B];
                        routed_src1_idx <= routing[3:0];
                        routed_src2_idx <= routing[7:4];

                        state <= S_EXEC_DISPATCH;
                    end
                end

                // ------------------------------------------------------------
                // 10. EXECUTION DISPATCH
                // ------------------------------------------------------------
                S_EXEC_DISPATCH: begin
                    // Switch based on Effective OpCode
                    reg [7:0] eff_A = rle_active ? rle_template_A : raw_A;

                    case (eff_A)
                        `OP_PATTERN: begin
                            // Hardware Recursion
                            call_stack[sp] <= pc;
                            sp <= sp + 1;
                            pc <= pattern_lut[raw_C[7:0]]; // Jump to Pattern
                            fetch_flush <= 1; // Invalidate Prefetch Buffer
                            
                            // Return to Fetch logic
                            hdr_byte_idx <= 0;
                            state <= S_FETCH_HEADER;
                        end

                        `OP_RETURN: begin
                            // Hardware Return
                            sp <= sp - 1;
                            pc <= call_stack[sp-1];
                            fetch_flush <= 1;
                            
                            hdr_byte_idx <= 0;
                            state <= S_FETCH_HEADER;
                        end

                        `OP_COPY: begin
                            rle_active <= 1;
                            rle_counter <= raw_C; // Count
        
                            rle_template_A <= raw_B;            // Template A arrived in header byte B
                            rle_template_B <= raw_inputs[0][7:0]; // Template B (Variation) in the first word D
                            rle_template_C <= raw_inputs[1];      // Template C (Constant) in the second word D
        
                            state <= S_TRANSLATE;
                        end

                        `OP_DATA_INPUT: begin
                            dma_cnt <= 0;
                            arbiter_sel <= 1; // FSM needs bus
                            state <= S_DMA_INPUT;
                        end

                        `OP_OUTPUT: begin
                            dma_cnt <= 0;
                            arbiter_sel <= 1;
                            state <= S_DMA_OUTPUT;
                        end

                        default: begin
                            // Fundamental Math Operation
                            state <= S_PREP_MATH;
                        end
                    endcase
                end

                // ------------------------------------------------------------
                // 11. DMA: INPUT LOAD (DDR -> ALU BRAM)
                // ------------------------------------------------------------
                S_DMA_INPUT: begin
                    // Read from DDR Input Buffer
                    fsm_mem_addr <= ptr_input + (dma_cnt * 4);
                    fsm_mem_req <= 1;
                    
                    if (mem_valid) begin
                        dma_wr_en <= 1; // Write to ALU BRAM
                        // Data is routed directly: mem_rdata -> alu.dma_wr_data
                        
                        if (dma_cnt == `VECTOR_LEN - 1) begin
                            // Done loading vector
                            hdr_byte_idx <= 0;
                            state <= S_FETCH_HEADER;
                        end else begin
                            dma_cnt <= dma_cnt + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 12. DMA: OUTPUT STORE (ALU BRAM -> DDR)
                // ------------------------------------------------------------
                S_DMA_OUTPUT: begin
                    // Cycle 1: Request read from ALU BRAM
                    dma_rd_req <= 1;
                    
                    // Cycle 2: Write to DDR (Output Buffer)
                    fsm_mem_addr <= ptr_output + (dma_cnt * 4);
                    fsm_mem_wdata <= dma_rd_data;
                    fsm_mem_we <= 1;
                    fsm_mem_req <= 1;
                    
                    if (mem_grant) begin
                        if (dma_cnt == `VECTOR_LEN - 1) begin
                            irq_done <= 1; // Signal completion
                            state <= S_DONE;
                        end else begin
                            dma_cnt <= dma_cnt + 1;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 13. PREPARE MATH EXECUTION
                // ------------------------------------------------------------
                S_PREP_MATH: begin
                    reg [15:0] eff_C = rle_active ? rle_template_C : raw_C;

                    // Configure Weight Streaming if required by OpCode LUT
                    if (alu_stream_en) begin
                        arbiter_sel <= 2; // Hand over bus to Streamer
                        // Calc Address: Base + (Constant_ID * VectorSize * 4 bytes)
                        stream_mem_addr <= ptr_weights + (eff_C * `VECTOR_LEN * 4);
                    end else begin
                        arbiter_sel <= 0; // Idle bus (or Fetcher background fetch)
                    end
                    
                    alu_start <= 1; // Trigger ALU
                    state <= S_RUN_MATH;
                end

                // ------------------------------------------------------------
                // 14. RUN MATH (Wait for ALU)
                // ------------------------------------------------------------
                S_RUN_MATH: begin
                    // Logic for Weight Streaming:
                    // If ALU stream_en is high, Arbiter (Mode 2) asserts mem_req based on ALU ready.
                    // Here we just manage the address pointer.
                    if (alu_stream_en && mem_grant && (arbiter_sel == 2)) begin
                        stream_mem_addr <= stream_mem_addr + 4; // Advance stream pointer
                    end

                    if (alu_done) begin
                        // Logic for RLE Loop
                        if (rle_active) begin
                            if (rle_counter == 1) begin
                                rle_active <= 0; // Loop finished
                                hdr_byte_idx <= 0;
                                state <= S_FETCH_HEADER;
                            end else begin
                                rle_counter <= rle_counter - 1;
                                state <= S_TRANSLATE; // Re-run translation/dispatch
                            end
                        end else begin
                            hdr_byte_idx <= 0;
                            state <= S_FETCH_HEADER;
                        end
                    end
                end

                // ------------------------------------------------------------
                // 15. DONE / SOFT RESET
                // ------------------------------------------------------------
                S_DONE: begin
                    led <= 4'b1111;
                    irq_done <= 1;
                    busy <= 0;
                    
                    // Soft Reset: Return to Boot to accept next command from Host
                    state <= S_BOOT;
                end

            endcase
        end
    end

endmodule