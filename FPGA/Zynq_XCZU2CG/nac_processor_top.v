`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_Processor_Top #(
    parameter C_M_AXI_ADDR_WIDTH = 40,
    parameter C_M_AXI_DATA_WIDTH = 32, // Рекомендуется 128 для макс. скорости, но 32 совместимо с текущим адаптером
    parameter DMA_BURST_LEN = 16,      // Words per burst
    parameter INT8_MODE = 1            // 1 = Enable Packing logic for INT8
)(
    input  wire        clk,       
    input  wire        rst_n,     

    // ========================================================================
    // AXI4-Lite Slave Interface (Control & Status from Zynq PS)
    // ========================================================================
    // Адреса: 0x00=Control, 0x04=Status, 0x08...=Pointers
    input  wire [5:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [5:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,

    // ========================================================================
    // AXI4 Master Interface (To Zynq S_AXI_HP / DDR4)
    // ========================================================================
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0]                    m_axi_awlen,
    output wire                          m_axi_awvalid,
    input  wire                          m_axi_awready,
    output wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output wire [C_M_AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output wire                          m_axi_wlast,
    output wire                          m_axi_wvalid,
    input  wire                          m_axi_wready,
    input  wire [1:0]                    m_axi_bresp,  
    input  wire                          m_axi_bvalid,
    output wire                          m_axi_bready,
    
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0]                    m_axi_arlen,
    output wire                          m_axi_arvalid,
    input  wire                          m_axi_arready,
    input  wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                    m_axi_rresp,  
    input  wire                          m_axi_rlast,
    input  wire                          m_axi_rvalid,
    output wire                          m_axi_rready,
    
    output wire [2:0] m_axi_awsize, output wire [1:0] m_axi_awburst,
    output wire [2:0] m_axi_arsize, output wire [1:0] m_axi_arburst,
    
    // AXI ID Signals (Required for Zynq HP Ports)
    output wire [5:0] m_axi_awid,
    output wire [5:0] m_axi_arid,
    input  wire [5:0] m_axi_bid, // Ignored
    input  wire [5:0] m_axi_rid, // Ignored

    // ========================================================================
    // Status Interface (External Pins)
    // ========================================================================
    output reg  [3:0]  led,
    output reg         irq_done
);

    // Hardwire unused IDs (Single Master)
    assign m_axi_awid = 6'd0;
    assign m_axi_arid = 6'd0;

    // ========================================================================
    // Internal Signals
    // ========================================================================
    
    // Control Signals from AXI-Lite Slave
    wire        cmd_start;       // Pulse start
    wire [31:0] reg_ptr_registry;
    wire [31:0] reg_ptr_code;
    wire [31:0] reg_ptr_weights;
    wire [31:0] reg_ptr_input;
    wire [31:0] reg_ptr_output;
    wire [31:0] reg_ptr_opmap;
    wire [31:0] reg_ptr_varmap;
    
    // Status Signal to AXI-Lite Slave
    reg [31:0] status_reg; // [0]=Busy, [1]=Done, [2]=Error
    reg        busy_flag;
    reg        error_flag;

    // Memory Arbiter Bus (Connecting Fetcher/FSM/Prefetcher to AXI Master)
    reg  [31:0] mem_addr;
    reg  [7:0]  mem_len;
    reg         mem_req;
    reg         mem_we;
    reg  [31:0] mem_wdata;
    reg         mem_wvalid;
    wire        mem_wready; 
    wire        mem_grant;
    wire        mem_valid;
    wire [31:0] mem_rdata;
    wire        axi_error; 

    // ========================================================================
    // 1. AXI-Lite Slave Instantiation
    // ========================================================================
    NAC_AXILite_Slave #(
        .C_S_AXI_DATA_WIDTH(32), 
        .C_S_AXI_ADDR_WIDTH(6)
    ) axi_ctrl (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(s_axi_awaddr), .S_AXI_AWVALID(s_axi_awvalid), .S_AXI_AWREADY(s_axi_awready),
        .S_AXI_WDATA(s_axi_wdata), .S_AXI_WSTRB(s_axi_wstrb), .S_AXI_WVALID(s_axi_wvalid), .S_AXI_WREADY(s_axi_wready),
        .S_AXI_BRESP(s_axi_bresp), .S_AXI_BVALID(s_axi_bvalid), .S_AXI_BREADY(s_axi_bready),
        .S_AXI_ARADDR(s_axi_araddr), .S_AXI_ARVALID(s_axi_arvalid), .S_AXI_ARREADY(s_axi_arready),
        .S_AXI_RDATA(s_axi_rdata), .S_AXI_RRESP(s_axi_rresp), .S_AXI_RVALID(s_axi_rvalid), .S_AXI_RREADY(s_axi_rready),
        
        // Internal Links
        .slv_start_pulse(cmd_start),
        .slv_ptr_registry(reg_ptr_registry),
        .slv_ptr_code(reg_ptr_code),
        .slv_ptr_weights(reg_ptr_weights),
        .slv_ptr_input(reg_ptr_input),
        .slv_ptr_output(reg_ptr_output),
        .slv_ptr_opmap(reg_ptr_opmap),
        .slv_ptr_varmap(reg_ptr_varmap),
        .slv_status_reg(status_reg)
    );

    // Update Status Register Logic
    always @(posedge clk) begin
        status_reg <= {29'd0, error_flag, irq_done, busy_flag};
    end

    // ========================================================================
    // 2. AXI Master Adapter Instantiation
    // ========================================================================
    NAC_AXI_Master_Adapter #(
        .C_M_AXI_ADDR_WIDTH(C_M_AXI_ADDR_WIDTH), .C_M_AXI_DATA_WIDTH(C_M_AXI_DATA_WIDTH)
    ) axi_bridge (
        .M_AXI_ACLK(clk), .M_AXI_ARESETN(rst_n),
        .sys_addr(mem_addr), .sys_len(mem_len), .sys_req(mem_req), .sys_we(mem_we),
        .sys_wdata(mem_wdata), .sys_wvalid(mem_wvalid), .sys_wready(mem_wready),
        .sys_grant(mem_grant), .sys_valid(mem_valid), .sys_rdata(mem_rdata), 
        .sys_error(axi_error),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst)
    );

    // ========================================================================
    // FSM & Execution Logic Definitions
    // ========================================================================
    
    // DMA Output Buffer (Write Burst FIFO) - Uses LUTRAM
    (* ram_style = "distributed" *) reg [31:0] dma_buf [0:DMA_BURST_LEN-1];
    reg [4:0]  dma_buf_ptr;
    reg [7:0]  dma_burst_target;

    // INT8 Packing Logic
    reg [1:0]  pack_cnt;
    reg [23:0] pack_reg; 
    reg [31:0] dma_wr_data_packed; 

    // Translation Lookaside Buffers (LUTs)
    (* ram_style = "distributed" *) reg [7:0] opcode_lut [0:255]; 
    (* ram_style = "distributed" *) reg [7:0] variation_lut [0:255];
    (* ram_style = "block" *)       reg [31:0] pattern_lut [0:255]; 

    // Registers
    reg [31:0] pc;
    reg [31:0] call_stack [0:`STACK_DEPTH-1];
    reg [4:0]  sp;

    reg [7:0] raw_A, raw_B, raw_D_len;
    reg [15:0] raw_C;
    reg [15:0] raw_inputs [0:`MAX_INPUTS-1];
    reg [2:0] hdr_byte_idx;
    reg [7:0] pay_byte_idx, temp_hi_byte;

    reg rle_active;
    reg [15:0] rle_counter;
    reg [7:0] rle_template_A, rle_template_B;
    reg [15:0] rle_template_C;

    reg [4:0] alu_hw_opcode;
    reg alu_stream_en, alu_start;
    wire alu_done, alu_stream_ready;
    reg [3:0] routed_src1_idx, routed_src2_idx;
    
    // DMA Signals
    reg [15:0] dma_cnt;
    reg        dma_wr_en;
    reg        dma_rd_req;
    wire [31:0] dma_rd_data;
    
    // Streaming / Prefetch Signals
    wire [15:0] eff_C = rle_active ? rle_template_C : raw_C;
    // Combinational address calculation
    wire [31:0] stream_base_addr = reg_ptr_weights + (eff_C * `VECTOR_LEN * 4);
    
    wire [31:0] stream_data_pipe;
    wire        stream_valid_pipe;
    wire [31:0] pf_mem_addr;
    wire [7:0]  pf_mem_len;
    wire        pf_mem_req;

    reg [1:0] arbiter_sel; 
    
    // FSM States
    localparam S_BOOT=0, S_UNPACK_OPMAP=1, S_UNPACK_VARMAP=2, S_LOAD_REGISTRY=3;
    localparam S_IDLE=4, S_FETCH_HEADER=5, S_FETCH_PAYLOAD=6, S_TRANSLATE=7, S_EXEC_DISPATCH=8;
    localparam S_DMA_INPUT_REQ=9, S_DMA_INPUT_WAIT=10; 
    localparam S_DMA_OUTPUT_FILL=11, S_DMA_OUTPUT_FLUSH=12; 
    localparam S_PREP_MATH=13, S_RUN_MATH=14, S_DONE=15, S_ERROR=16;
    
    reg [4:0] state;
    reg [2:0] word_unpack_cnt;
    reg [8:0] lut_load_idx;

    // ========================================================================
    // Helper: 4KB Boundary Calculators
    // ========================================================================
    
    // Input DMA
    wire [31:0] calc_input_addr = reg_ptr_input + (dma_cnt * (INT8_MODE ? 16 : 4)); // INT8 uses 16 bytes per 4-elem word
    wire [11:0] input_pg_off  = calc_input_addr[11:0];
    wire [12:0] input_bytes_rem = 13'h1000 - {1'b0, input_pg_off};
    wire [10:0] input_words_rem = input_bytes_rem[12:2];
    wire [7:0]  input_safe_burst = (input_words_rem < DMA_BURST_LEN) ? input_words_rem[7:0] : DMA_BURST_LEN[7:0];

    // Output DMA
    wire [31:0] calc_output_addr = reg_ptr_output + (dma_cnt * 4);
    wire [11:0] output_pg_off = calc_output_addr[11:0];
    wire [12:0] output_bytes_rem = 13'h1000 - {1'b0, output_pg_off};
    wire [10:0] output_words_rem = output_bytes_rem[12:2];
    wire [7:0]  output_safe_burst = (output_words_rem < DMA_BURST_LEN) ? output_words_rem[7:0] : DMA_BURST_LEN[7:0];

    // ========================================================================
    // Submodules: Fetcher, Prefetcher, ALU
    // ========================================================================

    // 1. Fetcher
    reg fetch_flush, fetch_byte_req;
    wire [7:0] fetch_byte_data;
    wire fetch_byte_valid, fetch_busy, fetch_mem_req;
    wire [31:0] fetch_mem_addr;

    NAC_Byte_Fetcher fetcher (
        .clk(clk), .rst_n(rst_n),
        .flush(fetch_flush), .start_addr(pc), .busy(fetch_busy),
        .byte_req(fetch_byte_req), .byte_data(fetch_byte_data), .byte_valid(fetch_byte_valid),
        .mem_addr(fetch_mem_addr), .mem_req(fetch_mem_req),
        .mem_grant(mem_grant && (arbiter_sel == 2'd0)), .mem_valid(mem_valid), .mem_rdata(mem_rdata)
    );

    // 2. Stream Prefetcher
    NAC_Stream_Prefetcher #(
        .BURST_LEN(DMA_BURST_LEN), .FIFO_DEPTH(64) 
    ) prefetcher (
        .clk(clk), .rst_n(rst_n),
        .enable(alu_stream_en && (state == S_RUN_MATH)), 
        .start_addr(stream_base_addr), 
        .mem_addr(pf_mem_addr), .mem_len(pf_mem_len), .mem_req(pf_mem_req),
        .mem_grant(mem_grant && (arbiter_sel == 2'd2)), .mem_valid(mem_valid), .mem_rdata(mem_rdata),
        .stream_data(stream_data_pipe), .stream_valid(stream_valid_pipe), .stream_ready(alu_stream_ready)
    );

    // 3. ALU Core
    NAC_ALU_Core alu (
        .clk(clk), .rst_n(rst_n),
        .start(alu_start), .hw_opcode(alu_hw_opcode),
        .src1_id(raw_inputs[routed_src1_idx]), .src2_id_bram(raw_inputs[routed_src2_idx]),
        .dst_id(raw_inputs[routed_src1_idx]),
        .done(alu_done),
        .use_stream(alu_stream_en), .stream_data(stream_data_pipe),
        .stream_valid(stream_valid_pipe), .stream_ready(alu_stream_ready),
        .dma_wr_en(dma_wr_en), .dma_wr_data(dma_wr_data_packed), // Use Packed Data
        .dma_wr_offset(dma_cnt), .dma_rd_req(dma_rd_req), .dma_rd_data(dma_rd_data)
    );

    // ========================================================================
    // Memory Arbiter Logic
    // ========================================================================
    reg [31:0] fsm_mem_addr, fsm_mem_wdata;
    reg [7:0]  fsm_mem_len;
    reg fsm_mem_req, fsm_mem_we, fsm_mem_wvalid;

    always @(*) begin
        case (arbiter_sel)
            2'd0: begin // Fetcher
                mem_addr = fetch_mem_addr; mem_len = 0; mem_req = fetch_mem_req; 
                mem_we = 0; mem_wdata = 0; mem_wvalid = 0;
            end
            2'd1: begin // FSM
                mem_addr = fsm_mem_addr; mem_len = fsm_mem_len; 
                mem_req = fsm_mem_req; mem_we = fsm_mem_we; 
                mem_wdata = fsm_mem_wdata; mem_wvalid = fsm_mem_wvalid;
            end
            2'd2: begin // Prefetcher
                mem_addr = pf_mem_addr; mem_len = pf_mem_len; 
                mem_req = pf_mem_req; mem_we = 0; 
                mem_wdata = 0; mem_wvalid = 0;
            end
            default: begin mem_addr=0; mem_len=0; mem_req=0; mem_we=0; mem_wdata=0; mem_wvalid=0; end
        endcase
    end

    // ========================================================================
    // Main FSM
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_BOOT; pc <= 0; sp <= 0; busy_flag <= 0; irq_done <= 0; led <= 0;
            arbiter_sel <= 1; fetch_flush <= 1; fetch_byte_req <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0; rle_active <= 0; error_flag <= 0;
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0; fsm_mem_len <= 0;
            dma_buf_ptr <= 0; dma_cnt <= 0; dma_burst_target <= 0;
            pack_cnt <= 0; pack_reg <= 0;
        end else begin
            // Reset Pulses
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0;
            fetch_byte_req <= 0; fetch_flush <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0;

            if (axi_error) begin
                state <= S_ERROR;
                error_flag <= 1;
            end 
            else begin
                case (state)
                    S_BOOT: begin
                        arbiter_sel <= 1; busy_flag <= 0; led <= 4'b0001;
                        // Wait for start command from AXI-Lite
                        if (cmd_start) begin
                            busy_flag <= 1; irq_done <= 0; error_flag <= 0;
                            // Pointers are already valid on reg_ptr_* wires
                            state <= S_UNPACK_OPMAP; lut_load_idx <= 0; word_unpack_cnt <= 0;
                        end
                    end

                    S_UNPACK_OPMAP: begin
                        // Load Opcode LUT from DDR to distributed RAM
                        if (word_unpack_cnt == 0) begin fsm_mem_addr <= reg_ptr_opmap + lut_load_idx; fsm_mem_req <= 1; end
                        if (mem_valid) begin
                            opcode_lut[lut_load_idx] <= mem_rdata[7:0]; opcode_lut[lut_load_idx+1] <= mem_rdata[15:8];
                            opcode_lut[lut_load_idx+2] <= mem_rdata[23:16]; opcode_lut[lut_load_idx+3] <= mem_rdata[31:24];
                            lut_load_idx <= lut_load_idx + 4;
                            if (lut_load_idx >= 252) begin state <= S_UNPACK_VARMAP; lut_load_idx <= 0; end
                        end
                    end
                    
                    S_UNPACK_VARMAP: begin
                        // Load Variation LUT
                        if (word_unpack_cnt == 0) begin fsm_mem_addr <= reg_ptr_varmap + lut_load_idx; fsm_mem_req <= 1; end
                        if (mem_valid) begin
                            variation_lut[lut_load_idx] <= mem_rdata[7:0]; variation_lut[lut_load_idx+1] <= mem_rdata[15:8];
                            variation_lut[lut_load_idx+2] <= mem_rdata[23:16]; variation_lut[lut_load_idx+3] <= mem_rdata[31:24];
                            lut_load_idx <= lut_load_idx + 4;
                            if (lut_load_idx >= 252) begin state <= S_LOAD_REGISTRY; lut_load_idx <= 0; end
                        end
                    end
                    
                    S_LOAD_REGISTRY: begin
                        // Load Pattern Registry (Jump Table)
                        fsm_mem_addr <= reg_ptr_registry + (lut_load_idx * 4); fsm_mem_req <= 1;
                        if (mem_valid) begin
                            pattern_lut[lut_load_idx] <= mem_rdata; lut_load_idx <= lut_load_idx + 1;
                            if (lut_load_idx == 255) begin state <= S_IDLE; end
                        end
                    end

                    S_IDLE: begin
                        // Init Execution
                        pc <= reg_ptr_code; fetch_flush <= 1; sp <= 0; rle_active <= 0;
                        state <= S_FETCH_HEADER; hdr_byte_idx <= 0; arbiter_sel <= 0; led <= 4'b0010;
                    end

                    S_FETCH_HEADER: begin
                        arbiter_sel <= 0; fetch_byte_req <= 1;
                        if (fetch_byte_valid) begin
                            case (hdr_byte_idx)
                                0: raw_A <= fetch_byte_data; 1: raw_B <= fetch_byte_data;
                                2: raw_C[15:8] <= fetch_byte_data; 3: raw_C[7:0] <= fetch_byte_data;
                                4: raw_D_len <= fetch_byte_data;
                            endcase
                            pc <= pc + 1;
                            if (hdr_byte_idx == 4) begin
                                if (raw_D_len == 0) state <= S_TRANSLATE;
                                else begin state <= S_FETCH_PAYLOAD; pay_byte_idx <= 0; end
                            end else hdr_byte_idx <= hdr_byte_idx + 1;
                        end
                    end

                    S_FETCH_PAYLOAD: begin
                        fetch_byte_req <= 1;
                        if (fetch_byte_valid) begin
                            pc <= pc + 1;
                            if (pay_byte_idx[0] == 0) temp_hi_byte <= fetch_byte_data;
                            else raw_inputs[pay_byte_idx >> 1] <= {temp_hi_byte, fetch_byte_data};
                            pay_byte_idx <= pay_byte_idx + 1;
                            if (pay_byte_idx == (raw_D_len * 2) - 1) state <= S_TRANSLATE;
                        end
                    end

                    S_TRANSLATE: begin
                        reg [7:0] eff_A = rle_active ? rle_template_A : raw_A;
                        reg [7:0] eff_B = rle_active ? rle_template_B : raw_B;
                        if (eff_A < 10) state <= S_EXEC_DISPATCH; // System Command
                        else begin
                            reg [7:0] ctrl = opcode_lut[eff_A];
                            hw_opcode <= ctrl[4:0];
                            alu_stream_en <= ctrl[5];
                            reg [7:0] rout = variation_lut[eff_B];
                            routed_src1_idx <= rout[3:0];
                            routed_src2_idx <= rout[7:4];
                            state <= S_EXEC_DISPATCH;
                        end
                    end

                    S_EXEC_DISPATCH: begin
                        reg [7:0] eff_A = rle_active ? rle_template_A : raw_A;
                        case (eff_A)
                            `OP_PATTERN: begin
                                call_stack[sp] <= pc; sp <= sp + 1; pc <= pattern_lut[raw_C[7:0]];
                                fetch_flush <= 1; hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                            end
                            `OP_RETURN: begin
                                sp <= sp - 1; pc <= call_stack[sp-1]; fetch_flush <= 1;
                                hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                            end
                            `OP_COPY: begin
                                rle_active <= 1; rle_counter <= raw_C;
                                rle_template_A <= raw_B; rle_template_B <= raw_inputs[0][7:0]; rle_template_C <= raw_inputs[1];
                                state <= S_TRANSLATE;
                            end
                            `OP_DATA_INPUT: begin dma_cnt <= 0; arbiter_sel <= 1; pack_cnt <= 0; state <= S_DMA_INPUT_REQ; end
                            `OP_OUTPUT: begin 
                                dma_cnt <= 0; arbiter_sel <= 1; 
                                dma_buf_ptr <= 0; 
                                state <= S_DMA_OUTPUT_FILL; 
                            end
                            default: state <= S_PREP_MATH;
                        endcase
                    end

                    // --- DMA INPUT (BURST READ + PACKING) ---
                    S_DMA_INPUT_REQ: begin
                        // 1. Calculate how many words we still need to read
                        reg [15:0] rem_len;
                        rem_len = (`VECTOR_LEN - dma_cnt) * (INT8_MODE ? 4 : 1);
                        
                        // 2. Determine safe burst length (min of needed, max burst, and 4KB limit)
                        if (rem_len > input_safe_burst) fsm_mem_len <= input_safe_burst - 1;
                        else fsm_mem_len <= rem_len - 1;
                        
                        fsm_mem_addr <= calc_input_addr; 
                        fsm_mem_req <= 1; 
                        state <= S_DMA_INPUT_WAIT;
                    end

                    S_DMA_INPUT_WAIT: begin
                        if (mem_valid) begin
                            if (INT8_MODE) begin
                                // Packing Logic: 4x 8-bit -> 1x 32-bit
                                pack_reg <= {mem_rdata[7:0], pack_reg[23:8]}; 
                                pack_cnt <= pack_cnt + 1;
                                if (pack_cnt == 3) begin
                                    dma_wr_en <= 1;
                                    dma_wr_data_packed <= {mem_rdata[7:0], pack_reg};
                                    dma_cnt <= dma_cnt + 1;
                                    pack_cnt <= 0;
                                end
                            end else begin
                                dma_wr_en <= 1;
                                dma_wr_data_packed <= mem_rdata;
                                dma_cnt <= dma_cnt + 1;
                            end
                        end
                        
                        // RACE CONDITION FIX:
                        // Ensure transaction is fully closed before deciding next step
                        if (!mem_req && !mem_valid && !mem_grant) begin 
                             if (dma_cnt == `VECTOR_LEN) begin
                                 hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                             end else begin
                                 // Need more data
                                 state <= S_DMA_INPUT_REQ;
                             end
                        end
                    end

                    // --- DMA OUTPUT (BURST WRITE) ---
                    S_DMA_OUTPUT_FILL: begin
                        // On first entry (ptr=0), calculate how many words we can safely burst
                        if (dma_buf_ptr == 0) begin
                            dma_burst_target <= output_safe_burst;
                        end

                        dma_rd_req <= 1; 
                        if (dma_buf_ptr > 0) dma_buf[dma_buf_ptr - 1] <= dma_rd_data; 

                        // Stop filling if we hit the safe burst limit OR end of vector
                        if (dma_buf_ptr == dma_burst_target) begin
                            // Initiate Write
                            fsm_mem_addr <= calc_output_addr;
                            fsm_mem_len <= dma_burst_target - 1; 
                            fsm_mem_we <= 1; fsm_mem_req <= 1;
                            dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FLUSH;
                        end 
                        else if (dma_cnt == `VECTOR_LEN) begin
                            // Partial burst at end of vector
                            fsm_mem_addr <= calc_output_addr;
                            fsm_mem_len <= dma_buf_ptr - 1; 
                            fsm_mem_we <= 1; fsm_mem_req <= 1;
                            dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FLUSH;
                        end
                        else begin
                            dma_buf_ptr <= dma_buf_ptr + 1; dma_cnt <= dma_cnt + 1;
                        end
                    end

                    S_DMA_OUTPUT_FLUSH: begin
                        if (mem_wready) begin
                            fsm_mem_wdata <= dma_buf[dma_buf_ptr];
                            fsm_mem_wvalid <= 1; dma_buf_ptr <= dma_buf_ptr + 1;
                        end
                        if (mem_grant && !mem_wvalid) begin // Wait for write complete
                            if (dma_cnt == `VECTOR_LEN) begin irq_done <= 1; state <= S_DONE; end
                            else begin 
                                // Buffer flushed, return to FILL to get next chunk
                                dma_buf_ptr <= 0; 
                                state <= S_DMA_OUTPUT_FILL; 
                            end
                        end
                    end

                    S_PREP_MATH: begin
                        if (alu_stream_en) arbiter_sel <= 2;
                        else arbiter_sel <= 0;
                        alu_start <= 1; state <= S_RUN_MATH;
                    end

                    S_RUN_MATH: begin
                        if (alu_done) begin
                            if (rle_active) begin
                                if (rle_counter == 1) begin rle_active <= 0; hdr_byte_idx <= 0; state <= S_FETCH_HEADER; end
                                else begin rle_counter <= rle_counter - 1; state <= S_TRANSLATE; end
                            end else begin hdr_byte_idx <= 0; state <= S_FETCH_HEADER; end
                        end
                    end

                    S_DONE: begin led <= 4'b1111; busy_flag <= 0; state <= S_BOOT; end
                    S_ERROR: begin led <= 4'b1001; end
                endcase
            end
        end
    end
endmodule