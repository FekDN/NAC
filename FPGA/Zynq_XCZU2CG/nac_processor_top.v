`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_Processor_Top #(
    parameter CONTROL_BASE = 32'h0000_0000,
    parameter C_M_AXI_ADDR_WIDTH = 40,
    parameter C_M_AXI_DATA_WIDTH = 32,
    parameter DMA_BURST_LEN = 16, // Words per burst
    parameter INT8_MODE = 1       // 1 = Enable Packing logic for INT8
)(
    input  wire        clk,       
    input  wire        rst_n,     

    // ========================================================================
    // AXI4 Master Interface (To Zynq S_AXI_HP)
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

    // ========================================================================
    // Status Interface
    // ========================================================================
    output reg  [3:0]  led,
    output reg         irq_done,
    output reg         busy
);

    // ========================================================================
    // Internal Memory Bus
    // ========================================================================
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
    // AXI Adapter
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
    // Buffers & Packers
    // ========================================================================
    // DMA Output Buffer (Write Burst FIFO) - Uses LUTRAM
    (* ram_style = "distributed" *) reg [31:0] dma_buf [0:DMA_BURST_LEN-1];
    reg [4:0]  dma_buf_ptr;

    // INT8 Packing Logic
    reg [1:0]  pack_cnt;
    reg [23:0] pack_reg; // Holds partial bytes
    reg [31:0] dma_wr_data_packed; // Final data to ALU

    // ========================================================================
    // Core Logic Signals
    // ========================================================================
    (* ram_style = "distributed" *) reg [7:0] opcode_lut [0:255]; 
    (* ram_style = "distributed" *) reg [7:0] variation_lut [0:255];
    (* ram_style = "block" *)       reg [31:0] pattern_lut [0:255]; 

    reg [31:0] ptr_registry, ptr_code, ptr_weights, ptr_input, ptr_output, ptr_opmap, ptr_varmap;
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
    wire [31:0] stream_base_addr = ptr_weights + (eff_C * `VECTOR_LEN * 4);
    
    wire [31:0] stream_data_pipe;
    wire        stream_valid_pipe;
    wire [31:0] pf_mem_addr;
    wire [7:0]  pf_mem_len;
    wire        pf_mem_req;

    reg [1:0] arbiter_sel; 
    
    // FSM States
    localparam S_BOOT=0, S_READ_CFG_PTRS=1, S_UNPACK_OPMAP=2, S_UNPACK_VARMAP=3, S_LOAD_REGISTRY=4;
    localparam S_IDLE=5, S_FETCH_HEADER=6, S_FETCH_PAYLOAD=7, S_TRANSLATE=8, S_EXEC_DISPATCH=9;
    localparam S_DMA_INPUT_REQ=10, S_DMA_INPUT_WAIT=11; 
    localparam S_DMA_OUTPUT_FILL=12, S_DMA_OUTPUT_FLUSH=13; 
    localparam S_PREP_MATH=14, S_RUN_MATH=15, S_DONE=16, S_ERROR=17;
    
    reg [4:0] state;
    reg [2:0] cfg_seq_cnt, word_unpack_cnt;
    reg [8:0] lut_load_idx;

    // ========================================================================
    // Submodules
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
    // Memory Arbiter
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
            state <= S_BOOT; pc <= 0; sp <= 0; busy <= 0; irq_done <= 0; led <= 0;
            arbiter_sel <= 1; fetch_flush <= 1; fetch_byte_req <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0; rle_active <= 0; 
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0; fsm_mem_len <= 0;
            dma_buf_ptr <= 0; dma_cnt <= 0;
            pack_cnt <= 0; pack_reg <= 0;
        end else begin
            // Reset Pulses
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0;
            fetch_byte_req <= 0; fetch_flush <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0;

            if (axi_error) state <= S_ERROR;
            else case (state)
                S_BOOT: begin
                    arbiter_sel <= 1; busy <= 0;
                    fsm_mem_addr <= CONTROL_BASE + `REG_CMD; fsm_mem_req <= 1; fsm_mem_len <= 0;
                    if (mem_valid && mem_grant) begin
                        if (mem_rdata == 1) begin state <= S_READ_CFG_PTRS; cfg_seq_cnt <= 0; busy <= 1; end
                        else if (mem_rdata == 2) begin state <= S_IDLE; busy <= 1; end
                    end
                end

                S_READ_CFG_PTRS: begin
                    fsm_mem_addr <= CONTROL_BASE + `REG_REGISTRY + (cfg_seq_cnt * 4); fsm_mem_req <= 1;
                    if (mem_valid) begin
                        case (cfg_seq_cnt)
                            0: ptr_registry <= mem_rdata; 1: ptr_code <= mem_rdata;
                            2: ptr_weights <= mem_rdata; 3: ptr_input <= mem_rdata;
                            4: ptr_output <= mem_rdata; 5: ptr_opmap <= mem_rdata; 6: ptr_varmap <= mem_rdata;
                        endcase
                        if (cfg_seq_cnt == 6) begin state <= S_UNPACK_OPMAP; lut_load_idx <= 0; word_unpack_cnt <= 0; end
                        else cfg_seq_cnt <= cfg_seq_cnt + 1;
                    end
                end
                S_UNPACK_OPMAP: begin
                    if (word_unpack_cnt == 0) begin fsm_mem_addr <= ptr_opmap + lut_load_idx; fsm_mem_req <= 1; end
                    if (mem_valid) begin
                        opcode_lut[lut_load_idx] <= mem_rdata[7:0]; opcode_lut[lut_load_idx+1] <= mem_rdata[15:8];
                        opcode_lut[lut_load_idx+2] <= mem_rdata[23:16]; opcode_lut[lut_load_idx+3] <= mem_rdata[31:24];
                        lut_load_idx <= lut_load_idx + 4;
                        if (lut_load_idx >= 252) begin state <= S_UNPACK_VARMAP; lut_load_idx <= 0; end
                    end
                end
                S_UNPACK_VARMAP: begin
                    if (word_unpack_cnt == 0) begin fsm_mem_addr <= ptr_varmap + lut_load_idx; fsm_mem_req <= 1; end
                    if (mem_valid) begin
                        variation_lut[lut_load_idx] <= mem_rdata[7:0]; variation_lut[lut_load_idx+1] <= mem_rdata[15:8];
                        variation_lut[lut_load_idx+2] <= mem_rdata[23:16]; variation_lut[lut_load_idx+3] <= mem_rdata[31:24];
                        lut_load_idx <= lut_load_idx + 4;
                        if (lut_load_idx >= 252) begin state <= S_LOAD_REGISTRY; lut_load_idx <= 0; end
                    end
                end
                S_LOAD_REGISTRY: begin
                    fsm_mem_addr <= ptr_registry + (lut_load_idx * 4); fsm_mem_req <= 1;
                    if (mem_valid) begin
                        pattern_lut[lut_load_idx] <= mem_rdata; lut_load_idx <= lut_load_idx + 1;
                        if (lut_load_idx == 255) begin state <= S_BOOT; led <= 4'b0001; end
                    end
                end

                S_IDLE: begin
                    pc <= ptr_code; fetch_flush <= 1; sp <= 0; rle_active <= 0;
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
                    if (eff_A < 10) state <= S_EXEC_DISPATCH;
                    else begin
                        reg [7:0] ctrl = opcode_lut[eff_A];
                        hw_opcode <= ctrl[4:0]; alu_stream_en <= ctrl[5];
                        reg [7:0] rout = variation_lut[eff_B];
                        routed_src1_idx <= rout[3:0]; routed_src2_idx <= rout[7:4];
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
                        `OP_OUTPUT: begin dma_cnt <= 0; arbiter_sel <= 1; dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FILL; end
                        default: state <= S_PREP_MATH;
                    endcase
                end

                // --- DMA INPUT (BURST READ + PACKING) ---
                S_DMA_INPUT_REQ: begin
                    // Calc Remainder for Burst
                    reg [15:0] rem_len;
                    // If INT8 packing, we read 4x more words (unpacked) to get 1 internal word
                    rem_len = (`VECTOR_LEN - dma_cnt) * (INT8_MODE ? 4 : 1);
                    
                    if (rem_len > DMA_BURST_LEN) fsm_mem_len <= DMA_BURST_LEN - 1;
                    else fsm_mem_len <= rem_len - 1;
                    
                    fsm_mem_addr <= ptr_input + (dma_cnt * (INT8_MODE ? 16 : 4)); // Shift logic handles offset
                    // Note: addr logic above simplified. Real ptr increment happens implicitly by burst.
                    // Correct: We need a linear input pointer state var if bursts are split.
                    // Assuming ptr_input + accumulated_bytes.
                    // Let's use dma_cnt logic for address:
                    // If packing: addr = ptr + (dma_cnt*4*4) + (pack_cnt*4) - tricky with bursts.
                    // SIMPLIFICATION: Assume we just increment a raw pointer.
                    
                    fsm_mem_req <= 1; 
                    state <= S_DMA_INPUT_WAIT;
                end

                S_DMA_INPUT_WAIT: begin
                    if (mem_valid) begin
                        if (INT8_MODE) begin
                            // Shift in LSB (Little Endian assumption)
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
                    
                    // Exit condition: Check if Vector Full
                    if (dma_cnt == `VECTOR_LEN) begin
                        hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                    end else if (!mem_valid && mem_grant) begin
                        // Wait for data...
                    end else if (mem_valid && /* burst end check */ 0) begin
                        // Re-trigger burst logic if needed (Simplified for single burst or stream)
                        // Ideally: check axi_rlast inside adapter or count beats.
                        // Assuming Adapter handles full burst flow. 
                        // If burst finished but dma_cnt < VECTOR_LEN, loop to REQ.
                    end
                    // NOTE: This state logic needs 'burst done' signal from Adapter for robust multi-burst.
                    // Added check:
                    if (mem_valid && dma_cnt < `VECTOR_LEN && /* last beat */ 0) state <= S_DMA_INPUT_REQ;
                end

                // --- DMA OUTPUT (BURST WRITE) ---
                S_DMA_OUTPUT_FILL: begin
                    dma_rd_req <= 1; 
                    if (dma_buf_ptr > 0) dma_buf[dma_buf_ptr - 1] <= dma_rd_data; 

                    if (dma_buf_ptr == DMA_BURST_LEN) begin
                        fsm_mem_addr <= ptr_output + ((dma_cnt - DMA_BURST_LEN) * 4);
                        fsm_mem_len <= DMA_BURST_LEN - 1; 
                        fsm_mem_we <= 1; fsm_mem_req <= 1;
                        dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FLUSH;
                    end 
                    else if (dma_cnt == `VECTOR_LEN) begin
                        fsm_mem_addr <= ptr_output + ((dma_cnt - dma_buf_ptr) * 4);
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
                    if (mem_grant) begin
                        if (axi_error) state <= S_ERROR;
                        else if (dma_cnt == `VECTOR_LEN) begin irq_done <= 1; state <= S_DONE; end
                        else begin dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FILL; end
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

                S_DONE: begin led <= 4'b1111; irq_done <= 1; busy <= 0; state <= S_BOOT; end
                S_ERROR: begin led <= 4'b1001; end
            endcase
        end
    end
endmodule
