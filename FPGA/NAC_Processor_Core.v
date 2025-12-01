`include "nac_defines.vh"
`include "nac_hw_defines.vh"

module NAC_Processor_Core #(
    parameter C_M_AXI_ADDR_WIDTH = 32, 
    parameter C_M_AXI_DATA_WIDTH = 32,
    parameter DMA_BURST_LEN      = 16, // Words per burst
    parameter INT8_MODE          = 1   // 1 = Enable Packing logic for INT8
)(
    input  wire        clk,
    input  wire        rst_n,

    // ========================================================================
    // Configuration Interface
    // ========================================================================
    input  wire        cmd_start,
    input  wire [31:0] cfg_ptr_code,
    input  wire [31:0] cfg_ptr_weights,
    input  wire [31:0] cfg_ptr_input,
    input  wire [31:0] cfg_ptr_output,
    input  wire [31:0] cfg_ptr_opmap,
    input  wire [31:0] cfg_ptr_varmap,
    input  wire [31:0] cfg_ptr_registry,

    // Status Outputs
    output reg         status_busy,
    output reg         status_done,
    output reg         status_error,
    output reg         irq_done_pulse,

    // ========================================================================
    // AXI4 Master Interface
    // ========================================================================
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0]                    m_axi_awlen,
    output wire [2:0]                    m_axi_awsize,
    output wire [1:0]                    m_axi_awburst,
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
    output wire [2:0]                    m_axi_arsize,
    output wire [1:0]                    m_axi_arburst,
    output wire                          m_axi_arvalid,
    input  wire                          m_axi_arready,
    input  wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                    m_axi_rresp,
    input  wire                          m_axi_rlast,
    input  wire                          m_axi_rvalid,
    output wire                          m_axi_rready
);

    // ========================================================================
    // 1. Internal Signals & State Registers
    // ========================================================================
    
    // Memory Arbiter
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
    wire        sys_error; 

    // DMA Buffers & Controls
    (* ram_style = "distributed" *) reg [31:0] dma_buf [0:DMA_BURST_LEN-1];
    reg [4:0]  dma_buf_ptr;
    reg [7:0]  dma_burst_target;
    reg [1:0]  pack_cnt;
    reg [23:0] pack_reg; 
    reg [31:0] dma_wr_data_packed; 
    reg [15:0] dma_cnt;
    reg        dma_wr_en;
    reg        dma_rd_req;
    wire [31:0] dma_rd_data;

    // LUTs for translation
    (* ram_style = "distributed" *) reg [7:0] opcode_lut [0:255]; 
    (* ram_style = "distributed" *) reg [7:0] variation_lut [0:255];
    (* ram_style = "block" *)       reg [31:0] pattern_lut [0:255]; 

    // Instruction Decoder State
    reg [31:0] pc;
    reg [31:0] call_stack [0:`STACK_DEPTH-1];
    reg [4:0]  sp;
    reg [7:0]  raw_A, raw_B, raw_D_len;
    reg [15:0] raw_C;
    reg [15:0] raw_inputs [0:`MAX_INPUTS-1];
    reg [3:0]  hdr_byte_idx; // 5 bytes for header A,B,C(hi),C(lo),D
    reg [7:0]  pay_byte_idx, temp_hi_byte;

    // RLE (Copy) State
    reg        rle_active;
    reg [15:0] rle_counter;
    reg [7:0]  rle_template_A;
    reg [15:0] rle_template_B, rle_template_C;

    // ALU Interface
    reg [5:0]  alu_hw_opcode; 
    reg        alu_stream_en, alu_start;
    wire       alu_done, alu_stream_ready;
    reg [4:0]  routed_src1_idx, routed_src2_idx, routed_dst_idx;

    // Prefetcher Interface
    wire [15:0] eff_C = rle_active ? rle_template_C : raw_C;
    wire [31:0] stream_base_addr = cfg_ptr_weights + (eff_C * 2048); // Weight slot addressing
    wire [31:0] stream_data_pipe;
    wire        stream_valid_pipe;
    wire [31:0] pf_mem_addr;
    wire [7:0]  pf_mem_len;
    wire        pf_mem_req;

    // Counter for burst reading metadata parameters
    reg [$clog2(`MAX_DIMS):0] meta_param_read_cnt; 

    // Arbiter & FSM
    reg [1:0] arbiter_sel; 
    localparam S_BOOT=0, S_UNPACK_OPMAP=1, S_UNPACK_VARMAP=2, S_LOAD_REGISTRY=3;
    localparam S_IDLE=4, S_FETCH_HEADER=5, S_FETCH_PAYLOAD=6, S_TRANSLATE=7, S_EXEC_DISPATCH=8;
    localparam S_DMA_INPUT_REQ=9, S_DMA_INPUT_WAIT=10; 
    localparam S_DMA_OUTPUT_FILL=11, S_DMA_OUTPUT_FLUSH=12; 
    localparam S_PREP_MATH=13, S_RUN_MATH=14, S_DONE=15, S_ERROR=16;
    localparam S_EXEC_METADATA=17, 
               S_FETCH_META_PARAM_REQ=18, 
               S_FETCH_META_PARAM_WAIT=19;
    
    reg [4:0] state;
    reg [8:0] lut_load_idx;

    // ========================================================================
    // 2. Tensor Metadata Storage
    // ========================================================================
    // For each tensor slot, we store its shape, strides, and number of dimensions
    reg [31:0] tensor_shape [0:`TENSOR_SLOTS-1][0:`MAX_DIMS-1];
    reg [31:0] tensor_strides [0:`TENSOR_SLOTS-1][0:`MAX_DIMS-1];
    reg [3:0]  tensor_dims [0:`TENSOR_SLOTS-1];
    
    // Intermediate buffer for reading metadata parameters from DDR
    reg [31:0] meta_param_buf [0:`MAX_DIMS-1];
    wire [31:0] alu_dst_shape_out [0:`MAX_DIMS-1];
    wire [31:0] alu_dst_strides_out [0:`MAX_DIMS-1];
    wire [3:0]  alu_dst_dims_out;

    // ========================================================================
    // 3. Helpers and Submodule Instantiations
    // ========================================================================
    
    // 4KB Boundary Calculators
    wire [31:0] calc_input_addr = cfg_ptr_input + (dma_cnt * (INT8_MODE ? 16 : 4));
    wire [11:0] input_pg_off  = calc_input_addr[11:0];
    wire [12:0] input_bytes_rem = 13'h1000 - {1'b0, input_pg_off};
    wire [10:0] input_words_rem = input_bytes_rem[12:2];
    wire [7:0]  input_safe_burst = (input_words_rem < DMA_BURST_LEN) ? input_words_rem[7:0] : DMA_BURST_LEN[7:0];

    wire [31:0] calc_output_addr = cfg_ptr_output + (dma_cnt * 4);
    wire [11:0] output_pg_off = calc_output_addr[11:0];
    wire [12:0] output_bytes_rem = 13'h1000 - {1'b0, output_pg_off};
    wire [10:0] output_words_rem = output_bytes_rem[12:2];
    wire [7:0]  output_safe_burst = (output_words_rem < DMA_BURST_LEN) ? output_words_rem[7:0] : DMA_BURST_LEN[7:0];

    // AXI Master Adapter Instantiation
    NAC_AXI_Master_Adapter #(
        .C_M_AXI_ADDR_WIDTH(C_M_AXI_ADDR_WIDTH), .C_M_AXI_DATA_WIDTH(C_M_AXI_DATA_WIDTH)
    ) axi_bridge (
        .M_AXI_ACLK(clk), .M_AXI_ARESETN(rst_n),
        .sys_addr(mem_addr), .sys_len(mem_len), .sys_req(mem_req), .sys_we(mem_we),
        .sys_wdata(mem_wdata), .sys_wvalid(mem_wvalid), .sys_wready(mem_wready),
        .sys_grant(mem_grant), .sys_valid(mem_valid), .sys_rdata(mem_rdata), 
        .sys_error(sys_error),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst)
    );

    // Fetcher
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

    // Stream Prefetcher
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

    // ALU Core Instantiation (with new metadata interface)
    NAC_ALU_Core alu (
        .clk(clk), .rst_n(rst_n),
        .start(alu_start), .hw_opcode(alu_hw_opcode),
        .src1_id(raw_inputs[routed_src1_idx]), .src2_id_bram(raw_inputs[routed_src2_idx]),
        .dst_id(raw_inputs[routed_dst_idx]), .done(alu_done),
        .use_stream(alu_stream_en), .stream_data(stream_data_pipe),
        .stream_valid(stream_valid_pipe), .stream_ready(alu_stream_ready),
        .dma_wr_en(dma_wr_en), .dma_wr_data(dma_wr_data_packed),
        .dma_wr_offset(dma_cnt), .dma_rd_req(dma_rd_req), .dma_rd_data(dma_rd_data),
        
        // Pass metadata to ALU
        .src1_shape(tensor_shape[raw_inputs[routed_src1_idx] & `SLOT_MASK]),
        .src1_strides(tensor_strides[raw_inputs[routed_src1_idx] & `SLOT_MASK]),
        .src1_dims(tensor_dims[raw_inputs[routed_src1_idx] & `SLOT_MASK]),
        .src2_shape(tensor_shape[raw_inputs[routed_src2_idx] & `SLOT_MASK]),
        .src2_strides(tensor_strides[raw_inputs[routed_src2_idx] & `SLOT_MASK]),
        .src2_dims(tensor_dims[raw_inputs[routed_src2_idx] & `SLOT_MASK]),
        
        // Receive updated metadata from ALU
        .dst_shape_out(alu_dst_shape_out),
        .dst_strides_out(alu_dst_strides_out),
        .dst_dims_out(alu_dst_dims_out)
    );

    // Memory Arbiter
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
    // 4. Main FSM
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_BOOT; pc <= 0; sp <= 0; 
            status_busy <= 0; status_done <= 0; status_error <= 0; irq_done_pulse <= 0;
            arbiter_sel <= 1; fetch_flush <= 1; fetch_byte_req <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0; rle_active <= 0;
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0; fsm_mem_len <= 0;
            dma_buf_ptr <= 0; dma_cnt <= 0; dma_burst_target <= 0;
            pack_cnt <= 0; pack_reg <= 0;
            meta_param_read_cnt <= 0;

            // Initialize metadata for all tensor slots to a default 1D vector
            integer i, j;
            for (i=0; i<`TENSOR_SLOTS; i=i+1) begin
                tensor_dims[i] <= 1;
                tensor_shape[i][0] <= `VECTOR_LEN;
                tensor_strides[i][0] <= 1;
                for (j=1; j<`MAX_DIMS; j=j+1) begin
                    tensor_shape[i][j] <= 0;
                    tensor_strides[i][j] <= 0;
                end
            end
        end else begin
            // Reset pulses on every cycle
            fsm_mem_req <= 0; fsm_mem_we <= 0; fsm_mem_wvalid <= 0;
            fetch_byte_req <= 0; fetch_flush <= 0; alu_start <= 0;
            dma_wr_en <= 0; dma_rd_req <= 0;
            irq_done_pulse <= 0;

            if (sys_error) begin 
                state <= S_ERROR; 
                status_error <= 1; 
            end else begin
                case (state)
                    S_BOOT: begin
                        arbiter_sel <= 1; status_busy <= 0; status_done <= 0;
                        if (cmd_start) begin
                            status_busy <= 1; status_error <= 0;
                            state <= S_UNPACK_OPMAP; lut_load_idx <= 0;
                        end
                    end

                    S_UNPACK_OPMAP: begin
                        fsm_mem_addr <= cfg_ptr_opmap + lut_load_idx; fsm_mem_req <= 1;
                        if (mem_valid) begin
                            opcode_lut[lut_load_idx]   <= mem_rdata[7:0];   opcode_lut[lut_load_idx+1] <= mem_rdata[15:8];
                            opcode_lut[lut_load_idx+2] <= mem_rdata[23:16]; opcode_lut[lut_load_idx+3] <= mem_rdata[31:24];
                            lut_load_idx <= lut_load_idx + 4;
                            if (lut_load_idx >= 252) begin state <= S_UNPACK_VARMAP; lut_load_idx <= 0; end
                        end
                    end
                    
                    S_UNPACK_VARMAP: begin
                        fsm_mem_addr <= cfg_ptr_varmap + lut_load_idx; fsm_mem_req <= 1;
                        if (mem_valid) begin
                            variation_lut[lut_load_idx]   <= mem_rdata[7:0];   variation_lut[lut_load_idx+1] <= mem_rdata[15:8];
                            variation_lut[lut_load_idx+2] <= mem_rdata[23:16]; variation_lut[lut_load_idx+3] <= mem_rdata[31:24];
                            lut_load_idx <= lut_load_idx + 4;
                            if (lut_load_idx >= 252) begin state <= S_LOAD_REGISTRY; lut_load_idx <= 0; end
                        end
                    end
                    
                    S_LOAD_REGISTRY: begin
                        fsm_mem_addr <= cfg_ptr_registry + (lut_load_idx * 4); fsm_mem_req <= 1;
                        if (mem_valid) begin
                            pattern_lut[lut_load_idx] <= mem_rdata; 
                            lut_load_idx <= lut_load_idx + 1;
                            if (lut_load_idx == 255) begin state <= S_IDLE; end
                        end
                    end

                    S_IDLE: begin
                        pc <= cfg_ptr_code; fetch_flush <= 1; sp <= 0; rle_active <= 0;
                        state <= S_FETCH_HEADER; hdr_byte_idx <= 0; arbiter_sel <= 0;
                    end

                    S_FETCH_HEADER: begin
                        arbiter_sel <= 0; fetch_byte_req <= 1;
                        if (fetch_byte_valid) begin
                            case (hdr_byte_idx)
                                0: raw_A <= fetch_byte_data;
                                1: raw_B <= fetch_byte_data;
                                2: raw_C[15:8] <= fetch_byte_data;
                                3: raw_C[7:0] <= fetch_byte_data;
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
                        reg [15:0] eff_B_16bit = rle_active ? rle_template_B : {8'd0, raw_B};
                        
                        if (eff_A < 10) begin
                            state <= S_EXEC_DISPATCH; // System Command
                        end else begin
                            reg [7:0] ctrl = opcode_lut[eff_A];
                            alu_hw_opcode <= ctrl[5:0];
                            alu_stream_en <= ctrl[6];
                            
                            reg [7:0] rout = variation_lut[eff_B_16bit[7:0]];
                            routed_src1_idx <= rout[3:0];
                            routed_src2_idx <= rout[7:4];
                            routed_dst_idx  <= rout[3:0]; // Default: dst = src1
                            
                            if (alu_hw_opcode >= `H_VIEW && alu_hw_opcode <= `H_SPLIT) begin
                                if (eff_C != 0) begin
                                    state <= S_FETCH_META_PARAM_REQ; // Needs params from DDR, start burst read
                                end else begin
                                    state <= S_EXEC_METADATA; // No params needed (e.g., simple transpose)
                                end
                            end else begin
                                state <= S_EXEC_DISPATCH; // Normal math operation
                            end
                        end
                    end

                    // =================================================================
                    // === NEW BLOCK FOR BURST READING METADATA PARAMETERS ===
                    // =================================================================
                    S_FETCH_META_PARAM_REQ: begin
                        // State 1: Request a read burst
                        arbiter_sel <= 1; // FSM requests the bus
                        fsm_mem_we <= 0;  // We are reading

                        // Set the address based on the constant slot C
                        fsm_mem_addr <= cfg_ptr_weights + (eff_C * 2048);
                        
                        // Request to read `MAX_DIMS` 32-bit words.
                        // AXI len is (number of transfers - 1).
                        fsm_mem_len <= `MAX_DIMS - 1;
                        
                        // Assert the request for one cycle
                        fsm_mem_req <= 1;
                        
                        // Reset the received word counter
                        meta_param_read_cnt <= 0;
                        
                        // Transition to the waiting state
                        state <= S_FETCH_META_PARAM_WAIT;
                    end

                    S_FETCH_META_PARAM_WAIT: begin
                        // State 2: Wait for and save the data
                        
                        // Keep the arbiter for ourselves until we have all the data
                        arbiter_sel <= 1; 

                        // If valid data has arrived...
                        if (mem_valid) begin
                            // ...save it to the buffer at the current counter index
                            meta_param_buf[meta_param_read_cnt] <= mem_rdata;
                            
                            // Increment the received word counter
                            meta_param_read_cnt <= meta_param_read_cnt + 1;
                            
                            // Check if this was the last word in the burst
                            if (meta_param_read_cnt == (`MAX_DIMS - 1)) begin
                                // Yes, all parameters are loaded. Proceed to execution.
                                state <= S_EXEC_METADATA;
                            end
                        end
                    end
                    // =================================================================

                    S_EXEC_METADATA: begin
                        // This block is now guaranteed to have all necessary parameters in meta_param_buf
                        reg [4:0] dst_slot = raw_inputs[routed_dst_idx] & `SLOT_MASK;
                        reg [4:0] src_slot = raw_inputs[routed_src1_idx] & `SLOT_MASK;
                        integer i, j; // Iterators for loops

                        // By default, copy metadata to preserve untouched parts
                        tensor_dims[dst_slot] <= tensor_dims[src_slot];
                        for(j=0; j<`MAX_DIMS; j=j+1) begin
                            tensor_shape[dst_slot][j] <= tensor_shape[src_slot][j];
                            tensor_strides[dst_slot][j] <= tensor_strides[src_slot][j];
                        end

                        // Apply transformation based on the opcode
                        case (alu_hw_opcode)
                            
                            `H_VIEW: begin
                                reg [3:0] new_dims;
                                reg [3:0] inferred_dim_idx;
                                reg [31:0] total_elements, product_of_new_dims;
                                total_elements = 1;
                                for (i = 0; i < tensor_dims[src_slot]; i = i + 1) begin
                                    total_elements = total_elements * tensor_shape[src_slot][i];
                                end
                                new_dims = 0;
                                product_of_new_dims = 1;
                                inferred_dim_idx = 0;
                                for (i = 0; i < `MAX_DIMS; i = i + 1) begin
                                    if (meta_param_buf[i] != 32'hFFFFFFFF) begin
                                        tensor_shape[dst_slot][i] <= meta_param_buf[i];
                                        if (meta_param_buf[i] != 0) begin
                                            product_of_new_dims = product_of_new_dims * meta_param_buf[i];
                                            new_dims = new_dims + 1;
                                        end
                                    end else begin
                                        inferred_dim_idx = i;
                                        tensor_shape[dst_slot][i] <= 0;
                                    end
                                end
                                tensor_dims[dst_slot] <= new_dims;
                                if (product_of_new_dims > 0) begin
                                    tensor_shape[dst_slot][inferred_dim_idx] <= total_elements / product_of_new_dims;
                                end
                                tensor_strides[dst_slot][new_dims-1] <= 1;
                                for (i = new_dims - 2; i >= 0; i = i - 1) begin
                                    tensor_strides[dst_slot][i] <= tensor_strides[dst_slot][i+1] * tensor_shape[dst_slot][i+1];
                                end
                            end

                            `H_TRANSPOSE: begin
                                reg [31:0] dim0, dim1, temp_s;
                                dim0 = meta_param_buf[0];
                                dim1 = meta_param_buf[1];
                                temp_s = tensor_shape[dst_slot][dim0];
                                tensor_shape[dst_slot][dim0] <= tensor_shape[dst_slot][dim1];
                                tensor_shape[dst_slot][dim1] <= temp_s;
                                temp_s = tensor_strides[dst_slot][dim0];
                                tensor_strides[dst_slot][dim0] <= tensor_strides[dst_slot][dim1];
                                tensor_strides[dst_slot][dim1] <= temp_s;
                            end

                            `H_PERMUTE: begin
                                reg [31:0] new_shape [0:`MAX_DIMS-1];
                                reg [31:0] new_strides [0:`MAX_DIMS-1];
                                for (i = 0; i < tensor_dims[src_slot]; i = i + 1) begin
                                    new_shape[i] <= tensor_shape[src_slot][meta_param_buf[i]];
                                    new_strides[i] <= tensor_strides[src_slot][meta_param_buf[i]];
                                end
                                for (i = 0; i < tensor_dims[src_slot]; i = i + 1) begin
                                    tensor_shape[dst_slot][i] <= new_shape[i];
                                    tensor_strides[dst_slot][i] <= new_strides[i];
                                end
                            end

                            `H_UNSQUEEZE: begin
                                reg [31:0] dim_to_unsqueeze;
                                dim_to_unsqueeze = meta_param_buf[0];
                                tensor_dims[dst_slot] <= tensor_dims[src_slot] + 1;
                                for (i = tensor_dims[src_slot]; i > dim_to_unsqueeze; i = i - 1) begin
                                    tensor_shape[dst_slot][i] <= tensor_shape[src_slot][i-1];
                                    tensor_strides[dst_slot][i] <= tensor_strides[src_slot][i-1];
                                end
                                tensor_shape[dst_slot][dim_to_unsqueeze] <= 1;
                                if (dim_to_unsqueeze == tensor_dims[src_slot]) begin
                                    tensor_strides[dst_slot][dim_to_unsqueeze] <= 1;
                                end else begin
                                    tensor_strides[dst_slot][dim_to_unsqueeze] <= tensor_strides[src_slot][dim_to_unsqueeze];
                                end
                            end
                            
                            `H_CAT, `H_SPLIT: begin
                                // NOP: Metadata is already copied by default.
                            end

                        endcase
                        
                        // After executing, move to the next instruction
                        hdr_byte_idx <= 0;
                        state <= S_FETCH_HEADER;
                        arbiter_sel <= 0; // Return arbiter control to the Fetcher
                    end
                    
                    S_EXEC_DISPATCH: begin
                        reg [7:0] eff_A = rle_active ? rle_template_A : raw_A;
                        case (eff_A)
                            `OP_PATTERN: begin
                                call_stack[sp] <= pc; sp <= sp + 1; pc <= pattern_lut[raw_B];
                                fetch_flush <= 1; hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                            end
                            `OP_RETURN: begin
                                sp <= sp - 1; pc <= call_stack[sp-1]; fetch_flush <= 1;
                                hdr_byte_idx <= 0; state <= S_FETCH_HEADER;
                            end
                            `OP_COPY: begin
                                rle_active <= 1; rle_counter <= raw_C;
                                rle_template_A <= raw_B; rle_template_B <= raw_inputs[0];
                                rle_template_C <= raw_inputs[1];
                                state <= S_TRANSLATE;
                            end
                            `OP_DATA_INPUT: begin 
                                dma_cnt <= 0; arbiter_sel <= 1; pack_cnt <= 0;
                                tensor_dims[0]    <= 4;
                                tensor_shape[0][0]  <= 1;       tensor_strides[0][0] <= 32*32*3;
                                tensor_shape[0][1]  <= 3;       tensor_strides[0][1] <= 32*32;
                                tensor_shape[0][2]  <= 32;      tensor_strides[0][2] <= 32;
                                tensor_shape[0][3]  <= 32;      tensor_strides[0][3] <= 1;
                                state <= S_DMA_INPUT_REQ; 
                            end
                            `OP_OUTPUT: begin 
                                dma_cnt <= 0; arbiter_sel <= 1; dma_buf_ptr <= 0; 
                                state <= S_DMA_OUTPUT_FILL; 
                            end
                            default: state <= S_PREP_MATH;
                        endcase
                    end

                    S_DMA_INPUT_REQ: begin
                        reg [15:0] rem_len;
                        rem_len = (`VECTOR_LEN - dma_cnt) * (INT8_MODE ? 4 : 1);
                        
                        if (rem_len > input_safe_burst) fsm_mem_len <= input_safe_burst - 1;
                        else fsm_mem_len <= rem_len - 1;
                        
                        fsm_mem_addr <= calc_input_addr; fsm_mem_req <= 1; state <= S_DMA_INPUT_WAIT;
                    end

                    S_DMA_INPUT_WAIT: begin
                        if (mem_valid) begin
                            if (INT8_MODE) begin
                                pack_reg <= {mem_rdata[7:0], pack_reg[23:8]}; pack_cnt <= pack_cnt + 1;
                                if (pack_cnt == 3) begin
                                    dma_wr_en <= 1; dma_wr_data_packed <= {mem_rdata[7:0], pack_reg};
                                    dma_cnt <= dma_cnt + 1; pack_cnt <= 0;
                                end
                            end else begin
                                dma_wr_en <= 1; dma_wr_data_packed <= mem_rdata;
                                dma_cnt <= dma_cnt + 1;
                            end
                        end
                        
                        if (!mem_req && !mem_grant) begin 
                             if (dma_cnt == `VECTOR_LEN) begin
                                 hdr_byte_idx <= 0; state <= S_FETCH_HEADER; arbiter_sel <= 0;
                             end else begin
                                 state <= S_DMA_INPUT_REQ;
                             end
                        end
                    end

                    S_DMA_OUTPUT_FILL: begin
                        if (dma_buf_ptr == 0) begin
                            dma_burst_target <= output_safe_burst;
                        end

                        dma_rd_req <= 1; 
                        if (dma_buf_ptr > 0) dma_buf[dma_buf_ptr - 1] <= dma_rd_data; 

                        if (dma_buf_ptr == dma_burst_target) begin
                            fsm_mem_addr <= calc_output_addr; fsm_mem_len <= dma_burst_target - 1; 
                            fsm_mem_we <= 1; fsm_mem_req <= 1;
                            dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FLUSH;
                        end 
                        else if (dma_cnt == `VECTOR_LEN) begin
                            fsm_mem_addr <= calc_output_addr; fsm_mem_len <= dma_buf_ptr - 1; 
                            fsm_mem_we <= 1; fsm_mem_req <= 1;
                            dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FLUSH;
                        end else begin
                            dma_buf_ptr <= dma_buf_ptr + 1; dma_cnt <= dma_cnt + 1;
                        end
                    end

                    S_DMA_OUTPUT_FLUSH: begin
                        if (mem_wready) begin
                            fsm_mem_wdata <= dma_buf[dma_buf_ptr];
                            fsm_mem_wvalid <= 1; dma_buf_ptr <= dma_buf_ptr + 1;
                        end
                        if (mem_grant && !mem_wvalid) begin // Write is complete
                            if (dma_cnt == `VECTOR_LEN) begin 
                                irq_done_pulse <= 1; status_done <= 1; state <= S_DONE; 
                            end else begin 
                                dma_buf_ptr <= 0; state <= S_DMA_OUTPUT_FILL; 
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
                            reg [4:0] dst_slot = raw_inputs[routed_dst_idx] & `SLOT_MASK;
                            tensor_dims[dst_slot] <= alu_dst_dims_out;
                            integer m;
                            for(m=0; m<`MAX_DIMS; m=m+1) begin
                                tensor_shape[dst_slot][m] <= alu_dst_shape_out[m];
                                tensor_strides[dst_slot][m] <= alu_dst_strides_out[m];
                            end

                            if (rle_active) begin
                                if (rle_counter == 1) begin rle_active <= 0; hdr_byte_idx <= 0; state <= S_FETCH_HEADER; end
                                else begin rle_counter <= rle_counter - 1; state <= S_TRANSLATE; end
                            end else begin hdr_byte_idx <= 0; state <= S_FETCH_HEADER; end
                        end
                    end

                    S_DONE: begin 
                        status_busy <= 0; status_done <= 1; state <= S_BOOT; 
                    end
                    S_ERROR: begin 
                        status_error <= 1; // Latch error state
                    end
                endcase
            end
        end
    end

endmodule