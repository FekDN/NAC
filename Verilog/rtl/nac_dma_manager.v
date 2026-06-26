`include "nac_defs.vh"

// DMA/resource manager for MMAP commands.
//
// nac_mmap_engine decides when a tensor/resource is needed. This block owns the
// physical movement policy: allocate scratchpad banks, issue external-memory
// read/write descriptors, preserve pinned/static weights, and free banks after
// last use. The external bus adapter can translate dma_* requests to AXI4,
// SD/eMMC, SATA, or another storage fabric without changing NAC/MMAP formats.
module nac_dma_manager #(
    parameter TARGETS = 256,
    parameter TARGET_BITS = 8,
    parameter BANKS = 16,
    parameter BANK_BITS = 4,
    parameter ADDR_WIDTH = 64,
    parameter LEN_WIDTH = 32
) (
    input  wire clk,
    input  wire rst,

    input  wire cfg_we,
    input  wire [TARGET_BITS-1:0] cfg_target,
    input  wire [ADDR_WIDTH-1:0] cfg_ext_addr,
    input  wire [LEN_WIDTH-1:0] cfg_size_bytes,
    input  wire cfg_static,

    input  wire mark_dirty_valid,
    input  wire [TARGET_BITS-1:0] mark_dirty_target,

    input  wire preload_valid,
    output wire preload_ready,
    input  wire [15:0] preload_target,
    input  wire free_valid,
    output wire free_ready,
    input  wire [15:0] free_target,
    input  wire save_result_valid,
    output wire save_result_ready,
    input  wire [15:0] save_result_src,
    input  wire [15:0] save_result_target,
    input  wire forward_valid,
    output wire forward_ready,
    input  wire [15:0] forward_src,
    input  wire [15:0] forward_dst,

    output reg  bank_alloc_valid,
    input  wire bank_alloc_ready,
    input  wire [BANK_BITS-1:0] bank_alloc_id,
    output reg  bank_free_valid,
    input  wire bank_free_ready,
    output reg  [BANK_BITS-1:0] bank_free_id,

    output reg  dma_read_valid,
    input  wire dma_read_ready,
    output reg  [ADDR_WIDTH-1:0] dma_read_addr,
    output reg  [LEN_WIDTH-1:0] dma_read_len,
    output reg  [BANK_BITS-1:0] dma_read_bank,
    input  wire dma_read_done,
    input  wire dma_read_error,

    output reg  dma_write_valid,
    input  wire dma_write_ready,
    output reg  [ADDR_WIDTH-1:0] dma_write_addr,
    output reg  [LEN_WIDTH-1:0] dma_write_len,
    output reg  [BANK_BITS-1:0] dma_write_bank,
    input  wire dma_write_done,
    input  wire dma_write_error,

    output reg  desc_update_valid,
    output reg  [TARGET_BITS-1:0] desc_update_target,
    output reg  [BANK_BITS-1:0] desc_update_bank,
    output reg  [LEN_WIDTH-1:0] desc_update_size,
    output reg  desc_invalidate_valid,
    output reg  [TARGET_BITS-1:0] desc_invalidate_target,

    output wire busy,
    output reg  error,
    output reg  [7:0] error_code
);
    localparam S_IDLE       = 4'd0;
    localparam S_ALLOC      = 4'd1;
    localparam S_READ_REQ   = 4'd2;
    localparam S_READ_WAIT  = 4'd3;
    localparam S_FREE_REQ   = 4'd4;
    localparam S_WRITE_REQ  = 4'd5;
    localparam S_WRITE_WAIT = 4'd6;
    localparam S_DONE       = 4'd7;
    localparam S_ERROR      = 4'd8;

    localparam ERR_BAD_TARGET = 8'd1;
    localparam ERR_DMA_READ   = 8'd2;
    localparam ERR_DMA_WRITE  = 8'd3;
    localparam ERR_BANK_FREE  = 8'd4;

    reg [3:0] state;
    reg [TARGET_BITS-1:0] active_target;
    reg [TARGET_BITS-1:0] active_src;
    reg [TARGET_BITS-1:0] active_dst;
    reg [BANK_BITS-1:0] active_bank;
    reg [ADDR_WIDTH-1:0] ext_addr [0:TARGETS-1];
    reg [LEN_WIDTH-1:0] size_bytes [0:TARGETS-1];
    reg [BANK_BITS-1:0] target_bank [0:TARGETS-1];
    reg target_loaded [0:TARGETS-1];
    reg target_static [0:TARGETS-1];
    reg target_dirty [0:TARGETS-1];
    reg [1:0] active_op;

    localparam OP_PRELOAD = 2'd0;
    localparam OP_FREE    = 2'd1;
    localparam OP_SAVE    = 2'd2;
    localparam OP_FORWARD = 2'd3;

    assign busy = (state != S_IDLE);
    assign preload_ready = (state == S_IDLE);
    assign free_ready = (state == S_IDLE) && !preload_valid && !save_result_valid && !forward_valid;
    assign save_result_ready = (state == S_IDLE) && !preload_valid;
    assign forward_ready = (state == S_IDLE) && !preload_valid && !save_result_valid;

    integer i;

    task fail;
        input [7:0] code;
        begin
            error <= 1'b1;
            error_code <= code;
            state <= S_ERROR;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            active_target <= {TARGET_BITS{1'b0}};
            active_src <= {TARGET_BITS{1'b0}};
            active_dst <= {TARGET_BITS{1'b0}};
            active_bank <= {BANK_BITS{1'b0}};
            active_op <= OP_PRELOAD;
            bank_alloc_valid <= 1'b0;
            bank_free_valid <= 1'b0;
            bank_free_id <= {BANK_BITS{1'b0}};
            dma_read_valid <= 1'b0;
            dma_read_addr <= {ADDR_WIDTH{1'b0}};
            dma_read_len <= {LEN_WIDTH{1'b0}};
            dma_read_bank <= {BANK_BITS{1'b0}};
            dma_write_valid <= 1'b0;
            dma_write_addr <= {ADDR_WIDTH{1'b0}};
            dma_write_len <= {LEN_WIDTH{1'b0}};
            dma_write_bank <= {BANK_BITS{1'b0}};
            desc_update_valid <= 1'b0;
            desc_update_target <= {TARGET_BITS{1'b0}};
            desc_update_bank <= {BANK_BITS{1'b0}};
            desc_update_size <= {LEN_WIDTH{1'b0}};
            desc_invalidate_valid <= 1'b0;
            desc_invalidate_target <= {TARGET_BITS{1'b0}};
            error <= 1'b0;
            error_code <= 8'd0;
            for (i = 0; i < TARGETS; i = i + 1) begin
                ext_addr[i] <= {ADDR_WIDTH{1'b0}};
                size_bytes[i] <= {LEN_WIDTH{1'b0}};
                target_bank[i] <= {BANK_BITS{1'b0}};
                target_loaded[i] <= 1'b0;
                target_static[i] <= 1'b0;
                target_dirty[i] <= 1'b0;
            end
        end else begin
            bank_alloc_valid <= 1'b0;
            bank_free_valid <= 1'b0;
            dma_read_valid <= 1'b0;
            dma_write_valid <= 1'b0;
            desc_update_valid <= 1'b0;
            desc_invalidate_valid <= 1'b0;

            if (cfg_we) begin
                ext_addr[cfg_target] <= cfg_ext_addr;
                size_bytes[cfg_target] <= cfg_size_bytes;
                target_static[cfg_target] <= cfg_static;
                target_dirty[cfg_target] <= 1'b0;
            end

            if (mark_dirty_valid && target_loaded[mark_dirty_target])
                target_dirty[mark_dirty_target] <= 1'b1;

            case (state)
                S_IDLE: begin
                    if (preload_valid) begin
                        active_op <= OP_PRELOAD;
                        active_target <= preload_target[TARGET_BITS-1:0];
                        if (size_bytes[preload_target[TARGET_BITS-1:0]] == {LEN_WIDTH{1'b0}}) begin
                            fail(ERR_BAD_TARGET);
                        end else if (target_loaded[preload_target[TARGET_BITS-1:0]]) begin
                            desc_update_valid <= 1'b1;
                            desc_update_target <= preload_target[TARGET_BITS-1:0];
                            desc_update_bank <= target_bank[preload_target[TARGET_BITS-1:0]];
                            desc_update_size <= size_bytes[preload_target[TARGET_BITS-1:0]];
                            state <= S_DONE;
                        end else begin
                            state <= S_ALLOC;
                        end
                    end else if (save_result_valid) begin
                        active_op <= OP_SAVE;
                        active_src <= save_result_src[TARGET_BITS-1:0];
                        active_dst <= save_result_target[TARGET_BITS-1:0];
                        if (!target_loaded[save_result_src[TARGET_BITS-1:0]] ||
                            size_bytes[save_result_target[TARGET_BITS-1:0]] == {LEN_WIDTH{1'b0}})
                            fail(ERR_BAD_TARGET);
                        else begin
                            active_bank <= target_bank[save_result_src[TARGET_BITS-1:0]];
                            state <= S_WRITE_REQ;
                        end
                    end else if (forward_valid) begin
                        active_op <= OP_FORWARD;
                        active_src <= forward_src[TARGET_BITS-1:0];
                        active_dst <= forward_dst[TARGET_BITS-1:0];
                        if (!target_loaded[forward_src[TARGET_BITS-1:0]])
                            fail(ERR_BAD_TARGET);
                        else begin
                            target_loaded[forward_dst[TARGET_BITS-1:0]] <= 1'b1;
                            target_bank[forward_dst[TARGET_BITS-1:0]] <= target_bank[forward_src[TARGET_BITS-1:0]];
                            target_dirty[forward_dst[TARGET_BITS-1:0]] <= target_dirty[forward_src[TARGET_BITS-1:0]];
                            desc_update_valid <= 1'b1;
                            desc_update_target <= forward_dst[TARGET_BITS-1:0];
                            desc_update_bank <= target_bank[forward_src[TARGET_BITS-1:0]];
                            desc_update_size <= size_bytes[forward_src[TARGET_BITS-1:0]];
                            state <= S_DONE;
                        end
                    end else if (free_valid) begin
                        active_op <= OP_FREE;
                        active_target <= free_target[TARGET_BITS-1:0];
                        if (!target_loaded[free_target[TARGET_BITS-1:0]]) begin
                            state <= S_DONE;
                        end else if (target_static[free_target[TARGET_BITS-1:0]]) begin
                            desc_update_valid <= 1'b1;
                            desc_update_target <= free_target[TARGET_BITS-1:0];
                            desc_update_bank <= target_bank[free_target[TARGET_BITS-1:0]];
                            desc_update_size <= size_bytes[free_target[TARGET_BITS-1:0]];
                            state <= S_DONE;
                        end else begin
                            active_bank <= target_bank[free_target[TARGET_BITS-1:0]];
                            state <= S_FREE_REQ;
                        end
                    end
                end

                S_ALLOC: begin
                    bank_alloc_valid <= 1'b1;
                    if (bank_alloc_ready) begin
                        active_bank <= bank_alloc_id;
                        state <= S_READ_REQ;
                    end
                end

                S_READ_REQ: begin
                    dma_read_valid <= 1'b1;
                    dma_read_addr <= ext_addr[active_target];
                    dma_read_len <= size_bytes[active_target];
                    dma_read_bank <= active_bank;
                    if (dma_read_ready)
                        state <= S_READ_WAIT;
                end

                S_READ_WAIT: begin
                    if (dma_read_error) begin
                        fail(ERR_DMA_READ);
                    end else if (dma_read_done) begin
                        target_loaded[active_target] <= 1'b1;
                        target_bank[active_target] <= active_bank;
                        target_dirty[active_target] <= 1'b0;
                        desc_update_valid <= 1'b1;
                        desc_update_target <= active_target;
                        desc_update_bank <= active_bank;
                        desc_update_size <= size_bytes[active_target];
                        state <= S_DONE;
                    end
                end

                S_FREE_REQ: begin
                    bank_free_valid <= 1'b1;
                    bank_free_id <= active_bank;
                    if (bank_free_ready) begin
                        target_loaded[active_target] <= 1'b0;
                        target_dirty[active_target] <= 1'b0;
                        desc_invalidate_valid <= 1'b1;
                        desc_invalidate_target <= active_target;
                        state <= S_DONE;
                    end
                end

                S_WRITE_REQ: begin
                    dma_write_valid <= 1'b1;
                    dma_write_addr <= ext_addr[active_dst];
                    dma_write_len <= size_bytes[active_dst];
                    dma_write_bank <= active_bank;
                    if (dma_write_ready)
                        state <= S_WRITE_WAIT;
                end

                S_WRITE_WAIT: begin
                    if (dma_write_error) begin
                        fail(ERR_DMA_WRITE);
                    end else if (dma_write_done) begin
                        target_loaded[active_dst] <= 1'b1;
                        target_bank[active_dst] <= active_bank;
                        target_dirty[active_dst] <= 1'b0;
                        desc_update_valid <= 1'b1;
                        desc_update_target <= active_dst;
                        desc_update_bank <= active_bank;
                        desc_update_size <= size_bytes[active_dst];
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                S_ERROR: begin
                end

                default: begin
                    fail(8'hff);
                end
            endcase
        end
    end
endmodule
