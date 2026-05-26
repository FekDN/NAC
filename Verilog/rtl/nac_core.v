`include "nac_defs.vh"

module nac_core #(
    parameter IDX_WIDTH = 10,
    parameter MAX_ARITY = 8,
    parameter MAX_CONSTS = 8,
    parameter DESC_WIDTH = 64,
    parameter PERM_ENTRIES = 256,
    parameter MAX_MMAP_CMDS_PER_TICK = 256,
    parameter MMAP_SLOT_WIDTH = 8
) (
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [15:0] num_outputs,

    // OPS byte stream. The loader feeds the OPS payload after the 'OPS ' tag
    // and instruction-count word have been consumed.
    input  wire ops_byte_valid,
    output wire ops_byte_ready,
    input  wire [7:0] ops_byte,

    // PERM table load: id -> arity + needs_const marker derived from PERM.
    input  wire perm_cfg_we,
    input  wire [7:0] perm_cfg_id,
    input  wire [3:0] perm_cfg_arity,
    input  wire perm_cfg_needs_consts,

    // Operation table load. Standard A ids are handled internally. Custom A ids
    // must be loaded from CMAP-derived standard metadata before start.
    input  wire op_cfg_we,
    input  wire [7:0] op_cfg_id,
    input  wire [7:0] op_cfg_kernel_class,

    // MMAP schedule load.
    input  wire mmap_cfg_we,
    input  wire [IDX_WIDTH-1:0] mmap_cfg_tick,
    input  wire [MMAP_SLOT_WIDTH-1:0] mmap_cfg_slot,
    input  wire mmap_cfg_valid,
    input  wire [7:0] mmap_cfg_action,
    input  wire [15:0] mmap_cfg_target,

    // Kernel command sideband. A tensor datapath or microcoded kernel controller
    // consumes this command and returns a result descriptor.
    output wire kernel_start,
    output wire [7:0] kernel_op_a,
    output wire [7:0] kernel_op_b,
    output wire [7:0] kernel_class,
    output wire [7:0] dsp_mode,
    output wire [IDX_WIDTH-1:0] kernel_instr_idx,
    output wire [3:0] kernel_c_count,
    output wire [MAX_CONSTS*16-1:0] kernel_c_flat,
    output wire [3:0] kernel_d_count,
    output wire [MAX_ARITY*16-1:0] kernel_d_flat,
    input  wire kernel_done,
    input  wire [DESC_WIDTH-1:0] kernel_result_desc,

    // Descriptor-table read ports for the tensor datapath. For each D offset,
    // the datapath computes ancestor = kernel_instr_idx + signed(D) and reads
    // the descriptor here.
    input  wire result_rd0_en,
    input  wire [IDX_WIDTH-1:0] result_rd0_idx,
    output wire result_rd0_valid,
    output wire [DESC_WIDTH-1:0] result_rd0_desc,
    input  wire result_rd1_en,
    input  wire [IDX_WIDTH-1:0] result_rd1_idx,
    output wire result_rd1_valid,
    output wire [DESC_WIDTH-1:0] result_rd1_desc,

    // Input/parameter/state descriptor provider.
    output wire input_req_valid,
    output wire [7:0] input_kind,
    output wire [15:0] input_id,
    input  wire input_req_ready,
    input  wire input_desc_valid,
    input  wire [DESC_WIDTH-1:0] input_desc,

    // MMAP command pulses for the memory subsystem.
    output wire preload_valid,
    input  wire preload_ready,
    output wire [15:0] preload_target,
    output wire free_valid,
    input  wire free_ready,
    output wire [15:0] free_target,
    output wire save_result_valid,
    input  wire save_result_ready,
    output wire [15:0] save_result_src,
    output wire [15:0] save_result_target,
    output wire forward_valid,
    input  wire forward_ready,
    output wire [15:0] forward_src,
    output wire [15:0] forward_dst,

    output wire busy,
    output wire done,
    output wire error
);
    reg core_started;
    reg orch_finished;

    wire [7:0] perm_lookup_id;
    wire [3:0] perm_arity;
    wire perm_needs_consts;
    wire perm_present;

    nac_perm_table #(
        .PERM_ENTRIES(PERM_ENTRIES),
        .ARITY_BITS(4)
    ) perm_table (
        .clk(clk),
        .cfg_we(perm_cfg_we),
        .cfg_id(perm_cfg_id),
        .cfg_arity(perm_cfg_arity),
        .cfg_needs_consts(perm_cfg_needs_consts),
        .lookup_id(perm_lookup_id),
        .lookup_arity(perm_arity),
        .lookup_needs_consts(perm_needs_consts),
        .lookup_present(perm_present)
    );

    wire instr_valid;
    wire instr_ready;
    wire [15:0] instr_index;
    wire [7:0] instr_a;
    wire [7:0] instr_b;
    wire [3:0] c_count;
    wire [MAX_CONSTS*16-1:0] c_flat;
    wire [3:0] d_count;
    wire [MAX_ARITY*16-1:0] d_flat;
    wire decoder_busy;
    wire decoder_error;
    wire [7:0] op_lookup_kernel_class;

    nac_op_table op_table (
        .clk(clk),
        .cfg_we(op_cfg_we),
        .cfg_op_id(op_cfg_id),
        .cfg_kernel_class(op_cfg_kernel_class),
        .lookup_op_id(instr_a),
        .lookup_kernel_class(op_lookup_kernel_class)
    );

    nac_abcd_decoder #(
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS),
        .ARITY_BITS(4)
    ) decoder (
        .clk(clk),
        .rst(rst),
        .start(start),
        .num_outputs(num_outputs),
        .ops_byte_valid(ops_byte_valid),
        .ops_byte_ready(ops_byte_ready),
        .ops_byte(ops_byte),
        .perm_lookup_id(perm_lookup_id),
        .perm_arity(perm_arity),
        .perm_needs_consts(perm_needs_consts),
        .perm_present(perm_present),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .instr_index(instr_index),
        .instr_a(instr_a),
        .instr_b(instr_b),
        .c_count(c_count),
        .c_flat(c_flat),
        .d_count(d_count),
        .d_flat(d_flat),
        .busy(decoder_busy),
        .error(decoder_error)
    );

    wire result_wr_valid;
    wire [IDX_WIDTH-1:0] result_wr_idx;
    wire [DESC_WIDTH-1:0] result_wr_desc;
    wire tick_commit_valid;
    wire [IDX_WIDTH-1:0] tick_commit_id;
    wire orch_done;
    wire orch_error;

    always @(posedge clk) begin
        if (rst) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
        end else if (start) begin
            core_started <= 1'b1;
            orch_finished <= 1'b0;
        end else if (decoder_error || orch_error || mmap_error) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
        end else if (orch_done) begin
            core_started <= 1'b0;
            orch_finished <= 1'b1;
        end else if (orch_finished && !mmap_busy) begin
            orch_finished <= 1'b0;
        end
    end

    nac_orch_fsm #(
        .IDX_WIDTH(IDX_WIDTH),
        .DESC_WIDTH(DESC_WIDTH),
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS)
    ) orch (
        .clk(clk),
        .rst(rst),
        .start(start),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .instr_index(instr_index),
        .instr_a(instr_a),
        .instr_b(instr_b),
        .c_count(c_count),
        .c_flat(c_flat),
        .d_count(d_count),
        .d_flat(d_flat),
        .op_table_kernel_class(op_lookup_kernel_class),
        .result_wr_valid(result_wr_valid),
        .result_wr_idx(result_wr_idx),
        .result_wr_desc(result_wr_desc),
        .tick_commit_valid(tick_commit_valid),
        .tick_commit_id(tick_commit_id),
        .kernel_start(kernel_start),
        .kernel_op_a(kernel_op_a),
        .kernel_op_b(kernel_op_b),
        .kernel_class(kernel_class),
        .dsp_mode(dsp_mode),
        .kernel_instr_idx(kernel_instr_idx),
        .kernel_c_count(kernel_c_count),
        .kernel_c_flat(kernel_c_flat),
        .kernel_d_count(kernel_d_count),
        .kernel_d_flat(kernel_d_flat),
        .kernel_done(kernel_done),
        .kernel_result_desc(kernel_result_desc),
        .input_req_valid(input_req_valid),
        .input_kind(input_kind),
        .input_id(input_id),
        .input_req_ready(input_req_ready),
        .input_desc_valid(input_desc_valid),
        .input_desc(input_desc),
        .done(orch_done),
        .error(orch_error)
    );

    wire mmap_busy;
    wire mmap_error;
    wire table_free_valid;
    wire table_copy_valid;
    wire [IDX_WIDTH-1:0] table_copy_src;
    wire [IDX_WIDTH-1:0] table_copy_dst;

    nac_mmap_engine #(
        .NUM_TICKS(1 << IDX_WIDTH),
        .TICK_WIDTH(IDX_WIDTH),
        .MAX_CMDS_PER_TICK(MAX_MMAP_CMDS_PER_TICK),
        .CMD_SLOT_WIDTH(MMAP_SLOT_WIDTH)
    ) mmap (
        .clk(clk),
        .rst(rst),
        .cfg_we(mmap_cfg_we),
        .cfg_tick(mmap_cfg_tick),
        .cfg_slot(mmap_cfg_slot),
        .cfg_valid(mmap_cfg_valid),
        .cfg_action(mmap_cfg_action),
        .cfg_target(mmap_cfg_target),
        .tick_valid(tick_commit_valid),
        .tick_id(tick_commit_id),
        .busy(mmap_busy),
        .preload_valid(preload_valid),
        .preload_ready(preload_ready),
        .preload_target(preload_target),
        .free_valid(free_valid),
        .free_ready(free_ready),
        .free_target(free_target),
        .save_result_valid(save_result_valid),
        .save_result_ready(save_result_ready),
        .save_result_src(save_result_src),
        .save_result_target(save_result_target),
        .forward_valid(forward_valid),
        .forward_ready(forward_ready),
        .forward_src(forward_src),
        .forward_dst(forward_dst),
        .error(mmap_error)
    );

    // Descriptor table is instantiated here so MMAP FORWARD/FREE semantics are
    // represented in the core even when an external tensor datapath is used.
    nac_result_table #(
        .ENTRIES(1 << IDX_WIDTH),
        .IDX_WIDTH(IDX_WIDTH),
        .DESC_WIDTH(DESC_WIDTH)
    ) result_table (
        .clk(clk),
        .rst(rst),
        .wr_valid(result_wr_valid),
        .wr_idx(result_wr_idx),
        .wr_desc(result_wr_desc),
        .rd0_en(result_rd0_en),
        .rd0_idx(result_rd0_idx),
        .rd0_valid(result_rd0_valid),
        .rd0_desc(result_rd0_desc),
        .rd1_en(result_rd1_en),
        .rd1_idx(result_rd1_idx),
        .rd1_valid(result_rd1_valid),
        .rd1_desc(result_rd1_desc),
        .free_valid(table_free_valid),
        .free_idx(free_target[IDX_WIDTH-1:0]),
        .forward_valid(table_copy_valid),
        .forward_src_idx(table_copy_src),
        .forward_dst_idx(table_copy_dst)
    );

    assign table_free_valid = free_valid & free_ready;
    assign table_copy_valid = (forward_valid & forward_ready) | (save_result_valid & save_result_ready);
    assign table_copy_src = forward_valid ? forward_src[IDX_WIDTH-1:0] : save_result_src[IDX_WIDTH-1:0];
    assign table_copy_dst = forward_valid ? forward_dst[IDX_WIDTH-1:0] : save_result_target[IDX_WIDTH-1:0];

    assign done = orch_finished && !mmap_busy;
    assign error = decoder_error | orch_error | mmap_error;
    assign busy = core_started | decoder_busy | mmap_busy;
endmodule
