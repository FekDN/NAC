`include "nac_defs.vh"
`include "nac_config.vh"
`include "nac_descriptor.vh"

module nac_core #(
    parameter [255:0] HW_CFG = `NAC_CFG_FPGA_TINY_EDGE,
    parameter IDX_WIDTH = `NAC_CFG_IDX_WIDTH(HW_CFG),
    parameter MAX_ARITY = `NAC_CFG_MAX_ARITY(HW_CFG),
    parameter MAX_CONSTS = `NAC_CFG_MAX_CONSTS(HW_CFG),
    parameter DESC_WIDTH = `NAC_DESC_MIN_WIDTH,
    parameter PERM_ENTRIES = 256,
    parameter MAX_MMAP_CMDS_PER_TICK = 256,
    parameter MMAP_SLOT_WIDTH = 8,
    parameter CONTEXTS = `NAC_CFG_CONTEXTS(HW_CFG),
    parameter CONTEXT_BITS = (CONTEXTS <= 2) ? 1 :
                             (CONTEXTS <= 4) ? 2 :
                             (CONTEXTS <= 8) ? 3 : 4,
    parameter ENABLE_INSTR_CACHE = `NAC_CFG_ENABLE_INSTR_CACHE(HW_CFG),
    parameter ENABLE_ECC = `NAC_CFG_ENABLE_ECC(HW_CFG),
    parameter WATCHDOG_TIMEOUT = `NAC_CFG_WATCHDOG_TIMEOUT(HW_CFG)
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire cmd_load_model,
    input  wire cmd_run_inference,
    input  wire cmd_clear_model,
    input  wire [CONTEXT_BITS-1:0] context_id,

    input  wire csr_we,
    input  wire [7:0] csr_addr,
    input  wire [63:0] csr_wdata,
    output wire [63:0] csr_rdata,

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
    input  wire [7:0] op_cfg_dsp_mode,
    input  wire op_cfg_uses_dsp,
    input  wire op_cfg_multi_pass,

    // MMAP schedule load.
    input  wire mmap_cfg_we,
    input  wire [IDX_WIDTH-1:0] mmap_cfg_tick,
    input  wire [MMAP_SLOT_WIDTH-1:0] mmap_cfg_slot,
    input  wire mmap_cfg_valid,
    input  wire [7:0] mmap_cfg_action,
    input  wire [15:0] mmap_cfg_target,
    input  wire mmap_cfg_static,

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

    output wire branch_req_valid,
    output wire [IDX_WIDTH-1:0] branch_predicate_idx,
    output wire [IDX_WIDTH-1:0] branch_true_target,
    output wire [IDX_WIDTH-1:0] branch_false_target,
    input  wire branch_req_ready,
    input  wire branch_resp_valid,
    input  wire branch_take_true,

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

    output wire model_configured,
    output wire watchdog_timeout,
    output wire busy,
    output wire done,
    output wire error
);
    reg core_started;
    reg orch_finished;
    reg load_mode;
    reg cache_run_mode;
    reg [CONTEXT_BITS-1:0] active_context;

    wire start_i = (start === 1'b1);
    wire cmd_load_model_i = (cmd_load_model === 1'b1);
    wire cmd_run_inference_i = (cmd_run_inference === 1'b1);
    wire cmd_clear_model_i = (cmd_clear_model === 1'b1);
    wire csr_we_i = (csr_we === 1'b1);
    wire branch_req_ready_i = (branch_req_ready === 1'b1);
    wire branch_resp_valid_i = (branch_resp_valid === 1'b1);
    wire branch_take_true_i = (branch_take_true === 1'b1);
    wire context_id_known = ((^context_id) === 1'b0) || ((^context_id) === 1'b1);
    wire [CONTEXT_BITS-1:0] context_i = context_id_known ? context_id : {CONTEXT_BITS{1'b0}};
    wire core_rst;
    wire decoder_rst;
    wire watchdog_reset_pulse;
    wire [7:0] csr_quant_flags;
    wire [15:0] csr_num_inputs;
    wire [15:0] csr_num_outputs;
    wire [15:0] csr_d_model;
    wire [11*64-1:0] csr_section_offsets;
    wire csr_active_valid;

    nac_csr_bank #(
        .CONTEXTS(CONTEXTS),
        .CONTEXT_BITS(CONTEXT_BITS)
    ) csr_bank (
        .clk(clk),
        .rst(rst),
        .clear_context(cmd_clear_model_i),
        .clear_context_id(context_i),
        .we(csr_we_i),
        .wr_context(context_i),
        .addr(csr_addr),
        .wdata(csr_wdata),
        .rd_context(context_i),
        .rdata(csr_rdata),
        .active_quant_flags(csr_quant_flags),
        .active_num_inputs(csr_num_inputs),
        .active_num_outputs(csr_num_outputs),
        .active_d_model(csr_d_model),
        .active_section_offsets(csr_section_offsets),
        .active_valid(csr_active_valid)
    );

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

    wire dec_instr_valid;
    wire dec_instr_ready;
    wire [15:0] dec_instr_index;
    wire [7:0] dec_instr_a;
    wire [7:0] dec_instr_b;
    wire [3:0] dec_c_count;
    wire [MAX_CONSTS*16-1:0] dec_c_flat;
    wire [3:0] dec_d_count;
    wire [MAX_ARITY*16-1:0] dec_d_flat;
    wire orch_instr_valid;
    wire orch_instr_ready;
    wire [15:0] orch_instr_index;
    wire [7:0] orch_instr_a;
    wire [7:0] orch_instr_b;
    wire [3:0] orch_c_count;
    wire [MAX_CONSTS*16-1:0] orch_c_flat;
    wire [3:0] orch_d_count;
    wire [MAX_ARITY*16-1:0] orch_d_flat;
    wire decoder_busy;
    wire decoder_error;
    wire [7:0] op_lookup_kernel_class;
    wire [7:0] op_lookup_dsp_mode;
    wire op_lookup_uses_dsp;
    wire op_lookup_multi_pass;
    wire op_lookup_present;

    nac_op_table op_table (
        .clk(clk),
        .cfg_we(op_cfg_we),
        .cfg_op_id(op_cfg_id),
        .cfg_kernel_class(op_cfg_kernel_class),
        .cfg_dsp_mode(op_cfg_dsp_mode),
        .cfg_uses_dsp(op_cfg_uses_dsp),
        .cfg_multi_pass(op_cfg_multi_pass),
        .lookup_op_id(orch_instr_a),
        .lookup_kernel_class(op_lookup_kernel_class),
        .lookup_dsp_mode(op_lookup_dsp_mode),
        .lookup_uses_dsp(op_lookup_uses_dsp),
        .lookup_multi_pass(op_lookup_multi_pass),
        .lookup_present(op_lookup_present)
    );

    nac_abcd_decoder #(
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS),
        .ARITY_BITS(4)
    ) decoder (
        .clk(clk),
        .rst(decoder_rst),
        .start(start_i | cmd_load_model_i),
        .ops_byte_valid(ops_byte_valid),
        .ops_byte_ready(ops_byte_ready),
        .ops_byte(ops_byte),
        .perm_lookup_id(perm_lookup_id),
        .perm_arity(perm_arity),
        .perm_needs_consts(perm_needs_consts),
        .perm_present(perm_present),
        .instr_valid(dec_instr_valid),
        .instr_ready(dec_instr_ready),
        .instr_index(dec_instr_index),
        .instr_a(dec_instr_a),
        .instr_b(dec_instr_b),
        .c_count(dec_c_count),
        .c_flat(dec_c_flat),
        .d_count(dec_d_count),
        .d_flat(dec_d_flat),
        .busy(decoder_busy),
        .error(decoder_error)
    );

    wire cache_load_done;
    wire cache_load_error;
    wire cache_run_active;
    wire cache_run_done;
    wire cache_run_error;
    wire cache_instr_valid;
    wire cache_instr_ready;
    wire [15:0] cache_instr_index;
    wire [7:0] cache_instr_a;
    wire [7:0] cache_instr_b;
    wire [3:0] cache_c_count;
    wire [MAX_CONSTS*16-1:0] cache_c_flat;
    wire [3:0] cache_d_count;
    wire [MAX_ARITY*16-1:0] cache_d_flat;
    wire cache_model_configured;
    wire cache_dec_ready;
    wire use_cache_run = ENABLE_INSTR_CACHE && cache_run_mode;
    wire branch_redirect_valid;
    wire [IDX_WIDTH-1:0] branch_redirect_target;

    nac_instruction_cache #(
        .CONTEXTS(CONTEXTS),
        .CONTEXT_BITS(CONTEXT_BITS),
        .IDX_WIDTH(IDX_WIDTH),
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS)
    ) instr_cache (
        .clk(clk),
        .rst(core_rst),
        .clear_context(cmd_clear_model_i),
        .clear_context_id(context_i),
        .load_start(ENABLE_INSTR_CACHE ? cmd_load_model_i : 1'b0),
        .load_context(context_i),
        .dec_valid(dec_instr_valid),
        .dec_ready(cache_dec_ready),
        .dec_index(dec_instr_index),
        .dec_a(dec_instr_a),
        .dec_b(dec_instr_b),
        .dec_c_count(dec_c_count),
        .dec_c_flat(dec_c_flat),
        .dec_d_count(dec_d_count),
        .dec_d_flat(dec_d_flat),
        .load_done(cache_load_done),
        .load_error(cache_load_error),
        .run_start(ENABLE_INSTR_CACHE ? cmd_run_inference_i : 1'b0),
        .run_context(context_i),
        .redirect_valid(branch_redirect_valid),
        .redirect_index(branch_redirect_target),
        .instr_valid(cache_instr_valid),
        .instr_ready(cache_instr_ready),
        .instr_index(cache_instr_index),
        .instr_a(cache_instr_a),
        .instr_b(cache_instr_b),
        .instr_c_count(cache_c_count),
        .instr_c_flat(cache_c_flat),
        .instr_d_count(cache_d_count),
        .instr_d_flat(cache_d_flat),
        .run_active(cache_run_active),
        .run_done(cache_run_done),
        .run_error(cache_run_error),
        .model_configured(cache_model_configured)
    );

    assign dec_instr_ready = (ENABLE_INSTR_CACHE && load_mode) ? cache_dec_ready :
                             (use_cache_run ? 1'b0 : orch_instr_ready);
    assign cache_instr_ready = use_cache_run ? orch_instr_ready : 1'b0;
    assign orch_instr_valid = use_cache_run ? cache_instr_valid : dec_instr_valid;
    assign orch_instr_index = use_cache_run ? cache_instr_index : dec_instr_index;
    assign orch_instr_a = use_cache_run ? cache_instr_a : dec_instr_a;
    assign orch_instr_b = use_cache_run ? cache_instr_b : dec_instr_b;
    assign orch_c_count = use_cache_run ? cache_c_count : dec_c_count;
    assign orch_c_flat = use_cache_run ? cache_c_flat : dec_c_flat;
    assign orch_d_count = use_cache_run ? cache_d_count : dec_d_count;
    assign orch_d_flat = use_cache_run ? cache_d_flat : dec_d_flat;

    wire result_wr_valid;
    wire [IDX_WIDTH-1:0] result_wr_idx;
    wire [DESC_WIDTH-1:0] result_wr_desc;
    wire tick_commit_valid;
    wire [IDX_WIDTH-1:0] tick_commit_id;
    wire orch_done;
    wire orch_error;

    always @(posedge clk) begin
        if (core_rst) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
            load_mode <= 1'b0;
            cache_run_mode <= 1'b0;
            active_context <= {CONTEXT_BITS{1'b0}};
        end else if (cmd_clear_model_i) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
            load_mode <= 1'b0;
            cache_run_mode <= 1'b0;
        end else if (cmd_load_model_i) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
            load_mode <= ENABLE_INSTR_CACHE;
            cache_run_mode <= 1'b0;
            active_context <= context_i;
        end else if (start_i) begin
            core_started <= 1'b1;
            orch_finished <= 1'b0;
            load_mode <= 1'b0;
            cache_run_mode <= 1'b0;
            active_context <= {CONTEXT_BITS{1'b0}};
        end else if (cmd_run_inference_i) begin
            core_started <= 1'b1;
            orch_finished <= 1'b0;
            load_mode <= 1'b0;
            cache_run_mode <= ENABLE_INSTR_CACHE;
            active_context <= context_i;
        end else if (cache_load_done || cache_load_error) begin
            load_mode <= 1'b0;
        end else if (decoder_error || orch_error || mmap_error || cache_load_error || cache_run_error) begin
            core_started <= 1'b0;
            orch_finished <= 1'b0;
            load_mode <= 1'b0;
            cache_run_mode <= 1'b0;
        end else if (orch_done) begin
            core_started <= 1'b0;
            orch_finished <= 1'b1;
            cache_run_mode <= 1'b0;
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
        .rst(core_rst),
        .start(start_i | cmd_run_inference_i),
        .instr_valid(orch_instr_valid),
        .instr_ready(orch_instr_ready),
        .instr_index(orch_instr_index),
        .instr_a(orch_instr_a),
        .instr_b(orch_instr_b),
        .c_count(orch_c_count),
        .c_flat(orch_c_flat),
        .d_count(orch_d_count),
        .d_flat(orch_d_flat),
        .op_table_kernel_class(op_lookup_kernel_class),
        .op_table_dsp_mode(op_lookup_dsp_mode),
        .op_table_uses_dsp(op_lookup_uses_dsp),
        .op_table_multi_pass(op_lookup_multi_pass),
        .op_table_present(op_lookup_present),
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
        .branch_req_valid(branch_req_valid),
        .branch_predicate_idx(branch_predicate_idx),
        .branch_true_target(branch_true_target),
        .branch_false_target(branch_false_target),
        .branch_req_ready(branch_req_ready_i),
        .branch_resp_valid(branch_resp_valid_i),
        .branch_take_true(branch_take_true_i),
        .branch_redirect_valid(branch_redirect_valid),
        .branch_redirect_target(branch_redirect_target),
        .done(orch_done),
        .error(orch_error)
    );

    wire mmap_busy;
    wire mmap_error;
    wire mmap_runtime_rst;
    wire table_free_valid;
    wire table_copy_valid;
    wire [IDX_WIDTH-1:0] table_copy_src;
    wire [IDX_WIDTH-1:0] table_copy_dst;
    wire result_single_error;
    wire result_double_error;

    nac_mmap_engine #(
        .NUM_TICKS(1 << IDX_WIDTH),
        .TICK_WIDTH(IDX_WIDTH),
        .MAX_CMDS_PER_TICK(MAX_MMAP_CMDS_PER_TICK),
        .CMD_SLOT_WIDTH(MMAP_SLOT_WIDTH)
    ) mmap (
        .clk(clk),
        .rst(mmap_runtime_rst),
        .cfg_we(mmap_cfg_we),
        .cfg_tick(mmap_cfg_tick),
        .cfg_slot(mmap_cfg_slot),
        .cfg_valid(mmap_cfg_valid),
        .cfg_action(mmap_cfg_action),
        .cfg_target(mmap_cfg_target),
        .cfg_static(mmap_cfg_static === 1'b1),
        .clear_static(cmd_load_model_i | cmd_clear_model_i),
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
        .DESC_WIDTH(DESC_WIDTH),
        .ENABLE_ECC(ENABLE_ECC)
    ) result_table (
        .clk(clk),
        .rst(core_rst),
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
        .forward_dst_idx(table_copy_dst),
        .single_error(result_single_error),
        .double_error(result_double_error)
    );

    assign table_free_valid = free_valid & free_ready;
    assign table_copy_valid = (forward_valid & forward_ready) | (save_result_valid & save_result_ready);
    assign table_copy_src = forward_valid ? forward_src[IDX_WIDTH-1:0] : save_result_src[IDX_WIDTH-1:0];
    assign table_copy_dst = forward_valid ? forward_dst[IDX_WIDTH-1:0] : save_result_target[IDX_WIDTH-1:0];

    wire raw_busy = core_started | decoder_busy | mmap_busy | orch_finished | load_mode | cache_run_mode;
    wire watchdog_progress =
        start_i | cmd_load_model_i | cmd_run_inference_i | cmd_clear_model_i |
        dec_instr_valid & dec_instr_ready |
        orch_instr_valid & orch_instr_ready |
        tick_commit_valid |
        kernel_start | kernel_done |
        input_req_valid & input_req_ready |
        branch_req_valid & branch_req_ready_i |
        branch_resp_valid_i |
        input_desc_valid |
        preload_valid & preload_ready |
        free_valid & free_ready |
        save_result_valid & save_result_ready |
        forward_valid & forward_ready |
        cache_load_done | cache_run_done;

    nac_watchdog #(
        .TIMEOUT_CYCLES(WATCHDOG_TIMEOUT),
        .COUNTER_WIDTH(32)
    ) watchdog (
        .clk(clk),
        .rst(rst),
        .clear(start_i | cmd_load_model_i | cmd_run_inference_i | cmd_clear_model_i),
        .enable(WATCHDOG_TIMEOUT != 0),
        .busy(raw_busy),
        .progress(watchdog_progress),
        .timeout(watchdog_timeout),
        .reset_pulse(watchdog_reset_pulse)
    );

    assign core_rst = rst | watchdog_reset_pulse;
    assign mmap_runtime_rst = core_rst | start_i | cmd_run_inference_i;
    assign decoder_rst = core_rst | cache_load_done | cache_load_error | cmd_clear_model_i;
    assign model_configured = cache_model_configured;
    assign done = orch_finished && !mmap_busy;
    assign error = decoder_error | orch_error | mmap_error | cache_load_error | cache_run_error |
                   result_double_error | watchdog_timeout;
    assign busy = raw_busy;
endmodule
