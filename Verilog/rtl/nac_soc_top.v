`include "nac_defs.vh"
`include "nac_config.vh"
`include "nac_descriptor.vh"

// Bare-metal NAC SoC integration shell.
//
// MEP is the master. It consumes ORCH/MEP bytecode, dispatches optional TISA
// preprocessing/postprocessing, launches NAC graph execution, and delegates
// physical tensor movement to the MMAP-driven DMA manager. Board-specific
// storage (AXI4, SD/eMMC, SATA, USB) and rich I/O controllers connect through
// explicit descriptor/stream ports instead of hidden top-level placeholders.
module nac_soc_top #(
    parameter [255:0] HW_CFG = `NAC_CFG_FPGA_SERVER_MAX,
    parameter IDX_WIDTH = `NAC_CFG_IDX_WIDTH(HW_CFG),
    parameter MAX_ARITY = `NAC_CFG_MAX_ARITY(HW_CFG),
    parameter MAX_CONSTS = `NAC_CFG_MAX_CONSTS(HW_CFG),
    parameter DESC_WIDTH = `NAC_DESC_MIN_WIDTH,
    parameter LANES = `NAC_CFG_LANES_1D(HW_CFG),
    parameter DATA_WIDTH = `NAC_CFG_DATA_WIDTH(HW_CFG),
    parameter ACC_WIDTH = `NAC_CFG_ACC_WIDTH(HW_CFG),
    parameter SRAM_BANKS = `NAC_CFG_SRAM_BANKS(HW_CFG),
    parameter BANK_BITS = (SRAM_BANKS <= 2) ? 1 :
                          (SRAM_BANKS <= 4) ? 2 :
                          (SRAM_BANKS <= 8) ? 3 : 4,
    parameter SPAD_ADDR_WIDTH = 12,
    parameter CONTEXTS = `NAC_CFG_CONTEXTS(HW_CFG),
    parameter CONTEXT_BITS = (CONTEXTS <= 2) ? 1 :
                             (CONTEXTS <= 4) ? 2 :
                             (CONTEXTS <= 8) ? 3 : 4,
    parameter DMA_TARGETS = 256,
    parameter DMA_TARGET_BITS = 8,
    parameter DMA_ADDR_WIDTH = 64,
    parameter DMA_LEN_WIDTH = 32
) (
    input  wire clk,
    input  wire rst_n,

    input  wire soc_start,
    input  wire core_cmd_load_model,
    input  wire core_cmd_clear_model,
    input  wire [CONTEXT_BITS-1:0] context_id,
    output wire soc_busy,
    output wire soc_done,
    output wire soc_error,

    // ORCH/MEP byte stream.
    input  wire [31:0] mep_plan_size,
    input  wire mep_plan_byte_valid,
    output wire mep_plan_byte_ready,
    input  wire [7:0] mep_plan_byte,

    // Core CSR/config loaders.
    input  wire csr_we,
    input  wire [7:0] csr_addr,
    input  wire [63:0] csr_wdata,
    output wire [63:0] csr_rdata,
    input  wire [15:0] num_outputs,
    input  wire ops_byte_valid,
    output wire ops_byte_ready,
    input  wire [7:0] ops_byte,
    input  wire perm_cfg_we,
    input  wire [7:0] perm_cfg_id,
    input  wire [3:0] perm_cfg_arity,
    input  wire perm_cfg_needs_consts,
    input  wire op_cfg_we,
    input  wire [7:0] op_cfg_id,
    input  wire [7:0] op_cfg_kernel_class,
    input  wire [7:0] op_cfg_dsp_mode,
    input  wire op_cfg_uses_dsp,
    input  wire op_cfg_multi_pass,
    input  wire mmap_cfg_we,
    input  wire [IDX_WIDTH-1:0] mmap_cfg_tick,
    input  wire [7:0] mmap_cfg_slot,
    input  wire mmap_cfg_valid,
    input  wire [7:0] mmap_cfg_action,
    input  wire [15:0] mmap_cfg_target,
    input  wire mmap_cfg_static,

    // Input/parameter/branch providers for NAC ABCD execution.
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

    // TISA manifest and runtime streams. MEP PREPROC_ENCODE/DECODE starts run.
    input  wire tisa_load_start,
    input  wire [31:0] tisa_manifest_size,
    input  wire tisa_manifest_byte_valid,
    output wire tisa_manifest_byte_ready,
    input  wire [7:0] tisa_manifest_byte,
    input  wire tisa_text_valid,
    output wire tisa_text_ready,
    input  wire [7:0] tisa_text_byte,
    input  wire tisa_text_last,
    output wire tisa_out_valid,
    input  wire tisa_out_ready,
    output wire [7:0] tisa_out_byte,
    output wire tisa_out_last,
    input  wire [63:0] tisa_result_ref_value,
    output wire tisa_requires_external_engine,

    // Generic MEP I/O peripheral interface.
    output wire mep_io_req_valid,
    input  wire mep_io_req_ready,
    output wire [7:0] mep_io_opcode,
    output wire [7:0] mep_io_out_key,
    output wire [7:0] mep_io_arg0,
    output wire [15:0] mep_io_const_id,
    input  wire mep_io_done,
    input  wire mep_io_error,
    input  wire [7:0] mep_io_result_key,
    input  wire [63:0] mep_io_result_value,
    input  wire mep_io_result_valid,

    // Training/storage peripherals controlled by MEP.
    output wire train_start,
    input  wire train_ready,
    output wire [7:0] train_model_id,
    output wire [7:0] train_loss_type,
    output wire [7:0] train_count_in,
    output wire [8*16-1:0] train_in_keys,
    output wire [7:0] train_target_count,
    output wire [8*16-1:0] train_target_keys,
    output wire [7:0] train_loss_key,
    output wire [7:0] train_lr_key,
    output wire [7:0] train_logits_key,
    output wire [15:0] train_head_weight_name_id,
    output wire [15:0] train_head_bias_name_id,
    input  wire train_done,
    input  wire train_error,
    output wire zero_grad_start,
    input  wire zero_grad_ready,
    output wire [7:0] zero_grad_model_id,
    input  wire zero_grad_done,
    input  wire zero_grad_error,
    output wire save_weights_start,
    input  wire save_weights_ready,
    output wire [7:0] save_model_id,
    output wire [7:0] save_path_key,
    output wire [7:0] save_type,
    input  wire save_weights_done,
    input  wire save_weights_error,

    // DMA target table and physical transfer interface.
    input  wire dma_cfg_we,
    input  wire [DMA_TARGET_BITS-1:0] dma_cfg_target,
    input  wire [DMA_ADDR_WIDTH-1:0] dma_cfg_ext_addr,
    input  wire [DMA_LEN_WIDTH-1:0] dma_cfg_size_bytes,
    input  wire dma_cfg_static,
    input  wire mark_dirty_valid,
    input  wire [DMA_TARGET_BITS-1:0] mark_dirty_target,
    output wire dma_read_valid,
    input  wire dma_read_ready,
    output wire [DMA_ADDR_WIDTH-1:0] dma_read_addr,
    output wire [DMA_LEN_WIDTH-1:0] dma_read_len,
    output wire [BANK_BITS-1:0] dma_read_bank,
    input  wire dma_read_done,
    input  wire dma_read_error,
    output wire dma_write_valid,
    input  wire dma_write_ready,
    output wire [DMA_ADDR_WIDTH-1:0] dma_write_addr,
    output wire [DMA_LEN_WIDTH-1:0] dma_write_len,
    output wire [BANK_BITS-1:0] dma_write_bank,
    input  wire dma_write_done,
    input  wire dma_write_error,

    // Scratchpad external data ports used by board-specific DMA/tensor fetch.
    input  wire spad_a_we,
    input  wire [BANK_BITS-1:0] spad_a_bank,
    input  wire [SPAD_ADDR_WIDTH-1:0] spad_a_addr,
    input  wire [DATA_WIDTH-1:0] spad_a_wdata,
    output wire [DATA_WIDTH-1:0] spad_a_rdata,
    input  wire spad_b_we,
    input  wire [BANK_BITS-1:0] spad_b_bank,
    input  wire [SPAD_ADDR_WIDTH-1:0] spad_b_addr,
    input  wire [DATA_WIDTH-1:0] spad_b_wdata,
    output wire [DATA_WIDTH-1:0] spad_b_rdata,

    // Tensor vectors feeding the integrated codec/DSP wrapper.
    input  wire [LANES*DATA_WIDTH-1:0] tensor_a_vec,
    input  wire [LANES*DATA_WIDTH-1:0] tensor_b_vec,
    input  wire tensor_weight_is_bfp,
    input  wire tensor_weight_is_palette,
    input  wire tensor_weight_is_sparse_2_4,
    input  wire tensor_result_is_rle,
    input  wire [LANES*4-1:0] tensor_bfp_mantissas,
    input  wire signed [7:0] tensor_bfp_exp,
    input  wire [16*DATA_WIDTH-1:0] tensor_palette,
    input  wire [LANES*4-1:0] tensor_palette_indices,
    input  wire [(LANES/2)*DATA_WIDTH-1:0] tensor_sparse_values,
    input  wire [(LANES/2)*2-1:0] tensor_sparse_indices,
    output wire dsp_result_valid,
    input  wire dsp_result_ready,
    output wire [LANES*DATA_WIDTH-1:0] dsp_result_vec,
    output wire signed [ACC_WIDTH-1:0] dsp_scalar_out,
    input  wire [DESC_WIDTH-1:0] dsp_result_desc,
    input  wire dsp_result_desc_valid,

    output wire [7:0] soc_error_flags
);
    wire rst = ~rst_n;

    // MEP packetizer.
    wire mep_pkt_valid;
    wire mep_pkt_ready;
    wire [7:0] mep_pkt_byte;
    wire [7:0] mep_pkt_opcode;
    wire [15:0] mep_pkt_instr_index;
    wire [15:0] mep_pkt_byte_index;
    wire mep_pkt_instr_start;
    wire mep_pkt_instr_end;
    wire mep_pkt_done;
    wire mep_pkt_error;

    nac_mep_packetizer mep_packetizer (
        .clk(clk),
        .rst(rst),
        .start(soc_start),
        .plan_size(mep_plan_size),
        .byte_valid(mep_plan_byte_valid),
        .byte_ready(mep_plan_byte_ready),
        .byte_in(mep_plan_byte),
        .out_valid(mep_pkt_valid),
        .out_ready(mep_pkt_ready),
        .out_byte(mep_pkt_byte),
        .out_opcode(mep_pkt_opcode),
        .out_instr_index(mep_pkt_instr_index),
        .out_byte_index(mep_pkt_byte_index),
        .out_instr_start(mep_pkt_instr_start),
        .out_instr_end(mep_pkt_instr_end),
        .done(mep_pkt_done),
        .error(mep_pkt_error)
    );

    // TISA tokenizer peripheral.
    wire mep_tisa_req_valid;
    wire mep_tisa_req_ready;
    wire mep_tisa_req_decode;
    wire [7:0] mep_tisa_proc_key;
    wire [7:0] mep_tisa_in_key;
    wire [7:0] mep_tisa_out_key;
    reg tisa_active;
    wire tisa_load_done;
    wire tisa_load_error;
    wire tisa_can_run_locally;
    wire tisa_run_start = mep_tisa_req_valid & mep_tisa_req_ready;
    wire tisa_run_done;
    wire tisa_run_error;
    wire mep_tisa_done = tisa_active & tisa_run_done;
    wire mep_tisa_error = tisa_run_error | tisa_load_error;

    assign mep_tisa_req_ready = tisa_can_run_locally && !tisa_active;

    always @(posedge clk) begin
        if (rst) begin
            tisa_active <= 1'b0;
        end else begin
            if (tisa_run_start)
                tisa_active <= 1'b1;
            else if (tisa_run_done || tisa_run_error)
                tisa_active <= 1'b0;
        end
    end

    nac_tisa_tokenizer tisa_tokenizer (
        .clk(clk),
        .rst(rst),
        .load_start(tisa_load_start),
        .manifest_size(tisa_manifest_size),
        .manifest_byte_valid(tisa_manifest_byte_valid),
        .manifest_byte_ready(tisa_manifest_byte_ready),
        .manifest_byte(tisa_manifest_byte),
        .load_done(tisa_load_done),
        .load_error(tisa_load_error),
        .requires_external_engine(tisa_requires_external_engine),
        .unsupported_opcode(),
        .can_run_locally(tisa_can_run_locally),
        .run_start(tisa_run_start),
        .text_valid(tisa_text_valid),
        .text_ready(tisa_text_ready),
        .text_byte(tisa_text_byte),
        .text_last(tisa_text_last),
        .out_valid(tisa_out_valid),
        .out_ready(tisa_out_ready),
        .out_byte(tisa_out_byte),
        .out_last(tisa_out_last),
        .run_done(tisa_run_done),
        .run_error(tisa_run_error)
    );

    // MEP master VM.
    wire mep_model_run_start;
    wire mep_model_run_ready;
    wire [7:0] mep_model_id;
    wire [7:0] mep_model_count_in;
    wire [8*16-1:0] mep_model_in_keys;
    wire [7:0] mep_model_count_out;
    wire [8*16-1:0] mep_model_out_keys;
    wire mep_model_run_done;
    wire mep_model_run_error;
    wire mep_done;
    wire mep_halted;
    wire mep_error;
    wire [7:0] mep_error_opcode;
    wire [63:0] mep_ctx0_value;
    wire mep_ctx0_valid;

    nac_mep_vm #(
        .MAX_INSTR_BYTES(64),
        .CONTEXT_VALUE_WIDTH(64)
    ) mep_master (
        .clk(clk),
        .rst(rst),
        .start(soc_start),
        .in_valid(mep_pkt_valid),
        .in_ready(mep_pkt_ready),
        .in_byte(mep_pkt_byte),
        .in_opcode(mep_pkt_opcode),
        .in_instr_index(mep_pkt_instr_index),
        .in_byte_index(mep_pkt_byte_index),
        .in_instr_start(mep_pkt_instr_start),
        .in_instr_end(mep_pkt_instr_end),
        .io_req_valid(mep_io_req_valid),
        .io_req_ready(mep_io_req_ready),
        .io_opcode(mep_io_opcode),
        .io_out_key(mep_io_out_key),
        .io_arg0(mep_io_arg0),
        .io_const_id(mep_io_const_id),
        .io_done(mep_io_done),
        .io_error(mep_io_error),
        .io_result_key(mep_io_result_key),
        .io_result_value(mep_io_result_value),
        .io_result_valid(mep_io_result_valid),
        .tisa_req_valid(mep_tisa_req_valid),
        .tisa_req_ready(mep_tisa_req_ready),
        .tisa_req_decode(mep_tisa_req_decode),
        .tisa_proc_key(mep_tisa_proc_key),
        .tisa_in_key(mep_tisa_in_key),
        .tisa_out_key(mep_tisa_out_key),
        .tisa_done(mep_tisa_done),
        .tisa_error(mep_tisa_error),
        .tisa_result_value(tisa_result_ref_value),
        .tisa_result_valid(mep_tisa_done),
        .model_run_start(mep_model_run_start),
        .model_run_ready(mep_model_run_ready),
        .model_id(mep_model_id),
        .model_count_in(mep_model_count_in),
        .model_in_keys(mep_model_in_keys),
        .model_count_out(mep_model_count_out),
        .model_out_keys(mep_model_out_keys),
        .model_run_done(mep_model_run_done),
        .model_run_error(mep_model_run_error),
        .train_start(train_start),
        .train_ready(train_ready),
        .train_model_id(train_model_id),
        .train_loss_type(train_loss_type),
        .train_count_in(train_count_in),
        .train_in_keys(train_in_keys),
        .train_target_count(train_target_count),
        .train_target_keys(train_target_keys),
        .train_loss_key(train_loss_key),
        .train_lr_key(train_lr_key),
        .train_logits_key(train_logits_key),
        .train_head_weight_name_id(train_head_weight_name_id),
        .train_head_bias_name_id(train_head_bias_name_id),
        .train_done(train_done),
        .train_error(train_error),
        .zero_grad_start(zero_grad_start),
        .zero_grad_ready(zero_grad_ready),
        .zero_grad_model_id(zero_grad_model_id),
        .zero_grad_done(zero_grad_done),
        .zero_grad_error(zero_grad_error),
        .save_weights_start(save_weights_start),
        .save_weights_ready(save_weights_ready),
        .save_model_id(save_model_id),
        .save_path_key(save_path_key),
        .save_type(save_type),
        .save_weights_done(save_weights_done),
        .save_weights_error(save_weights_error),
        .pc_redirect_valid(),
        .pc_redirect_ready(1'b1),
        .pc_redirect_offset(),
        .return_valid(),
        .return_count(),
        .return_keys(),
        .ctx0_value(mep_ctx0_value),
        .ctx0_valid(mep_ctx0_valid),
        .busy(soc_busy),
        .done(mep_done),
        .halted(mep_halted),
        .error(mep_error),
        .error_opcode(mep_error_opcode)
    );

    // NAC core and integrated codec/DSP.
    wire core_busy;
    wire core_done;
    wire core_error;
    wire core_model_configured;
    wire core_watchdog_timeout;
    wire kernel_start;
    wire [7:0] kernel_op_a;
    wire [7:0] kernel_op_b;
    wire [7:0] kernel_class;
    wire [7:0] dsp_mode;
    wire [IDX_WIDTH-1:0] kernel_instr_idx;
    wire [3:0] kernel_c_count;
    wire [MAX_CONSTS*16-1:0] kernel_c_flat;
    wire [3:0] kernel_d_count;
    wire [MAX_ARITY*16-1:0] kernel_d_flat;
    wire dsp_in_ready;
    wire dsp_error;
    wire kernel_done = dsp_result_valid & dsp_result_ready & dsp_result_desc_valid;
    wire [DESC_WIDTH-1:0] kernel_result_desc = dsp_result_desc;

    assign mep_model_run_ready = core_model_configured && !core_busy;
    assign mep_model_run_done = core_done;
    assign mep_model_run_error = core_error;

    wire preload_valid;
    wire preload_ready;
    wire [15:0] preload_target;
    wire free_valid;
    wire free_ready;
    wire [15:0] free_target;
    wire save_result_valid;
    wire save_result_ready;
    wire [15:0] save_result_src;
    wire [15:0] save_result_target;
    wire forward_valid;
    wire forward_ready;
    wire [15:0] forward_src;
    wire [15:0] forward_dst;

    nac_core #(
        .HW_CFG(HW_CFG),
        .IDX_WIDTH(IDX_WIDTH),
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS),
        .DESC_WIDTH(DESC_WIDTH),
        .CONTEXTS(CONTEXTS),
        .CONTEXT_BITS(CONTEXT_BITS)
    ) core_engine (
        .clk(clk),
        .rst(rst),
        .start(1'b0),
        .cmd_load_model(core_cmd_load_model),
        .cmd_run_inference(mep_model_run_start),
        .cmd_clear_model(core_cmd_clear_model),
        .context_id(context_id),
        .csr_we(csr_we),
        .csr_addr(csr_addr),
        .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata),
        .num_outputs(num_outputs),
        .ops_byte_valid(ops_byte_valid),
        .ops_byte_ready(ops_byte_ready),
        .ops_byte(ops_byte),
        .perm_cfg_we(perm_cfg_we),
        .perm_cfg_id(perm_cfg_id),
        .perm_cfg_arity(perm_cfg_arity),
        .perm_cfg_needs_consts(perm_cfg_needs_consts),
        .op_cfg_we(op_cfg_we),
        .op_cfg_id(op_cfg_id),
        .op_cfg_kernel_class(op_cfg_kernel_class),
        .op_cfg_dsp_mode(op_cfg_dsp_mode),
        .op_cfg_uses_dsp(op_cfg_uses_dsp),
        .op_cfg_multi_pass(op_cfg_multi_pass),
        .mmap_cfg_we(mmap_cfg_we),
        .mmap_cfg_tick(mmap_cfg_tick),
        .mmap_cfg_slot(mmap_cfg_slot),
        .mmap_cfg_valid(mmap_cfg_valid),
        .mmap_cfg_action(mmap_cfg_action),
        .mmap_cfg_target(mmap_cfg_target),
        .mmap_cfg_static(mmap_cfg_static),
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
        .result_rd0_en(1'b0),
        .result_rd0_idx({IDX_WIDTH{1'b0}}),
        .result_rd0_valid(),
        .result_rd0_desc(),
        .result_rd1_en(1'b0),
        .result_rd1_idx({IDX_WIDTH{1'b0}}),
        .result_rd1_valid(),
        .result_rd1_desc(),
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
        .branch_req_ready(branch_req_ready),
        .branch_resp_valid(branch_resp_valid),
        .branch_take_true(branch_take_true),
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
        .model_configured(core_model_configured),
        .watchdog_timeout(core_watchdog_timeout),
        .busy(core_busy),
        .done(core_done),
        .error(core_error)
    );

    nac_codec_dsp_pipeline #(
        .HW_CFG(HW_CFG),
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) codec_dsp (
        .clk(clk),
        .rst(rst),
        .cfg_mode(dsp_mode),
        .acc_in({ACC_WIDTH{1'b0}}),
        .in_valid(kernel_start),
        .in_ready(dsp_in_ready),
        .a_vec(tensor_a_vec),
        .b_mem_vec(tensor_b_vec),
        .weight_is_bfp(tensor_weight_is_bfp),
        .weight_is_palette(tensor_weight_is_palette),
        .weight_is_sparse_2_4(tensor_weight_is_sparse_2_4),
        .result_is_rle(tensor_result_is_rle),
        .weight_bfp_mantissas(tensor_bfp_mantissas),
        .weight_bfp_exp(tensor_bfp_exp),
        .weight_palette(tensor_palette),
        .weight_palette_indices(tensor_palette_indices),
        .weight_sparse_values(tensor_sparse_values),
        .weight_sparse_indices(tensor_sparse_indices),
        .out_valid(dsp_result_valid),
        .out_ready(dsp_result_ready & dsp_result_desc_valid),
        .out_vec(dsp_result_vec),
        .scalar_out(dsp_scalar_out),
        .sfu_req_valid(),
        .sfu_req_ready(1'b1),
        .sfu_req_mode(),
        .sfu_req_vec(),
        .sfu_resp_valid(1'b0),
        .sfu_resp_vec({LANES*DATA_WIDTH{1'b0}}),
        .rle_values(),
        .rle_counts(),
        .rle_is_zero_run(),
        .rle_token_count(),
        .rle_valid(),
        .error(dsp_error)
    );

    // MMAP-driven DMA manager and bank allocator.
    wire bank_alloc_req;
    wire bank_alloc_valid;
    wire bank_alloc_fail;
    wire [BANK_BITS-1:0] bank_alloc_id;
    wire bank_free_req;
    wire [BANK_BITS-1:0] bank_free_id;
    wire bank_free_error;
    wire dma_desc_update_valid;
    wire [DMA_TARGET_BITS-1:0] dma_desc_update_target;
    wire [BANK_BITS-1:0] dma_desc_update_bank;
    wire [DMA_LEN_WIDTH-1:0] dma_desc_update_size;
    wire dma_desc_invalidate_valid;
    wire [DMA_TARGET_BITS-1:0] dma_desc_invalidate_target;
    wire dma_manager_busy;
    wire dma_manager_error;
    wire [7:0] dma_manager_error_code;

    nac_bank_allocator #(
        .BANKS(SRAM_BANKS),
        .BANK_BITS(BANK_BITS)
    ) bank_allocator (
        .clk(clk),
        .rst(rst),
        .alloc_req(bank_alloc_req),
        .alloc_valid(bank_alloc_valid),
        .alloc_bank(bank_alloc_id),
        .alloc_fail(bank_alloc_fail),
        .free_req(bank_free_req),
        .free_bank(bank_free_id),
        .free_error(bank_free_error)
    );

    nac_dma_manager #(
        .TARGETS(DMA_TARGETS),
        .TARGET_BITS(DMA_TARGET_BITS),
        .BANKS(SRAM_BANKS),
        .BANK_BITS(BANK_BITS),
        .ADDR_WIDTH(DMA_ADDR_WIDTH),
        .LEN_WIDTH(DMA_LEN_WIDTH)
    ) dma_manager (
        .clk(clk),
        .rst(rst),
        .cfg_we(dma_cfg_we),
        .cfg_target(dma_cfg_target),
        .cfg_ext_addr(dma_cfg_ext_addr),
        .cfg_size_bytes(dma_cfg_size_bytes),
        .cfg_static(dma_cfg_static),
        .mark_dirty_valid(mark_dirty_valid),
        .mark_dirty_target(mark_dirty_target),
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
        .bank_alloc_valid(bank_alloc_req),
        .bank_alloc_ready(bank_alloc_valid),
        .bank_alloc_id(bank_alloc_id),
        .bank_free_valid(bank_free_req),
        .bank_free_ready(1'b1),
        .bank_free_id(bank_free_id),
        .dma_read_valid(dma_read_valid),
        .dma_read_ready(dma_read_ready),
        .dma_read_addr(dma_read_addr),
        .dma_read_len(dma_read_len),
        .dma_read_bank(dma_read_bank),
        .dma_read_done(dma_read_done),
        .dma_read_error(dma_read_error),
        .dma_write_valid(dma_write_valid),
        .dma_write_ready(dma_write_ready),
        .dma_write_addr(dma_write_addr),
        .dma_write_len(dma_write_len),
        .dma_write_bank(dma_write_bank),
        .dma_write_done(dma_write_done),
        .dma_write_error(dma_write_error),
        .desc_update_valid(dma_desc_update_valid),
        .desc_update_target(dma_desc_update_target),
        .desc_update_bank(dma_desc_update_bank),
        .desc_update_size(dma_desc_update_size),
        .desc_invalidate_valid(dma_desc_invalidate_valid),
        .desc_invalidate_target(dma_desc_invalidate_target),
        .busy(dma_manager_busy),
        .error(dma_manager_error),
        .error_code(dma_manager_error_code)
    );

    // Shared scratchpad. Port ownership is kept explicit for board adapters.
    wire spad_a_single_error;
    wire spad_a_double_error;
    wire spad_b_single_error;
    wire spad_b_double_error;

    nac_scratchpad #(
        .BANKS(SRAM_BANKS),
        .BANK_BITS(BANK_BITS),
        .ADDR_WIDTH(SPAD_ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ENABLE_ECC(`NAC_CFG_ENABLE_ECC(HW_CFG))
    ) scratchpad (
        .clk(clk),
        .a_we(spad_a_we),
        .a_bank(spad_a_bank),
        .a_addr(spad_a_addr),
        .a_wdata(spad_a_wdata),
        .a_rdata(spad_a_rdata),
        .b_we(spad_b_we),
        .b_bank(spad_b_bank),
        .b_addr(spad_b_addr),
        .b_wdata(spad_b_wdata),
        .b_rdata(spad_b_rdata),
        .a_single_error(spad_a_single_error),
        .a_double_error(spad_a_double_error),
        .b_single_error(spad_b_single_error),
        .b_double_error(spad_b_double_error)
    );

    assign soc_done = mep_done;
    assign soc_error = mep_pkt_error | mep_error | core_error | dsp_error |
                       dma_manager_error | bank_alloc_fail | bank_free_error |
                       spad_a_double_error | spad_b_double_error;
    assign soc_error_flags = {
        spad_a_double_error | spad_b_double_error,
        bank_alloc_fail | bank_free_error,
        dma_manager_error,
        dsp_error,
        core_error,
        mep_error,
        mep_pkt_error,
        tisa_load_error | tisa_run_error
    };
endmodule
