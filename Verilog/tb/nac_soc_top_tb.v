`timescale 1ns/1ps
`include "../rtl/nac_config.vh"
`include "../rtl/nac_descriptor.vh"

module nac_soc_top_tb;
    localparam LANES = 4;
    localparam DATA_WIDTH = 16;
    localparam ACC_WIDTH = 40;
    localparam IDX_WIDTH = 10;
    localparam MAX_ARITY = 4;
    localparam MAX_CONSTS = 4;
    localparam DESC_WIDTH = `NAC_DESC_MIN_WIDTH;
    localparam BANK_BITS = 2;
    localparam SPAD_ADDR_WIDTH = 8;
    localparam CONTEXT_BITS = 1;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n;
    reg soc_start;
    reg [7:0] mep_byte;
    reg mep_valid;
    wire mep_ready;
    wire soc_busy;
    wire soc_done;
    wire soc_error;

    nac_soc_top #(
        .HW_CFG(`NAC_CFG_FPGA_TINY_EDGE),
        .IDX_WIDTH(IDX_WIDTH),
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS),
        .DESC_WIDTH(DESC_WIDTH),
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SRAM_BANKS(4),
        .BANK_BITS(BANK_BITS),
        .SPAD_ADDR_WIDTH(SPAD_ADDR_WIDTH),
        .CONTEXTS(2),
        .CONTEXT_BITS(CONTEXT_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .soc_start(soc_start),
        .core_cmd_load_model(1'b0),
        .core_cmd_clear_model(1'b0),
        .context_id({CONTEXT_BITS{1'b0}}),
        .soc_busy(soc_busy),
        .soc_done(soc_done),
        .soc_error(soc_error),
        .mep_plan_size(32'd1),
        .mep_plan_byte_valid(mep_valid),
        .mep_plan_byte_ready(mep_ready),
        .mep_plan_byte(mep_byte),
        .csr_we(1'b0),
        .csr_addr(8'd0),
        .csr_wdata(64'd0),
        .csr_rdata(),
        .num_outputs(16'd0),
        .ops_byte_valid(1'b0),
        .ops_byte_ready(),
        .ops_byte(8'd0),
        .perm_cfg_we(1'b0),
        .perm_cfg_id(8'd0),
        .perm_cfg_arity(4'd0),
        .perm_cfg_needs_consts(1'b0),
        .op_cfg_we(1'b0),
        .op_cfg_id(8'd0),
        .op_cfg_kernel_class(8'd0),
        .op_cfg_dsp_mode(8'd0),
        .op_cfg_uses_dsp(1'b0),
        .op_cfg_multi_pass(1'b0),
        .mmap_cfg_we(1'b0),
        .mmap_cfg_tick({IDX_WIDTH{1'b0}}),
        .mmap_cfg_slot(8'd0),
        .mmap_cfg_valid(1'b0),
        .mmap_cfg_action(8'd0),
        .mmap_cfg_target(16'd0),
        .mmap_cfg_static(1'b0),
        .input_req_valid(),
        .input_kind(),
        .input_id(),
        .input_req_ready(1'b1),
        .input_desc_valid(1'b0),
        .input_desc({DESC_WIDTH{1'b0}}),
        .branch_req_valid(),
        .branch_predicate_idx(),
        .branch_true_target(),
        .branch_false_target(),
        .branch_req_ready(1'b1),
        .branch_resp_valid(1'b0),
        .branch_take_true(1'b0),
        .tisa_load_start(1'b0),
        .tisa_manifest_size(32'd0),
        .tisa_manifest_byte_valid(1'b0),
        .tisa_manifest_byte_ready(),
        .tisa_manifest_byte(8'd0),
        .tisa_text_valid(1'b0),
        .tisa_text_ready(),
        .tisa_text_byte(8'd0),
        .tisa_text_last(1'b0),
        .tisa_out_valid(),
        .tisa_out_ready(1'b1),
        .tisa_out_byte(),
        .tisa_out_last(),
        .tisa_result_ref_value(64'd0),
        .tisa_requires_external_engine(),
        .mep_io_req_valid(),
        .mep_io_req_ready(1'b1),
        .mep_io_opcode(),
        .mep_io_out_key(),
        .mep_io_arg0(),
        .mep_io_const_id(),
        .mep_io_done(1'b0),
        .mep_io_error(1'b0),
        .mep_io_result_key(8'd0),
        .mep_io_result_value(64'd0),
        .mep_io_result_valid(1'b0),
        .train_start(),
        .train_ready(1'b1),
        .train_model_id(),
        .train_loss_type(),
        .train_count_in(),
        .train_in_keys(),
        .train_target_count(),
        .train_target_keys(),
        .train_loss_key(),
        .train_lr_key(),
        .train_logits_key(),
        .train_head_weight_name_id(),
        .train_head_bias_name_id(),
        .train_done(1'b0),
        .train_error(1'b0),
        .zero_grad_start(),
        .zero_grad_ready(1'b1),
        .zero_grad_model_id(),
        .zero_grad_done(1'b0),
        .zero_grad_error(1'b0),
        .save_weights_start(),
        .save_weights_ready(1'b1),
        .save_model_id(),
        .save_path_key(),
        .save_type(),
        .save_weights_done(1'b0),
        .save_weights_error(1'b0),
        .dma_cfg_we(1'b0),
        .dma_cfg_target(8'd0),
        .dma_cfg_ext_addr(64'd0),
        .dma_cfg_size_bytes(32'd0),
        .dma_cfg_static(1'b0),
        .mark_dirty_valid(1'b0),
        .mark_dirty_target(8'd0),
        .dma_read_valid(),
        .dma_read_ready(1'b1),
        .dma_read_addr(),
        .dma_read_len(),
        .dma_read_bank(),
        .dma_read_done(1'b0),
        .dma_read_error(1'b0),
        .dma_write_valid(),
        .dma_write_ready(1'b1),
        .dma_write_addr(),
        .dma_write_len(),
        .dma_write_bank(),
        .dma_write_done(1'b0),
        .dma_write_error(1'b0),
        .spad_a_we(1'b0),
        .spad_a_bank({BANK_BITS{1'b0}}),
        .spad_a_addr({SPAD_ADDR_WIDTH{1'b0}}),
        .spad_a_wdata({DATA_WIDTH{1'b0}}),
        .spad_a_rdata(),
        .spad_b_we(1'b0),
        .spad_b_bank({BANK_BITS{1'b0}}),
        .spad_b_addr({SPAD_ADDR_WIDTH{1'b0}}),
        .spad_b_wdata({DATA_WIDTH{1'b0}}),
        .spad_b_rdata(),
        .tensor_a_vec({LANES*DATA_WIDTH{1'b0}}),
        .tensor_b_vec({LANES*DATA_WIDTH{1'b0}}),
        .tensor_weight_is_bfp(1'b0),
        .tensor_weight_is_palette(1'b0),
        .tensor_weight_is_sparse_2_4(1'b0),
        .tensor_result_is_rle(1'b0),
        .tensor_bfp_mantissas({LANES*4{1'b0}}),
        .tensor_bfp_exp(8'sd0),
        .tensor_palette({16*DATA_WIDTH{1'b0}}),
        .tensor_palette_indices({LANES*4{1'b0}}),
        .tensor_sparse_values({(LANES/2)*DATA_WIDTH{1'b0}}),
        .tensor_sparse_indices({(LANES/2)*2{1'b0}}),
        .dsp_result_valid(),
        .dsp_result_ready(1'b1),
        .dsp_result_vec(),
        .dsp_scalar_out(),
        .dsp_result_desc({DESC_WIDTH{1'b0}}),
        .dsp_result_desc_valid(1'b1),
        .soc_error_flags()
    );

    integer cycles;

    initial begin
        rst_n = 1'b0;
        soc_start = 1'b0;
        mep_byte = 8'hff; // EXEC_HALT
        mep_valid = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        soc_start = 1'b1;
        mep_valid = 1'b1;
        @(negedge clk);
        soc_start = 1'b0;
        while (!mep_ready) @(negedge clk);
        @(negedge clk);
        mep_valid = 1'b0;

        cycles = 0;
        while (!soc_done && !soc_error && cycles < 100) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (soc_error) $fatal(1, "nac_soc_top reported error");
        if (!soc_done) $fatal(1, "nac_soc_top timeout");

        $display("nac_soc_top_tb PASS");
        $finish;
    end
endmodule
