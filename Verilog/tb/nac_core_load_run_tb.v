`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_core_load_run_tb;
    localparam IDX_WIDTH = 3;
    localparam DESC_WIDTH = 64;
    localparam OPS_LEN = 14;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg cmd_load_model;
    reg cmd_run_inference;
    reg cmd_clear_model;
    reg context_id;
    reg [7:0] rom [0:OPS_LEN-1];
    reg [4:0] rp;
    wire ops_byte_valid = (rp < OPS_LEN);
    wire ops_byte_ready;
    wire [7:0] ops_byte = rom[rp];

    wire input_req_valid;
    wire [7:0] input_kind;
    wire [15:0] input_id;
    reg input_desc_valid;
    wire model_configured;
    wire done;
    wire error;

    nac_core #(
        .IDX_WIDTH(IDX_WIDTH),
        .MAX_ARITY(4),
        .MAX_CONSTS(4),
        .DESC_WIDTH(DESC_WIDTH),
        .MAX_MMAP_CMDS_PER_TICK(2),
        .MMAP_SLOT_WIDTH(1)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .cmd_load_model(cmd_load_model),
        .cmd_run_inference(cmd_run_inference),
        .cmd_clear_model(cmd_clear_model),
        .context_id(context_id),
        .csr_we(1'b0),
        .csr_addr(8'd0),
        .csr_wdata(64'd0),
        .csr_rdata(),
        .num_outputs(16'd1),
        .ops_byte_valid(ops_byte_valid),
        .ops_byte_ready(ops_byte_ready),
        .ops_byte(ops_byte),
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
        .mmap_cfg_slot(1'b0),
        .mmap_cfg_valid(1'b0),
        .mmap_cfg_action(8'd0),
        .mmap_cfg_target(16'd0),
        .mmap_cfg_static(1'b0),
        .kernel_start(),
        .kernel_op_a(),
        .kernel_op_b(),
        .kernel_class(),
        .dsp_mode(),
        .kernel_instr_idx(),
        .kernel_c_count(),
        .kernel_c_flat(),
        .kernel_d_count(),
        .kernel_d_flat(),
        .kernel_done(1'b0),
        .kernel_result_desc({DESC_WIDTH{1'b0}}),
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
        .input_req_ready(1'b1),
        .input_desc_valid(input_desc_valid),
        .input_desc(64'hfeed_cafe_1234_5678),
        .preload_valid(),
        .preload_ready(1'b1),
        .preload_target(),
        .free_valid(),
        .free_ready(1'b1),
        .free_target(),
        .save_result_valid(),
        .save_result_ready(1'b1),
        .save_result_src(),
        .save_result_target(),
        .forward_valid(),
        .forward_ready(1'b1),
        .forward_src(),
        .forward_dst(),
        .model_configured(model_configured),
        .watchdog_timeout(),
        .busy(),
        .done(done),
        .error(error)
    );

    always @(posedge clk) begin
        if (rst || cmd_load_model) begin
            rp <= 0;
        end else if (ops_byte_valid && ops_byte_ready) begin
            rp <= rp + 1'b1;
        end

        if (rst) begin
            input_desc_valid <= 1'b0;
        end else begin
            input_desc_valid <= input_req_valid;
        end
    end

    integer cycles;
    integer rp_after_load;
    integer run_count;

    initial begin
        // Instruction 0: INPUT B=1, C=[0, 7].
        rom[0] = `NAC_OP_INPUT;
        rom[1] = 8'd1;
        rom[2] = 8'd0;
        rom[3] = 8'd0;
        rom[4] = 8'd7;
        rom[5] = 8'd0;
        // Instruction 1: OUTPUT B=0, C=[2, 0], D=[-1].
        rom[6] = `NAC_OP_OUTPUT;
        rom[7] = 8'd0;
        rom[8] = 8'd2;
        rom[9] = 8'd0;
        rom[10] = 8'd0;
        rom[11] = 8'd0;
        rom[12] = 8'hff;
        rom[13] = 8'hff;

        rst = 1'b1;
        start = 1'b0;
        cmd_load_model = 1'b0;
        cmd_run_inference = 1'b0;
        cmd_clear_model = 1'b0;
        context_id = 1'b1;
        input_desc_valid = 1'b0;
        cycles = 0;
        rp_after_load = 0;
        run_count = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        @(negedge clk);
        cmd_load_model = 1'b1;
        @(negedge clk);
        cmd_load_model = 1'b0;

        while (!model_configured) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 120) $fatal(1, "model load timeout rp=%0d error=%0b", rp, error);
            if (error) $fatal(1, "core error during load");
        end
        if (rp != OPS_LEN) $fatal(1, "loader did not consume exact OPS stream");
        rp_after_load = rp;

        @(negedge clk);
        cmd_run_inference = 1'b1;
        @(negedge clk);
        cmd_run_inference = 1'b0;

        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 220) $fatal(1, "run timeout rp=%0d error=%0b", rp, error);
            if (error) $fatal(1, "core error during run");
            if (input_req_valid && (input_kind != 8'd1 || input_id != 16'd7)) begin
                $fatal(1, "bad cached input request");
            end
        end

        if (rp != rp_after_load) $fatal(1, "run phase consumed OPS bytes");
        run_count = run_count + 1;

        @(negedge clk);
        cmd_run_inference = 1'b1;
        @(negedge clk);
        cmd_run_inference = 1'b0;

        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 320) $fatal(1, "second run timeout rp=%0d error=%0b", rp, error);
            if (error) $fatal(1, "core error during second run");
        end

        if (rp != rp_after_load) $fatal(1, "second run consumed OPS bytes");
        run_count = run_count + 1;
        if (run_count != 2) $fatal(1, "cached run count mismatch");

        @(negedge clk);
        cmd_clear_model = 1'b1;
        @(negedge clk);
        cmd_clear_model = 1'b0;
        @(posedge clk);
        if (model_configured) $fatal(1, "clear model did not clear context");

        $display("nac_core_load_run_tb PASS");
        $finish;
    end
endmodule
