`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_core_control_flow_tb;
    localparam IDX_WIDTH = 3;
    localparam DESC_WIDTH = 96;
    localparam OPS_LEN = 30;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg cmd_load_model;
    reg cmd_run_inference;
    reg cmd_clear_model;
    reg context_id;
    reg [7:0] rom [0:OPS_LEN-1];
    reg [5:0] rp;
    wire ops_byte_valid = (rp < OPS_LEN);
    wire ops_byte_ready;
    wire [7:0] ops_byte = rom[rp];

    wire input_req_valid;
    wire [7:0] input_kind;
    wire [15:0] input_id;
    reg input_desc_valid;
    wire branch_req_valid;
    wire [IDX_WIDTH-1:0] branch_predicate_idx;
    wire [IDX_WIDTH-1:0] branch_true_target;
    wire [IDX_WIDTH-1:0] branch_false_target;
    reg branch_resp_valid;
    reg branch_take_true;
    wire done;
    wire error;
    wire model_configured;

    nac_core #(
        .IDX_WIDTH(IDX_WIDTH),
        .MAX_ARITY(4),
        .MAX_CONSTS(4),
        .DESC_WIDTH(DESC_WIDTH),
        .MAX_MMAP_CMDS_PER_TICK(1),
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
        .input_desc(96'h0000_0000_0123_4567_89ab_cdef),
        .branch_req_valid(branch_req_valid),
        .branch_predicate_idx(branch_predicate_idx),
        .branch_true_target(branch_true_target),
        .branch_false_target(branch_false_target),
        .branch_req_ready(1'b1),
        .branch_resp_valid(branch_resp_valid),
        .branch_take_true(branch_take_true),
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

    reg branch_resp_pending;
    integer seen_true;
    integer seen_false;

    always @(posedge clk) begin
        if (rst || cmd_load_model) begin
            rp <= 0;
        end else if (ops_byte_valid && ops_byte_ready) begin
            rp <= rp + 1'b1;
        end

        input_desc_valid <= !rst && input_req_valid;

        if (rst) begin
            branch_resp_pending <= 1'b0;
            branch_resp_valid <= 1'b0;
        end else begin
            branch_resp_pending <= branch_req_valid;
            branch_resp_valid <= branch_resp_pending;
        end

        if (!rst && input_req_valid) begin
            if (input_id == 16'd7) seen_true <= seen_true + 1;
            if (input_id == 16'd9) seen_false <= seen_false + 1;
        end
    end

    task run_once;
        input take_true;
        integer cycles;
        begin
            branch_take_true = take_true;
            @(negedge clk);
            cmd_run_inference = 1'b1;
            @(negedge clk);
            cmd_run_inference = 1'b0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (cycles > 160) $fatal(1, "control-flow run timeout");
                if (error) $fatal(1, "control-flow run error");
            end
        end
    endtask

    integer load_cycles;

    initial begin
        // Instruction 0: CONTROL_FLOW C=[3,1,1], D=[0].
        rom[0] = `NAC_OP_CONTROL_FLOW;
        rom[1] = 8'd0;
        rom[2] = 8'd3;
        rom[3] = 8'd0;
        rom[4] = 8'd1;
        rom[5] = 8'd0;
        rom[6] = 8'd1;
        rom[7] = 8'd0;
        rom[8] = 8'd0;
        rom[9] = 8'd0;
        // Instruction 1: true branch INPUT B=1, id=7.
        rom[10] = `NAC_OP_INPUT;
        rom[11] = 8'd1;
        rom[12] = 8'd0;
        rom[13] = 8'd0;
        rom[14] = 8'd7;
        rom[15] = 8'd0;
        // Instruction 2: false branch INPUT B=1, id=9.
        rom[16] = `NAC_OP_INPUT;
        rom[17] = 8'd1;
        rom[18] = 8'd0;
        rom[19] = 8'd0;
        rom[20] = 8'd9;
        rom[21] = 8'd0;
        // Instruction 3: OUTPUT B=0, C=[2,0], D=[-1].
        rom[22] = `NAC_OP_OUTPUT;
        rom[23] = 8'd0;
        rom[24] = 8'd2;
        rom[25] = 8'd0;
        rom[26] = 8'd0;
        rom[27] = 8'd0;
        rom[28] = 8'hff;
        rom[29] = 8'hff;

        rst = 1'b1;
        start = 1'b0;
        cmd_load_model = 1'b0;
        cmd_run_inference = 1'b0;
        cmd_clear_model = 1'b0;
        context_id = 1'b0;
        input_desc_valid = 1'b0;
        branch_resp_valid = 1'b0;
        branch_take_true = 1'b0;
        branch_resp_pending = 1'b0;
        seen_true = 0;
        seen_false = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        @(negedge clk);
        cmd_load_model = 1'b1;
        @(negedge clk);
        cmd_load_model = 1'b0;

        load_cycles = 0;
        while (!model_configured) begin
            @(posedge clk);
            load_cycles = load_cycles + 1;
            if (load_cycles > 160) $fatal(1, "control-flow load timeout");
            if (error) $fatal(1, "control-flow load error");
        end

        run_once(1'b0);
        if (seen_false != 1 || seen_true != 0) $fatal(1, "false branch did not execute exclusively");

        run_once(1'b1);
        if (seen_false != 1 || seen_true != 1) begin
            $fatal(1, "true branch did not skip false branch seen_true=%0d seen_false=%0d true_target=%0d false_target=%0d",
                   seen_true, seen_false, branch_true_target, branch_false_target);
        end
        if (branch_true_target != 3'd1 || branch_false_target != 3'd2) $fatal(1, "bad branch targets");

        $display("nac_core_control_flow_tb PASS");
        $finish;
    end
endmodule
