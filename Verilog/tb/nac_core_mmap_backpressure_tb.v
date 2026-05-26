`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_core_mmap_backpressure_tb;
    localparam IDX_WIDTH = 2;
    localparam DESC_WIDTH = 64;
    localparam OPS_LEN = 14;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] rom [0:OPS_LEN-1];
    reg [4:0] rp;
    wire ops_byte_valid = (rp < OPS_LEN);
    wire ops_byte_ready;
    wire [7:0] ops_byte = rom[rp];

    reg mmap_cfg_we;
    reg [IDX_WIDTH-1:0] mmap_cfg_tick;
    reg mmap_cfg_slot;
    reg mmap_cfg_valid;
    reg [7:0] mmap_cfg_action;
    reg [15:0] mmap_cfg_target;

    wire input_req_valid;
    wire [7:0] input_kind;
    wire [15:0] input_id;
    reg input_desc_valid;

    wire free_valid;
    reg free_ready;
    wire [15:0] free_target;

    reg result_rd0_en;
    reg [IDX_WIDTH-1:0] result_rd0_idx;
    wire result_rd0_valid;
    wire [DESC_WIDTH-1:0] result_rd0_desc;

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
        .mmap_cfg_we(mmap_cfg_we),
        .mmap_cfg_tick(mmap_cfg_tick),
        .mmap_cfg_slot(mmap_cfg_slot),
        .mmap_cfg_valid(mmap_cfg_valid),
        .mmap_cfg_action(mmap_cfg_action),
        .mmap_cfg_target(mmap_cfg_target),
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
        .result_rd0_en(result_rd0_en),
        .result_rd0_idx(result_rd0_idx),
        .result_rd0_valid(result_rd0_valid),
        .result_rd0_desc(result_rd0_desc),
        .result_rd1_en(1'b0),
        .result_rd1_idx({IDX_WIDTH{1'b0}}),
        .result_rd1_valid(),
        .result_rd1_desc(),
        .input_req_valid(input_req_valid),
        .input_kind(input_kind),
        .input_id(input_id),
        .input_req_ready(1'b1),
        .input_desc_valid(input_desc_valid),
        .input_desc(64'h0123_4567_89ab_cdef),
        .preload_valid(),
        .preload_ready(1'b1),
        .preload_target(),
        .free_valid(free_valid),
        .free_ready(free_ready),
        .free_target(free_target),
        .save_result_valid(),
        .save_result_ready(1'b1),
        .save_result_src(),
        .save_result_target(),
        .forward_valid(),
        .forward_ready(1'b1),
        .forward_src(),
        .forward_dst(),
        .busy(),
        .done(done),
        .error(error)
    );

    always @(posedge clk) begin
        if (rst || start) begin
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
        mmap_cfg_we = 1'b0;
        mmap_cfg_tick = {IDX_WIDTH{1'b0}};
        mmap_cfg_slot = 1'b0;
        mmap_cfg_valid = 1'b0;
        mmap_cfg_action = 8'd0;
        mmap_cfg_target = 16'd0;
        input_desc_valid = 1'b0;
        free_ready = 1'b0;
        result_rd0_en = 1'b0;
        result_rd0_idx = {IDX_WIDTH{1'b0}};
        cycles = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        @(negedge clk);
        mmap_cfg_we = 1'b1;
        mmap_cfg_tick = 2'd0;
        mmap_cfg_slot = 1'b0;
        mmap_cfg_valid = 1'b1;
        mmap_cfg_action = `NAC_MMAP_FREE;
        mmap_cfg_target = 16'd0;
        @(negedge clk);
        mmap_cfg_we = 1'b0;

        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (!free_valid) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 100) $fatal(1, "free_valid timeout");
            if (error) $fatal(1, "core error before free");
        end

        if (done) $fatal(1, "core done asserted while MMAP FREE is stalled");
        if (free_target != 16'd0) $fatal(1, "bad free target");

        result_rd0_en = 1'b1;
        result_rd0_idx = 2'd0;
        @(posedge clk);
        @(negedge clk);
        if (!result_rd0_valid) $fatal(1, "descriptor was freed before FREE command was accepted");
        if (result_rd0_desc != 64'h0123_4567_89ab_cdef) $fatal(1, "descriptor changed while FREE was stalled");

        @(negedge clk);
        free_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        if (result_rd0_valid) begin
            $fatal(1, "descriptor was not freed after FREE command was accepted free_valid=%0b free_ready=%0b table_free=%0b ram_valid=%0b",
                   free_valid, free_ready, dut.table_free_valid, dut.result_table.valid_ram[0]);
        end

        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 160) $fatal(1, "done timeout after MMAP drain");
            if (error) $fatal(1, "core error after free");
        end

        $display("nac_core_mmap_backpressure_tb PASS");
        $finish;
    end
endmodule
