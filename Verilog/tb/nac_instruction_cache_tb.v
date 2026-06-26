`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_instruction_cache_tb;
    localparam MAX_ARITY = 4;
    localparam MAX_CONSTS = 4;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg clear_context;
    reg clear_context_id;
    reg load_start;
    reg load_context;
    reg dec_valid;
    wire dec_ready;
    reg [15:0] dec_index;
    reg [7:0] dec_a;
    reg [7:0] dec_b;
    reg [3:0] dec_c_count;
    reg [MAX_CONSTS*16-1:0] dec_c_flat;
    reg [3:0] dec_d_count;
    reg [MAX_ARITY*16-1:0] dec_d_flat;
    wire load_done;
    wire load_error;
    reg run_start;
    reg run_context;
    reg redirect_valid;
    reg [1:0] redirect_index;
    wire instr_valid;
    reg instr_ready;
    wire [15:0] instr_index;
    wire [7:0] instr_a;
    wire [7:0] instr_b;
    wire [3:0] instr_c_count;
    wire [MAX_CONSTS*16-1:0] instr_c_flat;
    wire [3:0] instr_d_count;
    wire [MAX_ARITY*16-1:0] instr_d_flat;
    wire run_active;
    wire run_done;
    wire run_error;
    wire model_configured;

    nac_instruction_cache #(
        .CONTEXTS(2),
        .CONTEXT_BITS(1),
        .IDX_WIDTH(2),
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .clear_context(clear_context),
        .clear_context_id(clear_context_id),
        .load_start(load_start),
        .load_context(load_context),
        .dec_valid(dec_valid),
        .dec_ready(dec_ready),
        .dec_index(dec_index),
        .dec_a(dec_a),
        .dec_b(dec_b),
        .dec_c_count(dec_c_count),
        .dec_c_flat(dec_c_flat),
        .dec_d_count(dec_d_count),
        .dec_d_flat(dec_d_flat),
        .load_done(load_done),
        .load_error(load_error),
        .run_start(run_start),
        .run_context(run_context),
        .redirect_valid(redirect_valid),
        .redirect_index(redirect_index),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .instr_index(instr_index),
        .instr_a(instr_a),
        .instr_b(instr_b),
        .instr_c_count(instr_c_count),
        .instr_c_flat(instr_c_flat),
        .instr_d_count(instr_d_count),
        .instr_d_flat(instr_d_flat),
        .run_active(run_active),
        .run_done(run_done),
        .run_error(run_error),
        .model_configured(model_configured)
    );

    task push_instr;
        input [15:0] idx;
        input [7:0] a;
        input [7:0] b;
        input [3:0] cc;
        input [63:0] cf;
        input [3:0] dc;
        input [63:0] df;
        begin
            @(negedge clk);
            dec_index = idx;
            dec_a = a;
            dec_b = b;
            dec_c_count = cc;
            dec_c_flat = cf;
            dec_d_count = dc;
            dec_d_flat = df;
            dec_valid = 1'b1;
            @(posedge clk);
            if (!dec_ready) $fatal(1, "cache did not accept decode instruction");
            @(negedge clk);
            dec_valid = 1'b0;
        end
    endtask

    integer seen;

    initial begin
        rst = 1'b1;
        clear_context = 1'b0;
        clear_context_id = 1'b0;
        load_start = 1'b0;
        load_context = 1'b1;
        dec_valid = 1'b0;
        dec_index = 16'd0;
        dec_a = 8'd0;
        dec_b = 8'd0;
        dec_c_count = 4'd0;
        dec_c_flat = 64'd0;
        dec_d_count = 4'd0;
        dec_d_flat = 64'd0;
        run_start = 1'b0;
        run_context = 1'b1;
        redirect_valid = 1'b0;
        redirect_index = 2'd0;
        instr_ready = 1'b0;
        seen = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        @(negedge clk);
        load_start = 1'b1;
        @(negedge clk);
        load_start = 1'b0;

        push_instr(16'd0, `NAC_OP_INPUT, 8'd1, 4'd2, {32'd0, 16'd7, 16'd2}, 4'd0, 64'd0);
        push_instr(16'd1, `NAC_OP_OUTPUT, 8'd0, 4'd2, {32'd0, 16'd0, 16'd2}, 4'd1, 64'h0000_0000_0000_ffff);
        @(posedge clk);
        if (!load_done || load_error || !model_configured) $fatal(1, "cache load failed");

        @(negedge clk);
        run_start = 1'b1;
        instr_ready = 1'b1;
        @(negedge clk);
        run_start = 1'b0;

        while (!run_done) begin
            @(posedge clk);
            if (instr_valid && instr_ready) begin
                if (seen == 0 && (instr_a != `NAC_OP_INPUT || instr_index != 16'd0)) begin
                    $fatal(1, "bad cached input instruction");
                end
                if (seen == 1 && (instr_a != `NAC_OP_OUTPUT || instr_index != 16'd1)) begin
                    $fatal(1, "bad cached output instruction");
                end
                seen = seen + 1;
            end
            if (run_error) $fatal(1, "cache run error");
            if (seen > 2) $fatal(1, "cache emitted too many instructions");
        end
        if (seen != 2) $fatal(1, "cache emitted wrong instruction count");

        @(negedge clk);
        run_context = 1'b0;
        run_start = 1'b1;
        @(negedge clk);
        run_start = 1'b0;
        @(posedge clk);
        if (!run_error) $fatal(1, "cache accepted unconfigured context");

        $display("nac_instruction_cache_tb PASS");
        $finish;
    end
endmodule
