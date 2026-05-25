`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_dsp_pipeline_tb;
    localparam LANES = 4;
    localparam DATA_WIDTH = 32;
    localparam FRAC_BITS = 0;
    localparam ACC_WIDTH = 72;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg [7:0] mode;
    reg signed [ACC_WIDTH-1:0] acc_in;
    reg in_valid;
    wire in_ready;
    reg [LANES*DATA_WIDTH-1:0] a_vec;
    reg [LANES*DATA_WIDTH-1:0] b_vec;
    wire out_valid;
    reg out_ready;
    wire [LANES*DATA_WIDTH-1:0] out_vec;
    wire signed [ACC_WIDTH-1:0] scalar_out;
    wire sfu_req_valid;
    wire [7:0] sfu_req_mode;
    wire [LANES*DATA_WIDTH-1:0] sfu_req_vec;

    nac_dsp_pipeline #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .cfg_mode(mode),
        .acc_in(acc_in),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .a_vec(a_vec),
        .b_vec(b_vec),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_vec(out_vec),
        .scalar_out(scalar_out),
        .sfu_req_valid(sfu_req_valid),
        .sfu_req_ready(1'b1),
        .sfu_req_mode(sfu_req_mode),
        .sfu_req_vec(sfu_req_vec),
        .sfu_resp_valid(1'b0),
        .sfu_resp_vec({LANES*DATA_WIDTH{1'b0}})
    );

    task put_lane;
        output [LANES*DATA_WIDTH-1:0] vec;
        input integer lane;
        input signed [DATA_WIDTH-1:0] value;
        begin
            vec[lane*DATA_WIDTH +: DATA_WIDTH] = value;
        end
    endtask

    task run_vec;
        input [7:0] t_mode;
        begin
            @(posedge clk);
            mode <= t_mode;
            in_valid <= 1'b1;
            @(posedge clk);
            in_valid <= 1'b0;
            wait(out_valid);
            @(posedge clk);
        end
    endtask

    initial begin
        rst = 1'b1;
        mode = `NAC_DSP_NOP;
        acc_in = 0;
        in_valid = 1'b0;
        out_ready = 1'b1;
        a_vec = 0;
        b_vec = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        put_lane(a_vec, 0, 1);
        put_lane(a_vec, 1, 2);
        put_lane(a_vec, 2, 3);
        put_lane(a_vec, 3, 4);
        put_lane(b_vec, 0, 10);
        put_lane(b_vec, 1, 20);
        put_lane(b_vec, 2, 30);
        put_lane(b_vec, 3, 40);

        run_vec(`NAC_DSP_ADD);
        if ($signed(out_vec[0 +: 32]) != 11) $fatal(1, "ADD lane0 failed");
        if ($signed(out_vec[96 +: 32]) != 44) $fatal(1, "ADD lane3 failed");

        run_vec(`NAC_DSP_MUL);
        if ($signed(out_vec[0 +: 32]) != 10) $fatal(1, "MUL lane0 failed");
        if ($signed(out_vec[96 +: 32]) != 160) $fatal(1, "MUL lane3 failed");

        acc_in <= 5;
        run_vec(`NAC_DSP_MAC);
        if (scalar_out != 305) $fatal(1, "MAC sum failed: %0d", scalar_out);

        run_vec(`NAC_DSP_REDUCE_MAX);
        if ($signed(out_vec[0 +: 32]) != 4) $fatal(1, "REDUCE_MAX failed");

        put_lane(a_vec, 0, -4);
        put_lane(a_vec, 1, -1);
        put_lane(a_vec, 2, 0);
        put_lane(a_vec, 3, 5);
        run_vec(`NAC_DSP_RELU);
        if ($signed(out_vec[0 +: 32]) != 0) $fatal(1, "RELU lane0 failed");
        if ($signed(out_vec[96 +: 32]) != 5) $fatal(1, "RELU lane3 failed");

        $display("nac_dsp_pipeline_tb PASS");
        $finish;
    end
endmodule
