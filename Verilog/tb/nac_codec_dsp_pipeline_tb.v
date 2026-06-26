`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_codec_dsp_pipeline_tb;
    localparam LANES = 4;
    localparam DATA_WIDTH = 16;
    localparam ACC_WIDTH = 40;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg [7:0] mode;
    reg signed [ACC_WIDTH-1:0] acc_in;
    reg in_valid;
    wire in_ready;
    reg [LANES*DATA_WIDTH-1:0] a_vec;
    reg [LANES*DATA_WIDTH-1:0] b_mem_vec;
    reg weight_is_bfp;
    reg weight_is_palette;
    reg weight_is_sparse_2_4;
    reg result_is_rle;
    reg [LANES*4-1:0] weight_bfp_mantissas;
    reg signed [7:0] weight_bfp_exp;
    reg [16*DATA_WIDTH-1:0] weight_palette;
    reg [LANES*4-1:0] weight_palette_indices;
    reg [(LANES/2)*DATA_WIDTH-1:0] weight_sparse_values;
    reg [(LANES/2)*2-1:0] weight_sparse_indices;
    wire out_valid;
    reg out_ready;
    wire [LANES*DATA_WIDTH-1:0] out_vec;
    wire signed [ACC_WIDTH-1:0] scalar_out;
    wire [LANES*DATA_WIDTH-1:0] rle_values;
    wire [LANES*8-1:0] rle_counts;
    wire [LANES-1:0] rle_is_zero_run;
    wire [7:0] rle_token_count;
    wire rle_valid;
    wire error;

    nac_codec_dsp_pipeline #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(0),
        .ACC_WIDTH(ACC_WIDTH),
        .ENABLE_PALETTE_DECODE(1),
        .ENABLE_STRUCTURED_SPARSITY(1),
        .ENABLE_RLE_ENCODE(1)
    ) dut (
        .clk(clk),
        .rst(rst),
        .cfg_mode(mode),
        .acc_in(acc_in),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .a_vec(a_vec),
        .b_mem_vec(b_mem_vec),
        .weight_is_bfp(weight_is_bfp),
        .weight_is_palette(weight_is_palette),
        .weight_is_sparse_2_4(weight_is_sparse_2_4),
        .result_is_rle(result_is_rle),
        .weight_bfp_mantissas(weight_bfp_mantissas),
        .weight_bfp_exp(weight_bfp_exp),
        .weight_palette(weight_palette),
        .weight_palette_indices(weight_palette_indices),
        .weight_sparse_values(weight_sparse_values),
        .weight_sparse_indices(weight_sparse_indices),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_vec(out_vec),
        .scalar_out(scalar_out),
        .sfu_req_valid(),
        .sfu_req_ready(1'b1),
        .sfu_req_mode(),
        .sfu_req_vec(),
        .sfu_resp_valid(1'b0),
        .sfu_resp_vec({LANES*DATA_WIDTH{1'b0}}),
        .rle_values(rle_values),
        .rle_counts(rle_counts),
        .rle_is_zero_run(rle_is_zero_run),
        .rle_token_count(rle_token_count),
        .rle_valid(rle_valid),
        .error(error)
    );

    task put_lane;
        output [LANES*DATA_WIDTH-1:0] vec;
        input integer lane;
        input signed [DATA_WIDTH-1:0] value;
        begin
            vec[lane*DATA_WIDTH +: DATA_WIDTH] = value;
        end
    endtask

    integer i;

    initial begin
        rst = 1'b1;
        mode = `NAC_DSP_MUL;
        acc_in = 0;
        in_valid = 1'b0;
        out_ready = 1'b1;
        a_vec = 0;
        b_mem_vec = 0;
        weight_is_bfp = 1'b0;
        weight_is_palette = 1'b0;
        weight_is_sparse_2_4 = 1'b0;
        result_is_rle = 1'b0;
        weight_bfp_mantissas = 0;
        weight_bfp_exp = 0;
        weight_palette = 0;
        weight_palette_indices = 0;
        weight_sparse_values = 0;
        weight_sparse_indices = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        put_lane(a_vec, 0, 10);
        put_lane(a_vec, 1, 20);
        put_lane(a_vec, 2, 30);
        put_lane(a_vec, 3, 40);
        weight_sparse_values[0 +: 16] = 16'd2;
        weight_sparse_values[16 +: 16] = 16'd3;
        weight_sparse_indices = {2'd2, 2'd0};
        weight_is_sparse_2_4 = 1'b1;
        result_is_rle = 1'b1;

        @(posedge clk);
        in_valid = 1'b1;
        @(posedge clk);
        in_valid = 1'b0;
        wait(out_valid);
        #1;
        if (error) $fatal(1, "codec sparse path reported error");
        if ($signed(out_vec[0 +: 16]) != 20 ||
            $signed(out_vec[16 +: 16]) != 0 ||
            $signed(out_vec[32 +: 16]) != 90 ||
            $signed(out_vec[48 +: 16]) != 0) begin
            $fatal(1, "codec sparse->DSP path failed");
        end
        if (!rle_valid || rle_token_count != 8'd4) begin
            $fatal(1, "RLE encode sideband failed");
        end
        @(posedge clk);
        wait(!out_valid);

        weight_is_sparse_2_4 = 1'b0;
        weight_is_palette = 1'b1;
        result_is_rle = 1'b0;
        for (i = 0; i < 16; i = i + 1) begin
            weight_palette[i*DATA_WIDTH +: DATA_WIDTH] = i + 1;
        end
        weight_palette_indices = {4'd3, 4'd2, 4'd1, 4'd0};

        @(posedge clk);
        in_valid = 1'b1;
        @(posedge clk);
        in_valid = 1'b0;
        wait(out_valid);
        #1;
        if (error) $fatal(1, "codec palette path reported error");
        if ($signed(out_vec[0 +: 16]) != 10 ||
            $signed(out_vec[16 +: 16]) != 40 ||
            $signed(out_vec[32 +: 16]) != 90 ||
            $signed(out_vec[48 +: 16]) != 160) begin
            $fatal(1, "codec palette->DSP path failed");
        end

        $display("nac_codec_dsp_pipeline_tb PASS");
        $finish;
    end
endmodule
