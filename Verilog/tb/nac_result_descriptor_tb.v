`timescale 1ns/1ps
`include "../rtl/nac_descriptor.vh"

module nac_result_descriptor_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg wr_valid;
    reg [1:0] wr_idx;
    reg [95:0] wr_desc;
    reg rd0_en;
    reg [1:0] rd0_idx;
    wire rd0_valid;
    wire [95:0] rd0_desc;
    wire single_error;
    wire double_error;

    nac_result_table #(
        .ENTRIES(4),
        .IDX_WIDTH(2),
        .DESC_WIDTH(`NAC_DESC_MIN_WIDTH),
        .ENABLE_ECC(1)
    ) dut (
        .clk(clk),
        .rst(rst),
        .wr_valid(wr_valid),
        .wr_idx(wr_idx),
        .wr_desc(wr_desc),
        .rd0_en(rd0_en),
        .rd0_idx(rd0_idx),
        .rd0_valid(rd0_valid),
        .rd0_desc(rd0_desc),
        .rd1_en(1'b0),
        .rd1_idx(2'd0),
        .rd1_valid(),
        .rd1_desc(),
        .free_valid(1'b0),
        .free_idx(2'd0),
        .forward_valid(1'b0),
        .forward_src_idx(2'd0),
        .forward_dst_idx(2'd0),
        .single_error(single_error),
        .double_error(double_error)
    );

    initial begin
        rst = 1'b1;
        wr_valid = 1'b0;
        wr_idx = 2'd0;
        wr_desc = 96'd0;
        rd0_en = 1'b0;
        rd0_idx = 2'd0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        wr_desc[15:0] = 16'hcafe;
        wr_desc[`NAC_DESC_FLAG_IS_BFP] = 1'b1;
        wr_desc[`NAC_DESC_FLAG_IS_SPARSE_2_4] = 1'b1;
        wr_desc[`NAC_DESC_FLAG_IS_RLE] = 1'b1;
        @(negedge clk);
        wr_valid = 1'b1;
        wr_idx = 2'd1;
        @(negedge clk);
        wr_valid = 1'b0;

        rd0_idx = 2'd1;
        rd0_en = 1'b1;
        @(posedge clk);
        #1;
        if (!rd0_valid) $fatal(1, "descriptor read not valid");
        if (rd0_desc[15:0] != 16'hcafe ||
            !rd0_desc[`NAC_DESC_FLAG_IS_BFP] ||
            !rd0_desc[`NAC_DESC_FLAG_IS_SPARSE_2_4] ||
            !rd0_desc[`NAC_DESC_FLAG_IS_RLE]) begin
            $fatal(1, "descriptor flags were not preserved");
        end
        if (single_error || double_error) $fatal(1, "unexpected descriptor ECC error");

        $display("nac_result_descriptor_tb PASS");
        $finish;
    end
endmodule
