`timescale 1ns/1ps

module nac_scratchpad_ecc_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg a_we;
    reg a_bank;
    reg [1:0] a_addr;
    reg [31:0] a_wdata;
    wire [31:0] a_rdata;
    reg b_we;
    reg b_bank;
    reg [1:0] b_addr;
    reg [31:0] b_wdata;
    wire [31:0] b_rdata;
    wire a_single_error;
    wire a_double_error;
    wire b_single_error;
    wire b_double_error;

    nac_scratchpad #(
        .BANKS(2),
        .BANK_BITS(1),
        .ADDR_WIDTH(2),
        .DATA_WIDTH(32),
        .ENABLE_ECC(1)
    ) dut (
        .clk(clk),
        .a_we(a_we),
        .a_bank(a_bank),
        .a_addr(a_addr),
        .a_wdata(a_wdata),
        .a_rdata(a_rdata),
        .b_we(b_we),
        .b_bank(b_bank),
        .b_addr(b_addr),
        .b_wdata(b_wdata),
        .b_rdata(b_rdata),
        .a_single_error(a_single_error),
        .a_double_error(a_double_error),
        .b_single_error(b_single_error),
        .b_double_error(b_double_error)
    );

    initial begin
        a_we = 1'b0;
        a_bank = 1'b0;
        a_addr = 2'd1;
        a_wdata = 32'hdead_beef;
        b_we = 1'b0;
        b_bank = 1'b0;
        b_addr = 2'd0;
        b_wdata = 32'd0;

        @(negedge clk);
        a_we = 1'b1;
        @(negedge clk);
        a_we = 1'b0;
        dut.mem[1] = dut.mem[1] ^ 32'h0000_0010;
        @(posedge clk);
        #1;
        if (a_rdata != 32'hdead_beef || !a_single_error || a_double_error) begin
            $fatal(1, "scratchpad single-bit correction failed");
        end

        @(negedge clk);
        a_we = 1'b1;
        @(negedge clk);
        a_we = 1'b0;
        dut.mem[1] = dut.mem[1] ^ 32'h0000_0030;
        @(posedge clk);
        #1;
        if (!a_double_error) begin
            $fatal(1, "scratchpad double-bit detection failed");
        end

        $display("nac_scratchpad_ecc_tb PASS");
        $finish;
    end
endmodule
