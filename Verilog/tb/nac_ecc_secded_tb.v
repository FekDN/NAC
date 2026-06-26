`timescale 1ns/1ps

module nac_ecc_secded_tb;
    reg [31:0] data_in;
    reg [6:0] ecc_in;
    wire [6:0] ecc_out;
    wire [31:0] data_corrected;
    wire single_error;
    wire double_error;

    nac_ecc_secded #(
        .DATA_WIDTH(32),
        .ECC_WIDTH(7)
    ) dut (
        .data_in(data_in),
        .ecc_in(ecc_in),
        .ecc_out(ecc_out),
        .data_corrected(data_corrected),
        .single_error(single_error),
        .double_error(double_error)
    );

    reg [6:0] saved_ecc;

    initial begin
        data_in = 32'h1234_5678;
        ecc_in = 7'd0;
        #1;
        saved_ecc = ecc_out;

        ecc_in = saved_ecc;
        #1;
        if (data_corrected != 32'h1234_5678 || single_error || double_error) begin
            $fatal(1, "clean ECC read failed");
        end

        data_in = 32'h1234_5678 ^ 32'h0000_0020;
        ecc_in = saved_ecc;
        #1;
        if (data_corrected != 32'h1234_5678 || !single_error || double_error) begin
            $fatal(1, "single-bit ECC correction failed");
        end

        data_in = 32'h1234_5678 ^ 32'h0000_0060;
        ecc_in = saved_ecc;
        #1;
        if (!double_error || single_error) begin
            $fatal(1, "double-bit ECC detection failed");
        end

        $display("nac_ecc_secded_tb PASS");
        $finish;
    end
endmodule
