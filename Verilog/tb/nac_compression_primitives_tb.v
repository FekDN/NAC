`timescale 1ns/1ps

module nac_compression_primitives_tb;
    reg [15:0] bfp_mant;
    reg signed [7:0] bfp_exp;
    wire [31:0] bfp_values;
    wire [3:0] bfp_overflow;

    nac_bfp_microscale_decode #(
        .LANES(4),
        .MANT_WIDTH(4),
        .EXP_WIDTH(8),
        .OUT_WIDTH(8)
    ) bfp (
        .mantissas(bfp_mant),
        .shared_exp(bfp_exp),
        .values_out(bfp_values),
        .overflow_mask(bfp_overflow)
    );

    reg [31:0] sparse_vals;
    reg [7:0] sparse_idx;
    wire [63:0] sparse_dense;
    wire sparse_valid;

    nac_structured_sparsity_decode #(
        .GROUPS(2),
        .GROUP_SIZE(4),
        .NONZERO_PER_GROUP(2),
        .DATA_WIDTH(8)
    ) sparse (
        .compact_values(sparse_vals),
        .compact_indices(sparse_idx),
        .dense_values(sparse_dense),
        .valid_pattern(sparse_valid)
    );

    reg [127:0] palette;
    reg [15:0] palette_indices;
    wire [31:0] palette_values;

    nac_palette_decode #(
        .LANES(4),
        .DATA_WIDTH(8),
        .INDEX_WIDTH(4)
    ) pal (
        .palette(palette),
        .indices(palette_indices),
        .values_out(palette_values)
    );

    reg [63:0] dense_in;
    wire [63:0] enc_values;
    wire [63:0] enc_counts;
    wire [7:0] enc_is_zero_run;
    wire [7:0] enc_token_count;
    reg [63:0] dec_values;
    reg [63:0] dec_counts;
    reg [7:0] dec_is_zero_run;
    reg [7:0] dec_token_count;
    wire [63:0] dense_out;
    wire decode_error;

    nac_activation_rle_codec #(
        .LANES(8),
        .DATA_WIDTH(8),
        .RUN_WIDTH(8)
    ) rle (
        .dense_in(dense_in),
        .enc_values(enc_values),
        .enc_counts(enc_counts),
        .enc_is_zero_run(enc_is_zero_run),
        .enc_token_count(enc_token_count),
        .dec_values(dec_values),
        .dec_counts(dec_counts),
        .dec_is_zero_run(dec_is_zero_run),
        .dec_token_count(dec_token_count),
        .dense_out(dense_out),
        .decode_error(decode_error)
    );

    integer i;

    initial begin
        bfp_mant = {4'h8, 4'hf, 4'h7, 4'h1};
        bfp_exp = 8'sd1;
        #1;
        if (bfp_values[0 +: 8] != 8'd2 ||
            bfp_values[8 +: 8] != 8'd14 ||
            bfp_values[16 +: 8] != 8'hfe ||
            bfp_values[24 +: 8] != 8'hf0 ||
            bfp_overflow != 4'b0000) begin
            $fatal(1, "BFP positive exponent decode failed");
        end

        bfp_exp = -8'sd1;
        #1;
        if (bfp_values[8 +: 8] != 8'd3 ||
            bfp_values[16 +: 8] != 8'hff ||
            bfp_values[24 +: 8] != 8'hfc) begin
            $fatal(1, "BFP negative exponent decode failed");
        end

        bfp_exp = 8'sd6;
        #1;
        if (!bfp_overflow[1] || !bfp_overflow[3]) begin
            $fatal(1, "BFP overflow detection failed");
        end

        sparse_vals = {8'hdd, 8'hcc, 8'hbb, 8'haa};
        sparse_idx = {2'd3, 2'd1, 2'd2, 2'd0};
        #1;
        if (!sparse_valid ||
            sparse_dense[0 +: 8] != 8'haa ||
            sparse_dense[16 +: 8] != 8'hbb ||
            sparse_dense[40 +: 8] != 8'hcc ||
            sparse_dense[56 +: 8] != 8'hdd) begin
            $fatal(1, "structured sparsity decode failed");
        end
        sparse_idx = {2'd3, 2'd1, 2'd1, 2'd1};
        #1;
        if (sparse_valid) $fatal(1, "structured sparsity duplicate index accepted");

        palette = 128'd0;
        for (i = 0; i < 16; i = i + 1) begin
            palette[i*8 +: 8] = i * 3;
        end
        palette_indices = {4'd15, 4'd10, 4'd5, 4'd0};
        #1;
        if (palette_values[0 +: 8] != 8'd0 ||
            palette_values[8 +: 8] != 8'd15 ||
            palette_values[16 +: 8] != 8'd30 ||
            palette_values[24 +: 8] != 8'd45) begin
            $fatal(1, "palette decode failed");
        end

        dense_in = {8'd0, 8'd0, 8'd8, 8'd7, 8'd0, 8'd5, 8'd0, 8'd0};
        dec_values = 64'd0;
        dec_counts = 64'd0;
        dec_is_zero_run = 8'd0;
        dec_token_count = 8'd0;
        #1;
        if (enc_token_count != 8'd6) $fatal(1, "RLE token count failed");
        dec_values = enc_values;
        dec_counts = enc_counts;
        dec_is_zero_run = enc_is_zero_run;
        dec_token_count = enc_token_count;
        #1;
        if (decode_error || dense_out != dense_in) $fatal(1, "RLE round-trip failed");

        dec_counts[0 +: 8] = 8'd0;
        #1;
        if (!decode_error) $fatal(1, "RLE bad count accepted");

        $display("nac_compression_primitives_tb PASS");
        $finish;
    end
endmodule
