module nac_activation_rle_codec #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 16,
    parameter RUN_WIDTH = 8
) (
    input  wire [LANES*DATA_WIDTH-1:0] dense_in,
    output reg  [LANES*DATA_WIDTH-1:0] enc_values,
    output reg  [LANES*RUN_WIDTH-1:0] enc_counts,
    output reg  [LANES-1:0] enc_is_zero_run,
    output reg  [7:0] enc_token_count,

    input  wire [LANES*DATA_WIDTH-1:0] dec_values,
    input  wire [LANES*RUN_WIDTH-1:0] dec_counts,
    input  wire [LANES-1:0] dec_is_zero_run,
    input  wire [7:0] dec_token_count,
    output reg  [LANES*DATA_WIDTH-1:0] dense_out,
    output reg  decode_error
);
    integer i;
    integer k;
    integer token_index;
    integer write_pos;
    integer run_count;

    always @* begin
        enc_values = {LANES*DATA_WIDTH{1'b0}};
        enc_counts = {LANES*RUN_WIDTH{1'b0}};
        enc_is_zero_run = {LANES{1'b0}};
        enc_token_count = 8'd0;
        token_index = 0;

        for (i = 0; i < LANES; i = i + 1) begin
            if (dense_in[i*DATA_WIDTH +: DATA_WIDTH] == {DATA_WIDTH{1'b0}}) begin
                if (token_index > 0 &&
                    enc_is_zero_run[token_index-1] &&
                    enc_counts[(token_index-1)*RUN_WIDTH +: RUN_WIDTH] != {RUN_WIDTH{1'b1}}) begin
                    enc_counts[(token_index-1)*RUN_WIDTH +: RUN_WIDTH] =
                        enc_counts[(token_index-1)*RUN_WIDTH +: RUN_WIDTH] + 1'b1;
                end else begin
                    enc_is_zero_run[token_index] = 1'b1;
                    enc_counts[token_index*RUN_WIDTH +: RUN_WIDTH] = {{RUN_WIDTH-1{1'b0}}, 1'b1};
                    token_index = token_index + 1;
                end
            end else begin
                enc_is_zero_run[token_index] = 1'b0;
                enc_counts[token_index*RUN_WIDTH +: RUN_WIDTH] = {{RUN_WIDTH-1{1'b0}}, 1'b1};
                enc_values[token_index*DATA_WIDTH +: DATA_WIDTH] =
                    dense_in[i*DATA_WIDTH +: DATA_WIDTH];
                token_index = token_index + 1;
            end
        end
        enc_token_count = token_index;

        dense_out = {LANES*DATA_WIDTH{1'b0}};
        decode_error = 1'b0;
        write_pos = 0;

        for (i = 0; i < LANES; i = i + 1) begin
            if (i < dec_token_count) begin
                run_count = dec_counts[i*RUN_WIDTH +: RUN_WIDTH];
                if (run_count == 0) begin
                    decode_error = 1'b1;
                end else if (dec_is_zero_run[i]) begin
                    if ((write_pos + run_count) > LANES) begin
                        decode_error = 1'b1;
                    end
                    write_pos = write_pos + run_count;
                end else begin
                    if (run_count != 1 || write_pos >= LANES) begin
                        decode_error = 1'b1;
                    end else begin
                        dense_out[write_pos*DATA_WIDTH +: DATA_WIDTH] =
                            dec_values[i*DATA_WIDTH +: DATA_WIDTH];
                    end
                    write_pos = write_pos + 1;
                end
            end
        end

        if (write_pos != LANES) begin
            decode_error = 1'b1;
        end
    end
endmodule
