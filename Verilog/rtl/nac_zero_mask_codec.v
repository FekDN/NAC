module nac_zero_mask_codec #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32
) (
    input  wire [LANES*DATA_WIDTH-1:0] dense_in,
    output reg  [LANES-1:0] nonzero_mask,
    output reg  [LANES*DATA_WIDTH-1:0] compact_out,
    output reg  [7:0] nonzero_count,

    input  wire [LANES-1:0] decode_mask,
    input  wire [LANES*DATA_WIDTH-1:0] compact_in,
    output reg  [LANES*DATA_WIDTH-1:0] dense_out
);
    integer i;
    integer write_pos;
    integer read_pos;

    always @* begin
        nonzero_mask = {LANES{1'b0}};
        compact_out = {LANES*DATA_WIDTH{1'b0}};
        nonzero_count = 8'd0;
        write_pos = 0;

        for (i = 0; i < LANES; i = i + 1) begin
            if (dense_in[i*DATA_WIDTH +: DATA_WIDTH] != {DATA_WIDTH{1'b0}}) begin
                nonzero_mask[i] = 1'b1;
                compact_out[write_pos*DATA_WIDTH +: DATA_WIDTH] = dense_in[i*DATA_WIDTH +: DATA_WIDTH];
                write_pos = write_pos + 1;
            end
        end
        nonzero_count = write_pos;

        dense_out = {LANES*DATA_WIDTH{1'b0}};
        read_pos = 0;
        for (i = 0; i < LANES; i = i + 1) begin
            if (decode_mask[i]) begin
                dense_out[i*DATA_WIDTH +: DATA_WIDTH] = compact_in[read_pos*DATA_WIDTH +: DATA_WIDTH];
                read_pos = read_pos + 1;
            end
        end
    end
endmodule
