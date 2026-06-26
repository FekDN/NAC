module nac_palette_decode #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 16,
    parameter INDEX_WIDTH = 4,
    parameter PALETTE_SIZE = (1 << INDEX_WIDTH)
) (
    input  wire [PALETTE_SIZE*DATA_WIDTH-1:0] palette,
    input  wire [LANES*INDEX_WIDTH-1:0] indices,
    output reg  [LANES*DATA_WIDTH-1:0] values_out
);
    integer i;
    integer palette_index;

    always @* begin
        values_out = {LANES*DATA_WIDTH{1'b0}};
        for (i = 0; i < LANES; i = i + 1) begin
            palette_index = indices[i*INDEX_WIDTH +: INDEX_WIDTH];
            values_out[i*DATA_WIDTH +: DATA_WIDTH] =
                palette[palette_index*DATA_WIDTH +: DATA_WIDTH];
        end
    end
endmodule
