module nac_structured_sparsity_decode #(
    parameter GROUPS = 4,
    parameter GROUP_SIZE = 4,
    parameter NONZERO_PER_GROUP = 2,
    parameter DATA_WIDTH = 16,
    parameter INDEX_BITS = (GROUP_SIZE <= 2) ? 1 :
                           (GROUP_SIZE <= 4) ? 2 :
                           (GROUP_SIZE <= 8) ? 3 :
                           (GROUP_SIZE <= 16) ? 4 : 5
) (
    input  wire [GROUPS*NONZERO_PER_GROUP*DATA_WIDTH-1:0] compact_values,
    input  wire [GROUPS*NONZERO_PER_GROUP*INDEX_BITS-1:0] compact_indices,
    output reg  [GROUPS*GROUP_SIZE*DATA_WIDTH-1:0] dense_values,
    output reg  valid_pattern
);
    integer g;
    integer n;
    integer lane_index;
    reg [31:0] seen;

    always @* begin
        dense_values = {GROUPS*GROUP_SIZE*DATA_WIDTH{1'b0}};
        valid_pattern = 1'b1;

        for (g = 0; g < GROUPS; g = g + 1) begin
            seen = 32'd0;
            for (n = 0; n < NONZERO_PER_GROUP; n = n + 1) begin
                lane_index = compact_indices[(g*NONZERO_PER_GROUP+n)*INDEX_BITS +: INDEX_BITS];
                if (lane_index >= GROUP_SIZE) begin
                    valid_pattern = 1'b0;
                end else if (seen[lane_index]) begin
                    valid_pattern = 1'b0;
                end else begin
                    seen[lane_index] = 1'b1;
                    dense_values[(g*GROUP_SIZE+lane_index)*DATA_WIDTH +: DATA_WIDTH] =
                        compact_values[(g*NONZERO_PER_GROUP+n)*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end
endmodule
