`include "nac_defs.vh"

module nac_reduction_tree #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32,
    parameter ACC_WIDTH = DATA_WIDTH + 8
) (
    input  wire [LANES*DATA_WIDTH-1:0] vec_in,
    output reg  signed [ACC_WIDTH-1:0] sum_out,
    output reg  signed [DATA_WIDTH-1:0] max_out
);
    integer i;
    reg signed [DATA_WIDTH-1:0] lane;

    always @* begin
        sum_out = {ACC_WIDTH{1'b0}};
        max_out = vec_in[DATA_WIDTH-1:0];
        for (i = 0; i < LANES; i = i + 1) begin
            lane = vec_in[i*DATA_WIDTH +: DATA_WIDTH];
            sum_out = sum_out + lane;
            if (lane > max_out) begin
                max_out = lane;
            end
        end
    end
endmodule
