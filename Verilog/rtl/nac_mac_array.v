`include "nac_defs.vh"

module nac_mac_array #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32,
    parameter ACC_WIDTH = (DATA_WIDTH*2) + 8
) (
    input  wire [LANES*DATA_WIDTH-1:0] a_vec,
    input  wire [LANES*DATA_WIDTH-1:0] b_vec,
    output wire [LANES*ACC_WIDTH-1:0]  prod_vec,
    output wire signed [ACC_WIDTH-1:0] dot_sum
);
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_prod
            wire signed [DATA_WIDTH-1:0] a_lane;
            wire signed [DATA_WIDTH-1:0] b_lane;
            wire signed [(DATA_WIDTH*2)-1:0] prod;
            wire signed [ACC_WIDTH-1:0] prod_ext;
            assign a_lane = a_vec[g*DATA_WIDTH +: DATA_WIDTH];
            assign b_lane = b_vec[g*DATA_WIDTH +: DATA_WIDTH];
            assign prod = a_lane * b_lane;
            assign prod_ext = prod;
            assign prod_vec[g*ACC_WIDTH +: ACC_WIDTH] = prod_ext;
        end
    endgenerate

    nac_reduction_tree #(
        .LANES(LANES),
        .DATA_WIDTH(ACC_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) prod_reduce (
        .vec_in(prod_vec),
        .sum_out(dot_sum),
        .max_out()
    );
endmodule
