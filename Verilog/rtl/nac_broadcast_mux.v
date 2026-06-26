module nac_broadcast_mux #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32
) (
    input  wire enable_broadcast,
    input  wire [LANES*DATA_WIDTH-1:0] vector_in,
    input  wire [DATA_WIDTH-1:0] scalar_in,
    output wire [LANES*DATA_WIDTH-1:0] vector_out
);
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_broadcast
            assign vector_out[g*DATA_WIDTH +: DATA_WIDTH] =
                enable_broadcast ? scalar_in : vector_in[g*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate
endmodule
