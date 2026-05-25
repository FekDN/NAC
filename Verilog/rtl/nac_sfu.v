`include "nac_defs.vh"

module nac_sfu #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS = 16
) (
    input  wire [7:0] mode,
    input  wire signed [DATA_WIDTH-1:0] x,
    output reg  signed [DATA_WIDTH-1:0] y
);
    localparam signed [DATA_WIDTH-1:0] ZERO      = {DATA_WIDTH{1'b0}};
    localparam signed [DATA_WIDTH-1:0] ONE       = ({{(DATA_WIDTH-1){1'b0}}, 1'b1} <<< FRAC_BITS);
    localparam signed [DATA_WIDTH-1:0] HALF      = (ONE >>> 1);
    reg signed [DATA_WIDTH-1:0] tmp;
    reg signed [(DATA_WIDTH*2)-1:0] wide_mul;

    always @* begin
        y = x;
        case (mode)
            `NAC_DSP_RELU: begin
                y = x[DATA_WIDTH-1] ? ZERO : x;
            end
            `NAC_DSP_HSIGMOID: begin
                tmp = (x / 6) + HALF;
                if (tmp < ZERO) y = ZERO;
                else if (tmp > ONE) y = ONE;
                else y = tmp;
            end
            `NAC_DSP_HSWISH: begin
                tmp = (x / 6) + HALF;
                if (tmp < ZERO) tmp = ZERO;
                else if (tmp > ONE) tmp = ONE;
                wide_mul = x * tmp;
                y = wide_mul >>> FRAC_BITS;
            end
            default: begin
                y = x;
            end
        endcase
    end
endmodule
