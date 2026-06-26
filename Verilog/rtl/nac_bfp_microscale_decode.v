module nac_bfp_microscale_decode #(
    parameter LANES = 16,
    parameter MANT_WIDTH = 4,
    parameter EXP_WIDTH = 8,
    parameter OUT_WIDTH = 16
) (
    input  wire [LANES*MANT_WIDTH-1:0] mantissas,
    input  wire signed [EXP_WIDTH-1:0] shared_exp,
    output reg  [LANES*OUT_WIDTH-1:0] values_out,
    output reg  [LANES-1:0] overflow_mask
);
    function [OUT_WIDTH-1:0] sat_value;
        input sign_bit;
        begin
            sat_value = sign_bit ? {1'b1, {OUT_WIDTH-1{1'b0}}} :
                                   {1'b0, {OUT_WIDTH-1{1'b1}}};
        end
    endfunction

    function [OUT_WIDTH:0] decode_one;
        input [MANT_WIDTH-1:0] mant_bits;
        input signed [EXP_WIDTH-1:0] exp_bits;
        reg signed [OUT_WIDTH-1:0] extended;
        reg signed [OUT_WIDTH-1:0] shifted;
        reg overflow;
        integer shift;
        begin
            extended = {{(OUT_WIDTH-MANT_WIDTH){mant_bits[MANT_WIDTH-1]}}, mant_bits};
            shifted = extended;
            overflow = 1'b0;

            if (exp_bits >= 0) begin
                shift = exp_bits;
                if (shift >= OUT_WIDTH) begin
                    overflow = (mant_bits != {MANT_WIDTH{1'b0}});
                    shifted = sat_value(mant_bits[MANT_WIDTH-1]);
                end else begin
                    shifted = extended <<< shift;
                    if ((shifted >>> shift) != extended) begin
                        overflow = 1'b1;
                        shifted = sat_value(mant_bits[MANT_WIDTH-1]);
                    end
                end
            end else begin
                shift = -exp_bits;
                if (shift >= OUT_WIDTH) begin
                    shifted = mant_bits[MANT_WIDTH-1] ? {OUT_WIDTH{1'b1}} :
                                                        {OUT_WIDTH{1'b0}};
                end else begin
                    shifted = extended >>> shift;
                end
            end

            decode_one = {overflow, shifted};
        end
    endfunction

    integer i;
    reg [OUT_WIDTH:0] decoded;

    always @* begin
        values_out = {LANES*OUT_WIDTH{1'b0}};
        overflow_mask = {LANES{1'b0}};
        for (i = 0; i < LANES; i = i + 1) begin
            decoded = decode_one(mantissas[i*MANT_WIDTH +: MANT_WIDTH], shared_exp);
            overflow_mask[i] = decoded[OUT_WIDTH];
            values_out[i*OUT_WIDTH +: OUT_WIDTH] = decoded[OUT_WIDTH-1:0];
        end
    end
endmodule
