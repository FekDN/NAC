`include "nac_defs.vh"

// Exact byte-level map used by the Python TISA reference:
//   bs = 33..126, 161..172, 174..255
//   byte_map = dict(zip(bs + missing, chr(bs) + chr(256+i)))
// The output is UTF-8 encoded because downstream tokenizer engines consume the
// same byte representation that the C++ VM stores in std::string fragments.
module nac_tisa_byte_map (
    input  wire [7:0] in_byte,
    output reg  [15:0] out_bytes,
    output reg  [1:0] out_len
);
    reg [10:0] codepoint;

    always @* begin
        if ((in_byte >= 8'd33 && in_byte <= 8'd126) ||
            (in_byte >= 8'd161 && in_byte <= 8'd172) ||
            (in_byte >= 8'd174)) begin
            codepoint = {3'd0, in_byte};
        end else if (in_byte <= 8'd32) begin
            codepoint = 11'd256 + {3'd0, in_byte};
        end else if (in_byte <= 8'd160) begin
            codepoint = 11'd289 + (in_byte - 8'd127);
        end else begin
            codepoint = 11'd323; // byte 173, the only missing value in 161..255.
        end

        out_bytes = 16'd0;
        if (codepoint < 11'd128) begin
            out_len = 2'd1;
            out_bytes[7:0] = codepoint[7:0];
        end else begin
            out_len = 2'd2;
            out_bytes[7:0] = {3'b110, codepoint[10:6]};
            out_bytes[15:8] = {2'b10, codepoint[5:0]};
        end
    end
endmodule
