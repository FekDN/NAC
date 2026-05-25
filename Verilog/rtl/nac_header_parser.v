`include "nac_defs.vh"

module nac_header_parser (
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire byte_valid,
    output wire byte_ready,
    input  wire [7:0] byte_in,

    output reg  done,
    output reg  error,
    output reg  [7:0] quant_flags,
    output reg  [15:0] num_inputs,
    output reg  [15:0] num_outputs,
    output reg  [15:0] d_model,
    output reg  [11*64-1:0] section_offsets
);
    reg [6:0] pos;
    reg running;
    reg [63:0] offset_shift;
    reg [3:0] offset_index;
    reg [2:0] offset_byte;

    assign byte_ready = running && !done && !error;

    always @(posedge clk) begin
        if (rst) begin
            pos <= 7'd0;
            running <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            quant_flags <= 8'd0;
            num_inputs <= 16'd0;
            num_outputs <= 16'd0;
            d_model <= 16'd0;
            section_offsets <= {11*64{1'b0}};
            offset_shift <= 64'd0;
            offset_index <= 4'd0;
            offset_byte <= 3'd0;
        end else begin
            if (start) begin
                pos <= 7'd0;
                running <= 1'b1;
                done <= 1'b0;
                error <= 1'b0;
                quant_flags <= 8'd0;
                num_inputs <= 16'd0;
                num_outputs <= 16'd0;
                d_model <= 16'd0;
                section_offsets <= {11*64{1'b0}};
                offset_shift <= 64'd0;
                offset_index <= 4'd0;
                offset_byte <= 3'd0;
            end else if (byte_valid && byte_ready) begin
                case (pos)
                    7'd0: if (byte_in != `NAC_MAGIC_N) error <= 1'b1;
                    7'd1: if (byte_in != `NAC_MAGIC_A) error <= 1'b1;
                    7'd2: if (byte_in != `NAC_MAGIC_C) error <= 1'b1;
                    7'd3: if (byte_in != `NAC_VERSION_V18) error <= 1'b1;
                    7'd4: quant_flags <= byte_in;
                    7'd5: num_inputs[7:0] <= byte_in;
                    7'd6: num_inputs[15:8] <= byte_in;
                    7'd7: num_outputs[7:0] <= byte_in;
                    7'd8: num_outputs[15:8] <= byte_in;
                    7'd10: d_model[7:0] <= byte_in;
                    7'd11: d_model[15:8] <= byte_in;
                    default: begin
                        if (pos >= 7'd12 && pos < 7'd100) begin
                            offset_shift[offset_byte*8 +: 8] <= byte_in;
                            if (offset_byte == 3'd7) begin
                                section_offsets[offset_index*64 +: 64] <= {
                                    byte_in,
                                    offset_shift[55:0]
                                };
                                offset_byte <= 3'd0;
                                offset_index <= offset_index + 4'd1;
                            end else begin
                                offset_byte <= offset_byte + 3'd1;
                            end
                        end
                    end
                endcase

                if (pos == (`NAC_HEADER_BYTES - 1)) begin
                    done <= !error;
                    running <= 1'b0;
                end else begin
                    pos <= pos + 7'd1;
                end
            end
        end
    end
endmodule
