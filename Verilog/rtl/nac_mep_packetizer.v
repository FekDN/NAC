`include "nac_defs.vh"

module nac_mep_packetizer (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [31:0] plan_size,

    input  wire byte_valid,
    output wire byte_ready,
    input  wire [7:0] byte_in,

    output reg  out_valid,
    input  wire out_ready,
    output reg  [7:0] out_byte,
    output reg  [7:0] out_opcode,
    output reg  [15:0] out_instr_index,
    output reg  [15:0] out_byte_index,
    output reg  out_instr_start,
    output reg  out_instr_end,

    output reg  done,
    output reg  error
);
    localparam DYN_NONE = 4'd0;
    localparam DYN_2A   = 4'd1;
    localparam DYN_38   = 4'd2;
    localparam DYN_39   = 4'd3;
    localparam DYN_3A   = 4'd4;
    localparam DYN_50   = 4'd5;
    localparam DYN_51   = 4'd6;
    localparam DYN_80   = 4'd7;
    localparam DYN_81   = 4'd8;
    localparam DYN_82   = 4'd9;
    localparam DYN_FE   = 4'd10;

    reg running;
    reg [31:0] bytes_seen;
    reg [15:0] instr_index;
    reg [15:0] pos;
    reg [15:0] total_len;
    reg [7:0] active_opcode;
    reg [3:0] dyn_kind;
    reg [7:0] saved_count;
    reg [7:0] saved_subop;

    assign byte_ready = running && !done && !error && (!out_valid || out_ready);

    function [15:0] fixed_total_len;
        input [7:0] op;
        begin
            case (op)
                8'h01: fixed_total_len = 16'd3;
                8'h02: fixed_total_len = 16'd5;
                8'h03: fixed_total_len = 16'd4;
                8'h04: fixed_total_len = 16'd4;
                8'h05: fixed_total_len = 16'd2;
                8'h10: fixed_total_len = 16'd4;
                8'h11: fixed_total_len = 16'd5;
                8'h12: fixed_total_len = 16'd5;
                8'h13: fixed_total_len = 16'd4;
                8'h14: fixed_total_len = 16'd4;
                8'h18: fixed_total_len = 16'd4;
                8'h1F: fixed_total_len = 16'd3;
                8'h20: fixed_total_len = 16'd4;
                8'h21: fixed_total_len = 16'd4;
                8'h22: fixed_total_len = 16'd5;
                8'h30: fixed_total_len = 16'd5;
                8'h31: fixed_total_len = 16'd6;
                8'h3B: fixed_total_len = 16'd4;
                8'h59: fixed_total_len = 16'd3;
                8'h5F: fixed_total_len = 16'd4;
                8'h60: fixed_total_len = 16'd4;
                8'h61: fixed_total_len = 16'd5;
                8'h62: fixed_total_len = 16'd4;
                8'h68: fixed_total_len = 16'd5;
                8'h70: fixed_total_len = 16'd5;
                8'h71: fixed_total_len = 16'd5;
                8'h83: fixed_total_len = 16'd2;
                8'h85: fixed_total_len = 16'd4;
                8'hA0: fixed_total_len = 16'd2;
                8'hA1: fixed_total_len = 16'd3;
                8'hA8: fixed_total_len = 16'd4;
                8'hA9: fixed_total_len = 16'd4;
                8'hAF: fixed_total_len = 16'd1;
                8'hE0: fixed_total_len = 16'd4;
                8'hE1: fixed_total_len = 16'd4;
                8'hF0: fixed_total_len = 16'd5;
                8'hFF: fixed_total_len = 16'd1;
                default: fixed_total_len = 16'd0;
            endcase
        end
    endfunction

    function [3:0] dynamic_kind;
        input [7:0] op;
        begin
            case (op)
                8'h2A: dynamic_kind = DYN_2A;
                8'h38: dynamic_kind = DYN_38;
                8'h39: dynamic_kind = DYN_39;
                8'h3A: dynamic_kind = DYN_3A;
                8'h50: dynamic_kind = DYN_50;
                8'h51: dynamic_kind = DYN_51;
                8'h80: dynamic_kind = DYN_80;
                8'h81: dynamic_kind = DYN_81;
                8'h82: dynamic_kind = DYN_82;
                8'hFE: dynamic_kind = DYN_FE;
                default: dynamic_kind = DYN_NONE;
            endcase
        end
    endfunction

    reg [15:0] next_total;
    reg end_now;
    reg [3:0] next_dyn;

    always @* begin
        next_total = total_len;
        next_dyn = dyn_kind;

        if (pos == 16'd0) begin
            next_total = fixed_total_len(byte_in);
            next_dyn = dynamic_kind(byte_in);
        end else begin
            case (dyn_kind)
                DYN_2A: if (pos == 16'd4) next_total = 16'd5 + byte_in;
                DYN_38: if (pos == 16'd1) next_total = (byte_in == 8'd1) ? 16'd6 :
                                                          ((byte_in == 8'd2 || byte_in == 8'd3) ? 16'd5 : 16'd4);
                DYN_39: if (pos == 16'd3) next_total = 16'd4 + byte_in + ((saved_subop == 8'd0) ? 16'd1 : 16'd0);
                DYN_3A: if (pos == 16'd1) next_total = 16'd4 + ((byte_in == 8'd1) ? 16'd1 : 16'd0);
                DYN_50: begin
                    if (pos == 16'd3) next_total = 16'd0;
                    if (pos == (16'd4 + saved_count)) next_total = 16'd5 + saved_count + byte_in;
                end
                DYN_51: begin
                    if (pos == 16'd4) next_total = 16'd0;
                    if (pos == (16'd5 + saved_count)) next_total = 16'd6 + saved_count + byte_in;
                end
                DYN_80: begin
                    if (pos == 16'd2) next_total = 16'd0;
                    if (pos == (16'd3 + saved_count)) next_total = 16'd4 + saved_count + byte_in;
                end
                DYN_81: if (pos == 16'd2) next_total = 16'd4 + byte_in;
                DYN_82: begin
                    if (pos == 16'd3) next_total = 16'd0;
                    if (pos == (16'd4 + saved_count)) next_total = 16'd12 + saved_count + byte_in;
                end
                DYN_FE: if (pos == 16'd1) next_total = 16'd2 + byte_in;
                default: begin
                end
            endcase
        end

        end_now = (next_total != 16'd0) && ((pos + 16'd1) == next_total);
    end

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0;
            bytes_seen <= 32'd0;
            instr_index <= 16'd0;
            pos <= 16'd0;
            total_len <= 16'd0;
            active_opcode <= 8'd0;
            dyn_kind <= DYN_NONE;
            saved_count <= 8'd0;
            saved_subop <= 8'd0;
            out_valid <= 1'b0;
            out_byte <= 8'd0;
            out_opcode <= 8'd0;
            out_instr_index <= 16'd0;
            out_byte_index <= 16'd0;
            out_instr_start <= 1'b0;
            out_instr_end <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (start) begin
                running <= (plan_size != 32'd0);
                bytes_seen <= 32'd0;
                instr_index <= 16'd0;
                pos <= 16'd0;
                total_len <= 16'd0;
                active_opcode <= 8'd0;
                dyn_kind <= DYN_NONE;
                saved_count <= 8'd0;
                saved_subop <= 8'd0;
                out_valid <= 1'b0;
                done <= (plan_size == 32'd0);
                error <= 1'b0;
            end else if (byte_valid && byte_ready) begin
                out_valid <= 1'b1;
                out_byte <= byte_in;
                out_opcode <= (pos == 16'd0) ? byte_in : active_opcode;
                out_instr_index <= instr_index;
                out_byte_index <= pos;
                out_instr_start <= (pos == 16'd0);
                out_instr_end <= end_now;

                if (pos == 16'd0) begin
                    if (fixed_total_len(byte_in) == 16'd0 && dynamic_kind(byte_in) == DYN_NONE) begin
                        error <= 1'b1;
                        running <= 1'b0;
                    end
                    active_opcode <= byte_in;
                    dyn_kind <= next_dyn;
                    total_len <= next_total;
                end else begin
                    total_len <= next_total;
                    case (dyn_kind)
                        DYN_39: if (pos == 16'd1) saved_subop <= byte_in;
                        DYN_50: if (pos == 16'd3) saved_count <= byte_in;
                        DYN_51: if (pos == 16'd4) saved_count <= byte_in;
                        DYN_80: if (pos == 16'd2) saved_count <= byte_in;
                        DYN_82: if (pos == 16'd3) saved_count <= byte_in;
                        default: begin
                        end
                    endcase
                end

                bytes_seen <= bytes_seen + 32'd1;
                if (end_now) begin
                    pos <= 16'd0;
                    total_len <= 16'd0;
                    dyn_kind <= DYN_NONE;
                    instr_index <= instr_index + 16'd1;
                    if ((bytes_seen + 32'd1) == plan_size) begin
                        done <= 1'b1;
                        running <= 1'b0;
                    end
                end else begin
                    pos <= pos + 16'd1;
                    if ((bytes_seen + 32'd1) == plan_size) begin
                        error <= 1'b1;
                        running <= 1'b0;
                    end
                end
            end
        end
    end
endmodule
