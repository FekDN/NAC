`include "nac_defs.vh"

module nac_tisa_packetizer (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [31:0] manifest_size,

    input  wire byte_valid,
    output wire byte_ready,
    input  wire [7:0] byte_in,

    output reg  instr_valid,
    input  wire instr_ready,
    output reg  [7:0] opcode,
    output reg  [31:0] payload_len,
    output reg  [31:0] payload_byte_index,
    output reg  payload_byte_valid,
    output reg  [7:0] payload_byte,

    output reg  done,
    output reg  error
);
    localparam S_IDLE    = 4'd0;
    localparam S_MAGIC   = 4'd1;
    localparam S_VERSION = 4'd2;
    localparam S_OPCODE  = 4'd3;
    localparam S_LEN     = 4'd4;
    localparam S_EMIT    = 4'd5;
    localparam S_PAYLOAD = 4'd6;
    localparam S_DONE    = 4'd7;

    (* fsm_safe_state = "default_state" *) reg [3:0] state;
    reg [1:0] magic_idx;
    reg [1:0] len_idx;
    reg [31:0] bytes_seen;
    reg [31:0] instr_end_pos;
    reg [31:0] payload_pos;

    assign byte_ready = (state != S_IDLE) && (state != S_EMIT) && !done && !error;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            magic_idx <= 2'd0;
            len_idx <= 2'd0;
            bytes_seen <= 32'd0;
            instr_end_pos <= 32'd0;
            instr_valid <= 1'b0;
            opcode <= 8'd0;
            payload_len <= 32'd0;
            payload_byte_index <= 32'd0;
            payload_pos <= 32'd0;
            payload_byte_valid <= 1'b0;
            payload_byte <= 8'd0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            payload_byte_valid <= 1'b0;

            if (start) begin
                state <= S_MAGIC;
                magic_idx <= 2'd0;
                len_idx <= 2'd0;
                instr_valid <= 1'b0;
                payload_len <= 32'd0;
                payload_byte_index <= 32'd0;
                payload_pos <= 32'd0;
                bytes_seen <= 32'd0;
                instr_end_pos <= 32'd0;
                done <= 1'b0;
                error <= (manifest_size < 32'd5);
            end else begin
                case (state)
                    S_IDLE: begin
                    end
                    S_MAGIC: begin
                        if (byte_valid && byte_ready) begin
                            bytes_seen <= bytes_seen + 32'd1;
                            case (magic_idx)
                                2'd0: if (byte_in != 8'h54) error <= 1'b1; // T
                                2'd1: if (byte_in != 8'h49) error <= 1'b1; // I
                                2'd2: if (byte_in != 8'h53) error <= 1'b1; // S
                                2'd3: if (byte_in != 8'h41) error <= 1'b1; // A
                            endcase
                            if (magic_idx == 2'd3) state <= S_VERSION;
                            magic_idx <= magic_idx + 2'd1;
                        end
                    end
                    S_VERSION: begin
                        if (byte_valid && byte_ready) begin
                            bytes_seen <= bytes_seen + 32'd1;
                            if (byte_in != `TISA_VERSION_V10) begin
                                error <= 1'b1;
                            end
                            if (manifest_size == 32'd5) begin
                                done <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                state <= S_OPCODE;
                            end
                        end
                    end
                    S_OPCODE: begin
                        if (byte_valid && byte_ready) begin
                            bytes_seen <= bytes_seen + 32'd1;
                            opcode <= byte_in;
                            payload_len <= 32'd0;
                            len_idx <= 2'd0;
                            state <= S_LEN;
                        end
                    end
                    S_LEN: begin
                        if (byte_valid && byte_ready) begin
                            bytes_seen <= bytes_seen + 32'd1;
                            payload_len[len_idx*8 +: 8] <= byte_in;
                            if (len_idx == 2'd3) begin
                                instr_end_pos <= bytes_seen + 32'd1 + {byte_in, payload_len[23:0]};
                                if ((bytes_seen + 32'd1 + {byte_in, payload_len[23:0]}) > manifest_size) begin
                                    error <= 1'b1;
                                end
                                state <= S_EMIT;
                            end else begin
                                len_idx <= len_idx + 2'd1;
                            end
                        end
                    end
                    S_EMIT: begin
                        instr_valid <= 1'b1;
                        if (instr_valid && instr_ready) begin
                            instr_valid <= 1'b0;
                            payload_byte_index <= 32'd0;
                            payload_pos <= 32'd0;
                            if (payload_len == 32'd0) begin
                                if (instr_end_pos == manifest_size) begin
                                    done <= 1'b1;
                                    state <= S_DONE;
                                end else begin
                                    state <= S_OPCODE;
                                end
                            end else state <= S_PAYLOAD;
                        end
                    end
                    S_PAYLOAD: begin
                        if (byte_valid && byte_ready) begin
                            bytes_seen <= bytes_seen + 32'd1;
                            payload_byte_valid <= 1'b1;
                            payload_byte <= byte_in;
                            payload_byte_index <= payload_pos;
                            if (payload_pos + 32'd1 >= payload_len) begin
                                if ((bytes_seen + 32'd1) == manifest_size) begin
                                    done <= 1'b1;
                                    state <= S_DONE;
                                end else begin
                                    state <= S_OPCODE;
                                end
                            end
                            payload_pos <= payload_pos + 32'd1;
                        end
                    end
                    S_DONE: begin
                        done <= 1'b1;
                    end
                    default: begin
                        error <= 1'b1;
                    end
                endcase
            end
        end
    end
endmodule
