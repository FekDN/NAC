`include "nac_defs.vh"

// TISA text frontend for the local hardware tokenizer path.
//
// Exact local primitives:
//   - 0x01 LOWERCASE for ASCII input bytes.
//   - 0x15 BYTE_ENCODE using the reference GPT-2/TISA byte map.
//
// Resource-backed stages are valid TISA but require the full VM/resource
// engines. This frontend marks them with requires_external_engine and rejects
// local run rather than emitting an approximate token stream.
module nac_tisa_tokenizer_frontend #(
    parameter MAX_CMDS = 16,
    parameter ENABLE_ASCII_LOWER = 1,
    parameter ENABLE_BYTE_ENCODE = 1
) (
    input  wire clk,
    input  wire rst,

    input  wire load_start,
    input  wire instr_valid,
    output wire instr_ready,
    input  wire [7:0] opcode,
    input  wire [31:0] payload_len,
    input  wire payload_byte_valid,
    input  wire [31:0] payload_byte_index,
    input  wire [7:0] payload_byte,
    input  wire packet_done,
    input  wire packet_error,

    output reg  load_done,
    output reg  load_error,
    output reg  requires_external_engine,
    output reg  [7:0] unsupported_opcode,
    output wire can_run_locally,

    input  wire run_start,
    input  wire text_valid,
    output wire text_ready,
    input  wire [7:0] text_byte,
    input  wire text_last,

    output reg  out_valid,
    input  wire out_ready,
    output reg  [7:0] out_byte,
    output reg  out_last,

    output reg  run_done,
    output reg  run_error
);
    reg load_active;
    reg [15:0] cmd_count;
    reg has_ascii_lower;
    reg has_byte_encode;
    reg seen_byte_encode;
    reg running;
    reg pending_valid;
    reg [7:0] pending_byte;
    reg pending_last;

    wire [7:0] lowered_byte =
        (has_ascii_lower && text_byte >= 8'h41 && text_byte <= 8'h5a) ?
        (text_byte + 8'h20) : text_byte;

    wire [15:0] mapped_bytes;
    wire [1:0] mapped_len;

    nac_tisa_byte_map byte_map_i (
        .in_byte(lowered_byte),
        .out_bytes(mapped_bytes),
        .out_len(mapped_len)
    );

    assign instr_ready = load_active && !load_error;
    assign can_run_locally = load_done && !load_error && !requires_external_engine;
    assign text_ready = running && !run_done && !run_error && can_run_locally &&
                        !out_valid && !pending_valid;

    task mark_external;
        input [7:0] op;
        begin
            requires_external_engine <= 1'b1;
            if (unsupported_opcode == 8'd0)
                unsupported_opcode <= op;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            load_active <= 1'b0;
            cmd_count <= 16'd0;
            has_ascii_lower <= 1'b0;
            has_byte_encode <= 1'b0;
            seen_byte_encode <= 1'b0;
            load_done <= 1'b0;
            load_error <= 1'b0;
            requires_external_engine <= 1'b0;
            unsupported_opcode <= 8'd0;
            running <= 1'b0;
            pending_valid <= 1'b0;
            pending_byte <= 8'd0;
            pending_last <= 1'b0;
            out_valid <= 1'b0;
            out_byte <= 8'd0;
            out_last <= 1'b0;
            run_done <= 1'b0;
            run_error <= 1'b0;
        end else begin
            if (load_start) begin
                load_active <= 1'b1;
                cmd_count <= 16'd0;
                has_ascii_lower <= 1'b0;
                has_byte_encode <= 1'b0;
                seen_byte_encode <= 1'b0;
                load_done <= 1'b0;
                load_error <= 1'b0;
                requires_external_engine <= 1'b0;
                unsupported_opcode <= 8'd0;
            end

            if (packet_error) begin
                load_error <= 1'b1;
                load_active <= 1'b0;
            end

            if (instr_valid && instr_ready) begin
                if (cmd_count >= MAX_CMDS[15:0]) begin
                    load_error <= 1'b1;
                    unsupported_opcode <= opcode;
                end else begin
                    cmd_count <= cmd_count + 16'd1;
                    case (opcode)
                        `TISA_OP_LOWERCASE: begin
                            if (payload_len != 32'd0) begin
                                load_error <= 1'b1;
                                unsupported_opcode <= opcode;
                            end else if (!ENABLE_ASCII_LOWER || seen_byte_encode) begin
                                mark_external(opcode);
                            end else begin
                                has_ascii_lower <= 1'b1;
                            end
                        end
                        `TISA_OP_BYTE_ENCODE: begin
                            if (payload_len != 32'd0) begin
                                load_error <= 1'b1;
                                unsupported_opcode <= opcode;
                            end else if (!ENABLE_BYTE_ENCODE || seen_byte_encode) begin
                                mark_external(opcode);
                            end else begin
                                has_byte_encode <= 1'b1;
                                seen_byte_encode <= 1'b1;
                            end
                        end
                        `TISA_OP_UNICODE_NORM,
                        `TISA_OP_REPLACE,
                        `TISA_OP_FILTER_CAT,
                        `TISA_OP_PREPEND,
                        `TISA_OP_PARTITION,
                        `TISA_OP_BPE_ENCODE,
                        `TISA_OP_WORDPIECE,
                        `TISA_OP_UNIGRAM,
                        `TISA_OP_COMPOSE: begin
                            mark_external(opcode);
                        end
                        default: begin
                            load_error <= 1'b1;
                            unsupported_opcode <= opcode;
                        end
                    endcase
                end
            end

            if (packet_done && load_active) begin
                load_done <= 1'b1;
                load_active <= 1'b0;
            end

            if (run_start) begin
                out_valid <= 1'b0;
                out_last <= 1'b0;
                pending_valid <= 1'b0;
                run_done <= 1'b0;
                run_error <= 1'b0;
                if (can_run_locally) begin
                    running <= 1'b1;
                end else begin
                    running <= 1'b0;
                    run_error <= 1'b1;
                end
            end else begin
                if (out_valid && out_ready) begin
                    if (pending_valid) begin
                        out_byte <= pending_byte;
                        out_last <= pending_last;
                        pending_valid <= 1'b0;
                        out_valid <= 1'b1;
                    end else begin
                        if (out_last) begin
                            running <= 1'b0;
                            run_done <= 1'b1;
                        end
                        out_valid <= 1'b0;
                        out_last <= 1'b0;
                    end
                end

                if (text_valid && text_ready) begin
                    if (has_ascii_lower && text_byte[7]) begin
                        running <= 1'b0;
                        run_error <= 1'b1;
                    end else if (has_byte_encode) begin
                        out_valid <= 1'b1;
                        out_byte <= mapped_bytes[7:0];
                        if (mapped_len == 2'd2) begin
                            out_last <= 1'b0;
                            pending_valid <= 1'b1;
                            pending_byte <= mapped_bytes[15:8];
                            pending_last <= text_last;
                        end else begin
                            out_last <= text_last;
                        end
                    end else begin
                        out_valid <= 1'b1;
                        out_byte <= lowered_byte;
                        out_last <= text_last;
                    end
                end
            end
        end
    end
endmodule
