`include "nac_defs.vh"

// Manifest loader + exact local TISA text frontend.
//
// The module accepts a binary TISA v1 manifest and caches the part of the
// pipeline that can be executed locally without loss of compatibility. Manifests
// containing regex/Unicode/vocab-dependent stages remain valid, but they raise
// requires_external_engine and are not executed by the local stream path.
module nac_tisa_tokenizer #(
    parameter MAX_CMDS = 16,
    parameter ENABLE_ASCII_LOWER = 1,
    parameter ENABLE_BYTE_ENCODE = 1
) (
    input  wire clk,
    input  wire rst,

    input  wire load_start,
    input  wire [31:0] manifest_size,
    input  wire manifest_byte_valid,
    output wire manifest_byte_ready,
    input  wire [7:0] manifest_byte,

    output wire load_done,
    output wire load_error,
    output wire requires_external_engine,
    output wire [7:0] unsupported_opcode,
    output wire can_run_locally,

    input  wire run_start,
    input  wire text_valid,
    output wire text_ready,
    input  wire [7:0] text_byte,
    input  wire text_last,

    output wire out_valid,
    input  wire out_ready,
    output wire [7:0] out_byte,
    output wire out_last,

    output wire run_done,
    output wire run_error
);
    wire instr_valid;
    wire instr_ready;
    wire [7:0] opcode;
    wire [31:0] payload_len;
    wire [31:0] payload_byte_index;
    wire payload_byte_valid;
    wire [7:0] payload_byte;
    wire packet_done;
    wire packet_error;

    nac_tisa_packetizer packetizer_i (
        .clk(clk),
        .rst(rst),
        .start(load_start),
        .manifest_size(manifest_size),
        .byte_valid(manifest_byte_valid),
        .byte_ready(manifest_byte_ready),
        .byte_in(manifest_byte),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .opcode(opcode),
        .payload_len(payload_len),
        .payload_byte_index(payload_byte_index),
        .payload_byte_valid(payload_byte_valid),
        .payload_byte(payload_byte),
        .done(packet_done),
        .error(packet_error)
    );

    nac_tisa_tokenizer_frontend #(
        .MAX_CMDS(MAX_CMDS),
        .ENABLE_ASCII_LOWER(ENABLE_ASCII_LOWER),
        .ENABLE_BYTE_ENCODE(ENABLE_BYTE_ENCODE)
    ) frontend_i (
        .clk(clk),
        .rst(rst),
        .load_start(load_start),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .opcode(opcode),
        .payload_len(payload_len),
        .payload_byte_valid(payload_byte_valid),
        .payload_byte_index(payload_byte_index),
        .payload_byte(payload_byte),
        .packet_done(packet_done),
        .packet_error(packet_error),
        .load_done(load_done),
        .load_error(load_error),
        .requires_external_engine(requires_external_engine),
        .unsupported_opcode(unsupported_opcode),
        .can_run_locally(can_run_locally),
        .run_start(run_start),
        .text_valid(text_valid),
        .text_ready(text_ready),
        .text_byte(text_byte),
        .text_last(text_last),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_byte(out_byte),
        .out_last(out_last),
        .run_done(run_done),
        .run_error(run_error)
    );
endmodule
