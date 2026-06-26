`timescale 1ns/1ps
`include "nac_defs.vh"

module nac_tisa_tokenizer_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg load_start;
    reg [31:0] manifest_size;
    reg [7:0] manifest_rom [0:31];
    reg [5:0] manifest_pos;
    wire manifest_byte_valid = (manifest_pos < manifest_size);
    wire manifest_byte_ready;
    wire [7:0] manifest_byte = manifest_rom[manifest_pos];

    wire load_done;
    wire load_error;
    wire requires_external_engine;
    wire [7:0] unsupported_opcode;
    wire can_run_locally;

    reg run_start;
    reg text_valid;
    wire text_ready;
    reg [7:0] text_byte;
    reg text_last;
    wire out_valid;
    reg out_ready;
    wire [7:0] out_byte;
    wire out_last;
    wire run_done;
    wire run_error;

    nac_tisa_tokenizer #(
        .MAX_CMDS(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .load_start(load_start),
        .manifest_size(manifest_size),
        .manifest_byte_valid(manifest_byte_valid),
        .manifest_byte_ready(manifest_byte_ready),
        .manifest_byte(manifest_byte),
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

    reg [7:0] out_buf [0:15];
    integer out_count;
    integer cycles;

    always @(posedge clk) begin
        if (rst || load_start) begin
            manifest_pos <= 0;
        end else if (manifest_byte_valid && manifest_byte_ready) begin
            manifest_pos <= manifest_pos + 1'b1;
        end

        if (out_valid && out_ready) begin
            out_buf[out_count] <= out_byte;
            out_count <= out_count + 1;
        end
    end

    task reset_dut;
        begin
            rst = 1'b1;
            load_start = 1'b0;
            manifest_size = 32'd0;
            manifest_pos = 0;
            run_start = 1'b0;
            text_valid = 1'b0;
            text_byte = 8'd0;
            text_last = 1'b0;
            out_ready = 1'b1;
            out_count = 0;
            repeat (3) @(posedge clk);
            rst = 1'b0;
            @(posedge clk);
        end
    endtask

    task send_text_byte;
        input [7:0] b;
        input last;
        begin
            @(negedge clk);
            text_byte = b;
            text_last = last;
            text_valid = 1'b1;
            while (!text_ready) @(negedge clk);
            @(negedge clk);
            text_valid = 1'b0;
            text_last = 1'b0;
        end
    endtask

    initial begin
        reset_dut();

        // Manifest: TISA v1, LOWERCASE(), BYTE_ENCODE().
        manifest_rom[0] = "T";
        manifest_rom[1] = "I";
        manifest_rom[2] = "S";
        manifest_rom[3] = "A";
        manifest_rom[4] = `TISA_VERSION_V10;
        manifest_rom[5] = `TISA_OP_LOWERCASE;
        manifest_rom[6] = 8'd0;
        manifest_rom[7] = 8'd0;
        manifest_rom[8] = 8'd0;
        manifest_rom[9] = 8'd0;
        manifest_rom[10] = `TISA_OP_BYTE_ENCODE;
        manifest_rom[11] = 8'd0;
        manifest_rom[12] = 8'd0;
        manifest_rom[13] = 8'd0;
        manifest_rom[14] = 8'd0;
        manifest_size = 32'd15;

        @(negedge clk);
        load_start = 1'b1;
        @(negedge clk);
        load_start = 1'b0;
        cycles = 0;
        while (!load_done && !load_error && cycles < 200) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (load_error) $fatal(1, "manifest load errored");
        if (!load_done) $fatal(1, "manifest load timeout");
        if (!can_run_locally || requires_external_engine)
            $fatal(1, "local tokenizer manifest was not accepted");

        @(negedge clk);
        run_start = 1'b1;
        @(negedge clk);
        run_start = 1'b0;
        send_text_byte("A", 1'b0);
        send_text_byte(" ", 1'b0);
        send_text_byte("B", 1'b1);
        cycles = 0;
        while (!run_done && !run_error && cycles < 100) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (run_error) $fatal(1, "tokenizer run errored");
        if (!run_done) $fatal(1, "tokenizer run timeout");
        if (out_count != 4) $fatal(1, "bad tokenizer output length");
        if (out_buf[0] != "a" || out_buf[1] != 8'hc4 ||
            out_buf[2] != 8'ha0 || out_buf[3] != "b")
            $fatal(1, "bad tokenizer output bytes");

        reset_dut();
        manifest_rom[0] = "T";
        manifest_rom[1] = "I";
        manifest_rom[2] = "S";
        manifest_rom[3] = "A";
        manifest_rom[4] = `TISA_VERSION_V10;
        manifest_rom[5] = `TISA_OP_BPE_ENCODE;
        manifest_rom[6] = 8'd1;
        manifest_rom[7] = 8'd0;
        manifest_rom[8] = 8'd0;
        manifest_rom[9] = 8'd0;
        manifest_rom[10] = 8'd0; // suffix length = 0
        manifest_size = 32'd11;

        @(negedge clk);
        load_start = 1'b1;
        @(negedge clk);
        load_start = 1'b0;
        cycles = 0;
        while (!load_done && !load_error && cycles < 200) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (!load_done || load_error) $fatal(1, "BPE manifest did not load");
        if (!requires_external_engine || unsupported_opcode != `TISA_OP_BPE_ENCODE)
            $fatal(1, "BPE manifest did not request external engine");

        $display("nac_tisa_tokenizer_tb PASS");
        $finish;
    end
endmodule
