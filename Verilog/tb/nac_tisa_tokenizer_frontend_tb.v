`timescale 1ns/1ps
`include "nac_defs.vh"

module nac_tisa_tokenizer_frontend_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg load_start;
    reg instr_valid;
    wire instr_ready;
    reg [7:0] opcode;
    reg [31:0] payload_len;
    reg payload_byte_valid;
    reg [31:0] payload_byte_index;
    reg [7:0] payload_byte;
    reg packet_done;
    reg packet_error;

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

    nac_tisa_tokenizer_frontend #(
        .MAX_CMDS(4)
    ) dut (
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

    reg [7:0] out_buf [0:15];
    integer out_count;
    integer cycles;

    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            out_buf[out_count] <= out_byte;
            out_count <= out_count + 1;
        end
    end

    task reset_dut;
        begin
            rst = 1'b1;
            load_start = 1'b0;
            instr_valid = 1'b0;
            opcode = 8'd0;
            payload_len = 32'd0;
            payload_byte_valid = 1'b0;
            payload_byte_index = 32'd0;
            payload_byte = 8'd0;
            packet_done = 1'b0;
            packet_error = 1'b0;
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

    task begin_load;
        begin
            @(negedge clk);
            load_start = 1'b1;
            @(negedge clk);
            load_start = 1'b0;
        end
    endtask

    task issue_instr;
        input [7:0] op;
        input [31:0] len;
        begin
            @(negedge clk);
            opcode = op;
            payload_len = len;
            instr_valid = 1'b1;
            while (!instr_ready) @(negedge clk);
            @(negedge clk);
            instr_valid = 1'b0;
            opcode = 8'd0;
            payload_len = 32'd0;
        end
    endtask

    task finish_load;
        begin
            @(negedge clk);
            packet_done = 1'b1;
            @(negedge clk);
            packet_done = 1'b0;
            cycles = 0;
            while (!load_done && !load_error && cycles < 20) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
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

        begin_load();
        issue_instr(`TISA_OP_LOWERCASE, 32'd0);
        issue_instr(`TISA_OP_BYTE_ENCODE, 32'd0);
        finish_load();
        if (!can_run_locally || load_error || requires_external_engine)
            $fatal(1, "ASCII lower + byte encode should be local");

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
        if (run_error) $fatal(1, "local ASCII run errored");
        if (!run_done) $fatal(1, "local ASCII run timeout");
        if (out_count != 4) $fatal(1, "unexpected byte-encoded length");
        if (out_buf[0] != "a" || out_buf[1] != 8'hc4 ||
            out_buf[2] != 8'ha0 || out_buf[3] != "b")
            $fatal(1, "bad ASCII lower + byte encode output");

        reset_dut();
        begin_load();
        issue_instr(`TISA_OP_BPE_ENCODE, 32'd1);
        finish_load();
        if (!load_done || load_error) $fatal(1, "BPE manifest should load");
        if (!requires_external_engine || can_run_locally || unsupported_opcode != `TISA_OP_BPE_ENCODE)
            $fatal(1, "BPE must require an external tokenizer engine");
        @(negedge clk);
        run_start = 1'b1;
        @(negedge clk);
        run_start = 1'b0;
        @(posedge clk);
        if (!run_error) $fatal(1, "local run accepted external-only manifest");

        reset_dut();
        begin_load();
        issue_instr(`TISA_OP_LOWERCASE, 32'd0);
        finish_load();
        if (!can_run_locally) $fatal(1, "lowercase-only ASCII frontend should be local");
        @(negedge clk);
        run_start = 1'b1;
        @(negedge clk);
        run_start = 1'b0;
        send_text_byte(8'hc3, 1'b1);
        @(posedge clk);
        if (!run_error) $fatal(1, "Unicode byte accepted by ASCII lowercase path");

        reset_dut();
        begin_load();
        issue_instr(`TISA_OP_BYTE_ENCODE, 32'd1);
        finish_load();
        if (!load_error || unsupported_opcode != `TISA_OP_BYTE_ENCODE)
            $fatal(1, "malformed BYTE_ENCODE payload was accepted");

        $display("nac_tisa_tokenizer_frontend_tb PASS");
        $finish;
    end
endmodule
