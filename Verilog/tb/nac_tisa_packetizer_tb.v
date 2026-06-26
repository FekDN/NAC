`timescale 1ns/1ps

module nac_tisa_packetizer_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] rom [0:15];
    reg [4:0] rp;
    wire byte_valid = (rp < 11);
    wire byte_ready;
    wire [7:0] byte_in = rom[rp];
    wire instr_valid;
    reg instr_ready;
    wire [7:0] opcode;
    wire [31:0] payload_len;
    wire [31:0] payload_byte_index;
    wire payload_byte_valid;
    wire [7:0] payload_byte;
    wire done;
    wire error;
    reg [31:0] manifest_size;

    nac_tisa_packetizer dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .manifest_size(manifest_size),
        .byte_valid(byte_valid),
        .byte_ready(byte_ready),
        .byte_in(byte_in),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .opcode(opcode),
        .payload_len(payload_len),
        .payload_byte_index(payload_byte_index),
        .payload_byte_valid(payload_byte_valid),
        .payload_byte(payload_byte),
        .done(done),
        .error(error)
    );

    always @(posedge clk) begin
        if (rst || start) rp <= 0;
        else if (byte_valid && byte_ready) rp <= rp + 1'b1;
    end

    integer instrs;
    integer payloads;
    integer cycles;

    initial begin
        rom[0] = "T"; rom[1] = "I"; rom[2] = "S"; rom[3] = "A"; rom[4] = 8'h01;
        rom[5] = 8'h20; // BPE_ENCODE
        rom[6] = 8'h01; rom[7] = 8'h00; rom[8] = 8'h00; rom[9] = 8'h00;
        rom[10] = 8'h7B;

        rst = 1'b1;
        start = 1'b0;
        manifest_size = 32'd11;
        instr_ready = 1'b1;
        instrs = 0;
        payloads = 0;
        cycles = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (!done && !error) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 200) $fatal(1, "TISA packetizer timeout");
            if (instr_valid && instr_ready) begin
                instrs = instrs + 1;
                if (opcode != 8'h20 || payload_len != 32'd1) $fatal(1, "bad TISA instruction");
            end
            if (payload_byte_valid) begin
                payloads = payloads + 1;
                if (payload_byte_index != 32'd0 || payload_byte != 8'h7B) $fatal(1, "bad TISA payload");
            end
        end

        if (error) $fatal(1, "TISA packetizer error");
        if (instrs != 1 || payloads != 1) $fatal(1, "bad TISA counts");

        @(negedge clk);
        rst = 1'b1;
        @(negedge clk);
        rst = 1'b0;
        rom[0] = 8'h54;
        rom[1] = 8'h49;
        rom[2] = 8'h53;
        rom[3] = 8'h41;
        rom[4] = 8'h02;
        manifest_size = 32'd5;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        cycles = 0;
        while (!error && cycles < 20) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        if (!error) $fatal(1, "bad TISA version was accepted");
        $display("nac_tisa_packetizer_tb PASS");
        $finish;
    end
endmodule
