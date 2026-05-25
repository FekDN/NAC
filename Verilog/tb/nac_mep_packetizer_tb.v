`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_mep_packetizer_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] rom [0:31];
    reg [5:0] rp;
    wire byte_valid = (rp < 14);
    wire byte_ready;
    wire [7:0] byte_in = rom[rp];
    wire out_valid;
    reg out_ready;
    wire [7:0] out_byte;
    wire [7:0] out_opcode;
    wire [15:0] out_instr_index;
    wire [15:0] out_byte_index;
    wire out_instr_start;
    wire out_instr_end;
    wire done;
    wire error;

    nac_mep_packetizer dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .plan_size(32'd14),
        .byte_valid(byte_valid),
        .byte_ready(byte_ready),
        .byte_in(byte_in),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_byte(out_byte),
        .out_opcode(out_opcode),
        .out_instr_index(out_instr_index),
        .out_byte_index(out_byte_index),
        .out_instr_start(out_instr_start),
        .out_instr_end(out_instr_end),
        .done(done),
        .error(error)
    );

    always @(posedge clk) begin
        if (rst || start) rp <= 0;
        else if (byte_valid && byte_ready) rp <= rp + 1'b1;
    end

    integer starts;
    integer ends;
    integer cycles;

    initial begin
        // 0x04 SRC_CONSTANT: op,out,cid_lo,cid_hi
        rom[0] = 8'h04; rom[1] = 8'h07; rom[2] = 8'h34; rom[3] = 8'h12;
        // 0x80 MODEL_RUN_STATIC: op,model,ci,in0,in1,co,out0
        rom[4] = 8'h80; rom[5] = 8'h00; rom[6] = 8'h02; rom[7] = 8'h10;
        rom[8] = 8'h11; rom[9] = 8'h01; rom[10] = 8'h20;
        // 0xFE EXEC_RETURN: op,count,key0
        rom[11] = 8'hFE; rom[12] = 8'h01; rom[13] = 8'h20;

        rst = 1'b1;
        start = 1'b0;
        out_ready = 1'b1;
        starts = 0;
        ends = 0;
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
            if (cycles > 200) $fatal(1, "MEP packetizer timeout");
            if (out_valid && out_ready) begin
                if (out_instr_start) starts = starts + 1;
                if (out_instr_end) ends = ends + 1;
                if (out_instr_index == 16'd1 && out_byte_index == 16'd0 && out_opcode != 8'h80)
                    $fatal(1, "bad opcode tracking");
            end
        end

        if (error) $fatal(1, "MEP packetizer error");
        if (starts != 3 || ends != 3) $fatal(1, "bad MEP boundaries starts=%0d ends=%0d", starts, ends);
        $display("nac_mep_packetizer_tb PASS");
        $finish;
    end
endmodule
