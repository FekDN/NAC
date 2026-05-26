`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_abcd_decoder_tb;
    localparam MAX_ARITY = 4;
    localparam MAX_CONSTS = 4;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] rom [0:31];
    reg [5:0] rp;
    wire ops_byte_valid;
    wire ops_byte_ready;
    wire [7:0] ops_byte;

    assign ops_byte_valid = (rp < 20);
    assign ops_byte = rom[rp];

    wire [7:0] perm_lookup_id;
    reg [3:0] perm_arity;
    reg perm_needs_consts;
    reg perm_present;
    wire instr_valid;
    reg instr_ready;
    wire [15:0] instr_index;
    wire [7:0] instr_a;
    wire [7:0] instr_b;
    wire [3:0] c_count;
    wire [MAX_CONSTS*16-1:0] c_flat;
    wire [3:0] d_count;
    wire [MAX_ARITY*16-1:0] d_flat;
    wire busy;
    wire error;

    nac_abcd_decoder #(
        .MAX_ARITY(MAX_ARITY),
        .MAX_CONSTS(MAX_CONSTS),
        .ARITY_BITS(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .num_outputs(16'd1),
        .ops_byte_valid(ops_byte_valid),
        .ops_byte_ready(ops_byte_ready),
        .ops_byte(ops_byte),
        .perm_lookup_id(perm_lookup_id),
        .perm_arity(perm_arity),
        .perm_needs_consts(perm_needs_consts),
        .perm_present(perm_present),
        .instr_valid(instr_valid),
        .instr_ready(instr_ready),
        .instr_index(instr_index),
        .instr_a(instr_a),
        .instr_b(instr_b),
        .c_count(c_count),
        .c_flat(c_flat),
        .d_count(d_count),
        .d_flat(d_flat),
        .busy(busy),
        .error(error)
    );

    always @* begin
        perm_arity = 4'd0;
        perm_needs_consts = 1'b0;
        perm_present = 1'b0;
        if (perm_lookup_id == 8'd1) begin
            perm_arity = 4'd2;
            perm_needs_consts = 1'b0;
            perm_present = 1'b1;
        end
    end

    always @(posedge clk) begin
        if (rst || start) begin
            rp <= 0;
        end else if (ops_byte_valid && ops_byte_ready) begin
            rp <= rp + 1'b1;
        end
    end

    integer seen;
    integer cycles;

    initial begin
        // Instruction 0: regular op A=11, B=1, D=[-1,-2].
        rom[0] = 8'd11;
        rom[1] = 8'd1;
        rom[2] = 8'hff;
        rom[3] = 8'hff;
        rom[4] = 8'hfe;
        rom[5] = 8'hff;
        // Instruction 1: OUTPUT B=0, num_outputs=1, C=[2,0], D=[-1].
        rom[6] = `NAC_OP_OUTPUT;
        rom[7] = 8'd0;
        rom[8] = 8'd2;
        rom[9] = 8'd0;
        rom[10] = 8'd0;
        rom[11] = 8'd0;
        rom[12] = 8'hff;
        rom[13] = 8'hff;
        rom[14] = 8'd0;
        rom[15] = 8'd0;
        rom[16] = 8'd0;
        rom[17] = 8'd0;
        rom[18] = 8'd0;
        rom[19] = 8'd0;

        rst = 1'b1;
        start = 1'b0;
        instr_ready = 1'b1;
        seen = 0;
        cycles = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (seen < 2) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 200) begin
                $fatal(1, "decoder timeout state=%0d rp=%0d valid=%0b ready=%0b a=%0d b=%0d c=%0d d=%0d error=%0b",
                       dut.state, rp, instr_valid, instr_ready, instr_a, instr_b, c_count, d_count, error);
            end
            if (instr_valid && instr_ready) begin
                if (seen == 0) begin
                    if (instr_a != 8'd11 || instr_b != 8'd1) $fatal(1, "bad regular opcode");
                    if (d_count != 4'd2) $fatal(1, "bad D count");
                    if (d_flat[0 +: 16] != 16'hffff) $fatal(1, "bad D0");
                    if (d_flat[16 +: 16] != 16'hfffe) $fatal(1, "bad D1");
                end else begin
                    if (instr_a != `NAC_OP_OUTPUT || instr_b != 8'd0) $fatal(1, "bad output opcode");
                    if (c_count != 4'd2 || d_count != 4'd1) $fatal(1, "bad output counts");
                end
                seen = seen + 1;
            end
            if (error) $fatal(1, "decoder error");
        end

        $display("nac_abcd_decoder_tb PASS");
        $finish;
    end
endmodule
