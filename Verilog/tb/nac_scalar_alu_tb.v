`timescale 1ns/1ps

module nac_scalar_alu_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg is_compare;
    reg [7:0] op_type;
    reg signed [63:0] a;
    reg signed [63:0] b;
    wire busy;
    wire done;
    wire error;
    wire [63:0] result;
    wire result_is_bool;

    nac_scalar_alu #(
        .WIDTH(64)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .is_compare(is_compare),
        .op_type(op_type),
        .a(a),
        .b(b),
        .busy(busy),
        .done(done),
        .error(error),
        .result(result),
        .result_is_bool(result_is_bool)
    );

    task run_op;
        input cmp;
        input [7:0] op;
        input signed [63:0] av;
        input signed [63:0] bv;
        input signed [63:0] expected;
        integer cycles;
        begin
            @(negedge clk);
            is_compare = cmp;
            op_type = op;
            a = av;
            b = bv;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            cycles = 0;
            while (!done && !error && cycles < 200) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            if (error) $fatal(1, "scalar alu error op=%0d", op);
            if (!done) $fatal(1, "scalar alu timeout op=%0d", op);
            #1;
            if ($signed(result) != expected)
                $fatal(1, "bad scalar result op=%0d got=%0d expected=%0d",
                       op, $signed(result), expected);
        end
    endtask

    initial begin
        rst = 1'b1;
        start = 1'b0;
        is_compare = 1'b0;
        op_type = 8'd0;
        a = 64'sd0;
        b = 64'sd0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        run_op(1'b0, 8'd0, 64'sd12, 64'sd7, 64'sd19);
        run_op(1'b0, 8'd1, 64'sd12, 64'sd7, 64'sd5);
        run_op(1'b0, 8'd2, 64'sd12, -64'sd7, -64'sd84);
        run_op(1'b0, 8'd3, -64'sd84, 64'sd7, -64'sd12);
        run_op(1'b0, 8'd3, 64'sd99, 64'sd0, 64'sd0);
        run_op(1'b0, 8'd5, 64'sd12, 64'sd7, 64'sd12);
        run_op(1'b0, 8'd6, 64'sd12, 64'sd7, 64'sd7);
        run_op(1'b1, 8'd2, 64'sd12, 64'sd7, 64'sd1);
        run_op(1'b1, 8'd3, 64'sd12, 64'sd7, 64'sd0);

        $display("nac_scalar_alu_tb PASS");
        $finish;
    end
endmodule
