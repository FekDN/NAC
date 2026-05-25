`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_kernel_sequencer_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] kernel_class;
    reg [7:0] base_dsp_mode;
    wire step_valid;
    reg step_ready;
    wire [7:0] step_dsp_mode;
    wire [3:0] step_id;
    wire step_last;
    wire busy;
    wire done;

    nac_kernel_sequencer dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .kernel_class(kernel_class),
        .base_dsp_mode(base_dsp_mode),
        .step_valid(step_valid),
        .step_ready(step_ready),
        .step_dsp_mode(step_dsp_mode),
        .step_id(step_id),
        .step_last(step_last),
        .busy(busy),
        .done(done)
    );

    integer count;
    integer cycles;

    initial begin
        rst = 1'b1;
        start = 1'b0;
        kernel_class = `NAC_KCLASS_SOFTMAX;
        base_dsp_mode = `NAC_DSP_PASS;
        step_ready = 1'b1;
        count = 0;
        cycles = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles > 100) begin
                $fatal(1, "sequencer timeout state=%0d pc=%0d count=%0d valid=%0b done=%0b",
                       dut.state, dut.pc, count, step_valid, done);
            end
            if (step_valid && step_ready) begin
                if (count == 0 && step_dsp_mode != `NAC_DSP_REDUCE_MAX) $fatal(1, "softmax pass0 failed");
                if (count == 2 && step_dsp_mode != `NAC_DSP_EXP) $fatal(1, "softmax pass2 failed");
                count = count + 1;
            end
        end

        if (count != 5) $fatal(1, "softmax pass count failed: %0d", count);
        $display("nac_kernel_sequencer_tb PASS");
        $finish;
    end
endmodule
