`timescale 1ns/1ps

module nac_watchdog_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg clear;
    reg enable;
    reg busy;
    reg progress;
    wire timeout;
    wire reset_pulse;

    nac_watchdog #(
        .TIMEOUT_CYCLES(3),
        .COUNTER_WIDTH(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .enable(enable),
        .busy(busy),
        .progress(progress),
        .timeout(timeout),
        .reset_pulse(reset_pulse)
    );

    initial begin
        rst = 1'b1;
        clear = 1'b0;
        enable = 1'b1;
        busy = 1'b0;
        progress = 1'b0;
        repeat (2) @(posedge clk);
        rst = 1'b0;

        busy = 1'b1;
        repeat (4) @(posedge clk);
        if (!timeout) $fatal(1, "watchdog did not latch timeout");

        clear = 1'b1;
        @(posedge clk);
        #1;
        clear = 1'b0;
        if (timeout) $fatal(1, "watchdog clear failed");

        busy = 1'b1;
        @(posedge clk);
        progress = 1'b1;
        @(posedge clk);
        progress = 1'b0;
        @(posedge clk);
        @(posedge clk);
        if (timeout) $fatal(1, "watchdog ignored progress");
        @(posedge clk);
        @(posedge clk);
        if (!timeout) $fatal(1, "watchdog did not time out after progress stopped");

        $display("nac_watchdog_tb PASS");
        $finish;
    end
endmodule
