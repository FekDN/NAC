`timescale 1ns/1ps

module nac_addr_gen_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg cfg_valid;
    reg [7:0] cfg_base;
    reg [3:0] cfg_limit0, cfg_limit1, cfg_limit2, cfg_limit3;
    reg signed [7:0] cfg_stride0, cfg_stride1, cfg_stride2, cfg_stride3;
    reg start;
    reg step_ready;
    wire step_valid;
    wire [7:0] addr;
    wire done;
    wire error;

    nac_addr_gen #(
        .ADDR_WIDTH(8),
        .DIM_WIDTH(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .cfg_valid(cfg_valid),
        .cfg_base(cfg_base),
        .cfg_limit0(cfg_limit0),
        .cfg_limit1(cfg_limit1),
        .cfg_limit2(cfg_limit2),
        .cfg_limit3(cfg_limit3),
        .cfg_stride0(cfg_stride0),
        .cfg_stride1(cfg_stride1),
        .cfg_stride2(cfg_stride2),
        .cfg_stride3(cfg_stride3),
        .start(start),
        .step_ready(step_ready),
        .step_valid(step_valid),
        .addr(addr),
        .done(done),
        .error(error)
    );

    task configure;
        input [7:0] base;
        input [3:0] l0;
        input [3:0] l1;
        input [3:0] l2;
        input [3:0] l3;
        input signed [7:0] s0;
        input signed [7:0] s1;
        input signed [7:0] s2;
        input signed [7:0] s3;
        begin
            @(negedge clk);
            cfg_base = base;
            cfg_limit0 = l0;
            cfg_limit1 = l1;
            cfg_limit2 = l2;
            cfg_limit3 = l3;
            cfg_stride0 = s0;
            cfg_stride1 = s1;
            cfg_stride2 = s2;
            cfg_stride3 = s3;
            cfg_valid = 1'b1;
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    initial begin
        rst = 1'b1;
        cfg_valid = 1'b0;
        cfg_base = 8'd0;
        cfg_limit0 = 4'd0;
        cfg_limit1 = 4'd0;
        cfg_limit2 = 4'd0;
        cfg_limit3 = 4'd0;
        cfg_stride0 = 8'sd0;
        cfg_stride1 = 8'sd0;
        cfg_stride2 = 8'sd0;
        cfg_stride3 = 8'sd0;
        start = 1'b0;
        step_ready = 1'b1;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        configure(8'd10, 4'd1, 4'd1, 4'd1, 4'd3, 8'sd0, 8'sd0, 8'sd0, 8'sd2);
        pulse_start();
        if (!step_valid || addr != 8'd10 || error) $fatal(1, "bad first address");
        @(negedge clk);
        if (!step_valid || addr != 8'd12 || error) $fatal(1, "bad second address");
        @(negedge clk);
        if (!step_valid || addr != 8'd14 || error) $fatal(1, "bad third address");
        @(negedge clk);
        if (!done || step_valid || error) $fatal(1, "normal sequence did not finish cleanly");

        configure(8'd0, 4'd1, 4'd1, 4'd1, 4'd0, 8'sd0, 8'sd0, 8'sd0, 8'sd1);
        pulse_start();
        if (!done || step_valid || error) $fatal(1, "zero dimension was not handled as empty");

        configure(8'd250, 4'd1, 4'd1, 4'd1, 4'd4, 8'sd0, 8'sd0, 8'sd0, 8'sd4);
        pulse_start();
        if (!step_valid || addr != 8'd250 || error) $fatal(1, "positive overflow first address failed");
        @(negedge clk);
        if (!step_valid || addr != 8'd254 || error) $fatal(1, "positive overflow second address failed");
        @(negedge clk);
        if (!error || !done || step_valid) $fatal(1, "positive address overflow was not detected");

        configure(8'd3, 4'd1, 4'd1, 4'd1, 4'd3, 8'sd0, 8'sd0, 8'sd0, -8'sd2);
        pulse_start();
        if (!step_valid || addr != 8'd3 || error) $fatal(1, "negative stride first address failed");
        @(negedge clk);
        if (!step_valid || addr != 8'd1 || error) $fatal(1, "negative stride second address failed");
        @(negedge clk);
        if (!error || !done || step_valid) $fatal(1, "negative address overflow was not detected");

        $display("nac_addr_gen_tb PASS");
        $finish;
    end
endmodule
