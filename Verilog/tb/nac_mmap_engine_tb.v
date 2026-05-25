`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_mmap_engine_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg cfg_we;
    reg [3:0] cfg_tick;
    reg [1:0] cfg_slot;
    reg cfg_valid;
    reg [7:0] cfg_action;
    reg [15:0] cfg_target;
    reg tick_valid;
    reg [3:0] tick_id;
    wire busy;
    wire preload_valid;
    wire [15:0] preload_target;
    wire free_valid;
    wire [15:0] free_target;
    wire save_result_valid;
    wire [15:0] save_result_src;
    wire [15:0] save_result_target;
    wire forward_valid;
    wire [15:0] forward_src;
    wire [15:0] forward_dst;

    nac_mmap_engine #(
        .NUM_TICKS(16),
        .TICK_WIDTH(4),
        .MAX_CMDS_PER_TICK(4)
    ) dut (
        .clk(clk),
        .rst(rst),
        .cfg_we(cfg_we),
        .cfg_tick(cfg_tick),
        .cfg_slot(cfg_slot),
        .cfg_valid(cfg_valid),
        .cfg_action(cfg_action),
        .cfg_target(cfg_target),
        .tick_valid(tick_valid),
        .tick_id(tick_id),
        .busy(busy),
        .preload_valid(preload_valid),
        .preload_target(preload_target),
        .free_valid(free_valid),
        .free_target(free_target),
        .save_result_valid(save_result_valid),
        .save_result_src(save_result_src),
        .save_result_target(save_result_target),
        .forward_valid(forward_valid),
        .forward_src(forward_src),
        .forward_dst(forward_dst)
    );

    task cfg;
        input [3:0] t;
        input [1:0] s;
        input [7:0] a;
        input [15:0] target;
        begin
            @(posedge clk);
            cfg_tick <= t;
            cfg_slot <= s;
            cfg_action <= a;
            cfg_target <= target;
            cfg_valid <= 1'b1;
            cfg_we <= 1'b1;
            @(posedge clk);
            cfg_we <= 1'b0;
        end
    endtask

    initial begin
        rst = 1'b1;
        cfg_we = 1'b0;
        cfg_tick = 0;
        cfg_slot = 0;
        cfg_valid = 0;
        cfg_action = 0;
        cfg_target = 0;
        tick_valid = 0;
        tick_id = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;

        cfg(4'd5, 2'd0, `NAC_MMAP_PRELOAD, 16'd9);
        cfg(4'd5, 2'd1, `NAC_MMAP_FREE, 16'd2);
        cfg(4'd5, 2'd2, `NAC_MMAP_FORWARD, 16'd6);
        cfg(4'd5, 2'd3, `NAC_MMAP_SAVE_RESULT, 16'd12);

        @(posedge clk);
        tick_id <= 4'd5;
        tick_valid <= 1'b1;
        @(posedge clk);
        tick_valid <= 1'b0;

        wait(preload_valid);
        if (preload_target != 16'd9) $fatal(1, "PRELOAD target failed");
        wait(free_valid);
        if (free_target != 16'd2) $fatal(1, "FREE target failed");
        wait(forward_valid);
        if (forward_src != 16'd5 || forward_dst != 16'd6) $fatal(1, "FORWARD target failed");
        wait(save_result_valid);
        if (save_result_src != 16'd5 || save_result_target != 16'd12) $fatal(1, "SAVE_RESULT target failed");

        $display("nac_mmap_engine_tb PASS");
        $finish;
    end
endmodule
