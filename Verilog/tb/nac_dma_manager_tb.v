`timescale 1ns/1ps

module nac_dma_manager_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg cfg_we;
    reg [7:0] cfg_target;
    reg [63:0] cfg_ext_addr;
    reg [31:0] cfg_size_bytes;
    reg cfg_static;
    reg mark_dirty_valid;
    reg [7:0] mark_dirty_target;
    reg preload_valid;
    wire preload_ready;
    reg [15:0] preload_target;
    reg free_valid;
    wire free_ready;
    reg [15:0] free_target;
    reg save_result_valid;
    wire save_result_ready;
    reg [15:0] save_result_src;
    reg [15:0] save_result_target;
    reg forward_valid;
    wire forward_ready;
    reg [15:0] forward_src;
    reg [15:0] forward_dst;
    wire bank_alloc_valid;
    reg bank_alloc_ready;
    reg [3:0] bank_alloc_id;
    wire bank_free_valid;
    reg bank_free_ready;
    wire [3:0] bank_free_id;
    wire dma_read_valid;
    reg dma_read_ready;
    wire [63:0] dma_read_addr;
    wire [31:0] dma_read_len;
    wire [3:0] dma_read_bank;
    reg dma_read_done;
    reg dma_read_error;
    wire dma_write_valid;
    reg dma_write_ready;
    wire [63:0] dma_write_addr;
    wire [31:0] dma_write_len;
    wire [3:0] dma_write_bank;
    reg dma_write_done;
    reg dma_write_error;
    wire desc_update_valid;
    wire [7:0] desc_update_target;
    wire [3:0] desc_update_bank;
    wire [31:0] desc_update_size;
    wire desc_invalidate_valid;
    wire [7:0] desc_invalidate_target;
    wire busy;
    wire error;
    wire [7:0] error_code;

    nac_dma_manager dut (
        .clk(clk),
        .rst(rst),
        .cfg_we(cfg_we),
        .cfg_target(cfg_target),
        .cfg_ext_addr(cfg_ext_addr),
        .cfg_size_bytes(cfg_size_bytes),
        .cfg_static(cfg_static),
        .mark_dirty_valid(mark_dirty_valid),
        .mark_dirty_target(mark_dirty_target),
        .preload_valid(preload_valid),
        .preload_ready(preload_ready),
        .preload_target(preload_target),
        .free_valid(free_valid),
        .free_ready(free_ready),
        .free_target(free_target),
        .save_result_valid(save_result_valid),
        .save_result_ready(save_result_ready),
        .save_result_src(save_result_src),
        .save_result_target(save_result_target),
        .forward_valid(forward_valid),
        .forward_ready(forward_ready),
        .forward_src(forward_src),
        .forward_dst(forward_dst),
        .bank_alloc_valid(bank_alloc_valid),
        .bank_alloc_ready(bank_alloc_ready),
        .bank_alloc_id(bank_alloc_id),
        .bank_free_valid(bank_free_valid),
        .bank_free_ready(bank_free_ready),
        .bank_free_id(bank_free_id),
        .dma_read_valid(dma_read_valid),
        .dma_read_ready(dma_read_ready),
        .dma_read_addr(dma_read_addr),
        .dma_read_len(dma_read_len),
        .dma_read_bank(dma_read_bank),
        .dma_read_done(dma_read_done),
        .dma_read_error(dma_read_error),
        .dma_write_valid(dma_write_valid),
        .dma_write_ready(dma_write_ready),
        .dma_write_addr(dma_write_addr),
        .dma_write_len(dma_write_len),
        .dma_write_bank(dma_write_bank),
        .dma_write_done(dma_write_done),
        .dma_write_error(dma_write_error),
        .desc_update_valid(desc_update_valid),
        .desc_update_target(desc_update_target),
        .desc_update_bank(desc_update_bank),
        .desc_update_size(desc_update_size),
        .desc_invalidate_valid(desc_invalidate_valid),
        .desc_invalidate_target(desc_invalidate_target),
        .busy(busy),
        .error(error),
        .error_code(error_code)
    );

    task cfg_target_desc;
        input [7:0] id;
        input [63:0] addr;
        input [31:0] size;
        input is_static;
        begin
            @(negedge clk);
            cfg_target = id;
            cfg_ext_addr = addr;
            cfg_size_bytes = size;
            cfg_static = is_static;
            cfg_we = 1'b1;
            @(negedge clk);
            cfg_we = 1'b0;
            cfg_static = 1'b0;
        end
    endtask

    integer cycles;
    integer updates;
    integer invalidates;
    integer frees;

    always @(posedge clk) begin
        if (rst) begin
            updates <= 0;
            invalidates <= 0;
            frees <= 0;
        end else begin
            if (desc_update_valid)
                updates <= updates + 1;
            if (desc_invalidate_valid)
                invalidates <= invalidates + 1;
            if (bank_free_valid && bank_free_ready)
                frees <= frees + 1;
        end
    end

    initial begin
        rst = 1'b1;
        cfg_we = 1'b0;
        cfg_target = 8'd0;
        cfg_ext_addr = 64'd0;
        cfg_size_bytes = 32'd0;
        cfg_static = 1'b0;
        mark_dirty_valid = 1'b0;
        mark_dirty_target = 8'd0;
        preload_valid = 1'b0;
        preload_target = 16'd0;
        free_valid = 1'b0;
        free_target = 16'd0;
        save_result_valid = 1'b0;
        save_result_src = 16'd0;
        save_result_target = 16'd0;
        forward_valid = 1'b0;
        forward_src = 16'd0;
        forward_dst = 16'd0;
        bank_alloc_ready = 1'b1;
        bank_alloc_id = 4'd3;
        bank_free_ready = 1'b1;
        dma_read_ready = 1'b1;
        dma_read_done = 1'b0;
        dma_read_error = 1'b0;
        dma_write_ready = 1'b1;
        dma_write_done = 1'b0;
        dma_write_error = 1'b0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        cfg_target_desc(8'd5, 64'h1000, 32'd64, 1'b0);
        cfg_target_desc(8'd6, 64'h2000, 32'd64, 1'b0);

        @(negedge clk);
        preload_target = 16'd5;
        preload_valid = 1'b1;
        @(negedge clk);
        preload_valid = 1'b0;
        wait(dma_read_valid);
        #1;
        if (dma_read_addr != 64'h1000 || dma_read_len != 32'd64 || dma_read_bank != 4'd3)
            $fatal(1, "bad DMA read descriptor");
        @(negedge clk);
        dma_read_done = 1'b1;
        @(negedge clk);
        dma_read_done = 1'b0;
        cycles = 0;
        while (updates < 1 && cycles < 20) begin @(posedge clk); cycles = cycles + 1; end
        if (updates < 1 || error) $fatal(1, "preload did not update descriptor");

        @(negedge clk);
        save_result_src = 16'd5;
        save_result_target = 16'd6;
        save_result_valid = 1'b1;
        @(negedge clk);
        save_result_valid = 1'b0;
        wait(dma_write_valid);
        #1;
        if (dma_write_addr != 64'h2000 || dma_write_len != 32'd64 || dma_write_bank != 4'd3)
            $fatal(1, "bad DMA write descriptor");
        @(negedge clk);
        dma_write_done = 1'b1;
        @(negedge clk);
        dma_write_done = 1'b0;
        cycles = 0;
        while (updates < 2 && cycles < 20) begin @(posedge clk); cycles = cycles + 1; end
        if (updates < 2 || error) $fatal(1, "save did not update descriptor");

        @(negedge clk);
        free_target = 16'd5;
        free_valid = 1'b1;
        @(negedge clk);
        free_valid = 1'b0;
        cycles = 0;
        while (invalidates < 1 && cycles < 20) begin @(posedge clk); cycles = cycles + 1; end
        if (frees != 1 || invalidates != 1 || bank_free_id != 4'd3)
            $fatal(1, "free did not release bank");

        bank_alloc_id = 4'd4;
        cfg_target_desc(8'd7, 64'h3000, 32'd128, 1'b1);
        @(negedge clk);
        preload_target = 16'd7;
        preload_valid = 1'b1;
        @(negedge clk);
        preload_valid = 1'b0;
        wait(dma_read_valid);
        @(negedge clk);
        dma_read_done = 1'b1;
        @(negedge clk);
        dma_read_done = 1'b0;
        cycles = 0;
        while (updates < 3 && cycles < 20) begin @(posedge clk); cycles = cycles + 1; end
        @(negedge clk);
        free_target = 16'd7;
        free_valid = 1'b1;
        @(negedge clk);
        free_valid = 1'b0;
        repeat (5) @(posedge clk);
        if (frees != 1) $fatal(1, "static target was physically freed");

        if (error) $fatal(1, "DMA manager error %0d", error_code);
        $display("nac_dma_manager_tb PASS");
        $finish;
    end
endmodule
