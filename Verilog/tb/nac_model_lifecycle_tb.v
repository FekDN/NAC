`timescale 1ns/1ps

module nac_model_lifecycle_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg model_configured;
    reg [11*64-1:0] section_offsets;
    reg [63:0] nac_file_size;
    reg pre_tisa_enable;
    reg post_tisa_enable;
    reg [1:0] pre_load_mask;
    reg [1:0] pre_free_mask;
    reg [1:0] post_load_mask;
    reg [1:0] post_free_mask;
    wire resource_req_valid;
    reg resource_req_ready;
    wire resource_req_is_free;
    wire [1:0] resource_req_phase;
    wire [1:0] resource_req_kind;
    wire [3:0] resource_req_section_index;
    wire [63:0] resource_req_offset;
    wire [63:0] resource_req_size;
    reg resource_done;
    reg resource_error;
    wire pre_tisa_start;
    reg pre_tisa_done;
    reg pre_tisa_error;
    wire model_run_start;
    reg model_run_done;
    reg model_run_error;
    wire post_tisa_start;
    reg post_tisa_done;
    reg post_tisa_error;
    wire busy;
    wire done;
    wire error;
    wire [3:0] state_debug;

    nac_model_lifecycle dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .model_configured(model_configured),
        .section_offsets(section_offsets),
        .nac_file_size(nac_file_size),
        .pre_tisa_enable(pre_tisa_enable),
        .post_tisa_enable(post_tisa_enable),
        .pre_load_mask(pre_load_mask),
        .pre_free_mask(pre_free_mask),
        .post_load_mask(post_load_mask),
        .post_free_mask(post_free_mask),
        .resource_req_valid(resource_req_valid),
        .resource_req_ready(resource_req_ready),
        .resource_req_is_free(resource_req_is_free),
        .resource_req_phase(resource_req_phase),
        .resource_req_kind(resource_req_kind),
        .resource_req_section_index(resource_req_section_index),
        .resource_req_offset(resource_req_offset),
        .resource_req_size(resource_req_size),
        .resource_done(resource_done),
        .resource_error(resource_error),
        .pre_tisa_start(pre_tisa_start),
        .pre_tisa_done(pre_tisa_done),
        .pre_tisa_error(pre_tisa_error),
        .model_run_start(model_run_start),
        .model_run_done(model_run_done),
        .model_run_error(model_run_error),
        .post_tisa_start(post_tisa_start),
        .post_tisa_done(post_tisa_done),
        .post_tisa_error(post_tisa_error),
        .busy(busy),
        .done(done),
        .error(error),
        .state_debug(state_debug)
    );

    integer loads;
    integer frees;
    integer pre_starts;
    integer model_starts;
    integer post_starts;
    integer cycles;

    always @(posedge clk) begin
        resource_done <= 1'b0;
        pre_tisa_done <= 1'b0;
        model_run_done <= 1'b0;
        post_tisa_done <= 1'b0;
        if (!rst) begin
            if (resource_req_valid && resource_req_ready) begin
                if (resource_req_is_free)
                    frees <= frees + 1;
                else
                    loads <= loads + 1;
                resource_done <= 1'b1;
            end
            if (pre_tisa_start) begin
                pre_starts <= pre_starts + 1;
                pre_tisa_done <= 1'b1;
            end
            if (model_run_start) begin
                model_starts <= model_starts + 1;
                model_run_done <= 1'b1;
            end
            if (post_tisa_start) begin
                post_starts <= post_starts + 1;
                post_tisa_done <= 1'b1;
            end
        end
    end

    initial begin
        rst = 1'b1;
        start = 1'b0;
        model_configured = 1'b1;
        section_offsets = {11*64{1'b0}};
        section_offsets[6*64 +: 64] = 64'd100; // PROC
        section_offsets[9*64 +: 64] = 64'd200; // RSRC
        section_offsets[10*64 +: 64] = 64'd320; // ARRS, next section for RSRC size
        nac_file_size = 64'd512;
        pre_tisa_enable = 1'b1;
        post_tisa_enable = 1'b1;
        pre_load_mask = 2'b11;
        pre_free_mask = 2'b11;
        post_load_mask = 2'b01;
        post_free_mask = 2'b01;
        resource_req_ready = 1'b1;
        resource_done = 1'b0;
        resource_error = 1'b0;
        pre_tisa_done = 1'b0;
        pre_tisa_error = 1'b0;
        model_run_done = 1'b0;
        model_run_error = 1'b0;
        post_tisa_done = 1'b0;
        post_tisa_error = 1'b0;
        loads = 0;
        frees = 0;
        pre_starts = 0;
        model_starts = 0;
        post_starts = 0;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        cycles = 0;
        while (!done && !error && cycles < 200) begin
            @(posedge clk);
            cycles = cycles + 1;
        end

        if (error) $fatal(1, "lifecycle error state=%0d", state_debug);
        if (!done) $fatal(1, "lifecycle timeout state=%0d loads=%0d frees=%0d pre=%0d model=%0d post=%0d",
                          state_debug, loads, frees, pre_starts, model_starts, post_starts);
        if (loads != 3) $fatal(1, "expected 3 resource loads, got %0d", loads);
        if (frees != 3) $fatal(1, "expected 3 resource frees, got %0d", frees);
        if (pre_starts != 1 || model_starts != 1 || post_starts != 1)
            $fatal(1, "bad phase starts pre=%0d model=%0d post=%0d",
                   pre_starts, model_starts, post_starts);

        $display("nac_model_lifecycle_tb PASS");
        $finish;
    end
endmodule
