`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_op_table_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg cfg_we;
    reg [7:0] cfg_op_id;
    reg [7:0] cfg_kernel_class;
    reg [7:0] cfg_dsp_mode;
    reg cfg_uses_dsp;
    reg cfg_multi_pass;
    reg [7:0] lookup_op_id;
    wire [7:0] lookup_kernel_class;
    wire [7:0] lookup_dsp_mode;
    wire lookup_uses_dsp;
    wire lookup_multi_pass;
    wire lookup_present;

    nac_op_table dut (
        .clk(clk),
        .cfg_we(cfg_we),
        .cfg_op_id(cfg_op_id),
        .cfg_kernel_class(cfg_kernel_class),
        .cfg_dsp_mode(cfg_dsp_mode),
        .cfg_uses_dsp(cfg_uses_dsp),
        .cfg_multi_pass(cfg_multi_pass),
        .lookup_op_id(lookup_op_id),
        .lookup_kernel_class(lookup_kernel_class),
        .lookup_dsp_mode(lookup_dsp_mode),
        .lookup_uses_dsp(lookup_uses_dsp),
        .lookup_multi_pass(lookup_multi_pass),
        .lookup_present(lookup_present)
    );

    initial begin
        cfg_we = 1'b0;
        cfg_op_id = 8'd0;
        cfg_kernel_class = `NAC_KCLASS_NONE;
        cfg_dsp_mode = `NAC_DSP_NOP;
        cfg_uses_dsp = 1'b0;
        cfg_multi_pass = 1'b0;
        lookup_op_id = 8'd200;

        #1;
        if (lookup_present) $fatal(1, "unconfigured ATen/CMAP id must be absent");

        @(negedge clk);
        cfg_we = 1'b1;
        cfg_op_id = 8'd200;
        cfg_kernel_class = `NAC_KCLASS_ELEMWISE;
        cfg_dsp_mode = `NAC_DSP_HSWISH;
        cfg_uses_dsp = 1'b1;
        cfg_multi_pass = 1'b0;

        @(negedge clk);
        cfg_we = 1'b0;
        lookup_op_id = 8'd200;
        #1;
        if (!lookup_present ||
            lookup_kernel_class != `NAC_KCLASS_ELEMWISE ||
            lookup_dsp_mode != `NAC_DSP_HSWISH ||
            !lookup_uses_dsp ||
            lookup_multi_pass) begin
            $fatal(1, "loaded ATen/CMAP recipe mismatch");
        end

        lookup_op_id = 8'd201;
        #1;
        if (lookup_present) $fatal(1, "different dynamic id must remain absent");

        $display("nac_op_table_tb PASS");
        $finish;
    end
endmodule
