`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_op_dispatch_tb;
    reg [7:0] op_a;
    reg [7:0] op_table_kernel_class;
    reg [7:0] op_table_dsp_mode;
    reg op_table_uses_dsp;
    reg op_table_multi_pass;
    reg op_table_present;
    wire [7:0] dsp_mode;
    wire [7:0] kernel_class;
    wire uses_dsp;
    wire multi_pass;
    wire supported;

    nac_op_dispatch dut (
        .op_a(op_a),
        .op_table_kernel_class(op_table_kernel_class),
        .op_table_dsp_mode(op_table_dsp_mode),
        .op_table_uses_dsp(op_table_uses_dsp),
        .op_table_multi_pass(op_table_multi_pass),
        .op_table_present(op_table_present),
        .dsp_mode(dsp_mode),
        .kernel_class(kernel_class),
        .uses_dsp(uses_dsp),
        .multi_pass(multi_pass),
        .supported(supported)
    );

    initial begin
        op_a = `NAC_STD_ADD;
        op_table_kernel_class = `NAC_KCLASS_NONE;
        op_table_dsp_mode = `NAC_DSP_NOP;
        op_table_uses_dsp = 1'b0;
        op_table_multi_pass = 1'b0;
        op_table_present = 1'b0;
        #1;
        if (!supported || dsp_mode != `NAC_DSP_ADD || !uses_dsp) $fatal(1, "standard ADD dispatch failed");

        op_a = 8'd200;
        op_table_kernel_class = `NAC_KCLASS_ELEMWISE;
        op_table_dsp_mode = `NAC_DSP_HSWISH;
        op_table_uses_dsp = 1'b1;
        op_table_multi_pass = 1'b0;
        op_table_present = 1'b1;
        #1;
        if (!supported || dsp_mode != `NAC_DSP_HSWISH || !uses_dsp) $fatal(1, "CMAP-derived dynamic ATen dispatch failed");

        op_a = 8'd202;
        op_table_kernel_class = `NAC_KCLASS_SOFTMAX;
        op_table_dsp_mode = `NAC_DSP_REDUCE_MAX;
        op_table_uses_dsp = 1'b1;
        op_table_multi_pass = 1'b1;
        op_table_present = 1'b1;
        #1;
        if (!supported || dsp_mode != `NAC_DSP_REDUCE_MAX || !multi_pass) $fatal(1, "CMAP-derived multi-pass dispatch failed");

        op_a = 8'd201;
        op_table_kernel_class = `NAC_KCLASS_NONE;
        op_table_dsp_mode = `NAC_DSP_NOP;
        op_table_uses_dsp = 1'b0;
        op_table_multi_pass = 1'b0;
        op_table_present = 1'b0;
        #1;
        if (supported) $fatal(1, "unknown op must not be supported");

        $display("nac_op_dispatch_tb PASS");
        $finish;
    end
endmodule
