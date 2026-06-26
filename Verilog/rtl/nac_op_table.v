`include "nac_defs.vh"

module nac_op_table #(
    parameter OP_ENTRIES = 256
) (
    input  wire clk,

    input  wire cfg_we,
    input  wire [7:0] cfg_op_id,
    input  wire [7:0] cfg_kernel_class,
    input  wire [7:0] cfg_dsp_mode,
    input  wire cfg_uses_dsp,
    input  wire cfg_multi_pass,

    input  wire [7:0] lookup_op_id,
    output wire [7:0] lookup_kernel_class,
    output wire [7:0] lookup_dsp_mode,
    output wire lookup_uses_dsp,
    output wire lookup_multi_pass,
    output wire lookup_present
);
    reg [7:0] class_ram [0:OP_ENTRIES-1];
    reg [7:0] dsp_mode_ram [0:OP_ENTRIES-1];
    reg uses_dsp_ram [0:OP_ENTRIES-1];
    reg multi_pass_ram [0:OP_ENTRIES-1];
    reg present_ram [0:OP_ENTRIES-1];
    integer i;

    initial begin
        for (i = 0; i < OP_ENTRIES; i = i + 1) begin
            class_ram[i] = `NAC_KCLASS_NONE;
            dsp_mode_ram[i] = `NAC_DSP_NOP;
            uses_dsp_ram[i] = 1'b0;
            multi_pass_ram[i] = 1'b0;
            present_ram[i] = 1'b0;
        end
    end

    always @(posedge clk) begin
        if (cfg_we) begin
            class_ram[cfg_op_id] <= cfg_kernel_class;
            dsp_mode_ram[cfg_op_id] <= cfg_dsp_mode;
            uses_dsp_ram[cfg_op_id] <= cfg_uses_dsp;
            multi_pass_ram[cfg_op_id] <= cfg_multi_pass;
            present_ram[cfg_op_id] <= 1'b1;
        end
    end

    assign lookup_kernel_class = class_ram[lookup_op_id];
    assign lookup_dsp_mode = dsp_mode_ram[lookup_op_id];
    assign lookup_uses_dsp = uses_dsp_ram[lookup_op_id];
    assign lookup_multi_pass = multi_pass_ram[lookup_op_id];
    assign lookup_present = present_ram[lookup_op_id];
endmodule
