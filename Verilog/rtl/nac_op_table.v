`include "nac_defs.vh"

module nac_op_table #(
    parameter OP_ENTRIES = 256
) (
    input  wire clk,

    input  wire cfg_we,
    input  wire [7:0] cfg_op_id,
    input  wire [7:0] cfg_kernel_class,

    input  wire [7:0] lookup_op_id,
    output wire [7:0] lookup_kernel_class
);
    reg [7:0] class_ram [0:OP_ENTRIES-1];
    integer i;

    initial begin
        for (i = 0; i < OP_ENTRIES; i = i + 1) begin
            class_ram[i] = `NAC_KCLASS_NONE;
        end
    end

    always @(posedge clk) begin
        if (cfg_we) begin
            class_ram[cfg_op_id] <= cfg_kernel_class;
        end
    end

    assign lookup_kernel_class = class_ram[lookup_op_id];
endmodule
