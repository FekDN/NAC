`include "nac_defs.vh"

module nac_op_dispatch (
    input  wire [7:0] op_a,
    input  wire [7:0] op_table_kernel_class,
    input  wire [7:0] op_table_dsp_mode,
    input  wire op_table_uses_dsp,
    input  wire op_table_multi_pass,
    input  wire op_table_present,
    output reg  [7:0] dsp_mode,
    output reg  [7:0] kernel_class,
    output reg  uses_dsp,
    output reg  multi_pass,
    output reg  supported
);
    always @* begin
        dsp_mode = `NAC_DSP_NOP;
        kernel_class = op_table_kernel_class;
        uses_dsp = 1'b0;
        multi_pass = 1'b0;
        supported = 1'b1;

        case (op_a)
            `NAC_STD_PASS,
            `NAC_STD_CLONE,
            `NAC_STD_VIEW: begin
                dsp_mode = `NAC_DSP_PASS;
                kernel_class = `NAC_KCLASS_RESHAPE;
                uses_dsp = 1'b0;
            end
            `NAC_STD_ADD: begin
                dsp_mode = `NAC_DSP_ADD;
                kernel_class = `NAC_KCLASS_ELEMWISE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_SUB: begin
                dsp_mode = `NAC_DSP_SUB;
                kernel_class = `NAC_KCLASS_ELEMWISE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_MUL: begin
                dsp_mode = `NAC_DSP_MUL;
                kernel_class = `NAC_KCLASS_ELEMWISE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_NEG: begin
                dsp_mode = `NAC_DSP_NEG;
                kernel_class = `NAC_KCLASS_ELEMWISE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_GT: begin
                dsp_mode = `NAC_DSP_GT;
                kernel_class = `NAC_KCLASS_COMPARE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_LE: begin
                dsp_mode = `NAC_DSP_LE;
                kernel_class = `NAC_KCLASS_COMPARE;
                uses_dsp = 1'b1;
            end
            `NAC_STD_MATMUL: begin
                dsp_mode = `NAC_DSP_MAC;
                kernel_class = `NAC_KCLASS_MATMUL;
                uses_dsp = 1'b1;
            end
            default: begin
                if (op_table_present) begin
                    kernel_class = op_table_kernel_class;
                    dsp_mode = op_table_dsp_mode;
                    uses_dsp = op_table_uses_dsp;
                    multi_pass = op_table_multi_pass;
                    supported = 1'b1;
                end else begin
                    dsp_mode = `NAC_DSP_NOP;
                    uses_dsp = 1'b0;
                    multi_pass = 1'b0;
                    supported = 1'b0;
                end
            end
        endcase
    end
endmodule
