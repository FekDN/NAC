`include "nac_defs.vh"

module nac_kernel_sequencer (
    input  wire clk,
    input  wire rst,

    input  wire start,
    input  wire [7:0] kernel_class,
    input  wire [7:0] base_dsp_mode,

    output reg  step_valid,
    input  wire step_ready,
    output reg  [7:0] step_dsp_mode,
    output reg  [3:0] step_id,
    output reg  step_last,

    output reg  busy,
    output reg  done
);
    localparam S_IDLE = 2'd0;
    localparam S_EMIT = 2'd1;
    localparam S_DONE = 2'd2;

    (* fsm_safe_state = "default_state" *) reg [1:0] state;
    reg [3:0] pc;
    reg [3:0] last_pc;

    function [7:0] mode_for;
        input [7:0] cls;
        input [7:0] base_mode;
        input [3:0] id;
        begin
            mode_for = base_mode;
            case (cls)
                `NAC_KCLASS_LAYERNORM: begin
                    case (id)
                        4'd0: mode_for = `NAC_DSP_REDUCE_SUM;   // mean
                        4'd1: mode_for = `NAC_DSP_SUB;          // x - mean
                        4'd2: mode_for = `NAC_DSP_MUL;          // squared diff
                        4'd3: mode_for = `NAC_DSP_REDUCE_SUM;   // variance
                        4'd4: mode_for = `NAC_DSP_RSQRT;        // inv std
                        4'd5: mode_for = `NAC_DSP_SCALE;        // affine scale
                        default: mode_for = `NAC_DSP_PASS;
                    endcase
                end
                `NAC_KCLASS_SOFTMAX: begin
                    case (id)
                        4'd0: mode_for = `NAC_DSP_REDUCE_MAX;
                        4'd1: mode_for = `NAC_DSP_SUB;
                        4'd2: mode_for = `NAC_DSP_EXP;
                        4'd3: mode_for = `NAC_DSP_REDUCE_SUM;
                        4'd4: mode_for = `NAC_DSP_SCALE;
                        default: mode_for = `NAC_DSP_PASS;
                    endcase
                end
                `NAC_KCLASS_MATMUL,
                `NAC_KCLASS_CONV2D: begin
                    mode_for = `NAC_DSP_MAC;
                end
                default: begin
                    mode_for = base_mode;
                end
            endcase
        end
    endfunction

    function [3:0] last_for;
        input [7:0] cls;
        begin
            case (cls)
                `NAC_KCLASS_LAYERNORM: last_for = 4'd5;
                `NAC_KCLASS_SOFTMAX:   last_for = 4'd4;
                default:               last_for = 4'd0;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            pc <= 4'd0;
            last_pc <= 4'd0;
            step_valid <= 1'b0;
            step_dsp_mode <= `NAC_DSP_NOP;
            step_id <= 4'd0;
            step_last <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    step_valid <= 1'b0;
                    busy <= 1'b0;
                    if (start) begin
                        pc <= 4'd0;
                        last_pc <= last_for(kernel_class);
                        step_valid <= 1'b1;
                        step_dsp_mode <= mode_for(kernel_class, base_dsp_mode, 4'd0);
                        step_id <= 4'd0;
                        step_last <= (last_for(kernel_class) == 4'd0);
                        busy <= 1'b1;
                        state <= S_EMIT;
                    end
                end
                S_EMIT: begin
                    if (step_valid && step_ready) begin
                        if (pc == last_pc) begin
                            step_valid <= 1'b0;
                            state <= S_DONE;
                        end else begin
                            pc <= pc + 4'd1;
                            step_dsp_mode <= mode_for(kernel_class, base_dsp_mode, pc + 4'd1);
                            step_id <= pc + 4'd1;
                            step_last <= ((pc + 4'd1) == last_pc);
                        end
                    end
                end
                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
