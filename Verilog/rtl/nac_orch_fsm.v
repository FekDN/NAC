`include "nac_defs.vh"

module nac_orch_fsm #(
    parameter IDX_WIDTH = 10,
    parameter DESC_WIDTH = 64,
    parameter MAX_ARITY = 8,
    parameter MAX_CONSTS = 8
) (
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire instr_valid,
    output reg  instr_ready,
    input  wire [15:0] instr_index,
    input  wire [7:0] instr_a,
    input  wire [7:0] instr_b,
    input  wire [3:0] c_count,
    input  wire [MAX_CONSTS*16-1:0] c_flat,
    input  wire [3:0] d_count,
    input  wire [MAX_ARITY*16-1:0] d_flat,

    input  wire [7:0] op_table_kernel_class,

    output reg  result_wr_valid,
    output reg  [IDX_WIDTH-1:0] result_wr_idx,
    output reg  [DESC_WIDTH-1:0] result_wr_desc,

    output reg  tick_commit_valid,
    output reg  [IDX_WIDTH-1:0] tick_commit_id,

    output reg  kernel_start,
    output reg  [7:0] kernel_op_a,
    output reg  [7:0] kernel_op_b,
    output reg  [7:0] kernel_class,
    output reg  [7:0] dsp_mode,
    output reg  [IDX_WIDTH-1:0] kernel_instr_idx,
    output reg  [3:0] kernel_c_count,
    output reg  [MAX_CONSTS*16-1:0] kernel_c_flat,
    output reg  [3:0] kernel_d_count,
    output reg  [MAX_ARITY*16-1:0] kernel_d_flat,
    input  wire kernel_done,
    input  wire [DESC_WIDTH-1:0] kernel_result_desc,

    output reg  input_req_valid,
    output reg  [7:0] input_kind,
    output reg  [15:0] input_id,
    input  wire input_req_ready,
    input  wire input_desc_valid,
    input  wire [DESC_WIDTH-1:0] input_desc,

    output reg  done,
    output reg  error
);
    localparam S_IDLE       = 4'd0;
    localparam S_FETCH      = 4'd1;
    localparam S_INPUT_REQ  = 4'd2;
    localparam S_INPUT_WAIT = 4'd3;
    localparam S_KERNEL     = 4'd4;
    localparam S_COMMIT     = 4'd5;
    localparam S_DONE       = 4'd6;
    localparam S_ERROR      = 4'd7;

    reg [3:0] state;
    reg [15:0] active_index;
    reg [7:0] active_a;
    reg [7:0] active_b;
    reg [DESC_WIDTH-1:0] pending_desc;

    wire [7:0] dispatch_dsp_mode;
    wire [7:0] dispatch_kernel_class;
    wire dispatch_uses_dsp;
    wire dispatch_multi_pass;
    wire dispatch_supported;

    nac_op_dispatch dispatch (
        .op_a(instr_a),
        .op_table_kernel_class(op_table_kernel_class),
        .dsp_mode(dispatch_dsp_mode),
        .kernel_class(dispatch_kernel_class),
        .uses_dsp(dispatch_uses_dsp),
        .multi_pass(dispatch_multi_pass),
        .supported(dispatch_supported)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            instr_ready <= 1'b0;
            result_wr_valid <= 1'b0;
            result_wr_idx <= {IDX_WIDTH{1'b0}};
            result_wr_desc <= {DESC_WIDTH{1'b0}};
            tick_commit_valid <= 1'b0;
            tick_commit_id <= {IDX_WIDTH{1'b0}};
            kernel_start <= 1'b0;
            kernel_op_a <= 8'd0;
            kernel_op_b <= 8'd0;
            kernel_class <= 8'd0;
            dsp_mode <= 8'd0;
            kernel_instr_idx <= {IDX_WIDTH{1'b0}};
            kernel_c_count <= 4'd0;
            kernel_c_flat <= {MAX_CONSTS*16{1'b0}};
            kernel_d_count <= 4'd0;
            kernel_d_flat <= {MAX_ARITY*16{1'b0}};
            input_req_valid <= 1'b0;
            input_kind <= 8'd0;
            input_id <= 16'd0;
            done <= 1'b0;
            error <= 1'b0;
            active_index <= 16'd0;
            active_a <= 8'd0;
            active_b <= 8'd0;
            pending_desc <= {DESC_WIDTH{1'b0}};
        end else begin
            result_wr_valid <= 1'b0;
            tick_commit_valid <= 1'b0;
            kernel_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    instr_ready <= 1'b0;
                    input_req_valid <= 1'b0;
                    done <= 1'b0;
                    error <= 1'b0;
                    if (start) begin
                        state <= S_FETCH;
                    end
                end

                S_FETCH: begin
                    instr_ready <= 1'b1;
                    if (instr_valid && instr_ready) begin
                        instr_ready <= 1'b0;
                        active_index <= instr_index;
                        active_a <= instr_a;
                        active_b <= instr_b;

                        if (instr_a == `NAC_OP_INPUT) begin
                            input_kind <= instr_b;
                            input_id <= (c_count >= 2) ? c_flat[16 +: 16] : 16'd0;
                            state <= S_INPUT_REQ;
                        end else if (instr_a == `NAC_OP_OUTPUT && instr_b == 8'd0) begin
                            pending_desc <= {DESC_WIDTH{1'b0}};
                            state <= S_COMMIT;
                        end else if (instr_a >= 8'd10) begin
                            if (!dispatch_supported) begin
                                state <= S_ERROR;
                            end else begin
                                kernel_op_a <= instr_a;
                                kernel_op_b <= instr_b;
                                kernel_class <= dispatch_kernel_class;
                                dsp_mode <= dispatch_dsp_mode;
                                kernel_instr_idx <= instr_index[IDX_WIDTH-1:0];
                                kernel_c_count <= c_count;
                                kernel_c_flat <= c_flat;
                                kernel_d_count <= d_count;
                                kernel_d_flat <= d_flat;
                                kernel_start <= 1'b1;
                                state <= S_KERNEL;
                            end
                        end else begin
                            state <= S_ERROR;
                        end
                    end
                end

                S_INPUT_REQ: begin
                    input_req_valid <= 1'b1;
                    if (input_req_valid && input_req_ready) begin
                        input_req_valid <= 1'b0;
                        state <= S_INPUT_WAIT;
                    end
                end

                S_INPUT_WAIT: begin
                    if (input_desc_valid) begin
                        pending_desc <= input_desc;
                        result_wr_valid <= 1'b1;
                        result_wr_idx <= active_index[IDX_WIDTH-1:0];
                        result_wr_desc <= input_desc;
                        state <= S_COMMIT;
                    end
                end

                S_KERNEL: begin
                    if (kernel_done) begin
                        pending_desc <= kernel_result_desc;
                        result_wr_valid <= 1'b1;
                        result_wr_idx <= active_index[IDX_WIDTH-1:0];
                        result_wr_desc <= kernel_result_desc;
                        state <= S_COMMIT;
                    end
                end

                S_COMMIT: begin
                    tick_commit_valid <= 1'b1;
                    tick_commit_id <= active_index[IDX_WIDTH-1:0];
                    if (active_a == `NAC_OP_OUTPUT && active_b == 8'd0) begin
                        state <= S_DONE;
                    end else begin
                        state <= S_FETCH;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_DONE;
                end

                S_ERROR: begin
                    error <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_ERROR;
                end
            endcase
        end
    end
endmodule
