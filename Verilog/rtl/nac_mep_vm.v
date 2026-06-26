`include "nac_defs.vh"

// Hardware MEP virtual machine.
//
// MEP is the SoC master. It consumes byte-accurate instruction packets from
// nac_mep_packetizer, keeps a 256-slot context table, executes scalar context
// math/logic in hardware, and dispatches long-latency work to peripherals:
// I/O, TISA, NAC model lifecycle, training, and storage.
//
// Flow-control opcodes expose a redirect request. The upstream ORCH storage
// reader must honor it by repositioning the packetizer input stream; this keeps
// branch/loop logic in hardware without extending MEP bytecode.
module nac_mep_vm #(
    parameter MAX_INSTR_BYTES = 64,
    parameter CONTEXT_VALUE_WIDTH = 64
) (
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire in_valid,
    output wire in_ready,
    input  wire [7:0] in_byte,
    input  wire [7:0] in_opcode,
    input  wire [15:0] in_instr_index,
    input  wire [15:0] in_byte_index,
    input  wire in_instr_start,
    input  wire in_instr_end,

    output reg  io_req_valid,
    input  wire io_req_ready,
    output reg  [7:0] io_opcode,
    output reg  [7:0] io_out_key,
    output reg  [7:0] io_arg0,
    output reg  [15:0] io_const_id,
    input  wire io_done,
    input  wire io_error,
    input  wire [7:0] io_result_key,
    input  wire [CONTEXT_VALUE_WIDTH-1:0] io_result_value,
    input  wire io_result_valid,

    output reg  tisa_req_valid,
    input  wire tisa_req_ready,
    output reg  tisa_req_decode,
    output reg  [7:0] tisa_proc_key,
    output reg  [7:0] tisa_in_key,
    output reg  [7:0] tisa_out_key,
    input  wire tisa_done,
    input  wire tisa_error,
    input  wire [CONTEXT_VALUE_WIDTH-1:0] tisa_result_value,
    input  wire tisa_result_valid,

    output reg  model_run_start,
    input  wire model_run_ready,
    output reg  [7:0] model_id,
    output reg  [7:0] model_count_in,
    output reg  [8*16-1:0] model_in_keys,
    output reg  [7:0] model_count_out,
    output reg  [8*16-1:0] model_out_keys,
    input  wire model_run_done,
    input  wire model_run_error,

    output reg  train_start,
    input  wire train_ready,
    output reg  [7:0] train_model_id,
    output reg  [7:0] train_loss_type,
    output reg  [7:0] train_count_in,
    output reg  [8*16-1:0] train_in_keys,
    output reg  [7:0] train_target_count,
    output reg  [8*16-1:0] train_target_keys,
    output reg  [7:0] train_loss_key,
    output reg  [7:0] train_lr_key,
    output reg  [7:0] train_logits_key,
    output reg  [15:0] train_head_weight_name_id,
    output reg  [15:0] train_head_bias_name_id,
    input  wire train_done,
    input  wire train_error,

    output reg  zero_grad_start,
    input  wire zero_grad_ready,
    output reg  [7:0] zero_grad_model_id,
    input  wire zero_grad_done,
    input  wire zero_grad_error,

    output reg  save_weights_start,
    input  wire save_weights_ready,
    output reg  [7:0] save_model_id,
    output reg  [7:0] save_path_key,
    output reg  [7:0] save_type,
    input  wire save_weights_done,
    input  wire save_weights_error,

    output reg  pc_redirect_valid,
    input  wire pc_redirect_ready,
    output reg  signed [15:0] pc_redirect_offset,

    output reg  return_valid,
    output reg  [7:0] return_count,
    output reg  [8*16-1:0] return_keys,

    output wire [CONTEXT_VALUE_WIDTH-1:0] ctx0_value,
    output wire ctx0_valid,
    output reg  busy,
    output reg  done,
    output reg  halted,
    output reg  error,
    output reg  [7:0] error_opcode
);
    localparam CTX_TYPE_EMPTY = 2'd0;
    localparam CTX_TYPE_INT   = 2'd1;
    localparam CTX_TYPE_BOOL  = 2'd2;
    localparam CTX_TYPE_REF   = 2'd3;

    localparam S_IDLE          = 5'd0;
    localparam S_COLLECT       = 5'd1;
    localparam S_EXEC          = 5'd2;
    localparam S_IO_REQ        = 5'd3;
    localparam S_IO_WAIT       = 5'd4;
    localparam S_TISA_REQ      = 5'd5;
    localparam S_TISA_WAIT     = 5'd6;
    localparam S_MODEL_REQ     = 5'd7;
    localparam S_MODEL_WAIT    = 5'd8;
    localparam S_TRAIN_REQ     = 5'd9;
    localparam S_TRAIN_WAIT    = 5'd10;
    localparam S_ZERO_REQ      = 5'd11;
    localparam S_ZERO_WAIT     = 5'd12;
    localparam S_SAVE_REQ      = 5'd13;
    localparam S_SAVE_WAIT     = 5'd14;
    localparam S_REDIRECT      = 5'd15;
    localparam S_ALU_START     = 5'd16;
    localparam S_ALU_WAIT      = 5'd17;

    reg [4:0] state;
    reg [7:0] instr [0:MAX_INSTR_BYTES-1];
    reg [7:0] instr_len;
    reg [7:0] active_opcode;
    reg [CONTEXT_VALUE_WIDTH-1:0] ctx_value [0:255];
    reg [1:0] ctx_type [0:255];
    reg ctx_valid_ram [0:255];
    reg [7:0] loop_counter_key;
    reg loop_active;
    reg [7:0] pending_out_key;

    assign in_ready = busy && (state == S_COLLECT) && (instr_len < MAX_INSTR_BYTES);
    assign ctx0_value = ctx_value[0];
    assign ctx0_valid = ctx_valid_ram[0];

    integer i;
    integer k;
    reg scalar_start;
    reg scalar_is_compare;
    reg [7:0] scalar_op_type;
    reg signed [CONTEXT_VALUE_WIDTH-1:0] scalar_a;
    reg signed [CONTEXT_VALUE_WIDTH-1:0] scalar_b;
    reg [7:0] scalar_dest_key;
    wire scalar_busy;
    wire scalar_done;
    wire scalar_error;
    wire [CONTEXT_VALUE_WIDTH-1:0] scalar_result;
    wire scalar_result_is_bool;

    nac_scalar_alu #(
        .WIDTH(CONTEXT_VALUE_WIDTH)
    ) scalar_alu (
        .clk(clk),
        .rst(rst),
        .start(scalar_start),
        .is_compare(scalar_is_compare),
        .op_type(scalar_op_type),
        .a(scalar_a),
        .b(scalar_b),
        .busy(scalar_busy),
        .done(scalar_done),
        .error(scalar_error),
        .result(scalar_result),
        .result_is_bool(scalar_result_is_bool)
    );

    function [15:0] le16_at;
        input [7:0] idx;
        begin
            le16_at = {instr[idx + 1], instr[idx]};
        end
    endfunction

    task set_error;
        input [7:0] op;
        begin
            error <= 1'b1;
            error_opcode <= op;
            busy <= 1'b0;
            state <= S_IDLE;
        end
    endtask

    task ctx_write;
        input [7:0] key;
        input [CONTEXT_VALUE_WIDTH-1:0] value;
        input [1:0] typ;
        begin
            ctx_value[key] <= value;
            ctx_type[key] <= typ;
            ctx_valid_ram[key] <= 1'b1;
        end
    endtask

    task finish_instr;
        begin
            instr_len <= 8'd0;
            active_opcode <= 8'd0;
            state <= S_COLLECT;
        end
    endtask

    task load_key_vector;
        input [7:0] base;
        input [7:0] count;
        output reg [8*16-1:0] keys_flat;
        integer vi;
        begin
            keys_flat = {8*16{1'b0}};
            for (vi = 0; vi < 16; vi = vi + 1) begin
                if (vi < count)
                    keys_flat[vi*8 +: 8] = instr[base + vi[7:0]];
            end
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            instr_len <= 8'd0;
            active_opcode <= 8'd0;
            loop_counter_key <= 8'd0;
            loop_active <= 1'b0;
            pending_out_key <= 8'd0;
            io_req_valid <= 1'b0;
            io_opcode <= 8'd0;
            io_out_key <= 8'd0;
            io_arg0 <= 8'd0;
            io_const_id <= 16'd0;
            tisa_req_valid <= 1'b0;
            tisa_req_decode <= 1'b0;
            tisa_proc_key <= 8'd0;
            tisa_in_key <= 8'd0;
            tisa_out_key <= 8'd0;
            model_run_start <= 1'b0;
            model_id <= 8'd0;
            model_count_in <= 8'd0;
            model_in_keys <= {8*16{1'b0}};
            model_count_out <= 8'd0;
            model_out_keys <= {8*16{1'b0}};
            train_start <= 1'b0;
            train_model_id <= 8'd0;
            train_loss_type <= 8'd0;
            train_count_in <= 8'd0;
            train_in_keys <= {8*16{1'b0}};
            train_target_count <= 8'd0;
            train_target_keys <= {8*16{1'b0}};
            train_loss_key <= 8'd0;
            train_lr_key <= 8'd0;
            train_logits_key <= 8'd0;
            train_head_weight_name_id <= 16'd0;
            train_head_bias_name_id <= 16'd0;
            zero_grad_start <= 1'b0;
            zero_grad_model_id <= 8'd0;
            save_weights_start <= 1'b0;
            save_model_id <= 8'd0;
            save_path_key <= 8'd0;
            save_type <= 8'd0;
            pc_redirect_valid <= 1'b0;
            pc_redirect_offset <= 16'sd0;
            return_valid <= 1'b0;
            return_count <= 8'd0;
            return_keys <= {8*16{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
            halted <= 1'b0;
            error <= 1'b0;
            error_opcode <= 8'd0;
            scalar_start <= 1'b0;
            scalar_is_compare <= 1'b0;
            scalar_op_type <= 8'd0;
            scalar_a <= {CONTEXT_VALUE_WIDTH{1'b0}};
            scalar_b <= {CONTEXT_VALUE_WIDTH{1'b0}};
            scalar_dest_key <= 8'd0;
            for (i = 0; i < 256; i = i + 1) begin
                ctx_value[i] <= {CONTEXT_VALUE_WIDTH{1'b0}};
                ctx_type[i] <= CTX_TYPE_EMPTY;
                ctx_valid_ram[i] <= 1'b0;
            end
        end else begin
            io_req_valid <= 1'b0;
            tisa_req_valid <= 1'b0;
            model_run_start <= 1'b0;
            train_start <= 1'b0;
            zero_grad_start <= 1'b0;
            save_weights_start <= 1'b0;
            pc_redirect_valid <= 1'b0;
            scalar_start <= 1'b0;

            if (start) begin
                busy <= 1'b1;
                done <= 1'b0;
                halted <= 1'b0;
                error <= 1'b0;
                error_opcode <= 8'd0;
                return_valid <= 1'b0;
                instr_len <= 8'd0;
                active_opcode <= 8'd0;
                loop_active <= 1'b0;
                state <= S_COLLECT;
            end else begin
                case (state)
                    S_IDLE: begin
                    end

                    S_COLLECT: begin
                        if (in_valid && in_ready) begin
                            instr[in_byte_index[7:0]] <= in_byte;
                            if (in_instr_start) begin
                                active_opcode <= in_opcode;
                                instr_len <= 8'd1;
                            end else begin
                                instr_len <= in_byte_index[7:0] + 8'd1;
                            end
                            if (in_instr_end)
                                state <= S_EXEC;
                        end
                    end

                    S_EXEC: begin
                        case (active_opcode)
                            8'h01: begin // SRC_CMD_ARG: out_key,arg_idx
                                io_opcode <= active_opcode;
                                io_out_key <= instr[1];
                                io_arg0 <= instr[2];
                                io_const_id <= 16'd0;
                                pending_out_key <= instr[1];
                                state <= S_IO_REQ;
                            end
                            8'h02: begin // SRC_USER_PROMPT
                                io_opcode <= active_opcode;
                                io_out_key <= instr[1];
                                io_arg0 <= instr[2];
                                io_const_id <= le16_at(3);
                                pending_out_key <= instr[1];
                                state <= S_IO_REQ;
                            end
                            8'h03: begin // SRC_FILE_CONTENT
                                io_opcode <= active_opcode;
                                io_out_key <= instr[1];
                                io_arg0 <= instr[3];
                                io_const_id <= {8'd0, instr[2]};
                                pending_out_key <= instr[1];
                                state <= S_IO_REQ;
                            end
                            8'h04: begin // SRC_CONSTANT: keep CNST id as a reference.
                                ctx_write(instr[1], {{(CONTEXT_VALUE_WIDTH-16){1'b0}}, le16_at(2)}, CTX_TYPE_REF);
                                finish_instr();
                            end
                            8'h1F: begin // RES_UNLOAD: invalidate context resource.
                                ctx_valid_ram[instr[2]] <= 1'b0;
                                ctx_type[instr[2]] <= CTX_TYPE_EMPTY;
                                finish_instr();
                            end
                            8'h20,
                            8'h21: begin // PREPROC_ENCODE / PREPROC_DECODE
                                tisa_req_decode <= (active_opcode == 8'h21);
                                tisa_proc_key <= instr[1];
                                tisa_in_key <= instr[2];
                                tisa_out_key <= instr[3];
                                pending_out_key <= instr[3];
                                state <= S_TISA_REQ;
                            end
                            8'h59: begin // SYS_COPY
                                if (!ctx_valid_ram[instr[2]]) begin
                                    set_error(active_opcode);
                                end else begin
                                    ctx_write(instr[1], ctx_value[instr[2]], ctx_type[instr[2]]);
                                    finish_instr();
                                end
                            end
                            8'h61: begin // scalar MATH_BINARY
                                if (!ctx_valid_ram[instr[3]] || !ctx_valid_ram[instr[4]]) begin
                                    set_error(active_opcode);
                                end else begin
                                    scalar_is_compare <= 1'b0;
                                    scalar_op_type <= instr[1];
                                    scalar_dest_key <= instr[2];
                                    scalar_a <= ctx_value[instr[3]];
                                    scalar_b <= ctx_value[instr[4]];
                                    state <= S_ALU_START;
                                end
                            end
                            8'h68: begin // scalar LOGIC_COMPARE: 0 eq,1 neq,2 gt,3 lt
                                if (!ctx_valid_ram[instr[3]] || !ctx_valid_ram[instr[4]]) begin
                                    set_error(active_opcode);
                                end else begin
                                    scalar_is_compare <= 1'b1;
                                    scalar_op_type <= instr[1];
                                    scalar_dest_key <= instr[2];
                                    scalar_a <= ctx_value[instr[3]];
                                    scalar_b <= ctx_value[instr[4]];
                                    state <= S_ALU_START;
                                end
                            end
                            `MEP_MODEL_RUN_STATIC: begin
                                model_id <= instr[1];
                                model_count_in <= instr[2];
                                load_key_vector(8'd3, instr[2], model_in_keys);
                                model_count_out <= instr[3 + instr[2]];
                                load_key_vector(8'd4 + instr[2], instr[3 + instr[2]], model_out_keys);
                                state <= S_MODEL_REQ;
                            end
                            `MEP_MODEL_TRAIN_STEP: begin
                                train_model_id <= instr[1];
                                train_loss_type <= instr[2];
                                train_count_in <= instr[3];
                                load_key_vector(8'd4, instr[3], train_in_keys);
                                k = 4 + instr[3];
                                train_target_count <= instr[k];
                                load_key_vector(k[7:0] + 8'd1, instr[k], train_target_keys);
                                train_loss_key <= instr[k + 1 + instr[k]];
                                train_lr_key <= instr[k + 2 + instr[k]];
                                train_logits_key <= instr[k + 3 + instr[k]];
                                train_head_weight_name_id <= {instr[k + 5 + instr[k]], instr[k + 4 + instr[k]]};
                                train_head_bias_name_id <= {instr[k + 7 + instr[k]], instr[k + 6 + instr[k]]};
                                state <= S_TRAIN_REQ;
                            end
                            8'h83: begin
                                zero_grad_model_id <= instr[1];
                                state <= S_ZERO_REQ;
                            end
                            8'h85: begin
                                save_model_id <= instr[1];
                                save_path_key <= instr[2];
                                save_type <= instr[3];
                                state <= S_SAVE_REQ;
                            end
                            `MEP_FLOW_LOOP_START: begin
                                loop_counter_key <= instr[1];
                                loop_active <= 1'b1;
                                finish_instr();
                            end
                            `MEP_FLOW_LOOP_END: begin
                                if (!loop_active || !ctx_valid_ram[loop_counter_key]) begin
                                    set_error(active_opcode);
                                end else if (ctx_value[loop_counter_key] > 1) begin
                                    ctx_value[loop_counter_key] <= ctx_value[loop_counter_key] - 1'b1;
                                    pc_redirect_offset <= -$signed({1'b0, le16_at(1)});
                                    state <= S_REDIRECT;
                                end else begin
                                    ctx_value[loop_counter_key] <= {CONTEXT_VALUE_WIDTH{1'b0}};
                                    loop_active <= 1'b0;
                                    finish_instr();
                                end
                            end
                            `MEP_FLOW_BRANCH_IF: begin
                                if (!ctx_valid_ram[instr[1]]) begin
                                    set_error(active_opcode);
                                end else if (ctx_value[instr[1]] != 0) begin
                                    pc_redirect_offset <= $signed(le16_at(2));
                                    state <= S_REDIRECT;
                                end else begin
                                    finish_instr();
                                end
                            end
                            8'hA9: begin // FLOW_BREAK_LOOP_IF cond_key,jump_offset
                                if (!ctx_valid_ram[instr[1]]) begin
                                    set_error(active_opcode);
                                end else if (ctx_value[instr[1]] != 0) begin
                                    loop_active <= 1'b0;
                                    pc_redirect_offset <= $signed(le16_at(2));
                                    state <= S_REDIRECT;
                                end else begin
                                    finish_instr();
                                end
                            end
                            8'hAF,
                            `MEP_EXEC_HALT: begin
                                busy <= 1'b0;
                                halted <= 1'b1;
                                done <= 1'b1;
                                state <= S_IDLE;
                            end
                            `MEP_EXEC_RETURN: begin
                                return_count <= instr[1];
                                return_keys <= {8*16{1'b0}};
                                //for (k = 0; k < 16; k = k + 1) begin
                                for (k = 0; k < instr[1] && k < 16; k = k + 1) begin
                                    //if (k < instr[1])
                                    return_keys[k*8 +: 8] <= instr[2 + k[7:0]];
                                end
                                return_valid <= 1'b1;
                                busy <= 1'b0;
                                done <= 1'b1;
                                state <= S_IDLE;
                            end
                            default: begin
                                set_error(active_opcode);
                            end
                        endcase
                    end

                    S_IO_REQ: begin
                        io_req_valid <= 1'b1;
                        if (io_req_ready)
                            state <= S_IO_WAIT;
                    end
                    S_IO_WAIT: begin
                        if (io_error) begin
                            set_error(active_opcode);
                        end else if (io_done) begin
                            if (io_result_valid)
                                ctx_write(io_result_key, io_result_value, CTX_TYPE_REF);
                            else
                                ctx_write(pending_out_key, {CONTEXT_VALUE_WIDTH{1'b0}}, CTX_TYPE_REF);
                            finish_instr();
                        end
                    end

                    S_TISA_REQ: begin
                        tisa_req_valid <= 1'b1;
                        if (tisa_req_ready)
                            state <= S_TISA_WAIT;
                    end
                    S_TISA_WAIT: begin
                        if (tisa_error) begin
                            set_error(active_opcode);
                        end else if (tisa_done) begin
                            ctx_write(tisa_out_key, tisa_result_valid ? tisa_result_value : {CONTEXT_VALUE_WIDTH{1'b0}}, CTX_TYPE_REF);
                            finish_instr();
                        end
                    end

                    S_MODEL_REQ: begin
                        model_run_start <= 1'b1;
                        if (model_run_ready)
                            state <= S_MODEL_WAIT;
                    end
                    S_MODEL_WAIT: begin
                        if (model_run_error) begin
                            set_error(active_opcode);
                        end else if (model_run_done) begin
                            //for (k = 0; k < 16; k = k + 1) begin
                            //    if (k < model_count_out)
                            for (k = 0; k < model_count_out && k < 16; k = k + 1) begin
                                ctx_write(model_out_keys[k*8 +: 8], {{(CONTEXT_VALUE_WIDTH-16){1'b0}}, k[15:0]}, CTX_TYPE_REF);
                            end
                            finish_instr();
                        end
                    end

                    S_TRAIN_REQ: begin
                        train_start <= 1'b1;
                        if (train_ready)
                            state <= S_TRAIN_WAIT;
                    end
                    S_TRAIN_WAIT: begin
                        if (train_error) begin
                            set_error(active_opcode);
                        end else if (train_done) begin
                            ctx_write(train_loss_key, {CONTEXT_VALUE_WIDTH{1'b0}}, CTX_TYPE_REF);
                            finish_instr();
                        end
                    end

                    S_ZERO_REQ: begin
                        zero_grad_start <= 1'b1;
                        if (zero_grad_ready)
                            state <= S_ZERO_WAIT;
                    end
                    S_ZERO_WAIT: begin
                        if (zero_grad_error)
                            set_error(active_opcode);
                        else if (zero_grad_done)
                            finish_instr();
                    end

                    S_SAVE_REQ: begin
                        save_weights_start <= 1'b1;
                        if (save_weights_ready)
                            state <= S_SAVE_WAIT;
                    end
                    S_SAVE_WAIT: begin
                        if (save_weights_error)
                            set_error(active_opcode);
                        else if (save_weights_done)
                            finish_instr();
                    end

                    S_REDIRECT: begin
                        pc_redirect_valid <= 1'b1;
                        if (pc_redirect_ready)
                            finish_instr();
                    end

                    S_ALU_START: begin
                        scalar_start <= 1'b1;
                        state <= S_ALU_WAIT;
                    end

                    S_ALU_WAIT: begin
                        if (scalar_error) begin
                            set_error(active_opcode);
                        end else if (scalar_done) begin
                            ctx_write(
                                scalar_dest_key,
                                scalar_result,
                                scalar_result_is_bool ? CTX_TYPE_BOOL : CTX_TYPE_INT
                            );
                            finish_instr();
                        end
                    end

                    default: begin
                        set_error(active_opcode);
                    end
                endcase
            end
        end
    end
endmodule
