`timescale 1ns/1ps
`include "../rtl/nac_defs.vh"

module nac_mep_vm_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg start;
    reg [7:0] plan [0:63];
    reg [7:0] rp;
    reg [31:0] plan_size;

    wire byte_valid = (rp < plan_size);
    wire byte_ready;
    wire [7:0] byte_in = plan[rp];

    wire pkt_valid;
    wire pkt_ready;
    wire [7:0] pkt_byte;
    wire [7:0] pkt_opcode;
    wire [15:0] pkt_instr_index;
    wire [15:0] pkt_byte_index;
    wire pkt_instr_start;
    wire pkt_instr_end;
    wire pkt_done;
    wire pkt_error;

    nac_mep_packetizer packetizer (
        .clk(clk),
        .rst(rst),
        .start(start),
        .plan_size(plan_size),
        .byte_valid(byte_valid),
        .byte_ready(byte_ready),
        .byte_in(byte_in),
        .out_valid(pkt_valid),
        .out_ready(pkt_ready),
        .out_byte(pkt_byte),
        .out_opcode(pkt_opcode),
        .out_instr_index(pkt_instr_index),
        .out_byte_index(pkt_byte_index),
        .out_instr_start(pkt_instr_start),
        .out_instr_end(pkt_instr_end),
        .done(pkt_done),
        .error(pkt_error)
    );

    wire model_run_start;
    wire [7:0] model_id;
    wire [7:0] model_count_in;
    wire [8*16-1:0] model_in_keys;
    wire [7:0] model_count_out;
    wire [8*16-1:0] model_out_keys;
    reg model_run_done;

    wire return_valid;
    wire [7:0] return_count;
    wire [8*16-1:0] return_keys;
    wire [63:0] ctx0_value;
    wire ctx0_valid;
    wire vm_busy;
    wire vm_done;
    wire vm_halted;
    wire vm_error;
    wire [7:0] error_opcode;

    nac_mep_vm vm (
        .clk(clk),
        .rst(rst),
        .start(start),
        .in_valid(pkt_valid),
        .in_ready(pkt_ready),
        .in_byte(pkt_byte),
        .in_opcode(pkt_opcode),
        .in_instr_index(pkt_instr_index),
        .in_byte_index(pkt_byte_index),
        .in_instr_start(pkt_instr_start),
        .in_instr_end(pkt_instr_end),
        .io_req_valid(),
        .io_req_ready(1'b1),
        .io_opcode(),
        .io_out_key(),
        .io_arg0(),
        .io_const_id(),
        .io_done(1'b0),
        .io_error(1'b0),
        .io_result_key(8'd0),
        .io_result_value(64'd0),
        .io_result_valid(1'b0),
        .tisa_req_valid(),
        .tisa_req_ready(1'b1),
        .tisa_req_decode(),
        .tisa_proc_key(),
        .tisa_in_key(),
        .tisa_out_key(),
        .tisa_done(1'b0),
        .tisa_error(1'b0),
        .tisa_result_value(64'd0),
        .tisa_result_valid(1'b0),
        .model_run_start(model_run_start),
        .model_run_ready(1'b1),
        .model_id(model_id),
        .model_count_in(model_count_in),
        .model_in_keys(model_in_keys),
        .model_count_out(model_count_out),
        .model_out_keys(model_out_keys),
        .model_run_done(model_run_done),
        .model_run_error(1'b0),
        .train_start(),
        .train_ready(1'b1),
        .train_model_id(),
        .train_loss_type(),
        .train_count_in(),
        .train_in_keys(),
        .train_target_count(),
        .train_target_keys(),
        .train_loss_key(),
        .train_lr_key(),
        .train_logits_key(),
        .train_head_weight_name_id(),
        .train_head_bias_name_id(),
        .train_done(1'b0),
        .train_error(1'b0),
        .zero_grad_start(),
        .zero_grad_ready(1'b1),
        .zero_grad_model_id(),
        .zero_grad_done(1'b0),
        .zero_grad_error(1'b0),
        .save_weights_start(),
        .save_weights_ready(1'b1),
        .save_model_id(),
        .save_path_key(),
        .save_type(),
        .save_weights_done(1'b0),
        .save_weights_error(1'b0),
        .pc_redirect_valid(),
        .pc_redirect_ready(1'b1),
        .pc_redirect_offset(),
        .return_valid(return_valid),
        .return_count(return_count),
        .return_keys(return_keys),
        .ctx0_value(ctx0_value),
        .ctx0_valid(ctx0_valid),
        .busy(vm_busy),
        .done(vm_done),
        .halted(vm_halted),
        .error(vm_error),
        .error_opcode(error_opcode)
    );

    always @(posedge clk) begin
        if (rst || start)
            rp <= 8'd0;
        else if (byte_valid && byte_ready)
            rp <= rp + 8'd1;
    end

    integer model_starts;
    integer cycles;

    always @(posedge clk) begin
        if (rst) begin
            model_run_done <= 1'b0;
            model_starts <= 0;
        end else begin
            model_run_done <= 1'b0;
            if (model_run_start) begin
                model_starts <= model_starts + 1;
                model_run_done <= 1'b1;
            end
        end
    end

    initial begin
        rst = 1'b1;
        start = 1'b0;
        plan_size = 32'd29;

        // SRC_CONSTANT key1 = 6
        plan[0] = 8'h04; plan[1] = 8'd1; plan[2] = 8'd6; plan[3] = 8'd0;
        // SRC_CONSTANT key2 = 3
        plan[4] = 8'h04; plan[5] = 8'd2; plan[6] = 8'd3; plan[7] = 8'd0;
        // MATH_BINARY mul key3 = key1 * key2
        plan[8] = 8'h61; plan[9] = 8'd2; plan[10] = 8'd3; plan[11] = 8'd1; plan[12] = 8'd2;
        // MATH_BINARY div key4 = key3 / key2
        plan[13] = 8'h61; plan[14] = 8'd3; plan[15] = 8'd4; plan[16] = 8'd3; plan[17] = 8'd2;
        // LOGIC_COMPARE key5 = key4 > key2
        plan[18] = 8'h68; plan[19] = 8'd2; plan[20] = 8'd5; plan[21] = 8'd4; plan[22] = 8'd2;
        // MODEL_RUN_STATIC model0, inputs=[key4], outputs=[key6]
        plan[23] = 8'h80; plan[24] = 8'd0; plan[25] = 8'd1; plan[26] = 8'd4; plan[27] = 8'd1; plan[28] = 8'd6;
        // Return is omitted because MODEL_RUN_STATIC is enough to prove master dispatch here.

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        cycles = 0;
        while (!pkt_done && !pkt_error && !vm_error && cycles < 200) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        repeat (10) @(posedge clk);

        if (pkt_error) $fatal(1, "MEP packetizer error");
        if (vm_error) $fatal(1, "MEP VM error opcode=%02x", error_opcode);
        if (model_starts != 1) $fatal(1, "MODEL_RUN_STATIC was not dispatched once");
        if (model_id != 8'd0 || model_count_in != 8'd1 || model_in_keys[0 +: 8] != 8'd4 ||
            model_count_out != 8'd1 || model_out_keys[0 +: 8] != 8'd6)
            $fatal(1, "bad MODEL_RUN_STATIC payload");

        $display("nac_mep_vm_tb PASS");
        $finish;
    end
endmodule
