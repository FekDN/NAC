`include "nac_defs.vh"

module nac_instruction_cache #(
    parameter CONTEXTS = 2,
    parameter CONTEXT_BITS = 1,
    parameter IDX_WIDTH = 10,
    parameter MAX_ARITY = 8,
    parameter MAX_CONSTS = 8
) (
    input  wire clk,
    input  wire rst,
    input  wire clear_context,
    input  wire [CONTEXT_BITS-1:0] clear_context_id,

    input  wire load_start,
    input  wire [CONTEXT_BITS-1:0] load_context,
    input  wire dec_valid,
    output wire dec_ready,
    input  wire [15:0] dec_index,
    input  wire [7:0] dec_a,
    input  wire [7:0] dec_b,
    input  wire [3:0] dec_c_count,
    input  wire [MAX_CONSTS*16-1:0] dec_c_flat,
    input  wire [3:0] dec_d_count,
    input  wire [MAX_ARITY*16-1:0] dec_d_flat,
    output reg  load_done,
    output reg  load_error,

    input  wire run_start,
    input  wire [CONTEXT_BITS-1:0] run_context,
    input  wire redirect_valid,
    input  wire [IDX_WIDTH-1:0] redirect_index,
    output wire instr_valid,
    input  wire instr_ready,
    output wire [15:0] instr_index,
    output wire [7:0] instr_a,
    output wire [7:0] instr_b,
    output wire [3:0] instr_c_count,
    output wire [MAX_CONSTS*16-1:0] instr_c_flat,
    output wire [3:0] instr_d_count,
    output wire [MAX_ARITY*16-1:0] instr_d_flat,
    output reg  run_active,
    output reg  run_done,
    output reg  run_error,
    output wire model_configured
);
    localparam INSTR_DEPTH = (1 << IDX_WIDTH);
    localparam TOTAL_ENTRIES = CONTEXTS * INSTR_DEPTH;

    reg configured_ram [0:CONTEXTS-1];
    reg [IDX_WIDTH:0] instr_count_ram [0:CONTEXTS-1];
    reg [7:0] a_ram [0:TOTAL_ENTRIES-1];
    reg [7:0] b_ram [0:TOTAL_ENTRIES-1];
    reg [3:0] c_count_ram [0:TOTAL_ENTRIES-1];
    reg [MAX_CONSTS*16-1:0] c_flat_ram [0:TOTAL_ENTRIES-1];
    reg [3:0] d_count_ram [0:TOTAL_ENTRIES-1];
    reg [MAX_ARITY*16-1:0] d_flat_ram [0:TOTAL_ENTRIES-1];

    reg load_active;
    reg [CONTEXT_BITS-1:0] load_context_latched;
    reg [IDX_WIDTH-1:0] load_pc;
    reg [CONTEXT_BITS-1:0] run_context_latched;
    reg [IDX_WIDTH-1:0] run_pc;

    wire [IDX_WIDTH:0] load_count_next = {1'b0, load_pc} + {{IDX_WIDTH{1'b0}}, 1'b1};
    wire [IDX_WIDTH:0] active_run_count = instr_count_ram[run_context_latched];
    wire load_accept = load_active && dec_valid && dec_ready;
    wire run_accept = run_active && instr_valid && instr_ready;
    wire [31:0] load_index = (load_context_latched * INSTR_DEPTH) + load_pc;
    wire [31:0] run_index = (run_context_latched * INSTR_DEPTH) + run_pc;

    assign dec_ready = load_active && !load_error;
    assign instr_valid = run_active && !run_error;
    assign instr_index = {{(16-IDX_WIDTH){1'b0}}, run_pc};
    assign instr_a = a_ram[run_index];
    assign instr_b = b_ram[run_index];
    assign instr_c_count = c_count_ram[run_index];
    assign instr_c_flat = c_flat_ram[run_index];
    assign instr_d_count = d_count_ram[run_index];
    assign instr_d_flat = d_flat_ram[run_index];
    assign model_configured = configured_ram[run_context];

    integer i;
    initial begin
        for (i = 0; i < CONTEXTS; i = i + 1) begin
            configured_ram[i] = 1'b0;
            instr_count_ram[i] = {(IDX_WIDTH+1){1'b0}};
        end
        for (i = 0; i < TOTAL_ENTRIES; i = i + 1) begin
            a_ram[i] = 8'd0;
            b_ram[i] = 8'd0;
            c_count_ram[i] = 4'd0;
            c_flat_ram[i] = {MAX_CONSTS*16{1'b0}};
            d_count_ram[i] = 4'd0;
            d_flat_ram[i] = {MAX_ARITY*16{1'b0}};
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            load_active <= 1'b0;
            load_context_latched <= {CONTEXT_BITS{1'b0}};
            load_pc <= {IDX_WIDTH{1'b0}};
            load_done <= 1'b0;
            load_error <= 1'b0;
            run_active <= 1'b0;
            run_context_latched <= {CONTEXT_BITS{1'b0}};
            run_pc <= {IDX_WIDTH{1'b0}};
            run_done <= 1'b0;
            run_error <= 1'b0;
            for (i = 0; i < CONTEXTS; i = i + 1) begin
                configured_ram[i] <= 1'b0;
                instr_count_ram[i] <= {(IDX_WIDTH+1){1'b0}};
            end
        end else begin
            load_done <= 1'b0;
            run_done <= 1'b0;

            if (clear_context) begin
                configured_ram[clear_context_id] <= 1'b0;
                instr_count_ram[clear_context_id] <= {(IDX_WIDTH+1){1'b0}};
                if (load_active && load_context_latched == clear_context_id) begin
                    load_active <= 1'b0;
                end
                if (run_active && run_context_latched == clear_context_id) begin
                    run_active <= 1'b0;
                end
            end

            if (load_start) begin
                load_active <= 1'b1;
                load_context_latched <= load_context;
                load_pc <= {IDX_WIDTH{1'b0}};
                load_error <= 1'b0;
                configured_ram[load_context] <= 1'b0;
                instr_count_ram[load_context] <= {(IDX_WIDTH+1){1'b0}};
            end else if (load_accept) begin
                if (load_count_next > INSTR_DEPTH) begin
                    load_error <= 1'b1;
                    load_active <= 1'b0;
                end else begin
                    a_ram[load_index] <= dec_a;
                    b_ram[load_index] <= dec_b;
                    c_count_ram[load_index] <= dec_c_count;
                    c_flat_ram[load_index] <= dec_c_flat;
                    d_count_ram[load_index] <= dec_d_count;
                    d_flat_ram[load_index] <= dec_d_flat;

                    if (dec_a == `NAC_OP_OUTPUT && dec_b == 8'd0) begin
                        instr_count_ram[load_context_latched] <= load_count_next;
                        configured_ram[load_context_latched] <= 1'b1;
                        load_active <= 1'b0;
                        load_done <= 1'b1;
                    end else begin
                        load_pc <= load_pc + {{(IDX_WIDTH-1){1'b0}}, 1'b1};
                    end
                end
            end

            if (run_start) begin
                if (!configured_ram[run_context]) begin
                    run_error <= 1'b1;
                    run_active <= 1'b0;
                end else begin
                    run_context_latched <= run_context;
                    run_pc <= {IDX_WIDTH{1'b0}};
                    run_active <= 1'b1;
                    run_error <= 1'b0;
                end
            end else if (redirect_valid && run_active) begin
                if ({1'b0, redirect_index} >= active_run_count) begin
                    run_error <= 1'b1;
                    run_active <= 1'b0;
                end else begin
                    run_pc <= redirect_index;
                end
            end else if (run_accept) begin
                if (({1'b0, run_pc} + {{IDX_WIDTH{1'b0}}, 1'b1}) >= active_run_count) begin
                    run_active <= 1'b0;
                    run_done <= 1'b1;
                end else begin
                    run_pc <= run_pc + {{(IDX_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end
endmodule
