`include "nac_defs.vh"

module nac_abcd_decoder #(
    parameter MAX_ARITY = 8,
    parameter MAX_CONSTS = 8,
    parameter ARITY_BITS = 4
) (
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [15:0] num_outputs,

    input  wire ops_byte_valid,
    output wire ops_byte_ready,
    input  wire [7:0] ops_byte,

    output wire [7:0] perm_lookup_id,
    input  wire [ARITY_BITS-1:0] perm_arity,
    input  wire perm_needs_consts,
    input  wire perm_present,

    output reg  instr_valid,
    input  wire instr_ready,
    output reg  [15:0] instr_index,
    output reg  [7:0] instr_a,
    output reg  [7:0] instr_b,
    output reg  [3:0] c_count,
    output reg  [MAX_CONSTS*16-1:0] c_flat,
    output reg  [3:0] d_count,
    output reg  [MAX_ARITY*16-1:0] d_flat,

    output reg  busy,
    output reg  error
);
    localparam S_IDLE       = 5'd0;
    localparam S_A          = 5'd1;
    localparam S_B          = 5'd2;
    localparam S_PERM       = 5'd3;
    localparam S_C_LEN_LO   = 5'd4;
    localparam S_C_LEN_HI   = 5'd5;
    localparam S_C_LO       = 5'd6;
    localparam S_C_HI       = 5'd7;
    localparam S_D_LO       = 5'd8;
    localparam S_D_HI       = 5'd9;
    localparam S_SYS_C_LO   = 5'd10;
    localparam S_SYS_C_HI   = 5'd11;
    localparam S_SYS_D_LO   = 5'd12;
    localparam S_SYS_D_HI   = 5'd13;
    localparam S_EMIT       = 5'd14;
    localparam S_ERROR      = 5'd15;

    reg [4:0] state;
    reg [15:0] tmp16;
    reg [3:0] c_target_count;
    reg [3:0] d_target_count;
    reg [3:0] c_idx;
    reg [3:0] d_idx;
    reg [15:0] sys_c_target_count;
    reg [15:0] sys_d_target_count;

    assign ops_byte_ready =
        busy && !instr_valid && (
            state == S_A || state == S_B ||
            state == S_C_LEN_LO || state == S_C_LEN_HI ||
            state == S_C_LO || state == S_C_HI ||
            state == S_D_LO || state == S_D_HI ||
            state == S_SYS_C_LO || state == S_SYS_C_HI ||
            state == S_SYS_D_LO || state == S_SYS_D_HI
        );

    assign perm_lookup_id = instr_b;

    wire accept = ops_byte_valid && ops_byte_ready;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            error <= 1'b0;
            instr_valid <= 1'b0;
            instr_index <= 16'd0;
            instr_a <= 8'd0;
            instr_b <= 8'd0;
            c_count <= 4'd0;
            d_count <= 4'd0;
            c_flat <= {MAX_CONSTS*16{1'b0}};
            d_flat <= {MAX_ARITY*16{1'b0}};
        end else begin
            if (start) begin
                state <= S_A;
                busy <= 1'b1;
                error <= 1'b0;
                instr_valid <= 1'b0;
                instr_index <= 16'd0;
                c_count <= 4'd0;
                d_count <= 4'd0;
                c_flat <= {MAX_CONSTS*16{1'b0}};
                d_flat <= {MAX_ARITY*16{1'b0}};
            end else begin
                case (state)
                    S_IDLE: begin
                        instr_valid <= 1'b0;
                        busy <= 1'b0;
                    end

                    S_A: begin
                        if (accept) begin
                            instr_a <= ops_byte;
                            instr_b <= 8'd0;
                            c_count <= 4'd0;
                            d_count <= 4'd0;
                            c_flat <= {MAX_CONSTS*16{1'b0}};
                            d_flat <= {MAX_ARITY*16{1'b0}};
                            state <= S_B;
                        end
                    end

                    S_B: begin
                        if (accept) begin
                            instr_b <= ops_byte;
                            if (instr_a < 8'd10) begin
                                if (instr_a == `NAC_OP_INPUT && ops_byte >= 8'd1 && ops_byte <= 8'd5) begin
                                    sys_c_target_count <= 16'd2;
                                    sys_d_target_count <= 16'd0;
                                    c_idx <= 4'd0;
                                    state <= S_SYS_C_LO;
                                end else if (instr_a == `NAC_OP_OUTPUT && ops_byte == 8'd0) begin
                                    sys_c_target_count <= num_outputs + 16'd1;
                                    sys_d_target_count <= num_outputs;
                                    c_idx <= 4'd0;
                                    d_idx <= 4'd0;
                                    if ((num_outputs + 16'd1) > MAX_CONSTS || num_outputs > MAX_ARITY) begin
                                        state <= S_ERROR;
                                    end else begin
                                        state <= S_SYS_C_LO;
                                    end
                                end else if (instr_a == `NAC_OP_OUTPUT && ops_byte == 8'd3) begin
                                    sys_c_target_count <= 16'd2;
                                    sys_d_target_count <= 16'd2;
                                    c_idx <= 4'd0;
                                    d_idx <= 4'd0;
                                    state <= S_SYS_C_LO;
                                end else if (instr_a == `NAC_OP_OUTPUT && ops_byte == 8'd4) begin
                                    sys_c_target_count <= 16'd0;
                                    sys_d_target_count <= 16'd1;
                                    d_idx <= 4'd0;
                                    state <= S_SYS_D_LO;
                                end else begin
                                    state <= S_EMIT;
                                end
                            end else begin
                                state <= S_PERM;
                            end
                        end
                    end

                    S_PERM: begin
                        if (instr_b != 8'd0 && !perm_present) begin
                            state <= S_ERROR;
                        end else if (perm_arity > MAX_ARITY) begin
                            state <= S_ERROR;
                        end else begin
                            d_target_count <= perm_arity;
                            d_idx <= 4'd0;
                            if (perm_needs_consts) begin
                                state <= S_C_LEN_LO;
                            end else if (perm_arity == 0) begin
                                state <= S_EMIT;
                            end else begin
                                state <= S_D_LO;
                            end
                        end
                    end

                    S_C_LEN_LO: begin
                        if (accept) begin
                            tmp16[7:0] <= ops_byte;
                            state <= S_C_LEN_HI;
                        end
                    end

                    S_C_LEN_HI: begin
                        if (accept) begin
                            tmp16[15:8] <= ops_byte;
                            c_flat[0 +: 16] <= {ops_byte, tmp16[7:0]};
                            c_count <= 4'd1;
                            if ({ops_byte, tmp16[7:0]} > (MAX_CONSTS - 1)) begin
                                state <= S_ERROR;
                            end else if ({ops_byte, tmp16[7:0]} == 16'd0) begin
                                if (d_target_count == 0) state <= S_EMIT;
                                else state <= S_D_LO;
                            end else begin
                                c_target_count <= tmp16[3:0] + 4'd1;
                                c_idx <= 4'd1;
                                state <= S_C_LO;
                            end
                        end
                    end

                    S_C_LO: begin
                        if (accept) begin
                            tmp16[7:0] <= ops_byte;
                            state <= S_C_HI;
                        end
                    end

                    S_C_HI: begin
                        if (accept) begin
                            tmp16[15:8] <= ops_byte;
                            c_flat[c_idx*16 +: 16] <= {ops_byte, tmp16[7:0]};
                            c_count <= c_idx + 4'd1;
                            if ((c_idx + 4'd1) >= c_target_count) begin
                                if (d_target_count == 0) state <= S_EMIT;
                                else state <= S_D_LO;
                            end else begin
                                c_idx <= c_idx + 4'd1;
                                state <= S_C_LO;
                            end
                        end
                    end

                    S_D_LO: begin
                        if (accept) begin
                            tmp16[7:0] <= ops_byte;
                            state <= S_D_HI;
                        end
                    end

                    S_D_HI: begin
                        if (accept) begin
                            tmp16[15:8] <= ops_byte;
                            d_flat[d_idx*16 +: 16] <= {ops_byte, tmp16[7:0]};
                            d_count <= d_idx + 4'd1;
                            if ((d_idx + 4'd1) >= d_target_count) begin
                                state <= S_EMIT;
                            end else begin
                                d_idx <= d_idx + 4'd1;
                                state <= S_D_LO;
                            end
                        end
                    end

                    S_SYS_C_LO: begin
                        if (sys_c_target_count == 0) begin
                            if (sys_d_target_count == 0) state <= S_EMIT;
                            else state <= S_SYS_D_LO;
                        end else if (accept) begin
                            tmp16[7:0] <= ops_byte;
                            state <= S_SYS_C_HI;
                        end
                    end

                    S_SYS_C_HI: begin
                        if (accept) begin
                            tmp16[15:8] <= ops_byte;
                            c_flat[c_idx*16 +: 16] <= {ops_byte, tmp16[7:0]};
                            c_count <= c_idx + 4'd1;
                            if ((c_idx + 4'd1) >= sys_c_target_count[3:0]) begin
                                if (sys_d_target_count == 0) state <= S_EMIT;
                                else state <= S_SYS_D_LO;
                            end else begin
                                c_idx <= c_idx + 4'd1;
                                state <= S_SYS_C_LO;
                            end
                        end
                    end

                    S_SYS_D_LO: begin
                        if (accept) begin
                            tmp16[7:0] <= ops_byte;
                            state <= S_SYS_D_HI;
                        end
                    end

                    S_SYS_D_HI: begin
                        if (accept) begin
                            tmp16[15:8] <= ops_byte;
                            d_flat[d_idx*16 +: 16] <= {ops_byte, tmp16[7:0]};
                            d_count <= d_idx + 4'd1;
                            if ((d_idx + 4'd1) >= sys_d_target_count[3:0]) begin
                                state <= S_EMIT;
                            end else begin
                                d_idx <= d_idx + 4'd1;
                                state <= S_SYS_D_LO;
                            end
                        end
                    end

                    S_EMIT: begin
                        instr_valid <= 1'b1;
                        if (instr_valid && instr_ready) begin
                            instr_valid <= 1'b0;
                            instr_index <= instr_index + 16'd1;
                            state <= S_A;
                        end
                    end

                    S_ERROR: begin
                        error <= 1'b1;
                        busy <= 1'b0;
                        instr_valid <= 1'b0;
                        state <= S_IDLE;
                    end

                    default: begin
                        state <= S_ERROR;
                    end
                endcase
            end
        end
    end
endmodule
