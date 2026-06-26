`include "nac_defs.vh"

module nac_dsp_pipeline #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS = 16,
    parameter ACC_WIDTH = (DATA_WIDTH*2) + 8,
    parameter ENABLE_SPARSITY = 0,
    parameter SFU_TIMEOUT_CYCLES = 0,
    parameter ENABLE_RLE_ENCODE = 0
) (
    input  wire clk,
    input  wire rst,

    input  wire [7:0] cfg_mode,
    input  wire signed [ACC_WIDTH-1:0] acc_in,
    input  wire in_valid,
    output wire in_ready,
    input  wire [LANES*DATA_WIDTH-1:0] a_vec,
    input  wire [LANES*DATA_WIDTH-1:0] b_vec,

    output reg  out_valid,
    input  wire out_ready,
    output reg  [LANES*DATA_WIDTH-1:0] out_vec,
    output reg  signed [ACC_WIDTH-1:0] scalar_out,

    output reg  sfu_req_valid,
    input  wire sfu_req_ready,
    output reg  [7:0] sfu_req_mode,
    output reg  [LANES*DATA_WIDTH-1:0] sfu_req_vec,
    input  wire sfu_resp_valid,
    input  wire [LANES*DATA_WIDTH-1:0] sfu_resp_vec,
    input  wire result_is_rle,
    output wire [LANES*DATA_WIDTH-1:0] rle_values,
    output wire [LANES*8-1:0] rle_counts,
    output wire [LANES-1:0] rle_is_zero_run,
    output wire [7:0] rle_token_count,
    output wire rle_valid,
    output wire rle_error,
    output reg  error
);
    wire external_sfu_mode;
    wire accept;
    reg sfu_pending;

    assign external_sfu_mode = (cfg_mode == `NAC_DSP_RSQRT) || (cfg_mode == `NAC_DSP_EXP);
    assign in_ready = external_sfu_mode
                    ? (!sfu_pending && !sfu_req_valid)
                    : (!out_valid || out_ready);
    assign accept = in_valid && in_ready;

    wire [LANES*ACC_WIDTH-1:0] prod_vec;
    wire signed [ACC_WIDTH-1:0] dot_sum;
    wire [LANES-1:0] active_lane_mask;
    nac_mac_array #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ENABLE_SPARSITY(ENABLE_SPARSITY)
    ) mac_array (
        .a_vec(a_vec),
        .b_vec(b_vec),
        .prod_vec(prod_vec),
        .dot_sum(dot_sum),
        .active_lane_mask(active_lane_mask)
    );

    wire signed [DATA_WIDTH+8-1:0] sum_reduce;
    wire signed [DATA_WIDTH-1:0] max_reduce;
    nac_reduction_tree #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(DATA_WIDTH+8)
    ) lane_reduce (
        .vec_in(a_vec),
        .sum_out(sum_reduce),
        .max_out(max_reduce)
    );

    integer i;
    reg signed [DATA_WIDTH-1:0] a_lane;
    reg signed [DATA_WIDTH-1:0] b_lane;
    reg signed [(DATA_WIDTH*2)-1:0] wide_mul;
    reg signed [(DATA_WIDTH*2)-1:0] update_delta;
    reg signed [DATA_WIDTH-1:0] lr_lane;
    reg [LANES*DATA_WIDTH-1:0] comb_vec;
    reg signed [ACC_WIDTH-1:0] comb_scalar;
    wire [LANES*DATA_WIDTH-1:0] sfu_vec;
    reg [31:0] sfu_wait_counter;
    wire result_is_rle_i = (result_is_rle === 1'b1);
    wire [LANES*DATA_WIDTH-1:0] rle_dense_out_unused;
    wire rle_decode_error_unused;

    generate
        if (ENABLE_RLE_ENCODE) begin : gen_rle_encode
            nac_activation_rle_codec #(
                .LANES(LANES),
                .DATA_WIDTH(DATA_WIDTH),
                .RUN_WIDTH(8)
            ) activation_rle (
                .dense_in(out_vec),
                .enc_values(rle_values),
                .enc_counts(rle_counts),
                .enc_is_zero_run(rle_is_zero_run),
                .enc_token_count(rle_token_count),
                .dec_values({LANES*DATA_WIDTH{1'b0}}),
                .dec_counts({LANES*8{1'b0}}),
                .dec_is_zero_run({LANES{1'b0}}),
                .dec_token_count(8'd0),
                .dense_out(rle_dense_out_unused),
                .decode_error(rle_decode_error_unused)
            );
            assign rle_valid = out_valid && result_is_rle_i;
            assign rle_error = 1'b0;
        end else begin : gen_no_rle_encode
            assign rle_values = {LANES*DATA_WIDTH{1'b0}};
            assign rle_counts = {LANES*8{1'b0}};
            assign rle_is_zero_run = {LANES{1'b0}};
            assign rle_token_count = 8'd0;
            assign rle_valid = 1'b0;
            assign rle_error = 1'b0;
        end
    endgenerate

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_sfu
            nac_sfu #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) sfu_lane (
                .mode(cfg_mode),
                .x(a_vec[g*DATA_WIDTH +: DATA_WIDTH]),
                .y(sfu_vec[g*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

    always @* begin
        comb_vec = {LANES*DATA_WIDTH{1'b0}};
        comb_scalar = {ACC_WIDTH{1'b0}};

        case (cfg_mode)
            `NAC_DSP_NOP: begin
                comb_vec = {LANES*DATA_WIDTH{1'b0}};
            end
            `NAC_DSP_PASS: begin
                comb_vec = a_vec;
            end
            `NAC_DSP_ADD,
            `NAC_DSP_SUB,
            `NAC_DSP_MUL,
            `NAC_DSP_SCALE,
            `NAC_DSP_GT,
            `NAC_DSP_LE: begin
                for (i = 0; i < LANES; i = i + 1) begin
                    a_lane = a_vec[i*DATA_WIDTH +: DATA_WIDTH];
                    b_lane = b_vec[i*DATA_WIDTH +: DATA_WIDTH];
                    case (cfg_mode)
                        `NAC_DSP_ADD: comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = a_lane + b_lane;
                        `NAC_DSP_SUB: comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = a_lane - b_lane;
                        `NAC_DSP_MUL: begin
                            wide_mul = a_lane * b_lane;
                            comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = wide_mul >>> FRAC_BITS;
                        end
                        `NAC_DSP_SCALE: begin
                            wide_mul = a_lane * b_vec[DATA_WIDTH-1:0];
                            comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = wide_mul >>> FRAC_BITS;
                        end
                        `NAC_DSP_GT: comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = (a_lane > b_lane) ? {{(DATA_WIDTH-1){1'b0}}, 1'b1} : {DATA_WIDTH{1'b0}};
                        `NAC_DSP_LE: comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = (a_lane <= b_lane) ? {{(DATA_WIDTH-1){1'b0}}, 1'b1} : {DATA_WIDTH{1'b0}};
                        default: comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = {DATA_WIDTH{1'b0}};
                    endcase
                end
            end
            `NAC_DSP_WEIGHT_UPDATE: begin
                lr_lane = acc_in[DATA_WIDTH-1:0];
                for (i = 0; i < LANES; i = i + 1) begin
                    a_lane = a_vec[i*DATA_WIDTH +: DATA_WIDTH];
                    b_lane = b_vec[i*DATA_WIDTH +: DATA_WIDTH];
                    update_delta = b_lane * lr_lane;
                    comb_vec[i*DATA_WIDTH +: DATA_WIDTH] =
                        a_lane - (update_delta >>> FRAC_BITS);
                end
            end
            `NAC_DSP_NEG: begin
                for (i = 0; i < LANES; i = i + 1) begin
                    a_lane = a_vec[i*DATA_WIDTH +: DATA_WIDTH];
                    comb_vec[i*DATA_WIDTH +: DATA_WIDTH] = -a_lane;
                end
            end
            `NAC_DSP_RELU,
            `NAC_DSP_HSIGMOID,
            `NAC_DSP_HSWISH: begin
                comb_vec = sfu_vec;
            end
            `NAC_DSP_MAC: begin
                comb_scalar = dot_sum + acc_in;
                comb_vec[0 +: DATA_WIDTH] = comb_scalar[DATA_WIDTH-1:0];
            end
            `NAC_DSP_REDUCE_SUM: begin
                comb_scalar = sum_reduce;
                comb_vec[0 +: DATA_WIDTH] = sum_reduce[DATA_WIDTH-1:0];
            end
            `NAC_DSP_REDUCE_MAX: begin
                comb_scalar = max_reduce;
                comb_vec[0 +: DATA_WIDTH] = max_reduce;
            end
            default: begin
                comb_vec = a_vec;
            end
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_vec <= {LANES*DATA_WIDTH{1'b0}};
            scalar_out <= {ACC_WIDTH{1'b0}};
            sfu_req_valid <= 1'b0;
            sfu_req_mode <= `NAC_DSP_NOP;
            sfu_req_vec <= {LANES*DATA_WIDTH{1'b0}};
            sfu_pending <= 1'b0;
            sfu_wait_counter <= 32'd0;
            error <= 1'b0;
        end else if (sfu_pending) begin
            if (sfu_req_valid && sfu_req_ready) begin
                sfu_req_valid <= 1'b0;
            end
            if (sfu_resp_valid && (!out_valid || out_ready)) begin
                out_valid <= 1'b1;
                out_vec <= sfu_resp_vec;
                scalar_out <= {ACC_WIDTH{1'b0}};
                sfu_pending <= 1'b0;
                sfu_wait_counter <= 32'd0;
            end else if (SFU_TIMEOUT_CYCLES != 0) begin
                if (sfu_wait_counter >= (SFU_TIMEOUT_CYCLES - 1)) begin
                    error <= 1'b1;
                    sfu_req_valid <= 1'b0;
                    sfu_pending <= 1'b0;
                    sfu_wait_counter <= 32'd0;
                end else begin
                    sfu_wait_counter <= sfu_wait_counter + 32'd1;
                end
            end
        end else if (accept) begin
            if (external_sfu_mode) begin
                sfu_req_valid <= 1'b1;
                sfu_req_mode <= cfg_mode;
                sfu_req_vec <= a_vec;
                sfu_pending <= 1'b1;
                sfu_wait_counter <= 32'd0;
            end else begin
                out_valid <= 1'b1;
                out_vec <= comb_vec;
                scalar_out <= comb_scalar;
            end
        end else if (out_ready) begin
            out_valid <= 1'b0;
        end
    end
endmodule
