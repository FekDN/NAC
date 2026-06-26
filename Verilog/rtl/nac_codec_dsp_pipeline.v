`include "nac_defs.vh"
`include "nac_config.vh"

module nac_codec_dsp_pipeline #(
    parameter [255:0] HW_CFG = `NAC_CFG_FPGA_TINY_EDGE,
    parameter LANES = 16,
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS = 16,
    parameter ACC_WIDTH = (DATA_WIDTH*2) + 8,
    parameter ENABLE_BFP_DECODE = (`NAC_CFG_DATA_TYPE(HW_CFG) != `NAC_DTYPE_INT_FIXED),
    parameter ENABLE_PALETTE_DECODE = `NAC_CFG_ENABLE_COMPRESSION(HW_CFG),
    parameter ENABLE_STRUCTURED_SPARSITY = `NAC_CFG_ENABLE_SPARSITY(HW_CFG),
    parameter ENABLE_RLE_ENCODE = `NAC_CFG_ENABLE_COMPRESSION(HW_CFG)
) (
    input  wire clk,
    input  wire rst,

    input  wire [7:0] cfg_mode,
    input  wire signed [ACC_WIDTH-1:0] acc_in,
    input  wire in_valid,
    output wire in_ready,
    input  wire [LANES*DATA_WIDTH-1:0] a_vec,
    input  wire [LANES*DATA_WIDTH-1:0] b_mem_vec,

    input  wire weight_is_bfp,
    input  wire weight_is_palette,
    input  wire weight_is_sparse_2_4,
    input  wire result_is_rle,
    input  wire [LANES*4-1:0] weight_bfp_mantissas,
    input  wire signed [7:0] weight_bfp_exp,
    input  wire [16*DATA_WIDTH-1:0] weight_palette,
    input  wire [LANES*4-1:0] weight_palette_indices,
    input  wire [(LANES/2)*DATA_WIDTH-1:0] weight_sparse_values,
    input  wire [(LANES/2)*2-1:0] weight_sparse_indices,

    output wire out_valid,
    input  wire out_ready,
    output wire [LANES*DATA_WIDTH-1:0] out_vec,
    output wire signed [ACC_WIDTH-1:0] scalar_out,

    output wire sfu_req_valid,
    input  wire sfu_req_ready,
    output wire [7:0] sfu_req_mode,
    output wire [LANES*DATA_WIDTH-1:0] sfu_req_vec,
    input  wire sfu_resp_valid,
    input  wire [LANES*DATA_WIDTH-1:0] sfu_resp_vec,

    output wire [LANES*DATA_WIDTH-1:0] rle_values,
    output wire [LANES*8-1:0] rle_counts,
    output wire [LANES-1:0] rle_is_zero_run,
    output wire [7:0] rle_token_count,
    output wire rle_valid,
    output wire error
);
    wire codec_valid;
    wire codec_ready;
    wire [LANES*DATA_WIDTH-1:0] codec_a_vec;
    wire [LANES*DATA_WIDTH-1:0] codec_b_vec;
    wire codec_result_is_rle;
    wire codec_error;
    wire dsp_error;
    wire rle_error;

    nac_tensor_codec_pipeline #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .BFP_MANT_WIDTH(4),
        .ENABLE_BFP_DECODE(ENABLE_BFP_DECODE),
        .ENABLE_PALETTE_DECODE(ENABLE_PALETTE_DECODE),
        .ENABLE_STRUCTURED_SPARSITY(ENABLE_STRUCTURED_SPARSITY)
    ) tensor_codec (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .a_dense_in(a_vec),
        .b_dense_in(b_mem_vec),
        .weight_is_bfp(weight_is_bfp),
        .weight_is_palette(weight_is_palette),
        .weight_is_sparse_2_4(weight_is_sparse_2_4),
        .result_is_rle(result_is_rle),
        .weight_bfp_mantissas(weight_bfp_mantissas),
        .weight_bfp_exp(weight_bfp_exp),
        .weight_palette(weight_palette),
        .weight_palette_indices(weight_palette_indices),
        .weight_sparse_values(weight_sparse_values),
        .weight_sparse_indices(weight_sparse_indices),
        .out_valid(codec_valid),
        .out_ready(codec_ready),
        .a_dense_out(codec_a_vec),
        .b_dense_out(codec_b_vec),
        .result_is_rle_out(codec_result_is_rle),
        .error(codec_error)
    );

    nac_dsp_pipeline #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .ACC_WIDTH(ACC_WIDTH),
        .ENABLE_SPARSITY(ENABLE_STRUCTURED_SPARSITY),
        .ENABLE_RLE_ENCODE(ENABLE_RLE_ENCODE)
    ) dsp (
        .clk(clk),
        .rst(rst),
        .cfg_mode(cfg_mode),
        .acc_in(acc_in),
        .in_valid(codec_valid),
        .in_ready(codec_ready),
        .a_vec(codec_a_vec),
        .b_vec(codec_b_vec),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_vec(out_vec),
        .scalar_out(scalar_out),
        .sfu_req_valid(sfu_req_valid),
        .sfu_req_ready(sfu_req_ready),
        .sfu_req_mode(sfu_req_mode),
        .sfu_req_vec(sfu_req_vec),
        .sfu_resp_valid(sfu_resp_valid),
        .sfu_resp_vec(sfu_resp_vec),
        .result_is_rle(codec_result_is_rle),
        .rle_values(rle_values),
        .rle_counts(rle_counts),
        .rle_is_zero_run(rle_is_zero_run),
        .rle_token_count(rle_token_count),
        .rle_valid(rle_valid),
        .rle_error(rle_error),
        .error(dsp_error)
    );

    assign error = codec_error | dsp_error | rle_error;
endmodule
