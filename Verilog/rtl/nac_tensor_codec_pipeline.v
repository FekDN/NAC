module nac_tensor_codec_pipeline #(
    parameter LANES = 16,
    parameter DATA_WIDTH = 32,
    parameter BFP_MANT_WIDTH = 4,
    parameter ENABLE_BFP_DECODE = 0,
    parameter ENABLE_PALETTE_DECODE = 0,
    parameter ENABLE_STRUCTURED_SPARSITY = 0
) (
    input  wire clk,
    input  wire rst,

    input  wire in_valid,
    output wire in_ready,
    input  wire [LANES*DATA_WIDTH-1:0] a_dense_in,
    input  wire [LANES*DATA_WIDTH-1:0] b_dense_in,

    input  wire weight_is_bfp,
    input  wire weight_is_palette,
    input  wire weight_is_sparse_2_4,
    input  wire result_is_rle,

    input  wire [LANES*BFP_MANT_WIDTH-1:0] weight_bfp_mantissas,
    input  wire signed [7:0] weight_bfp_exp,
    input  wire [16*DATA_WIDTH-1:0] weight_palette,
    input  wire [LANES*4-1:0] weight_palette_indices,
    input  wire [(LANES/2)*DATA_WIDTH-1:0] weight_sparse_values,
    input  wire [(LANES/2)*2-1:0] weight_sparse_indices,

    output reg  out_valid,
    input  wire out_ready,
    output reg  [LANES*DATA_WIDTH-1:0] a_dense_out,
    output reg  [LANES*DATA_WIDTH-1:0] b_dense_out,
    output reg  result_is_rle_out,
    output reg  error
);
    localparam SPARSE_NONZERO = LANES / 2;

    wire weight_is_bfp_i = (weight_is_bfp === 1'b1);
    wire weight_is_palette_i = (weight_is_palette === 1'b1);
    wire weight_is_sparse_i = (weight_is_sparse_2_4 === 1'b1);
    wire result_is_rle_i = (result_is_rle === 1'b1);

    wire [LANES*DATA_WIDTH-1:0] dense_bfp_values;
    wire [LANES-1:0] dense_bfp_overflow;
    wire [LANES*DATA_WIDTH-1:0] dense_palette_values;
    wire [SPARSE_NONZERO*DATA_WIDTH-1:0] compact_bfp_values;
    wire [SPARSE_NONZERO-1:0] compact_bfp_overflow;
    wire [SPARSE_NONZERO*DATA_WIDTH-1:0] compact_palette_values;
    wire [LANES*DATA_WIDTH-1:0] sparse_dense_values;
    wire sparse_valid_pattern;
    reg  s1_valid;
    reg  [LANES*DATA_WIDTH-1:0] s1_a_dense;
    reg  [LANES*DATA_WIDTH-1:0] s1_dense_selected_pre;
    reg  [SPARSE_NONZERO*DATA_WIDTH-1:0] s1_compact_values;
    reg  [(LANES/2)*2-1:0] s1_sparse_indices;
    reg  s1_sparse;
    reg  s1_result_is_rle;
    reg  s1_error_pre;

    generate
        if (ENABLE_BFP_DECODE) begin : gen_bfp
            nac_bfp_microscale_decode #(
                .LANES(LANES),
                .MANT_WIDTH(BFP_MANT_WIDTH),
                .EXP_WIDTH(8),
                .OUT_WIDTH(DATA_WIDTH)
            ) dense_bfp (
                .mantissas(weight_bfp_mantissas),
                .shared_exp(weight_bfp_exp),
                .values_out(dense_bfp_values),
                .overflow_mask(dense_bfp_overflow)
            );

            nac_bfp_microscale_decode #(
                .LANES(SPARSE_NONZERO),
                .MANT_WIDTH(BFP_MANT_WIDTH),
                .EXP_WIDTH(8),
                .OUT_WIDTH(DATA_WIDTH)
            ) compact_bfp (
                .mantissas(weight_bfp_mantissas[SPARSE_NONZERO*BFP_MANT_WIDTH-1:0]),
                .shared_exp(weight_bfp_exp),
                .values_out(compact_bfp_values),
                .overflow_mask(compact_bfp_overflow)
            );
        end else begin : gen_no_bfp
            assign dense_bfp_values = {LANES*DATA_WIDTH{1'b0}};
            assign dense_bfp_overflow = {LANES{1'b0}};
            assign compact_bfp_values = {SPARSE_NONZERO*DATA_WIDTH{1'b0}};
            assign compact_bfp_overflow = {SPARSE_NONZERO{1'b0}};
        end
    endgenerate

    generate
        if (ENABLE_PALETTE_DECODE) begin : gen_palette
            nac_palette_decode #(
                .LANES(LANES),
                .DATA_WIDTH(DATA_WIDTH),
                .INDEX_WIDTH(4)
            ) dense_palette (
                .palette(weight_palette),
                .indices(weight_palette_indices),
                .values_out(dense_palette_values)
            );

            nac_palette_decode #(
                .LANES(SPARSE_NONZERO),
                .DATA_WIDTH(DATA_WIDTH),
                .INDEX_WIDTH(4)
            ) compact_palette (
                .palette(weight_palette),
                .indices(weight_palette_indices[SPARSE_NONZERO*4-1:0]),
                .values_out(compact_palette_values)
            );
        end else begin : gen_no_palette
            assign dense_palette_values = {LANES*DATA_WIDTH{1'b0}};
            assign compact_palette_values = {SPARSE_NONZERO*DATA_WIDTH{1'b0}};
        end
    endgenerate

    wire [SPARSE_NONZERO*DATA_WIDTH-1:0] compact_values_selected =
        (ENABLE_BFP_DECODE && weight_is_bfp_i) ? compact_bfp_values :
        (ENABLE_PALETTE_DECODE && weight_is_palette_i) ? compact_palette_values :
        weight_sparse_values;

    generate
        if (ENABLE_STRUCTURED_SPARSITY) begin : gen_sparse
            nac_structured_sparsity_decode #(
                .GROUPS(LANES/4),
                .GROUP_SIZE(4),
                .NONZERO_PER_GROUP(2),
                .DATA_WIDTH(DATA_WIDTH)
            ) sparse_decode (
                .compact_values(s1_compact_values),
                .compact_indices(s1_sparse_indices),
                .dense_values(sparse_dense_values),
                .valid_pattern(sparse_valid_pattern)
            );
        end else begin : gen_no_sparse
            assign sparse_dense_values = {LANES*DATA_WIDTH{1'b0}};
            assign sparse_valid_pattern = 1'b1;
        end
    endgenerate

    wire [LANES*DATA_WIDTH-1:0] dense_selected_pre =
        (ENABLE_BFP_DECODE && weight_is_bfp_i) ? dense_bfp_values :
        (ENABLE_PALETTE_DECODE && weight_is_palette_i) ? dense_palette_values :
        b_dense_in;

    wire format_error_pre =
        (ENABLE_BFP_DECODE && weight_is_bfp_i &&
            ((ENABLE_STRUCTURED_SPARSITY && weight_is_sparse_i) ? |compact_bfp_overflow : |dense_bfp_overflow));

    wire [LANES*DATA_WIDTH-1:0] dense_selected =
        (ENABLE_STRUCTURED_SPARSITY && s1_sparse) ? sparse_dense_values : s1_dense_selected_pre;

    wire format_error = s1_error_pre |
        (ENABLE_STRUCTURED_SPARSITY && s1_sparse && !sparse_valid_pattern);

    wire stage2_ready = !out_valid || out_ready;
    assign in_ready = !s1_valid || stage2_ready;

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
            s1_a_dense <= {LANES*DATA_WIDTH{1'b0}};
            s1_dense_selected_pre <= {LANES*DATA_WIDTH{1'b0}};
            s1_compact_values <= {SPARSE_NONZERO*DATA_WIDTH{1'b0}};
            s1_sparse_indices <= {(LANES/2)*2{1'b0}};
            s1_sparse <= 1'b0;
            s1_result_is_rle <= 1'b0;
            s1_error_pre <= 1'b0;
            out_valid <= 1'b0;
            a_dense_out <= {LANES*DATA_WIDTH{1'b0}};
            b_dense_out <= {LANES*DATA_WIDTH{1'b0}};
            result_is_rle_out <= 1'b0;
            error <= 1'b0;
        end else begin
            if (stage2_ready) begin
                if (s1_valid) begin
                    out_valid <= 1'b1;
                    a_dense_out <= s1_a_dense;
                    b_dense_out <= dense_selected;
                    result_is_rle_out <= s1_result_is_rle;
                    error <= format_error;
                end else begin
                    out_valid <= 1'b0;
                end
                s1_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                s1_valid <= 1'b1;
                s1_a_dense <= a_dense_in;
                s1_dense_selected_pre <= dense_selected_pre;
                s1_compact_values <= compact_values_selected;
                s1_sparse_indices <= weight_sparse_indices;
                s1_sparse <= weight_is_sparse_i;
                s1_result_is_rle <= result_is_rle_i;
                s1_error_pre <= format_error_pre;
            end
        end
    end
endmodule
