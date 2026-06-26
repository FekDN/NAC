`include "nac_defs.vh"
`include "nac_descriptor.vh"

module nac_result_table #(
    parameter ENTRIES = 1024,
    parameter IDX_WIDTH = 10,
    parameter DESC_WIDTH = `NAC_DESC_MIN_WIDTH,
    parameter ENABLE_ECC = 1,
    parameter ECC_WIDTH = (DESC_WIDTH <= 32) ? 7 : 8
) (
    input  wire clk,
    input  wire rst,

    input  wire wr_valid,
    input  wire [IDX_WIDTH-1:0] wr_idx,
    input  wire [DESC_WIDTH-1:0] wr_desc,

    input  wire rd0_en,
    input  wire [IDX_WIDTH-1:0] rd0_idx,
    output reg  rd0_valid,
    output reg  [DESC_WIDTH-1:0] rd0_desc,

    input  wire rd1_en,
    input  wire [IDX_WIDTH-1:0] rd1_idx,
    output reg  rd1_valid,
    output reg  [DESC_WIDTH-1:0] rd1_desc,

    input  wire free_valid,
    input  wire [IDX_WIDTH-1:0] free_idx,

    input  wire forward_valid,
    input  wire [IDX_WIDTH-1:0] forward_src_idx,
    input  wire [IDX_WIDTH-1:0] forward_dst_idx,

    output reg  single_error,
    output reg  double_error
);
    reg valid_ram [0:ENTRIES-1];
    reg [DESC_WIDTH-1:0] desc_ram [0:ENTRIES-1];
    reg [ECC_WIDTH-1:0] ecc_ram [0:ENTRIES-1];
    wire [ECC_WIDTH-1:0] wr_ecc;
    wire [DESC_WIDTH-1:0] rd0_corrected;
    wire [DESC_WIDTH-1:0] rd1_corrected;
    wire rd0_single_comb;
    wire rd0_double_comb;
    wire rd1_single_comb;
    wire rd1_double_comb;

    nac_ecc_secded #(
        .DATA_WIDTH(DESC_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_write (
        .data_in(wr_desc),
        .ecc_in({ECC_WIDTH{1'b0}}),
        .ecc_out(wr_ecc),
        .data_corrected(),
        .single_error(),
        .double_error()
    );

    nac_ecc_secded #(
        .DATA_WIDTH(DESC_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_rd0 (
        .data_in(desc_ram[rd0_idx]),
        .ecc_in(ecc_ram[rd0_idx]),
        .ecc_out(),
        .data_corrected(rd0_corrected),
        .single_error(rd0_single_comb),
        .double_error(rd0_double_comb)
    );

    nac_ecc_secded #(
        .DATA_WIDTH(DESC_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_rd1 (
        .data_in(desc_ram[rd1_idx]),
        .ecc_in(ecc_ram[rd1_idx]),
        .ecc_out(),
        .data_corrected(rd1_corrected),
        .single_error(rd1_single_comb),
        .double_error(rd1_double_comb)
    );

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < ENTRIES; i = i + 1) begin
                valid_ram[i] <= 1'b0;
                desc_ram[i] <= {DESC_WIDTH{1'b0}};
                ecc_ram[i] <= {ECC_WIDTH{1'b0}};
            end
            rd0_valid <= 1'b0;
            rd1_valid <= 1'b0;
            rd0_desc <= {DESC_WIDTH{1'b0}};
            rd1_desc <= {DESC_WIDTH{1'b0}};
            single_error <= 1'b0;
            double_error <= 1'b0;
        end else begin
            single_error <= ((rd0_en && ENABLE_ECC) ? rd0_single_comb : 1'b0) |
                            ((rd1_en && ENABLE_ECC) ? rd1_single_comb : 1'b0);
            double_error <= ((rd0_en && ENABLE_ECC) ? rd0_double_comb : 1'b0) |
                            ((rd1_en && ENABLE_ECC) ? rd1_double_comb : 1'b0);

            if (wr_valid) begin
                valid_ram[wr_idx] <= 1'b1;
                desc_ram[wr_idx] <= wr_desc;
                if (ENABLE_ECC) ecc_ram[wr_idx] <= wr_ecc;
            end

            if (free_valid) begin
                valid_ram[free_idx] <= 1'b0;
                desc_ram[free_idx] <= {DESC_WIDTH{1'b0}};
                ecc_ram[free_idx] <= {ECC_WIDTH{1'b0}};
            end

            if (forward_valid) begin
                valid_ram[forward_dst_idx] <= valid_ram[forward_src_idx];
                desc_ram[forward_dst_idx] <= desc_ram[forward_src_idx];
                ecc_ram[forward_dst_idx] <= ecc_ram[forward_src_idx];
            end

            if (rd0_en) begin
                rd0_valid <= valid_ram[rd0_idx];
                rd0_desc <= ENABLE_ECC ? rd0_corrected : desc_ram[rd0_idx];
            end else begin
                rd0_valid <= 1'b0;
            end

            if (rd1_en) begin
                rd1_valid <= valid_ram[rd1_idx];
                rd1_desc <= ENABLE_ECC ? rd1_corrected : desc_ram[rd1_idx];
            end else begin
                rd1_valid <= 1'b0;
            end
        end
    end
endmodule
