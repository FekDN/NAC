`include "nac_defs.vh"

module nac_result_table #(
    parameter ENTRIES = 1024,
    parameter IDX_WIDTH = 10,
    parameter DESC_WIDTH = 64
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
    input  wire [IDX_WIDTH-1:0] forward_dst_idx
);
    reg valid_ram [0:ENTRIES-1];
    reg [DESC_WIDTH-1:0] desc_ram [0:ENTRIES-1];

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < ENTRIES; i = i + 1) begin
                valid_ram[i] <= 1'b0;
                desc_ram[i] <= {DESC_WIDTH{1'b0}};
            end
            rd0_valid <= 1'b0;
            rd1_valid <= 1'b0;
            rd0_desc <= {DESC_WIDTH{1'b0}};
            rd1_desc <= {DESC_WIDTH{1'b0}};
        end else begin
            if (wr_valid) begin
                valid_ram[wr_idx] <= 1'b1;
                desc_ram[wr_idx] <= wr_desc;
            end

            if (free_valid) begin
                valid_ram[free_idx] <= 1'b0;
                desc_ram[free_idx] <= {DESC_WIDTH{1'b0}};
            end

            if (forward_valid) begin
                valid_ram[forward_dst_idx] <= valid_ram[forward_src_idx];
                desc_ram[forward_dst_idx] <= desc_ram[forward_src_idx];
            end

            if (rd0_en) begin
                rd0_valid <= valid_ram[rd0_idx];
                rd0_desc <= desc_ram[rd0_idx];
            end else begin
                rd0_valid <= 1'b0;
            end

            if (rd1_en) begin
                rd1_valid <= valid_ram[rd1_idx];
                rd1_desc <= desc_ram[rd1_idx];
            end else begin
                rd1_valid <= 1'b0;
            end
        end
    end
endmodule
