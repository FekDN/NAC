`include "nac_defs.vh"

module nac_systolic_array #(
    parameter DIM = 16,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 40,
    parameter FRAC_BITS = 0,
    parameter ENABLE_SPARSITY = 0
) (
    input  wire clk,
    input  wire rst,
    input  wire clear,
    input  wire in_valid,
    input  wire in_last,
    input  wire [DIM*DATA_WIDTH-1:0] a_west,
    input  wire [DIM*DATA_WIDTH-1:0] b_north,
    output reg  out_valid,
    output wire [DIM*DIM*ACC_WIDTH-1:0] c_flat,
    output wire [DIM*DIM-1:0] active_mac_mask
);
    localparam [15:0] FLUSH_CYCLES = (DIM <= 1) ? 16'd1 : ((DIM * 2) - 2);

    reg signed [DATA_WIDTH-1:0] a_pipe [0:DIM-1][0:DIM-1];
    reg signed [DATA_WIDTH-1:0] b_pipe [0:DIM-1][0:DIM-1];
    reg signed [ACC_WIDTH-1:0] acc [0:DIM-1][0:DIM-1];
    reg flushing;
    reg [15:0] flush_count;

    integer r;
    integer c;
    reg signed [(DATA_WIDTH*2)-1:0] prod;

    genvar gr;
    genvar gc;
    generate
        for (gr = 0; gr < DIM; gr = gr + 1) begin : gen_rows
            for (gc = 0; gc < DIM; gc = gc + 1) begin : gen_cols
                assign c_flat[(gr*DIM + gc)*ACC_WIDTH +: ACC_WIDTH] = acc[gr][gc];
                assign active_mac_mask[gr*DIM + gc] =
                    !ENABLE_SPARSITY ||
                    (a_pipe[gr][gc] != {DATA_WIDTH{1'b0}} && b_pipe[gr][gc] != {DATA_WIDTH{1'b0}});
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst || clear) begin
            for (r = 0; r < DIM; r = r + 1) begin
                for (c = 0; c < DIM; c = c + 1) begin
                    a_pipe[r][c] <= {DATA_WIDTH{1'b0}};
                    b_pipe[r][c] <= {DATA_WIDTH{1'b0}};
                    acc[r][c] <= {ACC_WIDTH{1'b0}};
                end
            end
            flushing <= 1'b0;
            flush_count <= 16'd0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;

            if (in_valid || flushing) begin
                for (r = 0; r < DIM; r = r + 1) begin
                    for (c = 0; c < DIM; c = c + 1) begin
                        if (!ENABLE_SPARSITY ||
                            (a_pipe[r][c] != {DATA_WIDTH{1'b0}} && b_pipe[r][c] != {DATA_WIDTH{1'b0}})) begin
                            prod = a_pipe[r][c] * b_pipe[r][c];
                            acc[r][c] <= acc[r][c] + (prod >>> FRAC_BITS);
                        end

                        if (c == 0) begin
                            a_pipe[r][c] <= in_valid ? a_west[r*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
                        end else begin
                            a_pipe[r][c] <= a_pipe[r][c-1];
                        end

                        if (r == 0) begin
                            b_pipe[r][c] <= in_valid ? b_north[c*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
                        end else begin
                            b_pipe[r][c] <= b_pipe[r-1][c];
                        end
                    end
                end

                if (in_valid && in_last) begin
                    flushing <= 1'b1;
                    flush_count <= 16'd0;
                end else if (flushing) begin
                    if (flush_count >= FLUSH_CYCLES) begin
                        flushing <= 1'b0;
                        out_valid <= 1'b1;
                    end else begin
                        flush_count <= flush_count + 16'd1;
                    end
                end
            end
        end
    end
endmodule
