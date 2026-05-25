`include "nac_defs.vh"

module nac_scratchpad #(
    parameter BANKS = 8,
    parameter BANK_BITS = 3,
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 32
) (
    input  wire clk,

    // Port A: DMA/MMAP side.
    input  wire a_we,
    input  wire [BANK_BITS-1:0] a_bank,
    input  wire [ADDR_WIDTH-1:0] a_addr,
    input  wire [DATA_WIDTH-1:0] a_wdata,
    output reg  [DATA_WIDTH-1:0] a_rdata,

    // Port B: DSP side.
    input  wire b_we,
    input  wire [BANK_BITS-1:0] b_bank,
    input  wire [ADDR_WIDTH-1:0] b_addr,
    input  wire [DATA_WIDTH-1:0] b_wdata,
    output reg  [DATA_WIDTH-1:0] b_rdata
);
    localparam DEPTH = (1 << ADDR_WIDTH);
    localparam TOTAL_WORDS = BANKS * DEPTH;

    reg [DATA_WIDTH-1:0] mem [0:TOTAL_WORDS-1];
    wire [ADDR_WIDTH+BANK_BITS-1:0] a_index = {a_bank, a_addr};
    wire [ADDR_WIDTH+BANK_BITS-1:0] b_index = {b_bank, b_addr};

    always @(posedge clk) begin
        if (a_we) begin
            mem[a_index] <= a_wdata;
        end
        a_rdata <= mem[a_index];

        if (b_we) begin
            mem[b_index] <= b_wdata;
        end
        b_rdata <= mem[b_index];
    end
endmodule
