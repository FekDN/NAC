`include "nac_defs.vh"

module nac_scratchpad #(
    parameter BANKS = 8,
    parameter BANK_BITS = 3,
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 32,
    parameter ENABLE_ECC = 1,
    parameter ECC_WIDTH = (DATA_WIDTH <= 32) ? 7 : 8
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
    output reg  [DATA_WIDTH-1:0] b_rdata,

    output reg  a_single_error,
    output reg  a_double_error,
    output reg  b_single_error,
    output reg  b_double_error
);
    localparam DEPTH = (1 << ADDR_WIDTH);
    localparam ENCODED_BANKS = (1 << BANK_BITS);
    localparam TOTAL_WORDS = ENCODED_BANKS * DEPTH;

    reg [DATA_WIDTH-1:0] mem [0:TOTAL_WORDS-1];
    reg [ECC_WIDTH-1:0] ecc_mem [0:TOTAL_WORDS-1];
    wire [ADDR_WIDTH+BANK_BITS-1:0] a_index = {a_bank, a_addr};
    wire [ADDR_WIDTH+BANK_BITS-1:0] b_index = {b_bank, b_addr};
    wire [ECC_WIDTH-1:0] a_write_ecc;
    wire [ECC_WIDTH-1:0] b_write_ecc;
    wire [DATA_WIDTH-1:0] a_corrected;
    wire [DATA_WIDTH-1:0] b_corrected;
    wire a_single_comb;
    wire a_double_comb;
    wire b_single_comb;
    wire b_double_comb;
    integer init_i;

    initial begin
        for (init_i = 0; init_i < TOTAL_WORDS; init_i = init_i + 1) begin
            mem[init_i] = {DATA_WIDTH{1'b0}};
            ecc_mem[init_i] = {ECC_WIDTH{1'b0}};
        end
        a_rdata = {DATA_WIDTH{1'b0}};
        b_rdata = {DATA_WIDTH{1'b0}};
        a_single_error = 1'b0;
        a_double_error = 1'b0;
        b_single_error = 1'b0;
        b_double_error = 1'b0;
    end

    nac_ecc_secded #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_a_write (
        .data_in(a_wdata),
        .ecc_in({ECC_WIDTH{1'b0}}),
        .ecc_out(a_write_ecc),
        .data_corrected(),
        .single_error(),
        .double_error()
    );

    nac_ecc_secded #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_b_write (
        .data_in(b_wdata),
        .ecc_in({ECC_WIDTH{1'b0}}),
        .ecc_out(b_write_ecc),
        .data_corrected(),
        .single_error(),
        .double_error()
    );

    nac_ecc_secded #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_a_read (
        .data_in(mem[a_index]),
        .ecc_in(ecc_mem[a_index]),
        .ecc_out(),
        .data_corrected(a_corrected),
        .single_error(a_single_comb),
        .double_error(a_double_comb)
    );

    nac_ecc_secded #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) ecc_b_read (
        .data_in(mem[b_index]),
        .ecc_in(ecc_mem[b_index]),
        .ecc_out(),
        .data_corrected(b_corrected),
        .single_error(b_single_comb),
        .double_error(b_double_comb)
    );

    always @(posedge clk) begin
        if (a_we) begin
            mem[a_index] <= a_wdata;
            if (ENABLE_ECC) ecc_mem[a_index] <= a_write_ecc;
        end
        a_rdata <= ENABLE_ECC ? a_corrected : mem[a_index];
        a_single_error <= ENABLE_ECC ? a_single_comb : 1'b0;
        a_double_error <= ENABLE_ECC ? a_double_comb : 1'b0;

        if (b_we) begin
            mem[b_index] <= b_wdata;
            if (ENABLE_ECC) ecc_mem[b_index] <= b_write_ecc;
        end
        b_rdata <= ENABLE_ECC ? b_corrected : mem[b_index];
        b_single_error <= ENABLE_ECC ? b_single_comb : 1'b0;
        b_double_error <= ENABLE_ECC ? b_double_comb : 1'b0;
    end
endmodule
