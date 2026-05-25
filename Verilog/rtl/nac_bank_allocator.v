`include "nac_defs.vh"

module nac_bank_allocator #(
    parameter BANKS = 8,
    parameter BANK_BITS = 3
) (
    input  wire clk,
    input  wire rst,

    input  wire alloc_req,
    output reg  alloc_valid,
    output reg  [BANK_BITS-1:0] alloc_bank,

    input  wire free_req,
    input  wire [BANK_BITS-1:0] free_bank
);
    reg [BANKS-1:0] used;
    integer i;
    reg found;

    always @(posedge clk) begin
        if (rst) begin
            used <= {BANKS{1'b0}};
            alloc_valid <= 1'b0;
            alloc_bank <= {BANK_BITS{1'b0}};
        end else begin
            alloc_valid <= 1'b0;

            if (free_req) begin
                used[free_bank] <= 1'b0;
            end

            if (alloc_req) begin
                found = 1'b0;
                for (i = 0; i < BANKS; i = i + 1) begin
                    if (!found && !used[i]) begin
                        found = 1'b1;
                        alloc_valid <= 1'b1;
                        alloc_bank <= i;
                        used[i] <= 1'b1;
                    end
                end
            end
        end
    end
endmodule
