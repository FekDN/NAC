`include "nac_defs.vh"

module nac_perm_table #(
    parameter PERM_ENTRIES = 256,
    parameter ARITY_BITS = 4
) (
    input  wire clk,

    input  wire cfg_we,
    input  wire [7:0] cfg_id,
    input  wire [ARITY_BITS-1:0] cfg_arity,
    input  wire cfg_needs_consts,

    input  wire [7:0] lookup_id,
    output wire [ARITY_BITS-1:0] lookup_arity,
    output wire lookup_needs_consts,
    output wire lookup_present
);
    reg [ARITY_BITS-1:0] arity_ram [0:PERM_ENTRIES-1];
    reg needs_ram [0:PERM_ENTRIES-1];
    reg present_ram [0:PERM_ENTRIES-1];

    integer i;
    initial begin
        for (i = 0; i < PERM_ENTRIES; i = i + 1) begin
            arity_ram[i] = {ARITY_BITS{1'b0}};
            needs_ram[i] = 1'b0;
            present_ram[i] = 1'b0;
        end
    end

    always @(posedge clk) begin
        if (cfg_we) begin
            arity_ram[cfg_id] <= cfg_arity;
            needs_ram[cfg_id] <= cfg_needs_consts;
            present_ram[cfg_id] <= 1'b1;
        end
    end

    assign lookup_arity = arity_ram[lookup_id];
    assign lookup_needs_consts = needs_ram[lookup_id];
    assign lookup_present = present_ram[lookup_id];
endmodule
