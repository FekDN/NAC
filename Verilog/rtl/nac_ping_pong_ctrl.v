module nac_ping_pong_ctrl #(
    parameter BANK_BITS = 3
) (
    input  wire clk,
    input  wire rst,
    input  wire enable,
    input  wire start,
    input  wire dma_done,
    input  wire compute_done,
    input  wire [BANK_BITS-1:0] ping_bank,
    input  wire [BANK_BITS-1:0] pong_bank,
    output reg  [BANK_BITS-1:0] dma_bank,
    output reg  [BANK_BITS-1:0] compute_bank,
    output reg  phase,
    output reg  swap_pulse
);
    always @(posedge clk) begin
        if (rst) begin
            phase <= 1'b0;
            dma_bank <= pong_bank;
            compute_bank <= ping_bank;
            swap_pulse <= 1'b0;
        end else begin
            swap_pulse <= 1'b0;

            if (start) begin
                phase <= 1'b0;
                dma_bank <= enable ? pong_bank : ping_bank;
                compute_bank <= ping_bank;
            end else if (enable && dma_done && compute_done) begin
                phase <= ~phase;
                dma_bank <= phase ? pong_bank : ping_bank;
                compute_bank <= phase ? ping_bank : pong_bank;
                swap_pulse <= 1'b1;
            end
        end
    end
endmodule
