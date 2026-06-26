`include "nac_defs.vh"

module nac_watchdog #(
    parameter TIMEOUT_CYCLES = 0,
    parameter COUNTER_WIDTH = 32
) (
    input  wire clk,
    input  wire rst,
    input  wire clear,
    input  wire enable,
    input  wire busy,
    input  wire progress,
    output reg  timeout,
    output reg  reset_pulse
);
    reg [COUNTER_WIDTH-1:0] counter;

    always @(posedge clk) begin
        if (rst || clear) begin
            counter <= {COUNTER_WIDTH{1'b0}};
            timeout <= 1'b0;
            reset_pulse <= 1'b0;
        end else begin
            reset_pulse <= 1'b0;

            if (!enable || TIMEOUT_CYCLES == 0 || !busy) begin
                counter <= {COUNTER_WIDTH{1'b0}};
            end else if (progress) begin
                counter <= {COUNTER_WIDTH{1'b0}};
            end else if (!timeout) begin
                if (counter >= (TIMEOUT_CYCLES - 1)) begin
                    timeout <= 1'b1;
                    reset_pulse <= 1'b1;
                    counter <= {COUNTER_WIDTH{1'b0}};
                end else begin
                    counter <= counter + {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end
endmodule
