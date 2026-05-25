`include "nac_defs.vh"

module nac_addr_gen #(
    parameter ADDR_WIDTH = 32,
    parameter DIM_WIDTH = 16
) (
    input  wire clk,
    input  wire rst,

    input  wire cfg_valid,
    input  wire [ADDR_WIDTH-1:0] cfg_base,
    input  wire [DIM_WIDTH-1:0] cfg_limit0,
    input  wire [DIM_WIDTH-1:0] cfg_limit1,
    input  wire [DIM_WIDTH-1:0] cfg_limit2,
    input  wire [DIM_WIDTH-1:0] cfg_limit3,
    input  wire signed [ADDR_WIDTH-1:0] cfg_stride0,
    input  wire signed [ADDR_WIDTH-1:0] cfg_stride1,
    input  wire signed [ADDR_WIDTH-1:0] cfg_stride2,
    input  wire signed [ADDR_WIDTH-1:0] cfg_stride3,

    input  wire start,
    input  wire step_ready,
    output reg  step_valid,
    output reg  [ADDR_WIDTH-1:0] addr,
    output reg  done
);
    reg [ADDR_WIDTH-1:0] base;
    reg [DIM_WIDTH-1:0] limit0, limit1, limit2, limit3;
    reg signed [ADDR_WIDTH-1:0] stride0, stride1, stride2, stride3;
    reg [DIM_WIDTH-1:0] i0, i1, i2, i3;
    reg running;

    always @(posedge clk) begin
        if (rst) begin
            base <= {ADDR_WIDTH{1'b0}};
            limit0 <= {DIM_WIDTH{1'b0}};
            limit1 <= {DIM_WIDTH{1'b0}};
            limit2 <= {DIM_WIDTH{1'b0}};
            limit3 <= {DIM_WIDTH{1'b0}};
            stride0 <= {ADDR_WIDTH{1'b0}};
            stride1 <= {ADDR_WIDTH{1'b0}};
            stride2 <= {ADDR_WIDTH{1'b0}};
            stride3 <= {ADDR_WIDTH{1'b0}};
            i0 <= {DIM_WIDTH{1'b0}};
            i1 <= {DIM_WIDTH{1'b0}};
            i2 <= {DIM_WIDTH{1'b0}};
            i3 <= {DIM_WIDTH{1'b0}};
            running <= 1'b0;
            step_valid <= 1'b0;
            addr <= {ADDR_WIDTH{1'b0}};
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (cfg_valid) begin
                base <= cfg_base;
                limit0 <= cfg_limit0;
                limit1 <= cfg_limit1;
                limit2 <= cfg_limit2;
                limit3 <= cfg_limit3;
                stride0 <= cfg_stride0;
                stride1 <= cfg_stride1;
                stride2 <= cfg_stride2;
                stride3 <= cfg_stride3;
            end

            if (start) begin
                i0 <= {DIM_WIDTH{1'b0}};
                i1 <= {DIM_WIDTH{1'b0}};
                i2 <= {DIM_WIDTH{1'b0}};
                i3 <= {DIM_WIDTH{1'b0}};
                running <= 1'b1;
                step_valid <= 1'b1;
                addr <= cfg_valid ? cfg_base : base;
            end else if (running && step_valid && step_ready) begin
                if (i3 + 1 < limit3) begin
                    i3 <= i3 + 1'b1;
                    addr <= addr + stride3;
                end else begin
                    i3 <= {DIM_WIDTH{1'b0}};
                    if (i2 + 1 < limit2) begin
                        i2 <= i2 + 1'b1;
                        addr <= base + (i0 * stride0) + (i1 * stride1) + ((i2 + 1'b1) * stride2);
                    end else begin
                        i2 <= {DIM_WIDTH{1'b0}};
                        if (i1 + 1 < limit1) begin
                            i1 <= i1 + 1'b1;
                            addr <= base + (i0 * stride0) + ((i1 + 1'b1) * stride1);
                        end else begin
                            i1 <= {DIM_WIDTH{1'b0}};
                            if (i0 + 1 < limit0) begin
                                i0 <= i0 + 1'b1;
                                addr <= base + ((i0 + 1'b1) * stride0);
                            end else begin
                                running <= 1'b0;
                                step_valid <= 1'b0;
                                done <= 1'b1;
                            end
                        end
                    end
                end
            end
        end
    end
endmodule
