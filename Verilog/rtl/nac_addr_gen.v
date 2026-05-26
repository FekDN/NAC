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
    output reg  done,
    output reg  error
);
    localparam CALC_WIDTH = ADDR_WIDTH + DIM_WIDTH + 3;

    reg [ADDR_WIDTH-1:0] base;
    reg [DIM_WIDTH-1:0] limit0, limit1, limit2, limit3;
    reg signed [ADDR_WIDTH-1:0] stride0, stride1, stride2, stride3;
    reg [DIM_WIDTH-1:0] i0, i1, i2, i3;
    reg running;
    reg signed [CALC_WIDTH-1:0] next_addr_calc;

    function signed [CALC_WIDTH-1:0] addr_from_indices;
        input [DIM_WIDTH-1:0] idx0;
        input [DIM_WIDTH-1:0] idx1;
        input [DIM_WIDTH-1:0] idx2;
        input [DIM_WIDTH-1:0] idx3;
        reg signed [CALC_WIDTH-1:0] base_ext;
        reg signed [CALC_WIDTH-1:0] idx0_ext;
        reg signed [CALC_WIDTH-1:0] idx1_ext;
        reg signed [CALC_WIDTH-1:0] idx2_ext;
        reg signed [CALC_WIDTH-1:0] idx3_ext;
        reg signed [CALC_WIDTH-1:0] stride0_ext;
        reg signed [CALC_WIDTH-1:0] stride1_ext;
        reg signed [CALC_WIDTH-1:0] stride2_ext;
        reg signed [CALC_WIDTH-1:0] stride3_ext;
        begin
            base_ext = {{(CALC_WIDTH-ADDR_WIDTH){1'b0}}, base};
            idx0_ext = {{(CALC_WIDTH-DIM_WIDTH){1'b0}}, idx0};
            idx1_ext = {{(CALC_WIDTH-DIM_WIDTH){1'b0}}, idx1};
            idx2_ext = {{(CALC_WIDTH-DIM_WIDTH){1'b0}}, idx2};
            idx3_ext = {{(CALC_WIDTH-DIM_WIDTH){1'b0}}, idx3};
            stride0_ext = {{(CALC_WIDTH-ADDR_WIDTH){stride0[ADDR_WIDTH-1]}}, stride0};
            stride1_ext = {{(CALC_WIDTH-ADDR_WIDTH){stride1[ADDR_WIDTH-1]}}, stride1};
            stride2_ext = {{(CALC_WIDTH-ADDR_WIDTH){stride2[ADDR_WIDTH-1]}}, stride2};
            stride3_ext = {{(CALC_WIDTH-ADDR_WIDTH){stride3[ADDR_WIDTH-1]}}, stride3};
            addr_from_indices = base_ext +
                                (idx0_ext * stride0_ext) +
                                (idx1_ext * stride1_ext) +
                                (idx2_ext * stride2_ext) +
                                (idx3_ext * stride3_ext);
        end
    endfunction

    function addr_in_range;
        input signed [CALC_WIDTH-1:0] value;
        begin
            addr_in_range = !value[CALC_WIDTH-1] &&
                            (value[CALC_WIDTH-1:ADDR_WIDTH] == {(CALC_WIDTH-ADDR_WIDTH){1'b0}});
        end
    endfunction

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
            error <= 1'b0;
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
                error <= 1'b0;
            end

            if (start) begin
                i0 <= {DIM_WIDTH{1'b0}};
                i1 <= {DIM_WIDTH{1'b0}};
                i2 <= {DIM_WIDTH{1'b0}};
                i3 <= {DIM_WIDTH{1'b0}};
                error <= 1'b0;
                if ((cfg_valid ? cfg_limit0 : limit0) == 0 ||
                    (cfg_valid ? cfg_limit1 : limit1) == 0 ||
                    (cfg_valid ? cfg_limit2 : limit2) == 0 ||
                    (cfg_valid ? cfg_limit3 : limit3) == 0) begin
                    running <= 1'b0;
                    step_valid <= 1'b0;
                    done <= 1'b1;
                end else begin
                    running <= 1'b1;
                    step_valid <= 1'b1;
                    addr <= cfg_valid ? cfg_base : base;
                end
            end else if (running && step_valid && step_ready) begin
                if (i3 + 1 < limit3) begin
                    next_addr_calc = addr_from_indices(i0, i1, i2, i3 + 1'b1);
                    if (!addr_in_range(next_addr_calc)) begin
                        running <= 1'b0;
                        step_valid <= 1'b0;
                        error <= 1'b1;
                        done <= 1'b1;
                    end else begin
                        i3 <= i3 + 1'b1;
                        addr <= next_addr_calc[ADDR_WIDTH-1:0];
                    end
                end else begin
                    i3 <= {DIM_WIDTH{1'b0}};
                    if (i2 + 1 < limit2) begin
                        next_addr_calc = addr_from_indices(i0, i1, i2 + 1'b1, {DIM_WIDTH{1'b0}});
                        if (!addr_in_range(next_addr_calc)) begin
                            running <= 1'b0;
                            step_valid <= 1'b0;
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            i2 <= i2 + 1'b1;
                            addr <= next_addr_calc[ADDR_WIDTH-1:0];
                        end
                    end else begin
                        i2 <= {DIM_WIDTH{1'b0}};
                        if (i1 + 1 < limit1) begin
                            next_addr_calc = addr_from_indices(i0, i1 + 1'b1, {DIM_WIDTH{1'b0}}, {DIM_WIDTH{1'b0}});
                            if (!addr_in_range(next_addr_calc)) begin
                                running <= 1'b0;
                                step_valid <= 1'b0;
                                error <= 1'b1;
                                done <= 1'b1;
                            end else begin
                                i1 <= i1 + 1'b1;
                                addr <= next_addr_calc[ADDR_WIDTH-1:0];
                            end
                        end else begin
                            i1 <= {DIM_WIDTH{1'b0}};
                            if (i0 + 1 < limit0) begin
                                next_addr_calc = addr_from_indices(i0 + 1'b1, {DIM_WIDTH{1'b0}}, {DIM_WIDTH{1'b0}}, {DIM_WIDTH{1'b0}});
                                if (!addr_in_range(next_addr_calc)) begin
                                    running <= 1'b0;
                                    step_valid <= 1'b0;
                                    error <= 1'b1;
                                    done <= 1'b1;
                                end else begin
                                    i0 <= i0 + 1'b1;
                                    addr <= next_addr_calc[ADDR_WIDTH-1:0];
                                end
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
