// Sequential scalar ALU for MEP context operations.
//
// 64-bit multiply/divide are intentionally iterative. A single-cycle
// combinational divider is not a realistic FPGA timing target, and a 64x64
// combinational multiplier would consume a large DSP/LUT footprint.
module nac_scalar_alu #(
    parameter WIDTH = 64
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire is_compare,
    input  wire [7:0] op_type,
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    output reg  busy,
    output reg  done,
    output reg  error,
    output reg  [WIDTH-1:0] result,
    output reg  result_is_bool
);
    localparam S_IDLE = 2'd0;
    localparam S_MUL  = 2'd1;
    localparam S_DIV  = 2'd2;

    reg [1:0] state;
    reg result_negative;
    reg [WIDTH-1:0] multiplicand;
    reg [WIDTH-1:0] multiplier;
    reg [WIDTH-1:0] product;
    reg [WIDTH-1:0] divisor;
    reg [WIDTH-1:0] dividend_shift;
    reg [WIDTH:0] remainder;
    reg [WIDTH-1:0] quotient;
    reg [7:0] count;

    wire [WIDTH-1:0] abs_a = a[WIDTH-1] ? (~a + {{(WIDTH-1){1'b0}}, 1'b1}) : a;
    wire [WIDTH-1:0] abs_b = b[WIDTH-1] ? (~b + {{(WIDTH-1){1'b0}}, 1'b1}) : b;

    reg [WIDTH:0] div_remainder_next;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            result <= {WIDTH{1'b0}};
            result_is_bool <= 1'b0;
            result_negative <= 1'b0;
            multiplicand <= {WIDTH{1'b0}};
            multiplier <= {WIDTH{1'b0}};
            product <= {WIDTH{1'b0}};
            divisor <= {WIDTH{1'b0}};
            dividend_shift <= {WIDTH{1'b0}};
            remainder <= {(WIDTH+1){1'b0}};
            quotient <= {WIDTH{1'b0}};
            count <= 8'd0;
        end else begin
            done <= 1'b0;
            error <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        result_is_bool <= is_compare;
                        if (is_compare) begin
                            case (op_type)
                                8'd0: result <= {{(WIDTH-1){1'b0}}, (a == b)};
                                8'd1: result <= {{(WIDTH-1){1'b0}}, (a != b)};
                                8'd2: result <= {{(WIDTH-1){1'b0}}, (a > b)};
                                8'd3: result <= {{(WIDTH-1){1'b0}}, (a < b)};
                                default: begin
                                    result <= {WIDTH{1'b0}};
                                    error <= 1'b1;
                                end
                            endcase
                            done <= 1'b1;
                            busy <= 1'b0;
                        end else begin
                            case (op_type)
                                8'd0: begin
                                    result <= a + b;
                                    done <= 1'b1;
                                    busy <= 1'b0;
                                end
                                8'd1: begin
                                    result <= a - b;
                                    done <= 1'b1;
                                    busy <= 1'b0;
                                end
                                8'd2: begin
                                    result_negative <= a[WIDTH-1] ^ b[WIDTH-1];
                                    multiplicand <= abs_a;
                                    multiplier <= abs_b;
                                    product <= {WIDTH{1'b0}};
                                    count <= WIDTH[7:0];
                                    state <= S_MUL;
                                end
                                8'd3: begin
                                    if (b == 0) begin
                                        result <= {WIDTH{1'b0}};
                                        done <= 1'b1;
                                        busy <= 1'b0;
                                    end else begin
                                        result_negative <= a[WIDTH-1] ^ b[WIDTH-1];
                                        divisor <= abs_b;
                                        dividend_shift <= abs_a;
                                        remainder <= {(WIDTH+1){1'b0}};
                                        quotient <= {WIDTH{1'b0}};
                                        count <= WIDTH[7:0];
                                        state <= S_DIV;
                                    end
                                end
                                8'd5: begin
                                    result <= (a > b) ? a : b;
                                    done <= 1'b1;
                                    busy <= 1'b0;
                                end
                                8'd6: begin
                                    result <= (a < b) ? a : b;
                                    done <= 1'b1;
                                    busy <= 1'b0;
                                end
                                default: begin
                                    result <= {WIDTH{1'b0}};
                                    error <= 1'b1;
                                    done <= 1'b1;
                                    busy <= 1'b0;
                                end
                            endcase
                        end
                    end
                end

                S_MUL: begin
                    if (multiplier[0])
                        product <= product + multiplicand;
                    multiplicand <= multiplicand << 1;
                    multiplier <= multiplier >> 1;
                    count <= count - 8'd1;
                    if (count == 8'd1) begin
                        state <= S_IDLE;
                        busy <= 1'b0;
                        done <= 1'b1;
                        result <= result_negative ? (~(product + (multiplier[0] ? multiplicand : {WIDTH{1'b0}})) + {{(WIDTH-1){1'b0}}, 1'b1}) :
                                                     (product + (multiplier[0] ? multiplicand : {WIDTH{1'b0}}));
                    end
                end

                S_DIV: begin
                    div_remainder_next = {remainder[WIDTH-1:0], dividend_shift[WIDTH-1]};
                    dividend_shift <= dividend_shift << 1;
                    quotient <= quotient << 1;
                    if (div_remainder_next >= {1'b0, divisor}) begin
                        remainder <= div_remainder_next - {1'b0, divisor};
                        quotient[0] <= 1'b1;
                    end else begin
                        remainder <= div_remainder_next;
                    end
                    count <= count - 8'd1;
                    if (count == 8'd1) begin
                        state <= S_IDLE;
                        busy <= 1'b0;
                        done <= 1'b1;
                        result <= result_negative ? (~((quotient << 1) |
                                  ((div_remainder_next >= {1'b0, divisor}) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}})) +
                                  {{(WIDTH-1){1'b0}}, 1'b1}) :
                                  ((quotient << 1) |
                                  ((div_remainder_next >= {1'b0, divisor}) ? {{(WIDTH-1){1'b0}}, 1'b1} : {WIDTH{1'b0}}));
                    end
                end

                default: begin
                    state <= S_IDLE;
                    busy <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                end
            endcase
        end
    end
endmodule
