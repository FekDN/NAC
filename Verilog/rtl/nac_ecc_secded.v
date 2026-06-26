module nac_ecc_secded #(
    parameter DATA_WIDTH = 32,
    parameter ECC_WIDTH = (DATA_WIDTH <= 32) ? 7 : 8
) (
    input  wire [DATA_WIDTH-1:0] data_in,
    input  wire [ECC_WIDTH-1:0] ecc_in,
    output wire [ECC_WIDTH-1:0] ecc_out,
    output reg  [DATA_WIDTH-1:0] data_corrected,
    output reg  single_error,
    output reg  double_error
);
    localparam HAM_BITS = ECC_WIDTH - 1;
    localparam CODE_BITS = DATA_WIDTH + HAM_BITS;

    function is_power_two;
        input integer value;
        begin
            is_power_two = (value > 0) && ((value & (value - 1)) == 0);
        end
    endfunction

    function integer code_position_for_data;
        input integer target_data_index;
        integer pos;
        integer data_index;
        begin
            code_position_for_data = 0;
            data_index = 0;
            for (pos = 1; pos <= CODE_BITS; pos = pos + 1) begin
                if (!is_power_two(pos)) begin
                    if (data_index == target_data_index) begin
                        code_position_for_data = pos;
                    end
                    data_index = data_index + 1;
                end
            end
        end
    endfunction

    function integer data_index_for_code_position;
        input integer wanted_pos;
        integer pos;
        integer data_index;
        begin
            data_index_for_code_position = -1;
            data_index = 0;
            for (pos = 1; pos <= CODE_BITS; pos = pos + 1) begin
                if (!is_power_two(pos)) begin
                    if (pos == wanted_pos) begin
                        data_index_for_code_position = data_index;
                    end
                    data_index = data_index + 1;
                end
            end
        end
    endfunction

    function [ECC_WIDTH-1:0] encode_ecc;
        input [DATA_WIDTH-1:0] data;
        integer bit_idx;
        integer parity_idx;
        integer code_pos;
        reg [HAM_BITS-1:0] ham;
        reg overall;
        begin
            ham = {HAM_BITS{1'b0}};
            for (parity_idx = 0; parity_idx < HAM_BITS; parity_idx = parity_idx + 1) begin
                for (bit_idx = 0; bit_idx < DATA_WIDTH; bit_idx = bit_idx + 1) begin
                    code_pos = code_position_for_data(bit_idx);
                    if (((code_pos >> parity_idx) & 1) != 0) begin
                        ham[parity_idx] = ham[parity_idx] ^ data[bit_idx];
                    end
                end
            end
            overall = ^{data, ham};
            encode_ecc = {overall, ham};
        end
    endfunction

    assign ecc_out = encode_ecc(data_in);

    integer syndrome_idx;
    integer corrected_index;
    reg [HAM_BITS-1:0] expected_ham;
    reg [HAM_BITS-1:0] syndrome;
    reg received_overall;
    reg overall_mismatch;

    always @* begin
        {received_overall, expected_ham} = encode_ecc(data_in);
        syndrome = expected_ham ^ ecc_in[HAM_BITS-1:0];
        received_overall = ^{data_in, ecc_in[HAM_BITS-1:0]};
        overall_mismatch = received_overall ^ ecc_in[ECC_WIDTH-1];
        data_corrected = data_in;
        single_error = 1'b0;
        double_error = 1'b0;

        if (syndrome == {HAM_BITS{1'b0}} && !overall_mismatch) begin
            single_error = 1'b0;
            double_error = 1'b0;
        end else if (syndrome == {HAM_BITS{1'b0}} && overall_mismatch) begin
            single_error = 1'b1;
        end else if (overall_mismatch) begin
            single_error = 1'b1;
            corrected_index = data_index_for_code_position(syndrome);
            if (corrected_index >= 0 && corrected_index < DATA_WIDTH) begin
                data_corrected[corrected_index] = ~data_in[corrected_index];
            end
        end else begin
            double_error = 1'b1;
        end
    end
endmodule
