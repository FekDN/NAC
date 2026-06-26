module nac_csr_bank #(
    parameter CONTEXTS = 2,
    parameter CONTEXT_BITS = 1
) (
    input  wire clk,
    input  wire rst,
    input  wire clear_context,
    input  wire [CONTEXT_BITS-1:0] clear_context_id,
    input  wire we,
    input  wire [CONTEXT_BITS-1:0] wr_context,
    input  wire [7:0] addr,
    input  wire [63:0] wdata,
    input  wire [CONTEXT_BITS-1:0] rd_context,
    output wire [63:0] rdata,

    output wire [7:0] active_quant_flags,
    output wire [15:0] active_num_inputs,
    output wire [15:0] active_num_outputs,
    output wire [15:0] active_d_model,
    output wire [11*64-1:0] active_section_offsets,
    output wire active_valid
);
    localparam CSR_QUANT_FLAGS = 8'h00;
    localparam CSR_NUM_INPUTS  = 8'h01;
    localparam CSR_NUM_OUTPUTS = 8'h02;
    localparam CSR_D_MODEL     = 8'h03;
    localparam CSR_SECTION0    = 8'h10;
    localparam CSR_VALID       = 8'h7f;

    reg valid_ram [0:CONTEXTS-1];
    reg [7:0] quant_flags_ram [0:CONTEXTS-1];
    reg [15:0] num_inputs_ram [0:CONTEXTS-1];
    reg [15:0] num_outputs_ram [0:CONTEXTS-1];
    reg [15:0] d_model_ram [0:CONTEXTS-1];
    reg [11*64-1:0] section_offsets_ram [0:CONTEXTS-1];

    assign active_quant_flags = quant_flags_ram[rd_context];
    assign active_num_inputs = num_inputs_ram[rd_context];
    assign active_num_outputs = num_outputs_ram[rd_context];
    assign active_d_model = d_model_ram[rd_context];
    assign active_section_offsets = section_offsets_ram[rd_context];
    assign active_valid = valid_ram[rd_context];

    integer i;
    wire rd_is_section = (addr >= CSR_SECTION0 && addr < (CSR_SECTION0 + 8'd11));
    wire [3:0] rd_section_idx = addr - CSR_SECTION0;

    initial begin
        for (i = 0; i < CONTEXTS; i = i + 1) begin
            valid_ram[i] = 1'b0;
            quant_flags_ram[i] = 8'd0;
            num_inputs_ram[i] = 16'd0;
            num_outputs_ram[i] = 16'd0;
            d_model_ram[i] = 16'd0;
            section_offsets_ram[i] = {11*64{1'b0}};
        end
    end

    assign rdata =
        (addr == CSR_QUANT_FLAGS) ? {56'd0, quant_flags_ram[rd_context]} :
        (addr == CSR_NUM_INPUTS)  ? {48'd0, num_inputs_ram[rd_context]} :
        (addr == CSR_NUM_OUTPUTS) ? {48'd0, num_outputs_ram[rd_context]} :
        (addr == CSR_D_MODEL)     ? {48'd0, d_model_ram[rd_context]} :
        rd_is_section             ? section_offsets_ram[rd_context][rd_section_idx*64 +: 64] :
        (addr == CSR_VALID)       ? {63'd0, valid_ram[rd_context]} :
                                    64'd0;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < CONTEXTS; i = i + 1) begin
                valid_ram[i] <= 1'b0;
                quant_flags_ram[i] <= 8'd0;
                num_inputs_ram[i] <= 16'd0;
                num_outputs_ram[i] <= 16'd0;
                d_model_ram[i] <= 16'd0;
                section_offsets_ram[i] <= {11*64{1'b0}};
            end
        end else begin
            if (clear_context) begin
                valid_ram[clear_context_id] <= 1'b0;
                quant_flags_ram[clear_context_id] <= 8'd0;
                num_inputs_ram[clear_context_id] <= 16'd0;
                num_outputs_ram[clear_context_id] <= 16'd0;
                d_model_ram[clear_context_id] <= 16'd0;
                section_offsets_ram[clear_context_id] <= {11*64{1'b0}};
            end

            if (we) begin
                if (addr == CSR_QUANT_FLAGS) begin
                    quant_flags_ram[wr_context] <= wdata[7:0];
                end else if (addr == CSR_NUM_INPUTS) begin
                    num_inputs_ram[wr_context] <= wdata[15:0];
                end else if (addr == CSR_NUM_OUTPUTS) begin
                    num_outputs_ram[wr_context] <= wdata[15:0];
                end else if (addr == CSR_D_MODEL) begin
                    d_model_ram[wr_context] <= wdata[15:0];
                end else if (addr >= CSR_SECTION0 && addr < (CSR_SECTION0 + 8'd11)) begin
                    section_offsets_ram[wr_context][(addr - CSR_SECTION0)*64 +: 64] <= wdata;
                end else if (addr == CSR_VALID) begin
                    valid_ram[wr_context] <= wdata[0];
                end
            end
        end
    end
endmodule
