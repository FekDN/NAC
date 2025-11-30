`timescale 1ns / 1ps

module NAC_AXILite_Slave # (
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
)(
    // Global Clock and Reset
    input wire  S_AXI_ACLK,
    input wire  S_AXI_ARESETN,

    // Write Address Channel
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    input wire [2 : 0] S_AXI_AWPROT, // Ignored in Lite
    input wire  S_AXI_AWVALID,
    output wire  S_AXI_AWREADY,

    // Write Data Channel
    input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    input wire  S_AXI_WVALID,
    output wire  S_AXI_WREADY,

    // Write Response Channel
    output wire [1 : 0] S_AXI_BRESP,
    output wire  S_AXI_BVALID,
    input wire  S_AXI_BREADY,

    // Read Address Channel
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    input wire [2 : 0] S_AXI_ARPROT, // Ignored in Lite
    input wire  S_AXI_ARVALID,
    output wire  S_AXI_ARREADY,

    // Read Data Channel
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    output wire [1 : 0] S_AXI_RRESP,
    output wire  S_AXI_RVALID,
    input wire  S_AXI_RREADY,

    // ========================================================================
    // User Ports (Connections to NAC_Processor_Top logic)
    // ========================================================================
    
    // Control / Status
    output reg         slv_start_pulse,    // Pulse 1 cycle when writing 1 to Reg0
    input  wire [31:0] slv_status_reg,     // Read-only status from FSM (Reg1)

    // Configuration Pointers (Read/Write Registers)
    output wire [31:0] slv_ptr_registry,   // Reg2 (0x08)
    output wire [31:0] slv_ptr_code,       // Reg3 (0x0C)
    output wire [31:0] slv_ptr_weights,    // Reg4 (0x10)
    output wire [31:0] slv_ptr_input,      // Reg5 (0x14)
    output wire [31:0] slv_ptr_output,     // Reg6 (0x18)
    output wire [31:0] slv_ptr_opmap,      // Reg7 (0x1C)
    output wire [31:0] slv_ptr_varmap      // Reg8 (0x20)
);

    // AXI4LITE signals
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_awaddr;
    reg  axi_awready;
    reg  axi_wready;
    reg [1 : 0] axi_bresp;
    reg  axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_araddr;
    reg  axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1 : 0] axi_rdata;
    reg [1 : 0] axi_rresp;
    reg  axi_rvalid;

    // Example-specific design signals
    // local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
    // ADDR_LSB is used for addressing 32/64 bit registers/memories
    // ADDR_LSB = 2 for 32 bits (n downto 2)
    // ADDR_LSB = 3 for 64 bits (n downto 3)
    localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
    localparam integer OPT_MEM_ADDR_BITS = 3; // Covers up to 16 registers (2^4)

    //----------------------------------------------------
    //-- Internal Registers
    //----------------------------------------------------
    // Index mapping based on nac_defines.vh:
    // 0: REG_CMD       (0x00) -> slv_reg0 (Pulse Gen)
    // 1: REG_STATUS    (0x04) -> Read Only (from slv_status_reg)
    // 2: REG_REGISTRY  (0x08) -> slv_reg2
    // 3: REG_CODE      (0x0C) -> slv_reg3
    // 4: REG_WEIGHTS   (0x10) -> slv_reg4
    // 5: REG_INPUT     (0x14) -> slv_reg5
    // 6: REG_OUTPUT    (0x18) -> slv_reg6
    // 7: REG_OPMAP     (0x1C) -> slv_reg7
    // 8: REG_VARMAP    (0x20) -> slv_reg8
    
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg0; // Command Storage (Optional)
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg2;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg3;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg4;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg5;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg6;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg7;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg8;
    
    wire slv_reg_rden;
    wire slv_reg_wren;
    reg [C_S_AXI_DATA_WIDTH-1:0] reg_data_out;
    integer byte_index;
    reg aw_en;

    // I/O Connections assignments
    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    // Output assignments
    assign slv_ptr_registry = slv_reg2;
    assign slv_ptr_code     = slv_reg3;
    assign slv_ptr_weights  = slv_reg4;
    assign slv_ptr_input    = slv_reg5;
    assign slv_ptr_output   = slv_reg6;
    assign slv_ptr_opmap    = slv_reg7;
    assign slv_ptr_varmap   = slv_reg8;

    // Write Address Ready (AWREADY) generation
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end else begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
                axi_awready <= 1'b1;
                aw_en <= 1'b0;
            end else if (S_AXI_BREADY && axi_bvalid) begin
                aw_en <= 1'b1;
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    // Address latching
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_awaddr <= 0;
        end else begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
                axi_awaddr <= S_AXI_AWADDR;
            end
        end
    end

    // Write Data Ready (WREADY) generation
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_wready <= 1'b0;
        end else begin
            if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end

    // Write Response Logic
    assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            slv_reg0 <= 0; slv_reg2 <= 0; slv_reg3 <= 0;
            slv_reg4 <= 0; slv_reg5 <= 0; slv_reg6 <= 0;
            slv_reg7 <= 0; slv_reg8 <= 0;
            slv_start_pulse <= 0;
        end else begin
            // Pulse default low
            slv_start_pulse <= 0; 
            
            if (slv_reg_wren) begin
                // Decoding: bits [5:2] (assuming 32-bit width => LSB=2)
                case ( axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS : ADDR_LSB] )
                    4'h0: begin // 0x00: Command Register
                        slv_reg0 <= S_AXI_WDATA;
                        // Trigger start if Bit 0 is written as 1
                        if (S_AXI_WDATA[0] == 1'b1) slv_start_pulse <= 1;
                    end
                    // 4'h1: 0x04 Status Register is Read-Only, write ignored
                    4'h2: slv_reg2 <= S_AXI_WDATA; // 0x08
                    4'h3: slv_reg3 <= S_AXI_WDATA; // 0x0C
                    4'h4: slv_reg4 <= S_AXI_WDATA; // 0x10
                    4'h5: slv_reg5 <= S_AXI_WDATA; // 0x14
                    4'h6: slv_reg6 <= S_AXI_WDATA; // 0x18
                    4'h7: slv_reg7 <= S_AXI_WDATA; // 0x1C
                    4'h8: slv_reg8 <= S_AXI_WDATA; // 0x20
                    default: begin
                        // Unknown address, do nothing
                    end
                endcase
            end
        end
    end

    // Write Response (BVALID/BRESP)
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_bvalid  <= 0;
            axi_bresp   <= 2'b0;
        end else begin
            if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b0; // 'OKAY' response
            end else if (S_AXI_BREADY && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // Read Address Ready (ARREADY)
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 32'b0;
        end else begin
            if (~axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_araddr  <= S_AXI_ARADDR;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end

    // Read Data Valid (RVALID) and Response
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_rvalid <= 0;
            axi_rresp  <= 0;
        end else begin
            if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b0; // 'OKAY'
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    // Memory Mapped Register Select and Read Logic
    assign slv_reg_rden = axi_arready & S_AXI_ARVALID & ~axi_rvalid;

    always @(*) begin
        // Address decoding for reading
        case ( axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS : ADDR_LSB] )
            4'h0   : reg_data_out <= slv_reg0;       // 0x00
            4'h1   : reg_data_out <= slv_status_reg; // 0x04 (Input from Logic)
            4'h2   : reg_data_out <= slv_reg2;       // 0x08
            4'h3   : reg_data_out <= slv_reg3;       // 0x0C
            4'h4   : reg_data_out <= slv_reg4;       // 0x10
            4'h5   : reg_data_out <= slv_reg5;       // 0x14
            4'h6   : reg_data_out <= slv_reg6;       // 0x18
            4'h7   : reg_data_out <= slv_reg7;       // 0x1C
            4'h8   : reg_data_out <= slv_reg8;       // 0x20
            default : reg_data_out <= 0;
        endcase
    end

    // Output Register generation
    always @( posedge S_AXI_ACLK ) begin
        if ( S_AXI_ARESETN == 1'b0 ) begin
            axi_rdata  <= 0;
        end else begin
            if (slv_reg_rden) begin
                axi_rdata <= reg_data_out;
            end
        end
    end

endmodule