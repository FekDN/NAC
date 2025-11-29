`timescale 1ns / 1ps

module NAC_AXI_Master_Adapter #(
    parameter C_M_AXI_ADDR_WIDTH = 40,
    parameter C_M_AXI_DATA_WIDTH = 32
)(
    input  wire M_AXI_ACLK,
    input  wire M_AXI_ARESETN,

    // Internal Enhanced Interface
    input  wire [31:0] sys_addr,
    input  wire [7:0]  sys_len,     // Burst Length (0 = Single, N = N+1 beats)
    input  wire        sys_req,     // Start Transaction
    input  wire        sys_we,      // Write Enable
    input  wire [31:0] sys_wdata,   // Write Data Stream
    input  wire        sys_wvalid,  // Write Data Valid (from internal FIFO)
    output reg         sys_wready,  // Ready to accept Write Data
    
    output reg         sys_grant,   // Command Accepted (Address Phase Done)
    output reg         sys_valid,   // Read Data Valid
    output reg         sys_last,    // High on last beat (Read or Write done)
    output reg  [31:0] sys_rdata,
    output reg         sys_error,

    // AXI4 Master Interface
    output reg [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output reg [7:0]                    m_axi_awlen, // Dynamic Length
    output reg                          m_axi_awvalid,
    input  wire                         m_axi_awready,
    output reg [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output reg [C_M_AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output reg                          m_axi_wlast,
    output reg                          m_axi_wvalid,
    input  wire                         m_axi_wready,
    input  wire [1:0]                   m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output reg                          m_axi_bready,
    
    output reg [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output reg [7:0]                    m_axi_arlen,   
    output reg                          m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output reg                          m_axi_rready,
    
    // Constant Sideband
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst
);

    assign m_axi_awsize  = 3'b010; // 4 bytes
    assign m_axi_awburst = 2'b01; // INCR
    assign m_axi_arsize  = 3'b010;
    assign m_axi_arburst = 2'b01; 

    localparam S_IDLE  = 3'd0;
    localparam S_READ  = 3'd1;
    localparam S_WRITE_ADDR = 3'd2;
    localparam S_WRITE_DATA = 3'd3;
    localparam S_WRITE_RESP = 3'd4;

    reg [2:0] state;
    reg [7:0] write_cnt;

    always @(posedge M_AXI_ACLK) begin
        if (!M_AXI_ARESETN) begin
            state <= S_IDLE;
            sys_grant <= 0; sys_valid <= 0; sys_last <= 0; sys_error <= 0; sys_wready <= 0;
            m_axi_awvalid <= 0; m_axi_wvalid <= 0; m_axi_arvalid <= 0; m_axi_rready <= 0; m_axi_bready <= 0;
            write_cnt <= 0;
        end else begin
            sys_grant <= 0; sys_valid <= 0; sys_last <= 0;

            case (state)
                S_IDLE: begin
                    if (sys_req) begin
                        sys_error <= 0;
                        if (sys_we) begin
                            // WRITE BURST START
                            m_axi_awaddr <= {8'h00, sys_addr};
                            m_axi_awlen  <= sys_len;
                            m_axi_awvalid <= 1;
                            write_cnt <= 0;
                            state <= S_WRITE_ADDR;
                        end else begin
                            // READ BURST START
                            m_axi_araddr <= {8'h00, sys_addr};
                            m_axi_arlen  <= sys_len; 
                            m_axi_arvalid <= 1;
                            m_axi_rready <= 1;
                            state <= S_READ;
                        end
                    end
                end

                // --- WRITE PATH ---
                S_WRITE_ADDR: begin
                    if (m_axi_awready && m_axi_awvalid) begin
                        m_axi_awvalid <= 0;
                        sys_wready <= 1; // Ask for first data
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    // Streaming data from internal FIFO to AXI
                    if (sys_wvalid && !m_axi_wvalid) begin
                        m_axi_wdata <= sys_wdata;
                        m_axi_wstrb <= 4'hF;
                        m_axi_wvalid <= 1;
                        m_axi_wlast <= (write_cnt == m_axi_awlen);
                        sys_wready <= 0; // Hold until AXI accepts
                    end

                    if (m_axi_wready && m_axi_wvalid) begin
                        m_axi_wvalid <= 0;
                        if (m_axi_wlast) begin
                            m_axi_bready <= 1;
                            state <= S_WRITE_RESP;
                        end else begin
                            write_cnt <= write_cnt + 1;
                            sys_wready <= 1; // Ready for next word
                        end
                    end
                end

                S_WRITE_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 0;
                        sys_grant <= 1; // Burst Complete
                        if (m_axi_bresp != 2'b00) sys_error <= 1;
                        state <= S_IDLE;
                    end
                end

                // --- READ PATH ---
                S_READ: begin
                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 0;
                        sys_grant <= 1; // Address Phase Done
                    end
                    if (m_axi_rvalid && m_axi_rready) begin
                        sys_rdata <= m_axi_rdata;
                        sys_valid <= 1; 
                        if (m_axi_rresp != 2'b00) sys_error <= 1;
                        if (m_axi_rlast) begin
                            sys_last <= 1;
                            m_axi_rready <= 0; 
                            state <= S_IDLE;
                        end
                    end
                end
            endcase
        end
    end
endmodule