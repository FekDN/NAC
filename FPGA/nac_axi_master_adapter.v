`timescale 1ns / 1ps

module NAC_AXI_Master_Adapter #(
    parameter C_M_AXI_ADDR_WIDTH = 40,
    parameter C_M_AXI_DATA_WIDTH = 32
)(
    input  wire M_AXI_ACLK,
    input  wire M_AXI_ARESETN,

    // ========================================================================
    // Internal Enhanced Interface (Synchronous to M_AXI_ACLK)
    // ========================================================================
    // Control
    input  wire [31:0] sys_addr,
    input  wire [7:0]  sys_len,     // Burst Length: 0=1 beat, 15=16 beats
    input  wire        sys_req,     // Start Transaction Strobe
    input  wire        sys_we,      // 1 = Write, 0 = Read

    // Write Data Stream (FIFO interface)
    input  wire [C_M_AXI_DATA_WIDTH-1:0] sys_wdata,
    input  wire        sys_wvalid,
    output reg         sys_wready,  // Combinatorial/Fast Ready

    // Read Data Stream / Status
    output reg         sys_grant,   // 1 = Command Accepted (Addr Phase Done), ready for next req
    output reg         sys_valid,   // 1 = Read Data Valid
    output reg         sys_last,    // 1 = Last beat of Read or Write transaction
    output reg  [C_M_AXI_DATA_WIDTH-1:0] sys_rdata,
    output reg         sys_error,   // 1 = AXI Error Response received

    // ========================================================================
    // AXI4 Master Interface
    // ========================================================================
    // Write Address
    output reg [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output reg [7:0]                    m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,
    output reg                          m_axi_awvalid,
    input  wire                         m_axi_awready,
    
    // Write Data
    output reg [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output reg [C_M_AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output reg                          m_axi_wlast,
    output reg                          m_axi_wvalid,
    input  wire                         m_axi_wready,
    
    // Write Response
    input  wire [1:0]                   m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output reg                          m_axi_bready,
    
    // Read Address
    output reg [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output reg [7:0]                    m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    output reg                          m_axi_arvalid,
    input  wire                         m_axi_arready,
    
    // Read Data
    input  wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output reg                          m_axi_rready
);

    // ========================================================================
    // Parameters & Constants
    // ========================================================================
    // Calculate AXI AxSIZE based on Data Width
    // 32-bit -> 2 (4 bytes), 64-bit -> 3 (8 bytes), 128-bit -> 4 (16 bytes)
    localparam [2:0] C_BURST_SIZE = (C_M_AXI_DATA_WIDTH == 128) ? 3'b100 :
                                    (C_M_AXI_DATA_WIDTH == 64)  ? 3'b011 : 
                                    3'b010;

    assign m_axi_awsize  = C_BURST_SIZE;
    assign m_axi_awburst = 2'b01; // INCR type
    assign m_axi_arsize  = C_BURST_SIZE;
    assign m_axi_arburst = 2'b01; // INCR type

    // FSM States
    localparam S_IDLE       = 3'd0;
    localparam S_ADDR_PHASE = 3'd1; // Sending AW or AR
    localparam S_WRITE_DATA = 3'd2; // Sending W data
    localparam S_WRITE_RESP = 3'd3; // Waiting for B response
    localparam S_READ_DATA  = 3'd4; // Receiving R data

    reg [2:0] state;
    reg [7:0] beat_cnt;
    
    // ========================================================================
    // Main FSM
    // ========================================================================
    always @(posedge M_AXI_ACLK) begin
        if (!M_AXI_ARESETN) begin
            state <= S_IDLE;
            sys_grant <= 0; sys_error <= 0; sys_last <= 0;
            m_axi_awvalid <= 0; m_axi_arvalid <= 0; m_axi_wvalid <= 0; 
            m_axi_bready <= 0; m_axi_rready <= 0;
            beat_cnt <= 0;
        end else begin
            // Default Pulses
            sys_grant <= 0;
            sys_last <= 0; 

            case (state)
                S_IDLE: begin
                    sys_error <= 0;
                    if (sys_req) begin
                        if (sys_we) begin
                            // Setup Write Address
                            m_axi_awaddr  <= {8'h00, sys_addr}; // Zero-pad high bits
                            m_axi_awlen   <= sys_len;
                            m_axi_awvalid <= 1;
                            
                            // Setup Write Data Counters
                            beat_cnt <= 0;
                            
                            state <= S_ADDR_PHASE;
                        end else begin
                            // Setup Read Address
                            m_axi_araddr  <= {8'h00, sys_addr};
                            m_axi_arlen   <= sys_len;
                            m_axi_arvalid <= 1;
                            m_axi_rready  <= 1; // Ready to accept data immediately
                            
                            state <= S_ADDR_PHASE;
                        end
                    end
                end

                S_ADDR_PHASE: begin
                    // --- Handle Write Address Handshake ---
                    if (sys_we) begin
                        // Transition to Data phase immediately to allow pipelining
                        // (Address and Data can happen in parallel in AXI, but here we sequence start)
                        if (m_axi_awready && m_axi_awvalid) begin
                            m_axi_awvalid <= 0;
                            state <= S_WRITE_DATA;
                        end
                    end 
                    // --- Handle Read Address Handshake ---
                    else begin
                        if (m_axi_arready && m_axi_arvalid) begin
                            m_axi_arvalid <= 0;
                            sys_grant <= 1; // Address accepted, read stream starts
                            state <= S_READ_DATA;
                        end
                    end
                end

                S_WRITE_DATA: begin
                    // Logic handled in separate "Write Data Channel" block below
                    // FSM transitions only when all data sent
                    if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
                        m_axi_wvalid <= 0; // Ensure valid drops after last beat
                        m_axi_bready <= 1; // Ready for response
                        state <= S_WRITE_RESP;
                    end
                end

                S_WRITE_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 0;
                        sys_grant <= 1; // Transaction totally done
                        sys_last  <= 1;
                        if (m_axi_bresp != 2'b00) sys_error <= 1;
                        state <= S_IDLE;
                    end
                end

                S_READ_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
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

    // ========================================================================
    // Read Data Path (Pass-through)
    // ========================================================================
    always @(posedge M_AXI_ACLK) begin
        if (!M_AXI_ARESETN) begin
            sys_valid <= 0;
            sys_rdata <= 0;
        end else begin
            // Pure register delay for timing closure
            sys_valid <= (m_axi_rvalid && m_axi_rready);
            sys_rdata <= m_axi_rdata;
        end
    end

    // ========================================================================
    // Write Data Path (Optimized Pipeline)
    // ========================================================================
    
    // We are ready for new data from SYS if:
    // 1. We are in the DATA state (or about to be)
    // 2. AND (The output register is empty OR the AXI slave is accepting data)
    always @(*) begin
        // Simple flow control: If we are in Write state, pass readiness back
        // If m_axi_wvalid is HIGH, we can only accept new data if m_axi_wready is HIGH.
        // If m_axi_wvalid is LOW, we can accept new data to fill the register.
        if (state == S_WRITE_DATA || (state == S_ADDR_PHASE && sys_we)) begin
            sys_wready = (!m_axi_wvalid || m_axi_wready);
        end else begin
            sys_wready = 0;
        end
    end

    always @(posedge M_AXI_ACLK) begin
        if (!M_AXI_ARESETN) begin
            m_axi_wvalid <= 0;
            m_axi_wlast  <= 0;
            m_axi_wdata  <= 0;
            m_axi_wstrb  <= {(C_M_AXI_DATA_WIDTH/8){1'b1}};
            beat_cnt     <= 0;
        end else if (state == S_WRITE_DATA || (state == S_ADDR_PHASE && sys_we && m_axi_awready)) begin
            
            // If the AXI slave accepted the previous data, we assume it's gone
            if (m_axi_wvalid && m_axi_wready) begin
                if (beat_cnt == m_axi_awlen) begin
                    // Last beat sent
                    m_axi_wvalid <= 0; 
                    m_axi_wlast <= 0;
                end else begin
                    beat_cnt <= beat_cnt + 1;
                    // If no new data is ready immediately, drop valid
                    if (!sys_wvalid) m_axi_wvalid <= 0;
                end
            end

            // Load new data into output register
            // Condition: We are allowed to load (sys_wready logic) AND data is available
            if (sys_wvalid && (!m_axi_wvalid || m_axi_wready)) begin
                m_axi_wdata  <= sys_wdata;
                m_axi_wstrb  <= {(C_M_AXI_DATA_WIDTH/8){1'b1}}; // Full word write
                m_axi_wvalid <= 1;
                
                // Look-ahead for LAST signal
                // Current beat_cnt represents the data currently *on the bus* (if valid)
                // or the data *about to be loaded*.
                // Logic: If we are loading the beat that matches awlen, mark wlast.
                if (m_axi_wvalid && m_axi_wready) begin
                    // We are advancing the counter this cycle
                    if ((beat_cnt + 1) == m_axi_awlen) m_axi_wlast <= 1;
                    else m_axi_wlast <= 0;
                end else begin
                    // Counter assumes current holding
                    if (beat_cnt == m_axi_awlen) m_axi_wlast <= 1;
                    else m_axi_wlast <= 0;
                end
            end
        end else if (state == S_IDLE) begin
            beat_cnt <= 0;
            m_axi_wlast <= 0;
            m_axi_wvalid <= 0;
        end
    end

endmodule