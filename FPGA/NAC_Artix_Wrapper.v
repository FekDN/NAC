`timescale 1ns / 1ps
`include "nac_defines.vh"

// ============================================================================
// NAC Artix Standalone System Wrapper
// ============================================================================
// This module integrates the NAC Processor Core with essential peripherals
// for a standalone FPGA board (like one based on Artix-7).
//
// System Flow:
// 1. On power-up, the MIG calibrates the DDR3 memory.
// 2. Once DDR3 is ready, the NAC_SD_Loader boots, reads a monolithic
//    binary image from the SD card, and writes it to DDR3 memory.
// 3. After the loader finishes, it de-asserts its `loader_active` signal.
// 4. This triggers the `nac_cmd_start` signal, launching the NAC_Processor_Core.
// 5. The NAC Core executes the neural network model directly from DDR3.
// ============================================================================

module NAC_Artix_Wrapper (
    // System Inputs (from board)
    input  wire       sys_clk_i,    // Main oscillator (e.g., 100 MHz)
    input  wire       sys_rst_n,    // Reset button (Active Low)

    // SD Card Interface (SPI Mode)
    output wire       sd_cs_n,
    output wire       sd_sck,
    output wire       sd_mosi,
    input  wire       sd_miso,
    input  wire       sd_cd_n,      // Card Detect (0 = inserted)

    // DDR3 Physical Interface
    // These pins must match the MIG IP core generated for your specific board.
    output wire [13:0] ddr3_addr,
    output wire [2:0]  ddr3_ba,
    output wire        ddr3_cas_n,
    output wire        ddr3_ck_n,
    output wire        ddr3_ck_p,
    output wire        ddr3_cke,
    output wire        ddr3_ras_n,
    output wire        ddr3_reset_n,
    output wire        ddr3_we_n,
    inout  wire [15:0] ddr3_dq,
    inout  wire [1:0]  ddr3_dqs_n,
    inout  wire [1:0]  ddr3_dqs_p,
    output wire [0:0]  ddr3_odt,
    output wire [1:0]  ddr3_dm,

    // Status LEDs
    output wire [3:0]  led
);

    // ========================================================================
    // 1. Clocking, Reset, and Memory Interface (MIG)
    // ========================================================================
    wire ui_clk;              // User Interface Clock from MIG (e.g., 100 MHz)
    wire ui_clk_sync_rst;     // Synchronous Active High reset from MIG
    wire init_calib_complete; // High when DDR3 is calibrated and ready
    
    // Internal AXI signals forming a common bus to the MIG
    wire [31:0] axi_mem_awaddr;
    wire [7:0]  axi_mem_awlen;
    wire [2:0]  axi_mem_awsize;
    wire [1:0]  axi_mem_awburst;
    wire        axi_mem_awvalid;
    wire        axi_mem_awready;
    wire [31:0] axi_mem_wdata;
    wire [3:0]  axi_mem_wstrb;
    wire        axi_mem_wlast;
    wire        axi_mem_wvalid;
    wire        axi_mem_wready;
    wire [1:0]  axi_mem_bresp;
    wire        axi_mem_bvalid;
    wire        axi_mem_bready;
    wire [31:0] axi_mem_araddr;
    wire [7:0]  axi_mem_arlen;
    wire [2:0]  axi_mem_arsize;
    wire [1:0]  axi_mem_arburst;
    wire        axi_mem_arvalid;
    wire        axi_mem_arready;
    wire [31:0] axi_mem_rdata;
    wire [1:0]  axi_mem_rresp;
    wire        axi_mem_rlast;
    wire        axi_mem_rvalid;
    wire        axi_mem_rready;

    // MIG IP CORE INSTANTIATION
    // IMPORTANT: The instance name ('ddr3_mig') and port connections must
    // match the IP core you generate in Vivado for your target board.
    // Ensure the "AXI4 Interface" is enabled in the MIG configuration.
    ddr3_mig u_mig_7series (
        // Physical DDR3 Interface
        .ddr3_addr          (ddr3_addr),
        .ddr3_ba            (ddr3_ba),
        .ddr3_cas_n         (ddr3_cas_n),
        .ddr3_ck_n          (ddr3_ck_n),
        .ddr3_ck_p          (ddr3_ck_p),
        .ddr3_cke           (ddr3_cke),
        .ddr3_ras_n         (ddr3_ras_n),
        .ddr3_reset_n       (ddr3_reset_n),
        .ddr3_we_n          (ddr3_we_n),
        .ddr3_dq            (ddr3_dq),
        .ddr3_dqs_n         (ddr3_dqs_n),
        .ddr3_dqs_p         (ddr3_dqs_p),
        .ddr3_odt           (ddr3_odt),
        .ddr3_dm            (ddr3_dm),
        
        // System Clock & Reset Inputs
        .sys_clk_i          (sys_clk_i),
        .sys_rst            (~sys_rst_n), // MIG expects Active High reset
        
        // User Interface Clock & Reset Outputs
        .ui_clk             (ui_clk),
        .ui_clk_sync_rst    (ui_clk_sync_rst),
        .init_calib_complete(init_calib_complete),
        
        // AXI4 Slave Interface (connected to our internal MUX)
        .s_axi_awid         (4'd0), 
        .s_axi_awaddr       (axi_mem_awaddr[27:0]), // MIG address width might be smaller (e.g., 28 bits for 256MB)
        .s_axi_awlen        (axi_mem_awlen), 
        .s_axi_awsize       (axi_mem_awsize), 
        .s_axi_awburst      (axi_mem_awburst),
        .s_axi_awvalid      (axi_mem_awvalid), 
        .s_axi_awready      (axi_mem_awready),
        .s_axi_wdata        (axi_mem_wdata), 
        .s_axi_wstrb        (axi_mem_wstrb), 
        .s_axi_wlast        (axi_mem_wlast),
        .s_axi_wvalid       (axi_mem_wvalid), 
        .s_axi_wready       (axi_mem_wready),
        .s_axi_bresp        (axi_mem_bresp),
        .s_axi_bvalid       (axi_mem_bvalid), 
        .s_axi_bready       (axi_mem_bready),
        .s_axi_arid         (4'd0), 
        .s_axi_araddr       (axi_mem_araddr[27:0]),
        .s_axi_arlen        (axi_mem_arlen), 
        .s_axi_arsize       (axi_mem_arsize), 
        .s_axi_arburst      (axi_mem_arburst),
        .s_axi_arvalid      (axi_mem_arvalid), 
        .s_axi_arready      (axi_mem_arready),
        .s_axi_rdata        (axi_mem_rdata), 
        .s_axi_rresp        (axi_mem_rresp), 
        .s_axi_rlast        (axi_mem_rlast),
        .s_axi_rvalid       (axi_mem_rvalid), 
        .s_axi_rready       (axi_mem_rready)
    );
    
    // ========================================================================
    // 2. SD Card Loader Instance
    // ========================================================================
    wire        loader_active;
    wire        sys_reset_req;
    
    wire [31:0] load_awaddr;
    wire [7:0]  load_awlen;
    wire [2:0]  load_awsize;
    wire [1:0]  load_awburst;
    wire        load_awvalid;
    wire [31:0] load_wdata;
    wire [3:0]  load_wstrb;
    wire        load_wlast;
    wire        load_wvalid;
    wire        load_bready;

    // Memory layout configuration (must match Python compiler)
    localparam DDR_BASE_ADDR = 32'h10000000;

    NAC_SD_Loader #(
        .CLK_FREQ_HZ(100000000),   // Frequency of ui_clk
        .SPI_FAST_KHZ(25000),     // 25 MHz for SD card reading
        .DDR_BASE_ADDR(DDR_BASE_ADDR)
    ) u_sd_loader (
        .clk(ui_clk),
        .rst_n(~ui_clk_sync_rst), // Loader is reset by MIG's synchronous reset
        
        // SD Hardware
        .sd_cd_n(sd_cd_n), 
        .sd_cs_n(sd_cs_n),
        .sd_sck(sd_sck), 
        .sd_mosi(sd_mosi), 
        .sd_miso(sd_miso),
        
        // Control
        .loader_active(loader_active),
        .system_reset(sys_reset_req),
        
        // AXI Master Write (connects to the AXI MUX)
        .m_axi_awaddr(load_awaddr), 
        .m_axi_awlen(load_awlen), 
        .m_axi_awsize(load_awsize), 
        .m_axi_awburst(load_awburst),
        .m_axi_awvalid(load_awvalid), 
        .m_axi_awready(axi_mem_awready && loader_active), // Grant only when loader is the active master
        .m_axi_wdata(load_wdata), 
        .m_axi_wstrb(load_wstrb), 
        .m_axi_wlast(load_wlast),
        .m_axi_wvalid(load_wvalid), 
        .m_axi_wready(axi_mem_wready && loader_active),
        .m_axi_bvalid(axi_mem_bvalid && loader_active), 
        .m_axi_bresp(), // bresp is read-only for master
        .m_axi_bready(load_bready)
    );

    // ========================================================================
    // 3. NAC Processor Core Instance
    // ========================================================================
    wire        nac_cmd_start;
    wire        nac_busy, nac_done, nac_error;
    
    wire [31:0] core_awaddr;
    wire [7:0]  core_awlen;
    wire [2:0]  core_awsize;
    wire [1:0]  core_awburst;
    wire        core_awvalid;
    wire [31:0] core_wdata;
    wire [3:0]  core_wstrb;
    wire        core_wlast;
    wire        core_wvalid;
    wire        core_bready;
    wire [31:0] core_araddr;
    wire [7:0]  core_arlen;
    wire [2:0]  core_arsize;
    wire [1:0]  core_arburst;
    wire        core_arvalid;
    wire        core_rready;

    // Autostart logic for the NAC Core
    assign nac_cmd_start = init_calib_complete && !loader_active && !sys_reset_req;

    // Reset logic for the NAC Core: keep in reset during MIG calibration, SD loading, or hot-swap events.
    wire nac_rst_n = sys_rst_n && init_calib_complete && !loader_active && !sys_reset_req;

    // Hardcoded memory map based on the fixed offsets defined in `nac_defines.vh`.
    // These addresses must match the monolithic SD card image created by the Python compiler.
    localparam CFG_PTR_OPMAP_ADDR    = DDR_BASE_ADDR + `PTR_OPMAP_OFFSET;
    localparam CFG_PTR_VARMAP_ADDR   = DDR_BASE_ADDR + `PTR_VARMAP_OFFSET;
    localparam CFG_PTR_REGISTRY_ADDR = DDR_BASE_ADDR + `PTR_REGISTRY_OFFSET;
    localparam CFG_PTR_CODE_ADDR     = DDR_BASE_ADDR + `PTR_CODE_OFFSET;
    localparam CFG_PTR_INPUT_ADDR    = DDR_BASE_ADDR + `PTR_INPUT_OFFSET;
    localparam CFG_PTR_OUTPUT_ADDR   = DDR_BASE_ADDR + `PTR_OUTPUT_OFFSET;
    localparam CFG_PTR_WEIGHTS_ADDR  = DDR_BASE_ADDR + `PTR_WEIGHTS_OFFSET;
    
    NAC_Processor_Core #(
        .C_M_AXI_ADDR_WIDTH(32), // 32-bit address for Artix-7 MIG
        .C_M_AXI_DATA_WIDTH(32)
    ) u_core (
        .clk(ui_clk),
        .rst_n(nac_rst_n),
        
        // Configuration Inputs (hardwired to predefined memory locations)
        .cmd_start(nac_cmd_start),
        .cfg_ptr_opmap    (CFG_PTR_OPMAP_ADDR),
        .cfg_ptr_varmap   (CFG_PTR_VARMAP_ADDR),
        .cfg_ptr_registry (CFG_PTR_REGISTRY_ADDR),
        .cfg_ptr_code     (CFG_PTR_CODE_ADDR), 
        .cfg_ptr_input    (CFG_PTR_INPUT_ADDR),
        .cfg_ptr_output   (CFG_PTR_OUTPUT_ADDR),
        .cfg_ptr_weights  (CFG_PTR_WEIGHTS_ADDR),
        
        // Status Outputs
        .status_busy(nac_busy),
        .status_done(nac_done),
        .status_error(nac_error),
        .irq_done_pulse(), // IRQ is not used in this standalone wrapper
        
        // AXI Master Interface (connects to the AXI MUX)
        .m_axi_awaddr(core_awaddr), .m_axi_awlen(core_awlen), .m_axi_awsize(core_awsize), .m_axi_awburst(core_awburst),
        .m_axi_awvalid(core_awvalid), .m_axi_awready(axi_mem_awready && !loader_active),
        .m_axi_wdata(core_wdata), .m_axi_wstrb(core_wstrb), .m_axi_wlast(core_wlast),
        .m_axi_wvalid(core_wvalid), .m_axi_wready(axi_mem_wready && !loader_active),
        .m_axi_bvalid(axi_mem_bvalid && !loader_active), .m_axi_bready(core_bready),
        .m_axi_bresp(axi_mem_bresp),
        .m_axi_araddr(core_araddr), .m_axi_arlen(core_arlen), .m_axi_arsize(core_arsize), .m_axi_arburst(core_arburst),
        .m_axi_arvalid(core_arvalid), .m_axi_arready(axi_mem_arready && !loader_active),
        .m_axi_rdata(axi_mem_rdata), .m_axi_rresp(axi_mem_rresp), .m_axi_rlast(axi_mem_rlast),
        .m_axi_rvalid(axi_mem_rvalid), .m_axi_rready(core_rready)
    );

    // ========================================================================
    // 4. AXI MUX (Simple Arbiter: SD Loader vs. NAC Core)
    // ========================================================================
    // Priority is given to the SD Loader. The NAC Core only gets bus access
    // after the loader has finished. The loader only performs writes.
    
    // Write Address Channel MUX
    assign axi_mem_awaddr  = loader_active ? load_awaddr  : core_awaddr;
    assign axi_mem_awlen   = loader_active ? load_awlen   : core_awlen;
    assign axi_mem_awsize  = loader_active ? load_awsize  : core_awsize;
    assign axi_mem_awburst = loader_active ? load_awburst : core_awburst;
    assign axi_mem_awvalid = loader_active ? load_awvalid : core_awvalid;
    
    // Write Data Channel MUX
    assign axi_mem_wdata   = loader_active ? load_wdata   : core_wdata;
    assign axi_mem_wstrb   = loader_active ? load_wstrb   : core_wstrb;
    assign axi_mem_wlast   = loader_active ? load_wlast   : core_wlast;
    assign axi_mem_wvalid  = loader_active ? load_wvalid  : core_wvalid;
    
    // Write Response Channel Demux
    assign axi_mem_bready  = loader_active ? load_bready  : core_bready;

    // Read Channels (Direct Connection)
    // The SD loader does not read from DDR, so the read channels are exclusively
    // used by the NAC Core. No MUX is needed.
    assign axi_mem_araddr  = core_araddr;
    assign axi_mem_arlen   = core_arlen;
    assign axi_mem_arsize  = core_arsize;
    assign axi_mem_arburst = core_arburst;
    assign axi_mem_arvalid = core_arvalid && !loader_active; // Ensure core doesn't request during load
    assign axi_mem_rready  = core_rready;

    // ========================================================================
    // 5. Status Indication (LEDs)
    // ========================================================================
    assign led[0] = init_calib_complete; // LED 0: System heartbeat (DDR ready)
    assign led[1] = loader_active;      // LED 1: Blinks while loading from SD card
    assign led[2] = nac_busy;           // LED 2: On during model inference
    assign led[3] = nac_done;           // LED 3: On when inference is complete

endmodule