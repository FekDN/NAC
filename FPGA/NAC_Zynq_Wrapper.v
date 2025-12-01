`timescale 1ns / 1ps

module NAC_Zynq_Wrapper #(
    // Параметры AXI-Lite (Control)
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 6, // 6 бит = 64 байта адресного пространства (достаточно для регистров)
    
    // Параметры AXI-Full (Data to DDR)
    parameter C_M_AXI_ADDR_WIDTH = 40, // 40 бит для UltraScale+ (или 32 для Zynq-7000)
    parameter C_M_AXI_DATA_WIDTH = 32,
    
    // Параметры ядра NAC
    parameter DMA_BURST_LEN      = 16,
    parameter INT8_MODE          = 1
)(
    // Глобальные сигналы
    input  wire  aclk,
    input  wire  aresetn,

    // ========================================================================
    // AXI4-Lite Slave Interface (Подключается к M_AXI_GP процессора Zynq)
    // ========================================================================
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire [2:0]                    s_axi_awprot,
    input  wire                          s_axi_awvalid,
    output wire                          s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  wire [3:0]                    s_axi_wstrb,
    input  wire                          s_axi_wvalid,
    output wire                          s_axi_wready,
    output wire [1:0]                    s_axi_bresp,
    output wire                          s_axi_bvalid,
    input  wire                          s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire [2:0]                    s_axi_arprot,
    input  wire                          s_axi_arvalid,
    output wire                          s_axi_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output wire [1:0]                    s_axi_rresp,
    output wire                          s_axi_rvalid,
    input  wire                          s_axi_rready,

    // ========================================================================
    // AXI4 Master Interface (Подключается к S_AXI_HP процессора Zynq)
    // ========================================================================
    // Write Address
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0]                    m_axi_awlen,
    output wire [2:0]                    m_axi_awsize,
    output wire [1:0]                    m_axi_awburst,
    output wire                          m_axi_awvalid,
    input  wire                          m_axi_awready,
    // Write Data
    output wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output wire [C_M_AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output wire                          m_axi_wlast,
    output wire                          m_axi_wvalid,
    input  wire                          m_axi_wready,
    // Write Response
    input  wire [1:0]                    m_axi_bresp,
    input  wire                          m_axi_bvalid,
    output wire                          m_axi_bready,
    // Read Address
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0]                    m_axi_arlen,
    output wire [2:0]                    m_axi_arsize,
    output wire [1:0]                    m_axi_arburst,
    output wire                          m_axi_arvalid,
    input  wire                          m_axi_arready,
    // Read Data
    input  wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  wire [1:0]                    m_axi_rresp,
    input  wire                          m_axi_rlast,
    input  wire                          m_axi_rvalid,
    output wire                          m_axi_rready,
    
    // Дополнительные сигналы ID и Cache (требуются для Zynq HP портов)
    output wire [5:0]                    m_axi_awid,
    output wire [5:0]                    m_axi_arid,
    output wire [3:0]                    m_axi_awcache,
    output wire [3:0]                    m_axi_arcache,
    output wire [2:0]                    m_axi_awprot,
    output wire [2:0]                    m_axi_arprot,
    // Входные ID (игнорируем)
    input  wire [5:0]                    m_axi_bid,
    input  wire [5:0]                    m_axi_rid,
    
    // Прерывание
    output wire                          irq_interrupt
);

    // Hardcode констант AXI для совместимости с Zynq
    assign m_axi_awid    = 6'd0;
    assign m_axi_arid    = 6'd0;
    assign m_axi_awcache = 4'b0011; // Bufferable
    assign m_axi_arcache = 4'b0011; // Bufferable
    assign m_axi_awprot  = 3'b000;
    assign m_axi_arprot  = 3'b000;

    // Внутренние сигналы соединения Slave -> Core
    wire        start_pulse;
    wire [31:0] ptr_code;
    wire [31:0] ptr_weights;
    wire [31:0] ptr_input;
    wire [31:0] ptr_output;
    wire [31:0] ptr_opmap;
    wire [31:0] ptr_varmap;
    wire [31:0] ptr_registry;
    
    // Сигналы Core -> Slave
    wire        status_busy;
    wire        status_done;
    wire        status_error;
    wire        core_irq_pulse;

    // 1. Инстанцируем AXI-Lite Slave (Регистровый файл)
    NAC_AXILite_Slave #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) axi_reg_file (
        .S_AXI_ACLK(aclk),
        .S_AXI_ARESETN(aresetn),
        // Write Address
        .S_AXI_AWADDR(s_axi_awaddr),
        .S_AXI_AWPROT(s_axi_awprot),
        .S_AXI_AWVALID(s_axi_awvalid),
        .S_AXI_AWREADY(s_axi_awready),
        // Write Data
        .S_AXI_WDATA(s_axi_wdata),
        .S_AXI_WSTRB(s_axi_wstrb),
        .S_AXI_WVALID(s_axi_wvalid),
        .S_AXI_WREADY(s_axi_wready),
        // Write Response
        .S_AXI_BRESP(s_axi_bresp),
        .S_AXI_BVALID(s_axi_bvalid),
        .S_AXI_BREADY(s_axi_bready),
        // Read Address
        .S_AXI_ARADDR(s_axi_araddr),
        .S_AXI_ARPROT(s_axi_arprot),
        .S_AXI_ARVALID(s_axi_arvalid),
        .S_AXI_ARREADY(s_axi_arready),
        // Read Data
        .S_AXI_RDATA(s_axi_rdata),
        .S_AXI_RRESP(s_axi_rresp),
        .S_AXI_RVALID(s_axi_rvalid),
        .S_AXI_RREADY(s_axi_rready),

        // Пользовательский интерфейс
        .slv_start_pulse(start_pulse),
        .slv_ptr_registry(ptr_registry),
        .slv_ptr_code(ptr_code),
        .slv_ptr_weights(ptr_weights),
        .slv_ptr_input(ptr_input),
        .slv_ptr_output(ptr_output),
        .slv_ptr_opmap(ptr_opmap),
        .slv_ptr_varmap(ptr_varmap),
        
        // Статус (собираем биты в 32-битное слово)
        // Bit 0: Busy, Bit 1: Done, Bit 2: Error
        .slv_status_reg({29'd0, status_error, status_done, status_busy})
    );

    // 2. Инстанцируем Ядро NAC
    NAC_Processor_Core #(
        .C_M_AXI_ADDR_WIDTH(C_M_AXI_ADDR_WIDTH),
        .C_M_AXI_DATA_WIDTH(C_M_AXI_DATA_WIDTH),
        .DMA_BURST_LEN(DMA_BURST_LEN),
        .INT8_MODE(INT8_MODE)
    ) nac_core (
        .clk(aclk),
        .rst_n(aresetn),
        
        // Конфигурация от AXI-Lite
        .cmd_start(start_pulse),
        .cfg_ptr_code(ptr_code),
        .cfg_ptr_weights(ptr_weights),
        .cfg_ptr_input(ptr_input),
        .cfg_ptr_output(ptr_output),
        .cfg_ptr_opmap(ptr_opmap),
        .cfg_ptr_varmap(ptr_varmap),
        .cfg_ptr_registry(ptr_registry),
        
        // Статусы
        .status_busy(status_busy),
        .status_done(status_done),
        .status_error(status_error),
        .irq_done_pulse(core_irq_pulse),
        
        // Интерфейс AXI Master
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    // Прерывание (уровень, пока статус Done=1)
    assign irq_interrupt = status_done;

endmodule