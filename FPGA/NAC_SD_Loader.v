`timescale 1ns / 1ps

module NAC_SD_Loader #(
    parameter CLK_FREQ_HZ   = 100000000, // Частота входного clk (от PLL/MMCM)
    parameter SPI_SLOW_KHZ  = 400,       // Скорость инициализации (макс 400kHz)
    parameter SPI_FAST_KHZ  = 20000,     // Рабочая скорость (20-25MHz)
    parameter START_SECTOR  = 32'd2048,  // Смещение на карте (обычно 1MB или 2048 секторов)
    parameter TOTAL_BYTES   = 32'd10485760, // Объем загрузки (10 MB)
    parameter DDR_BASE_ADDR = 32'h10000000  // Адрес в DDR3 (0x10000000 = 256MB offset)
)(
    input  wire        clk,
    input  wire        rst_n,

    // --- SD Card Hardware Interface ---
    input  wire        sd_cd_n,    // Card Detect (0 = карта вставлена)
    output reg         sd_cs_n,    // Chip Select
    output reg         sd_sck,     // SPI Clock
    output reg         sd_mosi,    // Master Out
    input  wire        sd_miso,    // Master In

    // --- AXI4 Master Write Interface (к MIG) ---
    output reg [31:0]  m_axi_awaddr,
    output reg [7:0]   m_axi_awlen,   // Burst Length: 127 (128 transfers)
    output wire [2:0]  m_axi_awsize,  // Burst Size: 3'b010 (4 bytes)
    output wire [1:0]  m_axi_awburst, // Burst Type: 2'b01 (INCR)
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    output reg [31:0]  m_axi_wdata,
    output wire [3:0]  m_axi_wstrb,   // Strobe: 4'b1111
    output reg         m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,

    // --- System Control ---
    output reg         loader_active, // 1 = идет загрузка (шина занята)
    output reg         system_reset   // Управление сбросом процессора/системы
);

    // ========================================================================
    // Константы и Команды SD
    // ========================================================================
    // CMD Format: {8'h40 | Index, Arg[31:0], CRC[7:0]}
    // CRC обязателен только для CMD0 и CMD8 в SPI режиме.
    
    localparam CMD0_RESET   = {8'h40, 32'h00000000, 8'h95}; // Go Idle
    localparam CMD8_CHECK   = {8'h48, 32'h000001AA, 8'h87}; // Check Voltage (2.7-3.6V), Pattern 0xAA
    localparam CMD55_APP    = {8'h77, 32'h00000000, 8'hFF}; // APP_CMD prefix (CRC ignored)
    localparam ACMD41_INIT  = {8'h69, 32'h40000000, 8'hFF}; // Init with HCS bit (30) set!
    
    // AXI константы
    assign m_axi_awsize  = 3'b010; // 4 байта (32 бита)
    assign m_axi_awburst = 2'b01;  // Инкрементный пакет
    assign m_axi_wstrb   = 4'b1111; // Пишем все байты

    // ========================================================================
    // Генерация SPI Clock
    // ========================================================================
    reg [15:0] clk_div;
    reg [15:0] clk_div_limit;
    reg        spi_tick_rise; // Строб переднего фронта (SCK 0->1) - чтение MISO
    reg        spi_tick_fall; // Строб заднего фронта (SCK 1->0) - изменение MOSI
    reg        fast_mode;     // 0 = 400kHz, 1 = 20MHz

    always @(posedge clk) begin
        if (fast_mode) clk_div_limit <= (CLK_FREQ_HZ / (SPI_FAST_KHZ * 1000)) / 2 - 1;
        else           clk_div_limit <= (CLK_FREQ_HZ / (SPI_SLOW_KHZ * 1000)) / 2 - 1;

        if (!rst_n) begin
            clk_div <= 0;
            sd_sck <= 0;
            spi_tick_rise <= 0;
            spi_tick_fall <= 0;
        end else begin
            spi_tick_rise <= 0;
            spi_tick_fall <= 0;
            
            if (clk_div >= clk_div_limit) begin
                clk_div <= 0;
                sd_sck <= ~sd_sck;
                if (sd_sck) spi_tick_fall <= 1; // Был 1, стал 0
                else        spi_tick_rise <= 1; // Был 0, стал 1
            end else begin
                clk_div <= clk_div + 1;
            end
        end
    end

    // ========================================================================
    // Детектор карты (Debounce + Hot Swap Logic)
    // ========================================================================
    reg [19:0] debounce_cnt;
    reg        cd_stable;
    wire       cd_raw_inserted = ~sd_cd_n; // 1 если карта вставлена
    
    // Сигналы сброса FSM
    reg        fsm_start;    // Импульс на старт загрузки
    reg        fsm_abort;    // Импульс на прерывание

    always @(posedge clk) begin
        if (!rst_n) begin
            debounce_cnt <= 0;
            cd_stable <= 0;
            fsm_start <= 0;
            fsm_abort <= 0;
        end else begin
            fsm_start <= 0;
            fsm_abort <= 0;

            if (cd_raw_inserted == cd_stable) begin
                debounce_cnt <= 0;
            end else begin
                debounce_cnt <= debounce_cnt + 1;
                // Ждем ~10-20ms (при 100MHz это около 1M-2M тактов, 20 бит хватает)
                if (debounce_cnt == 20'hFFFFF) begin
                    cd_stable <= cd_raw_inserted;
                    
                    if (cd_raw_inserted) begin
                        // Карта только что вставлена -> Старт загрузки
                        fsm_start <= 1; 
                    end else begin
                        // Карта извлечена -> Стоп
                        fsm_abort <= 1;
                    end
                end
            end
        end
    end

    // ========================================================================
    // Основной конечный автомат (Main FSM)
    // ========================================================================
    localparam S_IDLE           = 0;
    localparam S_POWERON_WAIT   = 1;  // Ждем 74+ тактов
    localparam S_SEND_CMD       = 2;  // Отправка 48 бит команды
    localparam S_WAIT_RESP      = 3;  // Ожидание R1 ответа (0xxx_xxxx)
    localparam S_READ_R7        = 4;  // Дочитывание CMD8 (еще 4 байта)
    localparam S_CMD_FINISH     = 5;  // 8 тактов пустышек после команды (CS=1)
    localparam S_CMD17_START    = 6;  // Подготовка чтения сектора
    localparam S_WAIT_TOKEN     = 7;  // Ждем 0xFE
    localparam S_READ_DATA      = 8;  // Читаем 512 байт
    localparam S_READ_CRC       = 9;  // Пропускаем 2 байта CRC
    localparam S_AXI_START      = 10; // Настройка AXI записи
    localparam S_AXI_WRITE      = 11; // Потоковая запись в DDR
    localparam S_DONE           = 12; // Успех

    reg [4:0]  state;
    reg [4:0]  return_state;      // Куда вернуться после универсальных состояний
    reg [2:0]  init_phase;        // 0=CMD0, 1=CMD8, 2=ACMD_PRE, 3=ACMD_EXE
    
    // Переменные данных
    reg [47:0] cmd_shifter;       // Сдвиговый регистр команды
    reg [5:0]  bit_cnt;           // Счётчик бит
    reg [7:0]  rx_byte;           // Принятый байт
    reg [31:0] r7_acc;            // Аккумулятор для ответа R7
    reg [15:0] wait_cnt;          // Таймаут счетчик
    reg [31:0] byte_addr;         // Текущий адрес DDR
    reg [31:0] sector_num;        // Текущий сектор SD
    reg [31:0] loaded_bytes;      // Счетчик загруженных байт

    // Буфер сектора (512 байт = 128 слов x 32 бита)
    reg [31:0] sector_buf [0:127];
    reg [31:0] word_assembly;     // Сборка 4 байт в слово
    reg [8:0]  byte_idx;          // 0..511
    reg [1:0]  align_cnt;         // 0..3
    reg [7:0]  axi_word_idx;      // 0..127

    // ========================================================================
    // Логика FSM
    // ========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            system_reset <= 1;  // По умолчанию держим ресет
            loader_active <= 1;
            sd_cs_n <= 1;
            sd_mosi <= 1;
        end else begin
            
            // --- Глобальная обработка событий карты ---
            if (fsm_abort) begin
                // Карта вытащена
                if (state != S_DONE) begin
                    // Если вытащили во время загрузки -> Ресет системы
                    system_reset <= 1;
                end 
                // Если S_DONE, то system_reset остается 0 (система работает)
                
                state <= S_IDLE;
                loader_active <= 1;
                sd_cs_n <= 1;
            end
            else if (fsm_start) begin
                // Карта вставлена -> Начинаем заново
                system_reset <= 1; // Жесткий сброс системы для новой загрузки
                state <= S_POWERON_WAIT;
                loader_active <= 1;
                
                // Сброс переменных
                wait_cnt <= 0;
                fast_mode <= 0;
                sd_cs_n <= 1;
                sd_mosi <= 1;
                byte_addr <= DDR_BASE_ADDR;
                sector_num <= START_SECTOR;
                loaded_bytes <= 0;
            end
            
            // --- Основной автомат ---
            else begin 
                case (state)
                    
                    // 0. Ожидание
                    S_IDLE: begin
                        sd_cs_n <= 1;
                        sd_mosi <= 1;
                        // Если мы здесь после успешной загрузки, system_reset = 0
                        // Если после сбоя или старта без карты, system_reset = 1
                    end

                    // 1. Power On Sequence (минимум 74 такта с CS=1)
                    S_POWERON_WAIT: begin
                        sd_cs_n <= 1;
                        sd_mosi <= 1;
                        if (spi_tick_rise) begin
                            wait_cnt <= wait_cnt + 1;
                            if (wait_cnt > 160) begin // ~80 тактов (rise+fall = 2 тика)
                                state <= S_SEND_CMD;
                                cmd_shifter <= CMD0_RESET;
                                bit_cnt <= 47;
                                init_phase <= 0;
                                return_state <= S_CMD_FINISH; // После CMD0 идем в "отдых"
                            end
                        end
                    end

                    // 2. Отправка команды (универсальное состояние)
                    S_SEND_CMD: begin
                        sd_cs_n <= 0; // CS Active Low
                        if (spi_tick_fall) begin
                            sd_mosi <= cmd_shifter[bit_cnt];
                            if (bit_cnt == 0) begin
                                state <= S_WAIT_RESP;
                                wait_cnt <= 0; // Используем для таймаута ответа
                            end else begin
                                bit_cnt <= bit_cnt - 1;
                            end
                        end
                    end

                    // 3. Ожидание R1 ответа (байт начинается с 0)
                    S_WAIT_RESP: begin
                        sd_mosi <= 1; // Отпускаем MOSI
                        if (spi_tick_rise) begin
                            rx_byte <= {rx_byte[6:0], sd_miso};
                            
                            // Проверка таймаута
                            wait_cnt <= wait_cnt + 1;
                            if (wait_cnt > 5000) begin 
                                state <= S_POWERON_WAIT; // Таймаут -> рестарт
                            end

                            // Мы ищем 0 в MSB. Так как мы сдвигаем каждый такт,
                            // нужно поймать момент, когда валидный байт сформирован.
                            // Для простоты здесь: ищем 8 тактов после первого нуля.
                            // Упрощенная реализация: если (rx_byte[7] == 0) - это ответ.
                            // (В реальном SD протоколе NCR min 1 byte, max 8 bytes).
                            
                            // Примечание: тут простая проверка. Если бит 7 == 0, считаем что ответ получен.
                            // Нужно убедиться, что это не "мусор" на линии до ответа.
                            // SD карта держит MISO в 1 пока занята.
                            
                            if (sd_miso == 0) begin
                                // Найден Start Bit. Нужно дочитать еще 7 бит.
                                // Но rx_byte уже сдвигается.
                                // Сделаем переход на дочитывание? Или проверим постфактум?
                                // Вариант: если последний принятый байт имеет 0 в MSB, это оно.
                            end
                            
                            // Проверка "на лету" (работает, если MISO подтянут к 1)
                            if (rx_byte[7] == 0 && wait_cnt > 8) begin // wait_cnt > 8 чтобы не поймать шум при переключении
                                // Анализ ответа
                                case (init_phase)
                                    0: begin // CMD0
                                        if (rx_byte == 8'h01) state <= S_CMD_FINISH; // Idle State
                                        else state <= S_POWERON_WAIT; // Ошибка, рестарт
                                    end
                                    1: begin // CMD8
                                        if (rx_byte == 8'h01) state <= S_READ_R7; // Idle, читаем хвост
                                        else state <= S_POWERON_WAIT; // Старая карта или ошибка
                                    end
                                    2: begin // CMD55
                                        if (rx_byte == 8'h01) state <= S_CMD_FINISH; // OK
                                        else state <= S_POWERON_WAIT;
                                    end
                                    3: begin // ACMD41
                                        if (rx_byte == 8'h00) begin
                                            // Инициализация завершена!
                                            state <= S_CMD_FINISH;
                                            fast_mode <= 1; // Включаем 20 MHz
                                        end else if (rx_byte == 8'h01) begin
                                            // In Idle State (Busy initialization) -> Повторить цикл
                                            state <= S_CMD_FINISH; 
                                            // Логика повтора будет в S_CMD_FINISH
                                        end else begin
                                            state <= S_POWERON_WAIT; // Ошибка
                                        end
                                    end
                                    4: begin // CMD17
                                        if (rx_byte == 8'h00) state <= S_WAIT_TOKEN; // OK, ждем данные
                                        else state <= S_POWERON_WAIT;
                                    end
                                    default: state <= S_POWERON_WAIT;
                                endcase
                            end
                        end
                    end

                    // 4. Дочитывание R7 (для CMD8)
                    S_READ_R7: begin
                        if (spi_tick_rise) begin
                            wait_cnt <= wait_cnt + 1; // Используем как счетчик бит
                            r7_acc <= {r7_acc[30:0], sd_miso};
                            // Нужно прочитать 32 бита
                            if (wait_cnt[5:0] == 32) begin // примерно 32 такта
                                // Проверка паттерна (последние 8 бит)
                                if (r7_acc[7:0] == 8'hAA) state <= S_CMD_FINISH;
                                else state <= S_POWERON_WAIT;
                                wait_cnt <= 0;
                            end
                        end
                    end

                    // 5. Завершение команды (8 тактов CS=1) и выбор следующей
                    S_CMD_FINISH: begin
                        sd_cs_n <= 1;
                        sd_mosi <= 1;
                        if (spi_tick_rise) begin
                            wait_cnt <= wait_cnt + 1;
                            if (wait_cnt >= 8) begin
                                wait_cnt <= 0;
                                case (init_phase)
                                    0: begin // Была CMD0 -> CMD8
                                        init_phase <= 1;
                                        cmd_shifter <= CMD8_CHECK;
                                        bit_cnt <= 47;
                                        state <= S_SEND_CMD;
                                    end
                                    1: begin // Была CMD8 -> CMD55
                                        init_phase <= 2;
                                        cmd_shifter <= CMD55_APP;
                                        bit_cnt <= 47;
                                        state <= S_SEND_CMD;
                                    end
                                    2: begin // Была CMD55 -> ACMD41
                                        init_phase <= 3;
                                        cmd_shifter <= ACMD41_INIT;
                                        bit_cnt <= 47;
                                        state <= S_SEND_CMD;
                                    end
                                    3: begin // Была ACMD41
                                        // Если мы здесь, значит ответ был получен.
                                        // Проверяем, перешли ли мы в fast_mode (признак успеха 0x00)
                                        if (fast_mode) begin
                                            init_phase <= 4; // Переходим к чтению
                                            state <= S_CMD17_START;
                                        end else begin
                                            // Если еще медленный режим, значит было 0x01 (Busy).
                                            // Повторяем цикл: CMD55 -> ACMD41
                                            init_phase <= 2;
                                            cmd_shifter <= CMD55_APP;
                                            bit_cnt <= 47;
                                            state <= S_SEND_CMD;
                                        end
                                    end
                                    default: state <= S_IDLE; // Shouldn't happen
                                endcase
                            end
                        end
                    end

                    // 6. Старт чтения сектора (CMD17)
                    S_CMD17_START: begin
                        if (loaded_bytes >= TOTAL_BYTES) begin
                            state <= S_DONE;
                        end else begin
                            init_phase <= 4; // Маркер для WAIT_RESP
                            cmd_shifter <= {8'h51, sector_num, 8'hFF};
                            bit_cnt <= 47;
                            state <= S_SEND_CMD;
                        end
                    end

                    // 7. Ожидание Data Token (0xFE)
                    S_WAIT_TOKEN: begin
                        sd_cs_n <= 0; // Держим CS активным!
                        if (spi_tick_rise) begin
                            rx_byte <= {rx_byte[6:0], sd_miso};
                            wait_cnt <= wait_cnt + 1;
                            
                            // Ищем 0xFE. Проверяем каждые 8 бит или скользящим окном?
                            // SD карта может выдать 0xFF много раз, потом 0xFE.
                            if (rx_byte == 8'hFE) begin
                                state <= S_READ_DATA;
                                byte_idx <= 0;
                                align_cnt <= 0;
                                wait_cnt <= 0;
                            end

                            if (wait_cnt > 60000) state <= S_POWERON_WAIT; // Timeout чтения
                        end
                    end

                    // 8. Чтение 512 байт данных
                    S_READ_DATA: begin
                        if (spi_tick_rise) begin
                            rx_byte <= {rx_byte[6:0], sd_miso};
                            
                            // Собираем байт каждые 8 тактов
                            if (align_cnt == 7) begin
                                // Байт готов. Пакуем в Little Endian (DDR).
                                // [B0, B1, B2, B3] -> Word
                                case (byte_idx[1:0])
                                    2'b00: word_assembly[7:0]   <= {rx_byte[6:0], sd_miso};
                                    2'b01: word_assembly[15:8]  <= {rx_byte[6:0], sd_miso};
                                    2'b10: word_assembly[23:16] <= {rx_byte[6:0], sd_miso};
                                    2'b11: begin
                                        word_assembly[31:24] <= {rx_byte[6:0], sd_miso};
                                        // Запись слова в буфер
                                        sector_buf[byte_idx[8:2]] <= { {rx_byte[6:0], sd_miso}, word_assembly[23:0] };
                                    end
                                endcase

                                byte_idx <= byte_idx + 1;
                                
                                if (byte_idx == 511) state <= S_READ_CRC;
                            end
                            
                            align_cnt <= align_cnt + 1;
                        end
                    end

                    // 9. Пропуск CRC (16 бит)
                    S_READ_CRC: begin
                        if (spi_tick_rise) begin
                            align_cnt <= align_cnt + 1;
                            if (align_cnt >= 15) begin // 16 тактов
                                sd_cs_n <= 1; // Отпускаем карту
                                state <= S_AXI_START;
                            end
                        end
                    end

                    // 10. Настройка AXI
                    S_AXI_START: begin
                        m_axi_awvalid <= 1;
                        m_axi_awaddr  <= byte_addr;
                        m_axi_awlen   <= 8'd127; // 128 words
                        m_axi_wvalid  <= 0;
                        axi_word_idx  <= 0;
                        state         <= S_AXI_WRITE;
                    end

                    // 11. Запись в DDR
                    S_AXI_WRITE: begin
                        // Address Channel
                        if (m_axi_awvalid && m_axi_awready) begin
                            m_axi_awvalid <= 0;
                            m_axi_wvalid  <= 1;
                            m_axi_wdata   <= sector_buf[0]; // Первое слово
                        end

                        // Data Channel
                        if (m_axi_wvalid && m_axi_wready) begin
                            if (axi_word_idx == 127) begin
                                m_axi_wlast  <= 1;
                                m_axi_wvalid <= 0;
                                m_axi_bready <= 1;
                            end else begin
                                axi_word_idx <= axi_word_idx + 1;
                                m_axi_wdata  <= sector_buf[axi_word_idx + 1];
                                if (axi_word_idx == 126) m_axi_wlast <= 1;
                            end
                        end

                        // Response Channel
                        if (m_axi_bvalid && m_axi_bready) begin
                            m_axi_bready <= 0;
                            // Обновляем счетчики
                            byte_addr <= byte_addr + 512;
                            loaded_bytes <= loaded_bytes + 512;
                            sector_num <= sector_num + 1;
                            
                            // Возврат к чтению следующего блока через паузу
                            wait_cnt <= 0;
                            state <= S_CMD_FINISH;
                            // Хак: устанавливаем init_phase=4, чтобы S_CMD_FINISH пнул нас в S_CMD17_START
                            init_phase <= 3; // (S_CMD_FINISH проверяет case 3 -> if fast -> case 4)
                            // Или просто напрямую пойдем:
                            // state <= S_CMD17_START; // Но лучше через CS=1 (Dummy)
                            // Для надежности используем S_CMD_FINISH с фазой:
                            // Но S_CMD_FINISH логика заточена под Init. Сделаем проще:
                            state <= S_CMD_FINISH;
                            init_phase <= 3; // В S_CMD_FINISH (case 3) стоит проверка fast_mode -> переход к CMD17
                        end
                    end

                    // 12. Готово
                    S_DONE: begin
                        loader_active <= 0; // Освобождаем шину AXI для других мастеров
                        system_reset  <= 0; // Запускаем процессор
                        sd_cs_n <= 1;
                        // Ждем извлечения карты (обработается в глобальном блоке)
                    end
                endcase
            end
        end
    end

endmodule