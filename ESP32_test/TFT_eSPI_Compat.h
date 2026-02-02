// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
// TFT_eSPI_Compat.h
// TFT_eSPI library API compatibility layer based on LovyanGFX.
// ТОЛЬКО ДЛЯ ДИСПЛЕЯ Sunton ESP32-2432S028 board (ST7789).

#pragma once

#define LGFX_USE_V1
#include <LovyanGFX.hpp>

// === Color constants (RGB565), full compatibility with TFT_eSPI ===
#define TFT_BLACK       0x0000
#define TFT_NAVY        0x000F
#define TFT_DARKGREEN   0x03E0
#define TFT_DARKCYAN    0x03EF
#define TFT_MAROON      0x7800
#define TFT_PURPLE      0x780F
#define TFT_OLIVE       0x7BE0
#define TFT_LIGHTGREY   0xD69A
#define TFT_DARKGREY    0x7BEF
#define TFT_BLUE        0x001F
#define TFT_GREEN       0x07E0
#define TFT_CYAN        0x07FF
#define TFT_RED         0xF800
#define TFT_MAGENTA     0xF81F
#define TFT_YELLOW      0xFFE0
#define TFT_WHITE       0xFFFF
#define TFT_ORANGE      0xFDA0
#define TFT_GREENYELLOW 0xAFE5
#define TFT_PINK        0xF81F

class TFT_eSPI : public lgfx::LGFX_Device
{
    lgfx::Panel_ST7789  _panel_instance;
    lgfx::Bus_SPI       _bus_instance;
    lgfx::Light_PWM     _light_instance;
    // lgfx::Touch_XPT2046 _touch_instance; // <-- УДАЛЕНО

public:
    TFT_eSPI()
    {
        // 1. Конфигурация шины SPI дисплея (HSPI) - НЕ ТРОГАЕМ
        {
            auto cfg = _bus_instance.config();
            cfg.spi_host     = HSPI_HOST;
            cfg.spi_mode     = 0;
            cfg.freq_write   = 80000000;
            cfg.freq_read    = 16000000;
            cfg.pin_sclk     = 14;
            cfg.pin_mosi     = 13;
            cfg.pin_miso     = 12;
            cfg.pin_dc       = 2;
            cfg.dma_channel  = SPI_DMA_CH_AUTO;
            _bus_instance.config(cfg);
            _panel_instance.setBus(&_bus_instance);
        }

        // 2. Конфигурация панели дисплея (ST7789) - НЕ ТРОГАЕМ
        {
            auto cfg = _panel_instance.config();
            cfg.pin_cs           = 15;
            cfg.pin_rst          = -1;
            cfg.pin_busy         = -1;
            cfg.panel_width      = 240;
            cfg.panel_height     = 320;
            cfg.offset_x         = 0;
            cfg.offset_y         = 0;
            cfg.offset_rotation  = 0;
            cfg.dummy_read_pixel = 16;
            cfg.readable         = true;
            cfg.invert           = false;
            cfg.rgb_order        = false;
            cfg.dlen_16bit       = false;
            cfg.bus_shared       = false;
            _panel_instance.config(cfg);
        }

        // 3. Конфигурация подсветки - НЕ ТРОГАЕМ
        {
            auto cfg = _light_instance.config();
            cfg.pin_bl       = 21;
            cfg.invert       = false;
            cfg.freq         = 12000;
            cfg.pwm_channel  = 7;
            _light_instance.config(cfg);
            _panel_instance.setLight(&_light_instance);
        }

        // 4. Конфигурация тачскрина - ПОЛНОСТЬЮ УДАЛЕНА
        
        setPanel(&_panel_instance);
    }

    // === Методы совместимости ===
    void init() { LGFX_Device::init(); setBrightness(255); }
    void setRotation(uint8_t r) { LGFX_Device::setRotation(r % 8); }
    // ... и так далее, все методы для рисования и текста остаются как были ...
    void fillScreen(uint16_t color) { LGFX_Device::fillScreen(color); }
    void drawPixel(int32_t x, int32_t y, uint16_t color) { LGFX_Device::drawPixel(x, y, color); }
    void drawLine(int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint16_t color) { LGFX_Device::drawLine(x0, y0, x1, y1, color); }
    void drawRect(int32_t x, int32_t y, int32_t w, int32_t h, uint16_t color) { LGFX_Device::drawRect(x, y, w, h, color); }
    void fillRect(int32_t x, int32_t y, int32_t w, int32_t h, uint16_t color) { LGFX_Device::fillRect(x, y, w, h, color); }
    void drawRoundRect(int32_t x, int32_t y, int32_t w, int32_t h, int32_t r, uint16_t color) { LGFX_Device::drawRoundRect(x, y, w, h, r, color); }
    void fillRoundRect(int32_t x, int32_t y, int32_t w, int32_t h, int32_t r, uint16_t color) { LGFX_Device::fillRoundRect(x, y, w, h, r, color); }
    void drawCircle(int32_t x, int32_t y, int32_t r, uint16_t color) { LGFX_Device::drawCircle(x, y, r, color); }
    void fillCircle(int32_t x, int32_t y, int32_t r, uint16_t color) { LGFX_Device::fillCircle(x, y, r, color); }
    void setTextColor(uint16_t c) { LGFX_Device::setTextColor(c); }
    void setTextColor(uint16_t c, uint16_t bg) { LGFX_Device::setTextColor(c, bg); }
    void setTextSize(uint8_t s) { LGFX_Device::setTextSize(s); }
    void setCursor(int32_t x, int32_t y) { LGFX_Device::setCursor(x, y); }
    using LGFX_Device::print;
    using LGFX_Device::println;
    using LGFX_Device::printf;
    void setBrightness(uint8_t brightness) { LGFX_Device::setBrightness(brightness); }
    int16_t width()  const { return LGFX_Device::width(); }
    int16_t height() const { return LGFX_Device::height(); }
};