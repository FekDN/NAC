// Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

#ifndef _CYD28_SD_H
#define _CYD28_SD_H

#include <Arduino.h>
#include "SD.h"
#include "SPI.h"

class CYD28_SD
{
public:
	CYD28_SD(){};
	~CYD28_SD(){};

	// --- ОСНОВНЫЕ ФУНКЦИИ ---
	void begin(int8_t sck, int8_t miso, int8_t mosi, int8_t cs);
	
	// --- ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛОМ МОДЕЛИ ---
    bool openFile(const char* path);
    void closeFile();
    bool seek(size_t position);                     // Версия с одним аргументом
    bool seek(uint32_t pos, SeekMode mode);         // ДОБАВЛЕНО: Версия с двумя аргументами
    size_t readData(uint8_t* buffer, size_t length);
    size_t getPosition();
    bool isFileOpen();
	size_t size();
	// --- ФУНКЦИИ СТАТУСА И ОТЛАДКИ ---
	void status(uint8_t *mount, uint8_t *type, uint64_t *size, 
				uint64_t *totalBytes, uint64_t *usedBytes);
	void printStatus(char *buf); 		
	
private:
    int8_t _cs_pin = -1;
    bool _is_initialized = false;
	const char *sdcardTypeLabels[5] = { "None", "MMC", "SD", "SDHC", "Unknown"};
    File _current_file; 
};

// 'extern' говорит, что эти объекты созданы где-то еще (в .cpp файле)
extern CYD28_SD sdcard;
extern SPIClass sd_spi;

#endif // _CYD28_SD_H
