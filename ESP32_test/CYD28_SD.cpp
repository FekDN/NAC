// CYD28_SD.cpp
// Облегченная версия для чтения NAC-моделей

#include "CYD28_SD.h"

// Глобальные объекты. 'extern' для них будет в .h файле
SPIClass sd_spi(VSPI);
CYD28_SD sdcard;

/**
 * @brief Инициализирует и монтирует SD-карту.
 */
void CYD28_SD::begin(int8_t sck, int8_t miso, int8_t mosi, int8_t cs)
{
	_cs_pin = cs;
	_is_initialized = false;

    // КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Мы больше НЕ вызываем sd_spi.begin() здесь.
    // Шину VSPI инициализирует LovyanGFX (tft.init()).
    // Мы просто говорим библиотеке SD использовать уже существующую конфигурацию.
    // sd_spi.begin(sck, miso, mosi, cs); // <-- СТРОКА УДАЛЕНА/ЗАКОММЕНТИРОВАНА

	if (!SD.begin(_cs_pin, sd_spi, 25000000))
	{
		Serial.println("Card Mount Failed");
		return;
	}
	uint8_t cardType = SD.cardType();

	if (cardType == CARD_NONE)
	{
		Serial.println("No SD card attached");
		return;
	}
    
    _is_initialized = true; 
    Serial.println("SD Card mounted successfully.");
}

/**
 * @brief Открывает файл для чтения и сохраняет его в _current_file.
 * @param path Путь к файлу.
 * @return true, если файл успешно открыт.
 */
bool CYD28_SD::openFile(const char* path) {
    if (!_is_initialized) {
        Serial.println("SD card not initialized. Cannot open file.");
        return false;
    }
    if (_current_file) {
        _current_file.close();
    }
    _current_file = SD.open(path, FILE_READ);
    if (!_current_file) {
        Serial.printf("Failed to open file for reading: %s\n", path);
        return false;
    }
    return true;
}

/**
 * @brief Закрывает текущий открытый файл.
 */
void CYD28_SD::closeFile() {
    if (_current_file) {
        _current_file.close();
    }
}

/**
 * @brief Перемещает указатель чтения в открытом файле.
 * @param position Смещение от начала файла.
 * @return true, если перемещение успешно.
 */
bool CYD28_SD::seek(size_t position) {
    if (!_current_file) return false;
    return _current_file.seek(position);
}
bool CYD28_SD::seek(uint32_t pos, SeekMode mode) {
    if (!_current_file) return false;
    return _current_file.seek(pos, mode);
}
/**
 * @brief Читает данные из текущей позиции в открытом файле.
 * @param buffer Указатель на буфер для записи данных.
 * @param length Количество байт для чтения.
 * @return Количество реально прочитанных байт.
 */
size_t CYD28_SD::readData(uint8_t* buffer, size_t length) {
    if (!_current_file) return 0;
    return _current_file.read(buffer, length);
}

/**
 * @brief Возвращает текущую позицию указателя в файле.
 */
size_t CYD28_SD::getPosition() {
    if (!_current_file) return 0;
    return _current_file.position();
}

/**
 * @brief Проверяет, открыт ли в данный момент какой-либо файл.
 */
bool CYD28_SD::isFileOpen() {
    return (bool)_current_file;
}

/**
 * @brief Возвращает размер открытого файла.
 */
size_t CYD28_SD::size() {
    if (!_current_file) return 0;
    return _current_file.size();
}

/**
 * @brief Получает статус SD-карты в виде сырых данных.
 */
void CYD28_SD::status(uint8_t *mount, uint8_t *type, uint64_t *size, 
			        uint64_t *totalBytes, uint64_t *usedBytes)
{
	if (mount) *mount = _is_initialized;
	if (!_is_initialized) return;

	if (type)		*type = SD.cardType();
	if (size)		*size = SD.cardSize();
	if (totalBytes) *totalBytes = SD.totalBytes();
	if (usedBytes)	*usedBytes = SD.usedBytes();
}

/**
 * @brief Форматирует статус SD-карты в текстовую строку.
 */
void CYD28_SD::printStatus(char *buf)
{
	if (buf == NULL) return;

	if (!_is_initialized)
	{
		snprintf(buf, 20, "SD CARD not found!");
	}
	else
	{
		uint8_t type = SD.cardType();
		uint64_t size = SD.cardSize() / 1048576;
		uint64_t totalBytes = SD.totalBytes() / 1048576;
		uint64_t usedBytes = SD.usedBytes() / 1048576;

		snprintf( buf, 255,
			"SD card found\n"
			"Type: %s\n"
			"Size: %llu MB\n"
			"Total: %llu MB\n"
			"Used: %llu MB\n", 
			sdcardTypeLabels[type], 
			size, 
			totalBytes, 
			usedBytes);
	}
}