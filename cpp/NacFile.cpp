// =============================================================================
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
// NacFile.cpp
// =============================================================================
#include "NacFile.h"
#include "platform.h"

NacFile sdcard;

static const char* TAG_FILE = "NacFile";

bool NacFile::openFile(const char* path) {
    closeFile();
    m_file.open(path, std::ios::in | std::ios::binary);
    if (!m_file.is_open()) {
        ESP_LOGE(TAG_FILE, "Cannot open '%s'", path);
        return false;
    }
    // Cache file size
    m_file.seekg(0, std::ios::end);
    m_size = (uint64_t)m_file.tellg();
    m_file.seekg(0, std::ios::beg);
#ifdef DBG
    ESP_LOGI(TAG_FILE, "Opened '%s' (%llu bytes)", path, (unsigned long long)m_size);
#endif
    return true;
}

void NacFile::closeFile() {
    if (m_file.is_open()) m_file.close();
    m_size = 0;
}

bool NacFile::seek(uint64_t pos) {
    m_file.clear();  // clear any previous EOF/error flags
    m_file.seekg((std::streamoff)pos, std::ios::beg);
    return !m_file.fail();
}

size_t NacFile::readData(uint8_t* buf, size_t len) {
    if (!buf || !len) return 0;
    m_file.read(reinterpret_cast<char*>(buf), (std::streamsize)len);
    std::streamsize got = m_file.gcount();
    return (size_t)(got < 0 ? 0 : got);
}

uint64_t NacFile::getPosition() {
    std::streampos p = m_file.tellg();
    return (p < 0) ? 0 : (uint64_t)p;
}

uint64_t NacFile::size() {
    return m_size;
}
