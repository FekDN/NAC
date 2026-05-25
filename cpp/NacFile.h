#pragma once
// =============================================================================
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
// NacFile.h — file I/O abstraction (replaces CYD28_SD on x64)
//
// Exposes exactly the subset of the CYD28_SD interface used by the NAC runtime.
// All accesses to the open file are NOT internally serialized — callers must
// hold g_sd_card_mutex (same contract as the original ESP32 code).
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <string>

class NacFile {
public:
    NacFile() = default;
    ~NacFile() { closeFile(); }

    // ── Initialization (matches sdcard.begin() — no-op on x64) ───────────────
    void begin(int, int, int, int) {}

    // ── File open/close ───────────────────────────────────────────────────────
    bool openFile(const char* path);
    void closeFile();
    bool isFileOpen() const { return m_file.is_open(); }

    // ── Positional I/O ────────────────────────────────────────────────────────
    // seek(): sets current file position, returns true on success.
    bool seek(uint64_t pos);

    // readData(): reads up to `len` bytes into `buf`.
    // Returns the number of bytes actually read (may be < len on EOF/error).
    size_t readData(uint8_t* buf, size_t len);

    // getPosition(): returns current file position.
    uint64_t getPosition();

    // size(): returns total file size in bytes.
    uint64_t size();

private:
    std::fstream m_file;
    uint64_t     m_size = 0;
};

// Global file handle (mirrors `extern CYD28_SD sdcard` in the original code).
extern NacFile sdcard;
