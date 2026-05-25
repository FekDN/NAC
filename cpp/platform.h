#pragma once
// =============================================================================
// Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
// platform.h — x64 compatibility shims for ESP32/Arduino/FreeRTOS APIs
//
// Drop-in replacement: source files that previously included Arduino.h /
// FreeRTOS headers need only include this file instead.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <cmath>
#include <cassert>
#ifdef _WIN32
#  include <malloc.h>   // _aligned_malloc / _aligned_free (MSVC and MinGW)
#endif
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <functional>

// ─── pgmspace / PROGMEM (AVR/ESP32 flash-address macros → no-ops on x64) ───
#define PROGMEM
#define pgm_read_dword(ptr)  (*(const uint32_t*)(ptr))
#define pgm_read_word(ptr)   (*(const uint16_t*)(ptr))
#define pgm_read_byte(ptr)   (*(const uint8_t*)(ptr))

// ─── Logging ─────────────────────────────────────────────────────────────────
// Thread-safe because printf on glibc is lock-protected.
// ESP_LOGD is silent by default; define NAC_VERBOSE_DEBUG to enable it.
#define ESP_LOGE(tag, fmt, ...) \
    fprintf(stderr, "[E][%s] " fmt "\n", (tag), ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) \
    fprintf(stderr, "[W][%s] " fmt "\n", (tag), ##__VA_ARGS__)
#define ESP_LOGI(tag, fmt, ...) \
    fprintf(stdout, "[I][%s] " fmt "\n", (tag), ##__VA_ARGS__)
#ifdef NAC_VERBOSE_DEBUG
#  define ESP_LOGD(tag, fmt, ...) \
    fprintf(stdout, "[D][%s] " fmt "\n", (tag), ##__VA_ARGS__)
#else
#  define ESP_LOGD(tag, fmt, ...) do {} while(0)
#endif

// ─── Arduino Serial ──────────────────────────────────────────────────────────
struct SerialType {
    void printf(const char* fmt, ...) const {
        va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
        fflush(stdout);
    }
    void print(const char* s)         const { fputs(s ? s : "", stdout); fflush(stdout); }
    void print(const std::string& s)  const { print(s.c_str()); }
    void println(const char* s = "")  const { ::printf("%s\n", s ? s : ""); fflush(stdout); }
    void println(const std::string& s) const { println(s.c_str()); }
    template<typename T>
    void println(T val) const { ::printf("%s\n", std::to_string(val).c_str()); }
    void begin(int /*baud*/) const {}
};
extern SerialType Serial;

// ─── Heap / Memory ───────────────────────────────────────────────────────────
// ESP32 heap_caps_* → standard allocators; capability flags are ignored.
#define MALLOC_CAP_8BIT      0
#define MALLOC_CAP_INTERNAL  0
#define MALLOC_CAP_SPIRAM    0

inline void* heap_caps_malloc(size_t size, int /*caps*/) {
#ifdef _WIN32
    return size ? _aligned_malloc(size, 16) : nullptr;
#else
    return size ? malloc(size) : nullptr;
#endif
}

inline void* heap_caps_aligned_alloc(size_t align, size_t size, int /*caps*/) {
    if (!size) return nullptr;
    if (align < sizeof(void*)) align = sizeof(void*);
#ifdef _WIN32
    // Windows: both MSVC and MinGW-w64 provide _aligned_malloc.
    // posix_memalign is NOT available on Windows, even with MinGW.
    return _aligned_malloc(size, align);
#else
    // Linux / macOS / other POSIX
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) return nullptr;
    return ptr;
#endif
}

inline void heap_caps_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
inline size_t heap_caps_get_free_size(int /*caps*/) { return SIZE_MAX; }
inline size_t heap_caps_get_largest_free_block(int /*caps*/) { return SIZE_MAX; }

// alloc_fast: defined in main.cpp (same role as on ESP32; prefers PSRAM on
// ESP32, here uses posix_memalign for 16-byte alignment).
void* alloc_fast(size_t size);

// ─── ESP SDK stubs ───────────────────────────────────────────────────────────
inline bool   psramFound()              { return false; }
inline size_t esp_get_free_heap_size()  { return SIZE_MAX; }

// ─── FreeRTOS: Semaphore / Mutex ─────────────────────────────────────────────
using SemaphoreHandle_t = std::mutex*;
inline SemaphoreHandle_t xSemaphoreCreateMutex() { return new std::mutex(); }
inline void xSemaphoreTake(SemaphoreHandle_t m, int /*timeout*/) { if (m) m->lock();   }
inline void xSemaphoreGive(SemaphoreHandle_t m)                  { if (m) m->unlock(); }
inline void vSemaphoreDelete(SemaphoreHandle_t m)                { delete m; }
#define portMAX_DELAY (-1)
#define pdTRUE   1
#define pdFALSE  0
#define pdPASS   1

// ─── FreeRTOS: Task Notifications ────────────────────────────────────────────
// Mirrors the "counting notification" pattern used by nac_memory_task:
//   Compute calls   xTaskNotifyGive(g_nac_memory_task_handle)  → give()
//   Memory task calls ulTaskNotifyTake(pdTRUE, portMAX_DELAY)  → take()
//
// Both reference g_nac_memory_task_handle (defined in main.cpp).
struct TaskNotify {
    std::mutex              mtx;
    std::condition_variable cv;
    std::atomic<uint32_t>   count{0};

    // Increment counter and wake the waiting thread.
    void give() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            count.fetch_add(1, std::memory_order_relaxed);
        }
        cv.notify_one();
    }

    // Block until count > 0, then atomically read+clear (mirrors pdTRUE).
    // Returns the accumulated count (≥ 1).
    uint32_t take() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] {
            return count.load(std::memory_order_relaxed) > 0;
        });
        return count.exchange(0, std::memory_order_relaxed);
    }
};

using TaskHandle_t = TaskNotify*;

// Forward-declared; defined in main.cpp:
extern TaskHandle_t g_nac_memory_task_handle;

inline void xTaskNotifyGive(TaskHandle_t h) { if (h) h->give(); }

// ulTaskNotifyTake always waits on g_nac_memory_task_handle — this matches
// the pattern: only nac_memory_task calls it, and it IS that task's handle.
inline uint32_t ulTaskNotifyTake(int /*pdTRUE*/, int /*portMAX_DELAY*/) {
    if (g_nac_memory_task_handle)
        return g_nac_memory_task_handle->take();
    return 0;
}

// ─── FreeRTOS: Event Groups ───────────────────────────────────────────────────
struct EventGroup {
    std::mutex              mtx;
    std::condition_variable cv;
    std::atomic<uint32_t>   bits{0};

    void set(uint32_t b) {
        { std::lock_guard<std::mutex> lk(mtx); bits.fetch_or(b, std::memory_order_relaxed); }
        cv.notify_all();
    }
    void clear(uint32_t b) {
        bits.fetch_and(~b, std::memory_order_relaxed);
    }
    // Returns the matched bits (clears them if clear_on_exit == true).
    uint32_t wait(uint32_t mask, bool clear_on_exit, bool wait_all,
                  int /*timeout_ms*/ = -1) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&] {
            uint32_t v = bits.load(std::memory_order_relaxed);
            return wait_all ? ((v & mask) == mask) : ((v & mask) != 0);
        });
        uint32_t matched = bits.load(std::memory_order_relaxed) & mask;
        if (clear_on_exit) bits.fetch_and(~mask, std::memory_order_relaxed);
        return matched;
    }
};

using EventGroupHandle_t = EventGroup*;
#define BIT0  (1u << 0)
#define BIT1  (1u << 1)
#define BIT2  (1u << 2)

inline EventGroupHandle_t xEventGroupCreate()           { return new EventGroup(); }
inline void xEventGroupSetBits(EventGroupHandle_t eg, uint32_t b)   { if (eg) eg->set(b); }
inline void xEventGroupClearBits(EventGroupHandle_t eg, uint32_t b) { if (eg) eg->clear(b); }
inline uint32_t xEventGroupWaitBits(EventGroupHandle_t eg,
                                     uint32_t bits, int clr, int all, int tmo) {
    if (eg) return eg->wait(bits, clr != 0, all != 0, tmo);
    return 0;
}

// ─── FreeRTOS: Task creation stub ────────────────────────────────────────────
// main.cpp uses std::thread directly; this is only here so that any leftover
// references compile.
inline int xTaskCreatePinnedToCore(void(*f)(void*), const char*, int,
                                    void* arg, int, void**, int) {
    (void)f; (void)arg;
    return pdPASS;
}
inline void vTaskDelete(void*) {}
inline void vTaskDelay(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
#define pdMS_TO_TICKS(ms) (ms)

// ─── Arduino millis() stub ───────────────────────────────────────────────────
inline unsigned long millis() {
    using namespace std::chrono;
    return (unsigned long)duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
}

// ─── Missing FreeRTOS stubs ──────────────────────────────────────────────────
inline uint32_t uxTaskGetStackHighWaterMark(void*) { return 0; }
