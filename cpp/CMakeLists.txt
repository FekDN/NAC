cmake_minimum_required(VERSION 3.16)
project(nac_executor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── ГЛОБАЛЬНАЯ НАСТРОЙКА ДЛЯ MINGW (ЖЕСТКО СТАТИЧЕСКАЯ СБОРКА) ───────────────
if(MINGW OR (WIN32 AND NOT MSVC))
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

# ── Per-compiler flags ────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(
        /W3 /wd4100 /wd4127 /wd4244 /wd4267 /wd4996 /utf-8
    )
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(
        -Wall -Wextra
        -Wno-unused-parameter
        -Wno-sign-compare
    )
endif()

option(NAC_VERBOSE_DEBUG "Enable verbose NAC debug output" OFF)
if(NAC_VERBOSE_DEBUG)
    add_compile_definitions(NAC_VERBOSE_DEBUG)
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    main.cpp
    platform.cpp
    NacFile.cpp
    TISA_VM.cpp
    op_kernels.cpp
    MEP_interpreter.cpp
)

add_executable(nac_executor ${SOURCES})
target_include_directories(nac_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── Linking & Optimizations ───────────────────────────────────────────────────
if(MSVC)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nac_executor PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_compile_options(nac_executor PRIVATE
        $<$<CONFIG:Release>:/O2 /arch:AVX2 /fp:precise /DNDEBUG /MT>
        $<$<CONFIG:Debug>:/Od /Zi /MTd /fsanitize=address>
    )

elseif(WIN32)
    # ── MinGW / GCC on Windows ──
    target_compile_options(nac_executor PRIVATE
        -fopenmp
        $<$<CONFIG:Release>:-Os -mavx2 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g3>
    )

    target_link_libraries(nac_executor PRIVATE -fopenmp)

else()
    # ── Linux / macOS ─────────────────────────────────────────────────────────
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nac_executor PRIVATE OpenMP::OpenMP_CXX)
    endif()

    find_package(Threads REQUIRED)
    target_link_libraries(nac_executor PRIVATE Threads::Threads)
    
    target_compile_options(nac_executor PRIVATE
        $<$<CONFIG:Release>:-Os -mavx2 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g3 -fsanitize=address,undefined>
    )
    target_link_options(nac_executor PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
endif()

install(TARGETS nac_executor RUNTIME DESTINATION bin)