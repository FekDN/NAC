@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: =============================================================================
:: build_windows.bat -- NAC Executor Windows Build Script
:: =============================================================================

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "BUILD_DIR=%SCRIPT_DIR%\build_windows"
set "BUILD_TYPE=Release"

:: -----------------------------------------------------------------------------
:: Parse arguments
:: -----------------------------------------------------------------------------
set "ARG=%~1"
if /I "%ARG%"==""        goto :args_ok
if /I "%ARG%"=="release" set "BUILD_TYPE=Release" & goto :args_ok
if /I "%ARG%"=="debug"   set "BUILD_TYPE=Debug"   & goto :args_ok
if /I "%ARG%"=="clean" (
    echo [build] Removing %BUILD_DIR% ...
    if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
    echo [build] Done.
    exit /b 0
)
echo Usage: build_windows.bat [release ^| debug ^| clean]
exit /b 1
:args_ok

echo.
echo ============================================================
echo  NAC Executor -- Windows Build  [%BUILD_TYPE%]
echo ============================================================
echo.

:: =============================================================================
:: 0. DOWNLOAD STB LIBRARIES IF MISSING
:: =============================================================================
echo [build] --- Step 0/3: Checking STB Image dependencies ---

set "STB_IMAGE=%SCRIPT_DIR%\stb_image.h"
set "STB_RESIZE=%SCRIPT_DIR%\stb_image_resize2.h"

if not exist "%STB_IMAGE%" (
    echo [build] Downloading stb_image.h...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -OutFile '%STB_IMAGE%'"
    if not exist "%STB_IMAGE%" (
        echo [build] ERROR: Failed to download stb_image.h
        exit /b 1
    )
)

if not exist "%STB_RESIZE%" (
    echo [build] Downloading stb_image_resize2.h...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h -OutFile '%STB_RESIZE%'"
    if not exist "%STB_RESIZE%" (
        echo [build] ERROR: Failed to download stb_image_resize2.h
        exit /b 1
    )
)
echo [build] STB Image headers are present.

:: =============================================================================
:: 1. FIND CMAKE
:: =============================================================================
echo [build] --- Step 1/3: Locating cmake ---
set "CMAKE_EXE="

where cmake.exe >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%C in ('where cmake.exe') do (
        if not defined CMAKE_EXE set "CMAKE_EXE=%%C"
    )
)

if not defined CMAKE_EXE (
    for %%P in (
        "%ProgramFiles%\CMake\bin\cmake.exe"
        "%ProgramFiles(x86)%\CMake\bin\cmake.exe"
        "%LOCALAPPDATA%\Programs\CMake\bin\cmake.exe"
    ) do (
        if exist "%%~P" set "CMAKE_EXE=%%~P"
    )
)

if not defined CMAKE_EXE (
    echo [build] Installing CMake via winget...
    winget install --id Kitware.CMake --silent ^
        --accept-package-agreements --accept-source-agreements
    set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
)

if not exist "%CMAKE_EXE%" (
    echo [build] ERROR: cmake not found.
    exit /b 1
)
echo [build] cmake: %CMAKE_EXE%

:: =============================================================================
:: 2. LOCATE OR INSTALL MSYS2 + MIN_GW
:: =============================================================================
echo [build] --- Step 2/3: Locating C++ compiler ---

set "MSYS2_ROOT="
for %%P in ("C:\msys64" "%USERPROFILE%\msys64" "%LOCALAPPDATA%\msys64") do (
    if exist "%%~P\usr\bin\bash.exe" set "MSYS2_ROOT=%%~P"
)

if not defined MSYS2_ROOT (
    echo [build] Installing MSYS2 via winget...
    winget install --id MSYS2.MSYS2 --silent ^
        --accept-package-agreements --accept-source-agreements
    if exist "C:\msys64\usr\bin\bash.exe" set "MSYS2_ROOT=C:\msys64"
)

if not defined MSYS2_ROOT (
    echo [build] ERROR: MSYS2 installation failed.
    exit /b 1
)

echo [build] MSYS2 detected: %MSYS2_ROOT%
set "MSYS2_BASH=%MSYS2_ROOT%\usr\bin\bash.exe"

:: -----------------------------------------------------------------------------
:: Install required packages only if missing
:: -----------------------------------------------------------------------------
echo [build] Checking required MSYS2 packages...

"%MSYS2_BASH%" -lc ^
"pacman -Q mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja > /dev/null 2>&1"

if errorlevel 1 (
    echo [build] Installing MinGW-w64 packages...

    "%MSYS2_BASH%" -lc ^
    "pacman -S --noconfirm --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja"

    if errorlevel 1 (
        echo [build] pacman failed. Removing lock and retrying...

        "%MSYS2_BASH%" -lc ^
        "rm -f /var/lib/pacman/db.lck && pacman -S --noconfirm --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja"

        if errorlevel 1 (
            echo.
            echo [build] ERROR: pacman failed inside MSYS2.
            echo   Open "MSYS2 MinGW UCRT64" and run:
            echo     pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake
            exit /b 1
        )
    )
) else (
    echo [build] Required MSYS2 packages already installed.
)

:: Add toolchain to PATH
set "PATH=%MSYS2_ROOT%\ucrt64\bin;%PATH%"

:: Select generator
set "CMAKE_GENERATOR=MinGW Makefiles"
set "TOOLCHAIN_LABEL=MinGW-w64/UCRT (MSYS2)"
if exist "%MSYS2_ROOT%\ucrt64\bin\ninja.exe" (
    set "CMAKE_GENERATOR=Ninja"
    set "TOOLCHAIN_LABEL=MinGW-w64/UCRT + Ninja"
)

:: =============================================================================
:: 3. BUILD
:: =============================================================================
echo [build] --- Step 3/3: Compilation ---
echo [build] Toolchain  : %TOOLCHAIN_LABEL%
echo [build] Generator  : %CMAKE_GENERATOR%
echo [build] Build type : %BUILD_TYPE%
echo.

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

"%CMAKE_EXE%" -G "%CMAKE_GENERATOR%" ^
    -S "%SCRIPT_DIR%" ^
    -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if errorlevel 1 (
    echo [build] ERROR: cmake configure failed.
    exit /b 1
)

:: Determine number of CPU cores
set "JOBS=4"
for /f "tokens=2 delims==" %%N in (
    'wmic cpu get NumberOfLogicalProcessors /value 2^>nul ^| findstr "="'
) do set "JOBS=%%N"

echo [build] Compiling with %JOBS% parallel jobs...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config %BUILD_TYPE% --parallel %JOBS%

if errorlevel 1 (
    echo [build] ERROR: Build failed.
    exit /b 1
)

:: Locate resulting binary
set "BIN=%BUILD_DIR%\nac_executor.exe"
if not exist "%BIN%" if exist "%BUILD_DIR%\%BUILD_TYPE%\nac_executor.exe" (
    set "BIN=%BUILD_DIR%\%BUILD_TYPE%\nac_executor.exe"
)

if not exist "%BIN%" (
    echo [build] ERROR: nac_executor.exe not found.
    exit /b 1
)

echo [build] Stripping binary...

where strip.exe >nul 2>&1
if not errorlevel 1 (
    strip.exe --strip-unneeded "%BIN%"
) else if exist "%MSYS2_ROOT%\ucrt64\bin\strip.exe" (
    "%MSYS2_ROOT%\ucrt64\bin\strip.exe" --strip-all "%BIN%"
)

echo.
echo ============================================================
echo  Build successful!
echo  %BIN%
echo ============================================================
echo.
echo Usage:
echo   %BIN% model.nac "your prompt"
echo   %BIN% --interactive
echo.

endlocal
pause