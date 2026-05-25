param(
    [string]$Top = "nac_dsp_pipeline_tb"
)

$ErrorActionPreference = "Stop"
$iverilogPath = $env:IVERILOG_BIN
if (-not $iverilogPath) {
    $cmd = Get-Command iverilog -ErrorAction SilentlyContinue
    if ($cmd) { $iverilogPath = $cmd.Source }
}
if (-not $iverilogPath -and (Test-Path "C:\iverilog\bin\iverilog.exe")) {
    $iverilogPath = "C:\iverilog\bin\iverilog.exe"
}
if (-not $iverilogPath) {
    Write-Error "iverilog was not found. Set IVERILOG_BIN or add iverilog to PATH."
}

$vvpPath = $env:VVP_BIN
if (-not $vvpPath) {
    $cmd = Get-Command vvp -ErrorAction SilentlyContinue
    if ($cmd) { $vvpPath = $cmd.Source }
}
if (-not $vvpPath -and (Test-Path "C:\iverilog\bin\vvp.exe")) {
    $vvpPath = "C:\iverilog\bin\vvp.exe"
}
if (-not $vvpPath) {
    Write-Error "vvp was not found. Set VVP_BIN or add vvp to PATH."
}

$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $root "build"
New-Item -ItemType Directory -Force $outDir | Out-Null
$out = Join-Path $outDir "$Top.vvp"

$rtl = Get-ChildItem (Join-Path $root "rtl") -Filter "*.v" | ForEach-Object { $_.FullName }
$tb = Join-Path $root "tb\$Top.v"

& $iverilogPath -g2005-sv -I (Join-Path $root "rtl") -o $out $tb $rtl
& $vvpPath $out
