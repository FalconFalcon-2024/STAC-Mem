$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path -LiteralPath $EnvFile) {
    foreach ($Line in Get-Content -LiteralPath $EnvFile) {
        $Value = $Line.Trim()
        if (-not $Value -or $Value.StartsWith("#") -or -not $Value.Contains("=")) {
            continue
        }
        $Name, $Setting = $Value.Split("=", 2)
        [Environment]::SetEnvironmentVariable($Name.Trim(), $Setting.Trim(), "Process")
    }
}

$Python = if ($env:STACMEM_PYTHON) {
    $env:STACMEM_PYTHON
} else {
    Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
$Port = if ($env:STACMEM_PORT) { $env:STACMEM_PORT } else { "8020" }
$Database = if ($env:STACMEM_DATABASE) {
    $env:STACMEM_DATABASE
} else {
    Join-Path $ProjectRoot "runtime\stacmem.sqlite3"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run .\scripts\quickstart.ps1 first."
}
if (-not $env:DASHSCOPE_API_KEY) {
    throw "DASHSCOPE_API_KEY is required for natural-language ingestion."
}

Set-Location $ProjectRoot
& $Python -m stacmem.standalone_cli `
    --config configs/standalone_qwen.toml `
    --database $Database `
    serve --port $Port
exit $LASTEXITCODE
