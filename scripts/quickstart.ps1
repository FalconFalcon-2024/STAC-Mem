$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python quickstart.py @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 quickstart.py @args
} else {
    throw "Python 3.11 or newer was not found on PATH."
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
