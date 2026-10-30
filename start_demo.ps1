param(
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found at $Python. Create .venv and install requirements.txt first."
}

Set-Location -LiteralPath $ProjectRoot
& $Python -c "import socket, sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.close()" $Port
if ($LASTEXITCODE -ne 0) {
    throw "127.0.0.1:$Port is unavailable. Choose a free port, for example: .\start_demo.ps1 -Port 8003"
}

Write-Host "Starting the project on http://127.0.0.1:$Port (single process, no reload subprocess)."
Write-Host "Open /health for readiness and /docs for API documentation."
& $Python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
