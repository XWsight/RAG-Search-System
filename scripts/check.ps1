$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python -m compileall -q rag_system tests scripts app.py api_app.py
    python scripts/scan_secrets.py
    python -m ruff check .
    python -m coverage run -m unittest discover -s tests -v
    python -m coverage report
    git diff --check
}
finally {
    Pop-Location
}
