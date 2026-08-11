$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnvironmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $virtualEnvironmentPython -PathType Leaf) {
    $virtualEnvironmentPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & $pythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-CheckedGit {
    param(
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed with exit code ${LASTEXITCODE}: git $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    Invoke-CheckedPython -m compileall -q rag_system tests scripts app.py api_app.py
    Invoke-CheckedPython scripts/scan_secrets.py
    Invoke-CheckedPython -m ruff check .
    Invoke-CheckedPython -m mypy
    Invoke-CheckedPython scripts/benchmark_sparse.py `
        evals/retrieval_cases.jsonl `
        evals/corpus/rag.md `
        evals/corpus/retrieval.md `
        evals/corpus/safety.md `
        evals/corpus/storage.md `
        --top-k 5 `
        --quality-gate evals/gates/bm25-smoke.json `
        --json-output reports/bm25-smoke.json `
        --markdown-output reports/bm25-smoke.md
    Invoke-CheckedPython -m coverage run -m unittest discover -s tests -v
    Invoke-CheckedPython -m coverage report
    Invoke-CheckedGit diff --check
}
finally {
    Pop-Location
}
