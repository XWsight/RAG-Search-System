$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python -m compileall -q rag_system tests scripts app.py api_app.py
    python scripts/scan_secrets.py
    python -m ruff check .
    python scripts/benchmark_sparse.py `
        evals/retrieval_cases.jsonl `
        evals/corpus/rag.md `
        evals/corpus/retrieval.md `
        evals/corpus/safety.md `
        evals/corpus/storage.md `
        --top-k 5 `
        --quality-gate evals/gates/bm25-smoke.json `
        --json-output reports/bm25-smoke.json `
        --markdown-output reports/bm25-smoke.md
    python -m coverage run -m unittest discover -s tests -v
    python -m coverage report
    git diff --check
}
finally {
    Pop-Location
}
