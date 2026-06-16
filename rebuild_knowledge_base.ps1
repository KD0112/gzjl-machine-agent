$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "[1/3] Checking docs directory..."
if (!(Test-Path -LiteralPath ".\docs")) {
    New-Item -ItemType Directory -Path ".\docs" | Out-Null
}

$docs = Get-ChildItem -LiteralPath ".\docs" -File | Where-Object {
    $_.Extension.ToLowerInvariant() -in @(".md", ".txt", ".pdf")
}

if ($docs.Count -eq 0) {
    throw "No .md, .txt, or .pdf files found in .\docs. Put your knowledge-base files there first."
}

Write-Host "Documents to index:"
$docs | ForEach-Object { Write-Host " - $($_.Name)" }

Write-Host "[2/3] Removing old Chroma vector database..."
if (Test-Path -LiteralPath ".\chroma_db") {
    Remove-Item -LiteralPath ".\chroma_db" -Recurse -Force
}

Write-Host "[3/3] Building new vector database..."
.\.venv\Scripts\python.exe build_index.py

Write-Host "Done. Restart Streamlit if it was already running."
