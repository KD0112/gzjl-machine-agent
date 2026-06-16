$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Starting Streamlit with project virtual environment..."
.\.venv\Scripts\streamlit.exe run app.py --server.port 8502
