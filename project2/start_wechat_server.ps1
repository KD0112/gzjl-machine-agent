param(
    [int]$Port = 8510,
    [string]$Token = "project2-agent-token",
    [ValidateSet("graph", "workflow")]
    [string]$Mode = "graph"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path (Split-Path -Parent $ProjectRoot) ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python runtime not found: $Python"
}

Set-Location $ProjectRoot
& $Python "wechat_server.py" --port $Port --token $Token --mode $Mode
