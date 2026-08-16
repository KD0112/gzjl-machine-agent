[CmdletBinding()]
param(
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$streamlitExe = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$logRoot = Join-Path $localAppData "Temp\jinlong-agent-sites"
$launchStamp = Get-Date -Format "yyyyMMdd-HHmmss"

if (-not (Test-Path -LiteralPath $streamlitExe)) {
    throw "Streamlit was not found in the project virtual environment: $streamlitExe"
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$apps = @(
    @{
        Name = "Main customer-service app"
        Port = 8502
        WorkDir = $projectRoot
        Script = "app.py"
    },
    @{
        Name = "Multi-tool Agent"
        Port = 8503
        WorkDir = Join-Path $projectRoot "project2"
        Script = "app.py"
    },
    @{
        Name = "RAG debug console"
        Port = 8504
        WorkDir = $projectRoot
        Script = "rag_debug_app.py"
    }
)

function Get-ListeningProcessId {
    param([int]$Port)

    $connection = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port } |
        Select-Object -First 1

    if ($null -eq $connection) {
        return $null
    }

    return $connection.OwningProcess
}

foreach ($app in $apps) {
    $existingProcessId = Get-ListeningProcessId -Port $app.Port
    if ($null -ne $existingProcessId) {
        Write-Host ("[{0}] Already running: http://127.0.0.1:{1} (PID {2})" -f $app.Name, $app.Port, $existingProcessId)
        continue
    }

    $stdoutPath = Join-Path $logRoot ("port-{0}-{1}.stdout.log" -f $app.Port, $launchStamp)
    $stderrPath = Join-Path $logRoot ("port-{0}-{1}.stderr.log" -f $app.Port, $launchStamp)
    $arguments = @(
        "run",
        $app.Script,
        "--server.address",
        "127.0.0.1",
        "--server.port",
        [string]$app.Port,
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none"
    )

    Start-Process `
        -FilePath $streamlitExe `
        -ArgumentList $arguments `
        -WorkingDirectory $app.WorkDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath | Out-Null

    $started = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($null -ne (Get-ListeningProcessId -Port $app.Port)) {
            $started = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $started) {
        $errorTail = Get-Content -LiteralPath $stderrPath -Tail 20 -ErrorAction SilentlyContinue
        throw "[{0}] Failed to start. Log: {1}`n{2}" -f $app.Name, $stderrPath, ($errorTail -join [Environment]::NewLine)
    }

    Write-Host ("[{0}] Started: http://127.0.0.1:{1}" -f $app.Name, $app.Port)
}

Write-Host ("Log directory: {0}" -f $logRoot)

if ($OpenBrowser) {
    foreach ($app in $apps) {
        Start-Process ("http://127.0.0.1:{0}/" -f $app.Port)
    }
}
