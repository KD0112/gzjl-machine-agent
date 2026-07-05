param(
    [int]$Port = 8510
)

$ErrorActionPreference = "Stop"

$cloudflared = Get-Command "cloudflared" -ErrorAction SilentlyContinue
if ($cloudflared) {
    Write-Host "Starting Cloudflare Tunnel for http://127.0.0.1:$Port"
    Write-Host "Copy the generated https://*.trycloudflare.com URL into the WeChat URL field."
    & $cloudflared.Source tunnel --url "http://127.0.0.1:$Port"
    exit
}

$ngrok = Get-Command "ngrok" -ErrorAction SilentlyContinue
if ($ngrok) {
    Write-Host "Starting ngrok for http://127.0.0.1:$Port"
    Write-Host "Copy the generated https://*.ngrok-free.app URL into the WeChat URL field."
    & $ngrok.Source http $Port
    exit
}

Write-Host "No tunnel tool found."
Write-Host "Install one of these first:"
Write-Host "  Cloudflare Tunnel: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
Write-Host "  ngrok:             https://ngrok.com/download"
Write-Host ""
Write-Host "After installing, run:"
Write-Host "  .\start_wechat_tunnel.ps1 -Port $Port"
