# am2Strata installer for Windows (x64): the latest release, checked against its SHA-256.
#
#   irm https://github.com/abhaymin/am2strata/releases/latest/download/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://github.com/abhaymin/am2strata/releases/latest/download/install.ps1))) -Version v0.1.0
#   ... -Uninstall      removes the program; your models and settings stay in %LOCALAPPDATA%\am2strata
#
# Installs into %LOCALAPPDATA%\am2strata\app, adds `am2strata` to your PATH and a Start menu shortcut. PDFs work out of
# the box (the download carries Poppler); the browser tool uses the Edge or Chrome already on this PC. No admin rights.
param([string]$Version = "", [switch]$Uninstall)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with its progress bar
$Repo = "abhaymin/am2strata"
$Data = if ($env:AM2STRATA_DATA) { $env:AM2STRATA_DATA } else { Join-Path $env:LOCALAPPDATA "am2strata" }
$Apps = Join-Path $Data "app"
$Bin = Join-Path $Data "bin"
$Shortcut = Join-Path ([Environment]::GetFolderPath("Programs")) "am2Strata.lnk"

function Set-UserPath([string]$dir, [bool]$add) {
    $p = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @($p -split ";" | Where-Object { $_ -and $_ -ne $dir })
    if ($add) { $parts += $dir }
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
}

if ($Uninstall) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Apps, $Bin, $Shortcut
    Set-UserPath $Bin $false
    Write-Host "Removed am2Strata. Your models and settings are still in $Data."
    return
}

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    throw "am2Strata's Windows download is for x64 PCs."
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (-not $Version) {
    $Version = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
}
$Ver = $Version.TrimStart("v")
$Name = "am2strata-$Ver-windows-x86_64"
$Url = "https://github.com/$Repo/releases/download/$Version"
$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("am2strata-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $Tmp | Out-Null
try {
    Write-Host "Downloading am2Strata $Version for Windows..."
    Invoke-WebRequest "$Url/$Name.zip" -OutFile "$Tmp\$Name.zip" -UseBasicParsing
    Invoke-WebRequest "$Url/$Name.zip.sha256" -OutFile "$Tmp\sum" -UseBasicParsing
    $want = ((Get-Content "$Tmp\sum" -Raw).Trim() -split "\s+")[0].ToLower()
    $got = (Get-FileHash "$Tmp\$Name.zip" -Algorithm SHA256).Hash.ToLower()
    if ($want -ne $got) { throw "checksum mismatch ($got, expected $want): nothing was installed" }

    New-Item -ItemType Directory -Force $Apps, $Bin | Out-Null
    Get-ChildItem $Apps -Directory -Filter "am2strata-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Expand-Archive "$Tmp\$Name.zip" -DestinationPath $Apps -Force
    $App = Join-Path $Apps $Name
    Set-Content -Encoding ASCII (Join-Path $Bin "am2strata.cmd") "@echo off`r`ncall `"$App\am2strata.cmd`" %*"
    Set-UserPath $Bin $true
    $env:Path = "$env:Path;$Bin"

    # Start menu: the server in a window of its own, then the web app in the default browser
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($Shortcut)
    $lnk.TargetPath = "$env:ComSpec"
    $lnk.Arguments = "/k start `"`" http://127.0.0.1:8000 & `"$App\am2strata.cmd`""
    $lnk.WorkingDirectory = $App
    $lnk.Description = "am2Strata: AI models on this PC"
    $lnk.Save()
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Tmp
}

Write-Host "Installed $Version in $App"
Write-Host ""
Write-Host "Next, in a new terminal:"
Write-Host "  am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M    # a small model to start with"
Write-Host "  am2strata                                                           # then open http://127.0.0.1:8000"
Write-Host "Or start it from the Start menu: am2Strata."
