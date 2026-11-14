# am2Strata installer for Windows (x64 and Arm64): the latest release, checked against its SHA-256. No admin rights.
#
#   irm https://github.com/abhaymin/am2strata/releases/latest/download/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://github.com/abhaymin/am2strata/releases/latest/download/install.ps1))) -Version v0.1.8
#
#   -Version X    install release X instead of the latest
#   -Dir DIR      install into DIR instead of %LOCALAPPDATA%\am2strata (models and settings stay there too)
#   -From DIR     install from a folder that already holds the download and its .sha256 (no network)
#   -Service      start the server when you sign in (a shortcut in your Startup folder; no admin rights)
#   -NoService    no start at sign-in (the default); also removes the one an earlier -Service made
#   -WithDeps     install what the optional features miss with winget: the command is shown first and runs only
#                 after you confirm (-Yes: without asking)
#   -Uninstall    remove the program, its commands and shortcuts; your models and settings stay
#
# Installs into %LOCALAPPDATA%\am2strata\app, adds `am2strata`, `am2strata-run` and `am2strata-workshop` to your PATH
# and a Start menu shortcut. PDFs work out of the box (the download carries Poppler); the browser tool uses the Edge
# or Chrome already on this PC.
param([string]$Version = "", [string]$Dir = "", [string]$From = "", [switch]$Service, [switch]$NoService,
      [switch]$WithDeps, [switch]$Yes, [switch]$Uninstall)

# "arm64" or "x86_64" from what Windows says in four places ("" when none is a 64-bit PC). Arm64 from any of them
# wins: an x64 PowerShell on Windows on Arm runs emulated and reports AMD64 for itself, and the native build is the
# right one there.
function Get-Am2StrataArch([string]$Process, [string]$Wow64, [string]$Os, [string]$Machine) {
    $all = @($Wow64, $Os, $Machine, $Process)
    foreach ($a in $all) { if ($a -match '^(ARM64|AARCH64)$') { return "arm64" } }
    foreach ($a in $all) { if ($a -match '^(AMD64|X64|X86_64)$') { return "x86_64" } }
    return ""
}

# everything in a function: `irm | iex` runs this file in your own session, and nothing here should stay set in it
function Install-Am2Strata {
    param([string]$Version, [string]$Dir, [string]$From, [bool]$Service, [bool]$NoService, [bool]$WithDeps, [bool]$Yes, [bool]$Uninstall)
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with its progress bar
    $Repo = "abhaymin/am2strata"
    $Data = if ($Dir) { $Dir } elseif ($env:AM2STRATA_DATA) { $env:AM2STRATA_DATA } else { Join-Path $env:LOCALAPPDATA "am2strata" }
    $Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)   # absolute, from where you are
    if ($Data -match '[%"&<>^|]') { throw "the folder $Data has a character the launchers cannot carry (one of % `" & < > ^ |): choose another" }
    $Apps = Join-Path $Data "app"
    $Bin = Join-Path $Data "bin"
    $Shortcut = Join-Path ([Environment]::GetFolderPath("Programs")) "am2Strata.lnk"
    $Startup = Join-Path ([Environment]::GetFolderPath("Startup")) "am2Strata server.lnk"
    $Mark = "written by am2Strata's install.ps1"

    function Set-UserPath([string]$dir, [bool]$add) {
        $p = [Environment]::GetEnvironmentVariable("Path", "User")
        $parts = @($p -split ";" | Where-Object { $_ -and $_ -ne $dir })
        if ($add) { $parts += $dir }
        [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    }
    # a shortcut is ours when it points into this install
    function Remove-OurShortcut([string]$path) {
        if (-not (Test-Path $path)) { return $false }
        $l = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
        if (([string]$l.Arguments + [string]$l.WorkingDirectory + [string]$l.TargetPath).Contains($Data)) { Remove-Item -Force $path; return $true }
        return $false
    }

    if ($Uninstall) {
        [void](Remove-OurShortcut $Shortcut)
        [void](Remove-OurShortcut $Startup)
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Apps, $Bin
        Set-UserPath $Bin $false
        Write-Host "Removed am2Strata. Your models and settings are still in $Data (delete that folder to remove them too)."
        return
    }

    $osArch = ""
    try { $osArch = [string][System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture } catch { $osArch = "" }
    $machineArch = ""
    try {
        $machineArch = [string](Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" -ErrorAction Stop).PROCESSOR_ARCHITECTURE
    } catch { $machineArch = "" }
    $arch = Get-Am2StrataArch $env:PROCESSOR_ARCHITECTURE $env:PROCESSOR_ARCHITEW6432 $osArch $machineArch
    if (-not $arch) { throw "am2Strata's Windows downloads are for 64-bit PCs (x64 and Arm64); this one says $env:PROCESSOR_ARCHITECTURE." }
    # an Arm64 PC also runs the x64 download (emulated, slower): the second choice when a release has no Arm64 one
    $labels = if ($arch -eq "arm64") { @("windows-arm64", "windows-x86_64") } else { @("windows-x86_64") }

    $Tmp = Join-Path ([IO.Path]::GetTempPath()) ("am2strata-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force $Tmp | Out-Null
    try {
        $zip = $null
        $sum = $null
        if ($From) {
            if (-not (Test-Path $From -PathType Container)) { throw "-From $From is not a folder" }
            foreach ($candidate in $labels) {
                $pattern = if ($Version) { "am2strata-" + $Version.TrimStart("v") + "-$candidate.zip" } else { "am2strata-*-$candidate.zip" }
                $found = Get-ChildItem $From -File -Filter $pattern | Sort-Object Name | Select-Object -Last 1
                if ($found) { $zip = $found.FullName; $Label = $candidate; break }
            }
            if (-not $zip) { throw "-From $From has no am2strata-<version>-$($labels[0]).zip" }
            $sum = "$zip.sha256"
            if (-not (Test-Path $sum)) { throw "no checksum file beside ${zip}: nothing was installed" }
            $Name = [IO.Path]::GetFileNameWithoutExtension($zip)
            $Version = "v" + $Name.Substring("am2strata-".Length, $Name.Length - "am2strata-".Length - $Label.Length - 1)
        } else {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            if (-not $Version) {
                $Version = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
                if (-not $Version) { throw "could not find the latest release: try -Version v0.1.8" }
            }
            if (-not $Version.StartsWith("v")) { $Version = "v$Version" }
            $Url = "https://github.com/$Repo/releases/download/$Version"
            foreach ($candidate in $labels) {
                $Name = "am2strata-" + $Version.TrimStart("v") + "-$candidate"
                Write-Host "Downloading am2Strata $Version for $candidate..."
                try {
                    Invoke-WebRequest "$Url/$Name.zip" -OutFile "$Tmp\$Name.zip" -UseBasicParsing
                    $zip = "$Tmp\$Name.zip"
                    $Label = $candidate
                    break
                } catch {
                    if ($candidate -eq $labels[-1]) { throw "no download $Name.zip in release $Version ($Url): $($_.Exception.Message)" }
                    Write-Host "Release $Version has no $candidate download; taking the x64 one, which Windows on Arm runs emulated."
                }
            }
            $sum = "$Tmp\sum"
            Invoke-WebRequest "$Url/$Name.zip.sha256" -OutFile $sum -UseBasicParsing
        }
        $want = ((Get-Content $sum -Raw).Trim() -split "\s+")[0].ToLower()
        $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if (-not $want -or $want -ne $got) { throw "checksum mismatch ($got, expected $want): nothing was installed" }
        Write-Host "Checked its SHA-256."

        # unpacked beside its final place, then moved: an install that stops half way leaves the old one working
        New-Item -ItemType Directory -Force $Apps, $Bin | Out-Null
        $Unpack = Join-Path $Apps ".unpack"
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Unpack
        Expand-Archive $zip -DestinationPath $Unpack -Force
        if (-not (Test-Path (Join-Path $Unpack "$Name\am2strata.cmd"))) { throw "the download has no $Name\am2strata.cmd: nothing was installed" }
        Get-ChildItem $Apps -Directory -Filter "am2strata-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        $App = Join-Path $Apps $Name
        Move-Item (Join-Path $Unpack $Name) $App
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Unpack

        # the commands: small .cmd files that carry the data folder (for -Dir) and call the installed program
        $head = "@echo off`r`nrem $Mark`r`nsetlocal`r`nif `"%AM2STRATA_DATA%`"==`"`" set `"AM2STRATA_DATA=$Data`"`r`n"
        Set-Content -Encoding ASCII (Join-Path $Bin "am2strata.cmd") ($head + "call `"$App\am2strata.cmd`" %*`r`nexit /b %ERRORLEVEL%")
        Set-Content -Encoding ASCII (Join-Path $Bin "am2strata-run.cmd") ($head + "`"$App\bin\am2strata-run.exe`" %*`r`nexit /b %ERRORLEVEL%")
        Set-Content -Encoding ASCII (Join-Path $Bin "am2strata-workshop.cmd") ($head + "`"$App\bin\am2strata-workshop.exe`" %*`r`nexit /b %ERRORLEVEL%")
        Set-UserPath $Bin $true
        if (-not (($env:Path -split ";") -contains $Bin)) { $env:Path = "$env:Path;$Bin" }

        # Start menu: the server in a window of its own, then the web app in the default browser
        $ws = New-Object -ComObject WScript.Shell
        $lnk = $ws.CreateShortcut($Shortcut)
        $lnk.TargetPath = "$env:ComSpec"
        $lnk.Arguments = "/k start `"`" http://127.0.0.1:8000 & `"$Bin\am2strata.cmd`""
        $lnk.WorkingDirectory = $App
        $lnk.Description = "am2Strata: AI models on this PC"
        $lnk.Save()

        if ($Service) {
            $lnk = $ws.CreateShortcut($Startup)
            $lnk.TargetPath = "$env:ComSpec"
            $lnk.Arguments = "/c `"$Bin\am2strata.cmd`""
            $lnk.WorkingDirectory = $App
            $lnk.WindowStyle = 7     # minimized
            $lnk.Description = "am2Strata's server, started when you sign in"
            $lnk.Save()
            Write-Host "The server starts when you sign in (a shortcut in your Startup folder: $Startup), once a model is there."
        } elseif ($NoService) {
            if (Remove-OurShortcut $Startup) { Write-Host "Removed the start at sign-in." }
        }
    } finally {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Tmp
    }
    Write-Host "Installed am2Strata $Version ($Label) in $App"
    if (Test-Path (Join-Path $Data "strata-*.json")) { Write-Host "Your models and settings in $Data were kept as they are." }

    # what the optional features need here: the product's own report, so the installer and `am2strata doctor` agree
    $run = Join-Path $App "bin\am2strata-run.exe"
    $dataDir = Join-Path $Data "data"
    Write-Host ""
    $ErrorActionPreference = "Continue"      # a native program writing to stderr is not this script's error
    $report = $null
    try { $report = & $run doctor --data-dir $dataDir 2>$null } catch { $report = $null }
    $pkgs = @()
    if ($LASTEXITCODE -eq 0 -and $report) {
        $report | ForEach-Object { Write-Host $_ }
        $line = & $run doctor --data-dir $dataDir --packages winget 2>$null
        if ($line) { $pkgs = @(([string]$line).Trim() -split "\s+" | Where-Object { $_ }) }
    } else {
        # a release from before `am2strata-run doctor`
        Write-Host "  ready     HTTPS                           built in"
        Write-Host "  ready     PDFs in chats                   the download carries Poppler"
        if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
            Write-Host "  ready     Video from API clients          ffmpeg found"
        } else {
            Write-Host "  missing   Video from API clients          needs ffmpeg (winget install -e --id Gyan.FFmpeg)"
            $pkgs = @("Gyan.FFmpeg")
        }
        Write-Host "  not here  Computer tab (sandbox desktop)  Linux only"
        Write-Host "  not here  Sandbox VM                      Linux on x86_64 only"
    }
    if ($WithDeps) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($pkgs.Count -eq 0) {
            Write-Host "-WithDeps: nothing to install."
        } elseif (-not $winget) {
            Write-Host "-WithDeps needs winget (App Installer, from the Microsoft Store), which this PC does not have."
        } else {
            Write-Host "-WithDeps: this installs what is missing:"
            foreach ($p in $pkgs) { Write-Host "  winget install -e --id $p" }
            $go = $Yes
            if (-not $go) { $go = (Read-Host "Run it now? [y/N]") -match '^(y|yes)$' }
            if ($go) {
                foreach ($p in $pkgs) { & winget install -e --id $p }
            } else {
                Write-Host "Not run. am2Strata works without it; the features above stay off until you install them."
            }
        }
    } elseif ($pkgs.Count -gt 0) {
        Write-Host "This installer does that for you when you run it again with -WithDeps (it shows the command and asks first)."
    }

    Write-Host ""
    Write-Host "Next, in a new terminal:"
    Write-Host "  am2strata pull mistralai/Ministral-3-3B-Instruct-2512-GGUF Q4_K_M    # a small model to start with"
    Write-Host "  am2strata                                                           # then open http://127.0.0.1:8000"
    Write-Host "  am2strata doctor                                                    # this report again, any time"
    Write-Host "Or start it from the Start menu: am2Strata. Uninstall: this installer with -Uninstall."
}

Install-Am2Strata -Version $Version -Dir $Dir -From $From -Service $Service.IsPresent -NoService $NoService.IsPresent `
    -WithDeps $WithDeps.IsPresent -Yes $Yes.IsPresent -Uninstall $Uninstall.IsPresent
