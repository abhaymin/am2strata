# What a new user does, against a download packaged from this checkout (deploy/package-portable.sh), on Windows:
# install with deploy/release/install.ps1, run the commands, then uninstall and see that it is gone and the data stayed.
#   .github/scripts/install-check.ps1 -Dist <dist folder> -Label windows-x86_64|windows-arm64      (CI: install-check.yml)
# Runs under Windows PowerShell 5.1 and PowerShell 7. It installs under %LOCALAPPDATA% and changes the user PATH: CI only.
param([Parameter(Mandatory = $true)][string]$Dist, [Parameter(Mandatory = $true)][string]$Label)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Install = Join-Path $Root "deploy\release\install.ps1"
$Ver = (Select-String -Path (Join-Path $Root "Cargo.toml") -Pattern '^version = "(.*)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$Dir = Join-Path $env:LOCALAPPDATA "am2strata"
$Bin = Join-Path $Dir "bin"
$Startup = Join-Path ([Environment]::GetFolderPath("Startup")) "am2Strata server.lnk"
$Menu = Join-Path ([Environment]::GetFolderPath("Programs")) "am2Strata.lnk"
function Step([string]$s) { Write-Host ""; Write-Host "== $s" }
function Die([string]$s) { throw "install-check: $s" }
# a native command's output as one string, whatever it exits with (the launcher's "no model yet" exits 1)
function Out([string]$exe, [string[]]$a) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { return ((& $exe @a 2>&1 | ForEach-Object { "$_" }) -join "`n") } finally { $ErrorActionPreference = $old }
}

Step "PowerShell $($PSVersionTable.PSVersion) on $env:PROCESSOR_ARCHITECTURE"

Step "the processor: an x64 PowerShell on Windows on Arm still gets the Arm64 download"
$errors = $null
$tokens = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Install, [ref]$tokens, [ref]$errors)
if ($errors.Count) { Die "install.ps1 does not parse: $($errors[0])" }
$fn = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Get-Am2StrataArch" }, $true)
Invoke-Expression $fn.Extent.Text
$cases = @(
    @("AMD64", "", "X64", "AMD64", "x86_64"), @("ARM64", "", "Arm64", "ARM64", "arm64"),
    @("AMD64", "", "X64", "ARM64", "arm64"), @("AMD64", "", "Arm64", "", "arm64"),
    @("x86", "ARM64", "", "", "arm64"), @("x86", "AMD64", "", "", "x86_64"), @("x86", "", "X86", "x86", ""))
foreach ($c in $cases) {
    $r = Get-Am2StrataArch $c[0] $c[1] $c[2] $c[3]
    if ($r -ne $c[4]) { Die "Get-Am2StrataArch $($c -join ',') gave '$r'" }
}

Step "a download whose checksum does not match is refused"
$Bad = Join-Path ([IO.Path]::GetTempPath()) ("am2bad-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory $Bad | Out-Null
Copy-Item (Join-Path $Dist "am2strata-$Ver-$Label.zip") $Bad
Set-Content (Join-Path $Bad "am2strata-$Ver-$Label.zip.sha256") ("0" * 64)
$refused = $false
try { & $Install -From $Bad } catch { $refused = "$_" -match "checksum mismatch"; Write-Host "refused: $_" }
if (-not $refused) { Die "a bad checksum was accepted" }
if (Test-Path (Join-Path $Dir "app")) { Die "something was installed after a bad checksum" }
Remove-Item -Recurse -Force $Bad

Step "install (as the one-line form runs it: a script block), picking this PC's download by itself"
$log = & ([scriptblock]::Create((Get-Content $Install -Raw))) -From $Dist 6>&1 | ForEach-Object { "$_" }
$log | ForEach-Object { Write-Host $_ }
$text = $log -join "`n"
if ($text -notmatch [regex]::Escape("Installed am2Strata v$Ver ($Label)")) { Die "it did not install $Label" }
if ($text -notmatch "ready\s+HTTPS") { Die "the installer printed no doctor report" }
$App = Join-Path $Dir "app\am2strata-$Ver-$Label"
if (-not (Test-Path "$App\poppler\bin\pdftotext.exe")) { Die "no bundled poppler" }
if (-not (Test-Path "$App\bin\am2strata-hybrid.exe")) { Die "no engine in the download" }
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (($userPath -split ";") -contains $Bin)) { Die "the user PATH does not have $Bin" }
if (-not (Test-Path $Menu)) { Die "no Start menu shortcut" }

Step "the commands"
$out = Out "$Bin\am2strata.cmd" @("--version")
Write-Host $out
if ($out.Trim() -ne "am2strata $Ver") { Die "am2strata --version said: $out" }
$out = Out "$Bin\am2strata-run.cmd" @("doctor")
Write-Host $out
if ($out -notmatch "ready\s+HTTPS") { Die "am2strata-run doctor did not report HTTPS" }
if ($out -notmatch "ready\s+Workshop") { Die "am2strata-run doctor did not find the Workshop" }
if ($out -notmatch "ready\s+PDFs in chats") { Die "doctor did not find the bundled Poppler" }
if ($out -notmatch "not here\s+Computer tab") { Die "doctor did not say the Computer tab is Linux only" }
if ((Out "$Bin\am2strata.cmd" @("doctor")) -notmatch "ready\s+HTTPS") { Die "am2strata doctor did not report" }
if ((Out "$Bin\am2strata-workshop.cmd" @("--help")) -notmatch "(?i)usage") { Die "am2strata-workshop does not start" }
$out = Out "$Bin\am2strata.cmd" @()
Write-Host $out
if ($out -notmatch "No model yet") { Die "the launcher did not ask for a model" }

Step "an update keeps models and settings; -Service and -NoService"
New-Item -ItemType Directory -Force (Join-Path $Dir "data") | Out-Null
Set-Content (Join-Path $Dir "data\settings-kept.txt") "kept"
& $Install -From $Dist -Service
if (-not (Test-Path $Startup)) { Die "-Service made no Startup shortcut" }
& $Install -From $Dist -NoService
if (Test-Path $Startup) { Die "-NoService left the Startup shortcut" }
if ((Get-Content (Join-Path $Dir "data\settings-kept.txt")) -ne "kept") { Die "an update lost the data" }

Step "uninstall: the program is gone, the data stays"
& $Install -Uninstall
foreach ($f in @((Join-Path $Dir "app"), $Bin, $Menu, $Startup)) { if (Test-Path $f) { Die "uninstall left $f" } }
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -contains $Bin) { Die "uninstall left $Bin on the user PATH" }
if ((Get-Content (Join-Path $Dir "data\settings-kept.txt")) -ne "kept") { Die "uninstall removed the data" }

Step "-Dir: another place, with its own data"
$Other = Join-Path ([IO.Path]::GetTempPath()) ("am2 other " + [guid]::NewGuid().ToString("N"))
& $Install -From $Dist -Dir $Other | Out-Null
$out = Out "$Other\bin\am2strata.cmd" @("--version")
if ($out.Trim() -ne "am2strata $Ver") { Die "-Dir: am2strata --version said: $out" }
if ((Out "$Other\bin\am2strata.cmd" @()) -notmatch "No model yet") { Die "-Dir: the launcher did not ask for a model" }
if (Test-Path (Join-Path $Dir "app")) { Die "-Dir also installed into the default folder" }
& $Install -Uninstall -Dir $Other
if ((Test-Path "$Other\app") -or (Test-Path "$Other\bin")) { Die "-Uninstall -Dir left files" }
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Other

Step "ok: $Label"
exit 0   # not the last program's exit code (the launcher's "no model yet" is 1)
