# Builds Mob Cam for Windows: PyInstaller onedir, then wraps it into an installer .exe via Inno Setup.
# Must run on Windows - PyInstaller does not cross-compile.
# Requires Inno Setup 6 (https://jrsoftware.org/isdl.php) with ISCC.exe on PATH or at its default install path.
# Usage: .\build_windows.ps1 [version]

param(
    [string]$Version = "0.1.0"
)
$ErrorActionPreference = "Stop"

$RootDir = $PSScriptRoot
$OutDir = Join-Path $RootDir "packaging\windows\output"

Write-Host "==> Building Mob Cam v$Version for Windows"

pip install --quiet -r requirements-build.txt

$IcoPath = Join-Path $RootDir "packaging\windows\mobcam.ico"
if (-not (Test-Path $IcoPath)) {
    Write-Host "==> No icon found, generating one from logo.png"
    python "$RootDir\packaging\windows\make_ico.py"
}

Remove-Item -Recurse -Force "$RootDir\build", "$RootDir\dist" -ErrorAction SilentlyContinue
pyinstaller "$RootDir\mobcam_windows.spec" --noconfirm

$Iscc = (Get-Command "ISCC.exe" -ErrorAction SilentlyContinue).Source
if (-not $Iscc) {
    $DefaultPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (Test-Path $DefaultPath) {
        $Iscc = $DefaultPath
    } else {
        throw "ISCC.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isdl.php"
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
& $Iscc "/DMyAppVersion=$Version" "$RootDir\packaging\windows\mobcam.iss"

Write-Host "==> Done. Artifact in $OutDir"
Get-ChildItem $OutDir
