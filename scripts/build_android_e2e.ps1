# Builds the :e2e capture-to-view APK with the shared CPU COLMAP SfM
# (libs/cross3dgs_sfm via scripts/build_android_sfm.ps1) packaged in.
param(
    [string]$VcpkgRoot = "",
    [switch]$SkipSfmBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sfmOut = Join-Path $repoRoot "build\android_sfm"
$jniRoot = Join-Path $sfmOut "jniLibs"
$jniLibrary = Join-Path $jniRoot "arm64-v8a\libcross3dgs_sfm_jni.so"

if (-not $SkipSfmBuild) {
    $sfmArgs = @{ Out = $sfmOut }
    if ($VcpkgRoot) { $sfmArgs.VcpkgRoot = $VcpkgRoot }
    & (Join-Path $PSScriptRoot "build_android_sfm.ps1") @sfmArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not (Test-Path -LiteralPath $jniLibrary -PathType Leaf)) {
    throw "SfM JNI library not found: $jniLibrary"
}

if (-not $env:JAVA_HOME) {
    $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
}
if (-not $env:ANDROID_HOME) {
    $env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA "Android\Sdk"
}

$gradle = Join-Path $repoRoot "android\gradlew.bat"
Push-Location (Join-Path $repoRoot "android")
try {
    & $gradle :e2e:assembleRelease `
        ("-PsfmJniDir=" + $jniRoot) `
        --no-daemon
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

$apk = Join-Path $repoRoot "android\e2e\build\outputs\apk\release\e2e-release.apk"
if (-not (Test-Path -LiteralPath $apk -PathType Leaf)) {
    throw "E2E APK not found after build: $apk"
}
$jar = Join-Path $env:JAVA_HOME "bin\jar.exe"
$entries = & $jar tf $apk
foreach ($required in @(
    "lib/arm64-v8a/libcross3dgs_sfm_jni.so",
    "lib/arm64-v8a/libcross3dgs_android.so"
)) {
    if ($entries -notcontains $required) {
        throw "E2E APK is missing required entry: $required"
    }
}

Write-Host "E2E APK: $apk"
Write-Host "APK SHA-256: $((Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash)"
Write-Host "SfM JNI SHA-256: $((Get-FileHash -LiteralPath $jniLibrary -Algorithm SHA256).Hash)"
