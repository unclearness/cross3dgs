# Builds libcross3dgs_sfm_jni.so: the shared CPU COLMAP sparse SfM
# (libs/cross3dgs_sfm, COLMAP fork in external/colmap) plus its JNI wrapper,
# for the Android :e2e app. Android counterpart of scripts/build_ios_sfm.sh.
#
# Output: <Out>\jniLibs\arm64-v8a\libcross3dgs_sfm_jni.so (stripped); pass
# <Out>\jniLibs to Gradle as -PsfmJniDir (scripts/build_android_e2e.ps1 does).
#
# Usage: .\scripts\build_android_sfm.ps1 [-VcpkgRoot <vcpkg root>] [-Out build\android_sfm]
# (-VcpkgRoot defaults to $env:VCPKG_ROOT)
param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Out = "",
    [switch]$BlasSmoke,
    [ValidateSet("portable", "tuned")][string]$Blas = "tuned"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sfmSource = Join-Path $repoRoot "libs\cross3dgs_sfm"
if (-not $Out) { $Out = Join-Path $repoRoot "build\android_sfm" }
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$buildDir = Join-Path $Out "build"

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "external\colmap\CMakeLists.txt"))) {
    throw "external/colmap is missing; run git submodule update --init external/colmap"
}
if (-not $VcpkgRoot -or
    -not (Test-Path -LiteralPath (Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"))) {
    throw "-VcpkgRoot (or VCPKG_ROOT) must contain a bootstrapped vcpkg: $VcpkgRoot"
}
if (-not $env:ANDROID_NDK_HOME) {
    $env:ANDROID_NDK_HOME = Join-Path $env:LOCALAPPDATA "Android\Sdk\ndk\28.2.13676358"
}
$llvm = Join-Path $env:ANDROID_NDK_HOME "toolchains\llvm\prebuilt\windows-x86_64"
$omp = Get-ChildItem -Path (Join-Path $llvm "lib\clang\*\lib\linux\aarch64\libomp.a") |
    Select-Object -First 1
if (-not $omp) { throw "NDK libomp.a (aarch64) not found under $llvm" }

$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) {
    $ninja = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ninja) { throw "ninja not found (PATH or Visual Studio CMake component)" }

$triplet = "arm64-android"
$vcpkgLib = ($buildDir -replace '\\', '/') + "/vcpkg_installed/$triplet/lib"
$cmakeArgs = @(
    "-S", $sfmSource,
    "-B", $buildDir,
    "-G", "Ninja",
    ("-DCMAKE_MAKE_PROGRAM=" + $ninja),
    ("-DCMAKE_TOOLCHAIN_FILE=" + $VcpkgRoot + "\scripts\buildsystems\vcpkg.cmake"),
    "-DVCPKG_TARGET_TRIPLET=$triplet",
    ("-DVCPKG_OVERLAY_TRIPLETS=" + (Join-Path $sfmSource "triplets")),
    ("-DVCPKG_OVERLAY_PORTS=" + (Join-Path $sfmSource "ports\android")),
    ("-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=" + $env:ANDROID_NDK_HOME + "\build\cmake\android.toolchain.cmake"),
    "-DANDROID_ABI=arm64-v8a",
    "-DANDROID_PLATFORM=android-28",
    "-DANDROID_STL=c++_static",
    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DVCPKG_MANIFEST_FEATURES=$(if ($Blas -eq 'portable') { 'portable-blas' } else { '' })",
    "-DCROSS3DGS_SFM_BUILD_BLAS_SMOKE=$($BlasSmoke.IsPresent.ToString().ToUpperInvariant())",
    ("-DLAPACK_LIBRARIES=" + $vcpkgLib + "/liblapack.a;" + $vcpkgLib + "/libf2c.a"),
    ("-DOpenMP_omp_LIBRARY=" + $omp.FullName)
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build $buildDir --target cross3dgs_sfm_jni
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BlasSmoke) {
    & cmake --build $buildDir --target cross3dgs_sfm_blas_smoke
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "BLAS correctness smoke: $(Join-Path $buildDir 'cross3dgs_sfm_blas_smoke')"
}

$jniDir = Join-Path $Out "jniLibs\arm64-v8a"
New-Item -ItemType Directory -Force $jniDir | Out-Null
$so = Join-Path $jniDir "libcross3dgs_sfm_jni.so"
Copy-Item -LiteralPath (Join-Path $buildDir "libcross3dgs_sfm_jni.so") -Destination $so -Force
& (Join-Path $llvm "bin\llvm-strip.exe") $so
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ("SfM JNI: " + $so + " (" + [math]::Round((Get-Item $so).Length / 1MB, 1) + " MB)")
Write-Host "SfM JNI SHA-256: $((Get-FileHash -LiteralPath $so -Algorithm SHA256).Hash)"
