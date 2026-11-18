# cross3dgs_sfm

CPU COLMAP sparse SfM for the capture-to-view apps. Android (`android/e2e`) and
iOS (`ios/e2e`) run this one implementation, so the SfM configuration exists
in exactly one place.

| Path | Role |
|---|---|
| `include/cross3dgs_sfm.h` | C API: `cross3dgs_sfm_run`, `cross3dgs_sfm_cancel`, log callback, exit codes, live preview |
| `src/cross3dgs_sfm.cc` | Feature extraction → matching → mapping with the recorded options and AR frame priors |
| `src/sfm_preview.{h,cc}` | Live preview renderer (see below) |
| `jni/cross3dgs_sfm_jni.cc` | Android-only JNI wrapper for `org.unclearness.cross3dgs.SfmNativeBridge` |
| `CMakeLists.txt` | Standalone project (iOS or Android only); adds the COLMAP fork from `external/colmap` |
| `vcpkg.json`, `vcpkg-configuration.json` | Dependency manifest and pinned vcpkg baseline; openblas/clapack on Android only (iOS uses Accelerate) |
| `triplets/` | `arm64-android` (API 28, static, 16 KB pages) and `arm64-ios-cross3dgs` |
| `ports/android/`, `ports/ios/` | Per-platform overlay ports: clapack and [OpenBLAS](ports/android/openblas/README.md) for Android, gklib without `system()` for iOS |

COLMAP comes from the `external/colmap` submodule
(`unclearness/colmap`, branch `android-port`): COLMAP 4.1.1 plus build fixes
for Android/iOS (CHOLMOD-free Eigen fallback, Android detection, optional
OpenGL lookup).

## Live preview

`cross3dgs_sfm_set_preview(fn, user, width, height)` makes the next runs
deliver RGBA frames (at most 4/s, on a library thread, none after
`cross3dgs_sfm_run` returns) with a two-line caption, so both apps show the
same picture while SfM works:

| Stage | Picture | Caption, second line |
|---|---|---|
| SIFT | latest extracted image (dimmed) with its keypoints; circle color and radius = detection scale | `Circle color/size = scale (cyan fine → yellow coarse)` |
| Matching | image-pair matrix in capture order, cell color = geometrically verified inliers (log, inferno; color bar right), dark red = verified without geometry, dark grey = not yet verified; beside it the newest verified pair, stacked, with up to 100 inlier matches drawn as lines and the pair marked in the matrix | `Color = match quality (inliers/pair, log; dark red = none)` |
| Mapping | growing sparse model and camera frusta on a turntable about the frame priors' gravity, newest camera orange; the final model is the last frame | `Frusta = cameras (orange = newest) · up = gravity` |

It only observes: SQLite reads go through a separate read-only connection
(the database is in WAL mode) and the model comes from copies taken in the
incremental pipeline's registration callbacks, so the outputs are unchanged.
It does share the CPU with SfM; the apps record whether it ran in
`pipeline_config.sfm_live_preview` and can turn it off per run
(Android `--ez sfm_preview false`, iOS `-sfm_preview NO`).

## Build

The library is not part of the desktop build; each platform has a script that
configures this directory with the vcpkg toolchain chained to the platform
toolchain:

```powershell
git submodule update --init external/colmap
.\scripts\build_android_sfm.ps1 [-VcpkgRoot <vcpkg root>]   # -> build\android_sfm\jniLibs\arm64-v8a\libcross3dgs_sfm_jni.so
```

```sh
scripts/build_ios_sfm.sh --vcpkg-root <vcpkg root>         # -> build-ios-sfm/libcross3dgs_sfm.a
```

`scripts/build_android_e2e.ps1` runs the Android step before building the APK;
see [`android/E2E_README.md`](../../android/E2E_README.md) and
[`ios/e2e/README.md`](../../ios/e2e/README.md).

Android OpenBLAS defaults to ARMv8-compatible CORTEXA57 NEON kernels and allocator
locking for concurrent FAISS callers, with one BLAS thread per caller. The
library logs its actual BLAS configuration, kernel and thread count at the
start of each run.
The `portable-blas` manifest feature selects the safe ARMV8 comparison build;
both variants retain allocator locking and single-threaded BLAS. The benchmark
sets explicit seed and 2/2/2 workers and verifies the actual iOS memory-bound
thread choices. Ordinary capture apps retain their existing defaults.
Rebuild the SfM library before packaging an APK; `-SkipSfmBuild` can otherwise
package the previous library.

To build and run the Android BLAS correctness smoke (does not install or
replace an app):

```powershell
.\scripts\build_android_sfm.ps1 -BlasSmoke
adb push build/android_sfm/build/cross3dgs_sfm_blas_smoke /data/local/tmp/cross3dgs_sfm_blas_smoke
adb shell chmod 755 /data/local/tmp/cross3dgs_sfm_blas_smoke
adb shell /data/local/tmp/cross3dgs_sfm_blas_smoke
```

This checks concurrent arithmetic correctness, not end-to-end performance.
A speed or quality comparison should use identical images and priors,
independent repetitions, and record peak memory and thermal/clock state.

## Configuration

Changing any option in `src/cross3dgs_sfm.cc` changes the recorded SfM
configuration of both apps (`ProjectEvidence.kt` names it
`incremental_colmap_cpu_sift8192_single_camera_v1`). The only per-platform
difference is that iOS bounds the extraction and matching thread counts by
`os_proc_available_memory()` to stay under the jetsam limit; Android keeps one
thread per core. Per-image SIFT features do not depend on the thread count;
matching and geometric verification vary run to run on both platforms
regardless of it.
