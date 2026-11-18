# Cross3DGS on Android

The Gradle project in this folder builds two apps that share the Cross3DGS
native trainer:

| App | Module | Application ID | What it does |
|---|---|---|---|
| Cross 3DGS | `:app` | `org.unclearness.cross3dgs` | Trains on bundled Mip-NeRF 360 scenes and evaluates the held-out views |
| Cross3DGS capture-to-view | `:e2e` | `org.unclearness.cross3dgs.e2e` | ARCore capture, frame extraction, SfM, training, and a free-view viewer ([E2E_README.md](E2E_README.md)) |

This page covers `:app`. Prerequisites are listed in the
[top-level README](../README.md#android).

## Requirements on the device

- Android 10 (API 29) or newer, arm64-v8a
- Vulkan 1.1 with compute workgroups of at least 256 invocations, a subgroup
  size from 8 to 128, and at least 16 KiB of shared memory
- Enough memory for the standard-quality setting (one million Gaussians); a
  device with 8 GB of RAM or more is recommended

## Build and install

The APK bundles the seven Mip-NeRF 360 scenes (`bicycle`, `bonsai`, `counter`,
`garden`, `kitchen`, `room`, `stump`; `images_4` and `sparse/0`), so download
them first. The APK is about 1.1 GB, and the first launch copies about 1.2 GB
into app storage.

```powershell
uv run python scripts/download_datasets.py mipnerf360
.\android\gradlew.bat :app:assembleRelease
adb install -r android\app\build\outputs\apk\release\app-release.apk
```

On Linux and macOS use `./android/gradlew`. The wrapper always uses `android/`
as the project folder, so it runs from the repository root. The release build
is signed with the debug key so that `adb run-as` can read its files.

## Train and evaluate from a computer

`scripts/benchmark_android.py` launches the app with the settings of a preset,
records device telemetry during the run, pulls the results, and with
`--evaluate` scores the held-out views:

```powershell
uv run python scripts/benchmark_android.py --serial <adb serial> `
    --config configs/standard_quality.json --scene bicycle `
    --output out/android/bicycle --evaluate
```

`--iterations 200` gives a quick smoke run. The output folder holds
`bench.json` (timings, memory, and the selected schedule), the renders and
targets of the held-out views, `splat.ply`, the logcat stream, telemetry, and
`common_eval.json` with PSNR, SSIM, and LPIPS.

## Use the app directly

Launch **Cross 3DGS**, pick a scene, set the number of updates, and tap
**Optimize**. The view shows the current training render; after evaluation,
the numbered buttons select a held-out camera and the mode button switches
between the render and its target.

The same settings can be passed as intent extras, for example:

```powershell
adb shell am start -n org.unclearness.cross3dgs/.MainActivity `
    --es scene garden --ei iterations 30000 --ei sh_degree 3 `
    --es density_strategy mcmc --ei gaussian_capacity 1000000 `
    --ef ssim_weight 0.2 --es view_order random --es adam_moments fp32 `
    --ez full_image_ssim true --ef mcmc_clone_offset 0.5 `
    --ef mcmc_gradient_sampling 1.5 --ei progress_every 100 --ez auto_start true
```

`views` limits the number of training views (0 = all), `training_view_seed`
sets the view-sampling seed, `init_seed` the initialization seed (default 42),
`tile_bin_capacity` the tile-Gaussian pairs budgeted per tile (default 2048,
as on the desktop trainer), and `backward_candidate` forces a gradient method
by name instead of the selected one.

## Outputs

Results are written per scene to the app's private storage:

```text
files/out/<scene>/
  bench.json
  splat.ply
  render/view_000.png ...
  target/view_000.png ...
```

Read them with `adb run-as`, from bash (PowerShell's `>` corrupts binary
output):

```sh
adb shell run-as org.unclearness.cross3dgs cat files/out/bicycle/bench.json
adb exec-out run-as org.unclearness.cross3dgs cat files/out/bicycle/splat.ply > splat.ply
```

**Export PLY** in the app copies the result to the shared
`Download/cross_3dgs/` folder, and **Share PLY** sends it to another app.
`splat.ply` uses the standard 3DGS layout and opens in common splat viewers,
for example the [SuperSplat editor](https://superspl.at/editor).
