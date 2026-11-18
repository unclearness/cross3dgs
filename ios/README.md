# Cross3DGS on iOS

This folder builds two apps that share the Cross3DGS trainer, running on the
GPU through MoltenVK:

| App | CMake target | Bundle ID | What it does |
|---|---|---|---|
| cross3dgs | `cross3dgs_ios` | `org.unclearness.cross3dgs` | Trains on a bundled Mip-NeRF 360 scene and evaluates the held-out views |
| cross3dgs_e2e | `cross3dgs_e2e_ios` | `org.unclearness.cross3dgs.e2e` | ARKit capture, frame extraction, SfM, training, and a free-view viewer ([e2e/README.md](e2e/README.md)) |

This page covers the training app. Prerequisites are listed in the
[top-level README](../README.md#ios).

The training app wraps the desktop CLI (`tools/train_cli`): launch arguments
pass straight to the trainer, with `@BUNDLE@` and `@DOCUMENTS@` replaced by
the app's bundle and Documents folders. Output, the console log, thermal and
memory telemetry, and a completion record (`run_status.json`) are written to
`Documents/`. Launched without arguments, the app only writes the device's
Vulkan capabilities to `Documents/caps.json`.

## Build and install

The app bundles the scenes in `CROSS_3DGS_IOS_BUNDLE_SCENES` (default
`bicycle`; for example `"bicycle;garden"`) from `data/360_v2`, so download them
first. CMake finds `MoltenVK.xcframework` under `$HOME/VulkanSDK/*/macOS/lib`,
or set `-DCROSS_3DGS_MOLTENVK_XCFRAMEWORK=<path>`.

```sh
uv run python scripts/download_datasets.py mipnerf360 --scenes bicycle
cmake -S ios -B build-ios -G Xcode -DCMAKE_SYSTEM_NAME=iOS \
      -DCROSS_3DGS_IOS_TEAM=<apple team id>
cmake --build build-ios --config Release --target cross3dgs_ios \
      -- -allowProvisioningUpdates
xcrun devicectl list devices                    # find the device identifier
xcrun devicectl device install app --device <udid> \
      build-ios/Release-iphoneos/cross3dgs.app
```

- Signing is automatic with the team of an Apple ID signed in to Xcode
  (Settings > Accounts). A free personal team works, with 7-day profiles.
- The app requests `com.apple.developer.kernel.increased-memory-limit`. If
  your signing account rejects it, configure with
  `-DCROSS_3DGS_IOS_INCREASED_MEMORY_LIMIT=OFF`.
- To check that it compiles without signing, append `CODE_SIGNING_ALLOWED=NO`
  to the build command instead of passing a team.

The iPhone must trust the Mac, have Developer Mode enabled (Settings > Privacy
& Security > Developer Mode, offered after the first install attempt), and
stay unlocked with the app in the foreground during a run; iOS suspends
background compute.

## Train and evaluate from a Mac

```sh
uv run python scripts/benchmark_ios.py --device <udid> \
    --config configs/standard_quality.json \
    --output out/ios/bicycle --evaluate
```

The script launches the app with the preset's arguments, waits for
`run_status.json`, copies `bench.json`, the held-out renders and targets,
`splat.ply`, the console log, and telemetry, and with `--evaluate` writes
`common_eval.json` with PSNR, SSIM, and LPIPS. `--iterations 200` gives a
quick smoke run, and `--caps-only` only records the capabilities.

On the iPhone 12 Pro (6 GB of memory), the preset's one-million-Gaussian
capacity plus every image held in host memory exceeds the app's memory limit,
and iOS ends the app right after the scene loads; the script then waits until
`--timeout-minutes`. Add `--extra-arg=--stream-train-targets`: the app reads
each image from disk only when it uploads it and frees the trainer before
evaluation. Training is unchanged.

To copy a single file by hand:

```sh
xcrun devicectl device copy from --device <udid> \
      --domain-type appDataContainer --domain-identifier org.unclearness.cross3dgs \
      --source Documents/out/bicycle/bench.json --destination bench.json
```
