# Cross3DGS capture-to-view app (Android)

The `:e2e` module runs the whole pipeline on the phone:

```text
Capture -> Extract frames -> SfM (SIFT, matching, mapping) -> 3DGS optimization -> Free-view viewer
```

It records a video with ARCore, extracts sharp frames, runs CPU COLMAP
([`libs/cross3dgs_sfm`](../libs/cross3dgs_sfm/README.md)) with the ARCore
camera positions, gravity, and intrinsics as priors, trains with the
Cross3DGS core, and opens an orbit viewer. It is a separate app from the
training app `:app` ([README.md](README.md)).

## Build

Prerequisites are listed in the [top-level README](../README.md#android).
The script builds the SfM library (`scripts/build_android_sfm.ps1`, which
needs `external/colmap`, a bootstrapped vcpkg, and NDK 28.2) and then the APK:

```powershell
git submodule update --init external/colmap
.\scripts\build_android_e2e.ps1 -VcpkgRoot <vcpkg root>
adb install -r android\e2e\build\outputs\apk\release\e2e-release.apk
```

`-SkipSfmBuild` reuses an SfM library built earlier. The device needs ARCore
(Google Play Services for AR).

## Use

1. Tap **Extract frames** to set how many frames to extract (8 to 240,
   default 60), and **3DGS settings** or **Preset** to choose the training
   budget:
   - Demo: SH degree 3, MCMC, 250k Gaussians, 3k updates
   - Full: the standard-quality setting, SH degree 3, MCMC, 1M Gaussians,
     30k updates
2. Tap **Capture**, walk slowly around the subject, and tap **Stop
   recording**. The remaining stages run automatically, and the status line
   shows the current step and its progress. During optimization the view
   shows the training photo, the current render of the same view, and the
   per-pixel loss.
3. In the viewer, drag to rotate, pinch to zoom, and drag with two fingers to
   pan. The view is level with gravity and orbits the subject.

**Run saved capture** processes an earlier capture again in a new run, and
**Replay frozen SfM input** reruns SfM onward from an earlier run's extracted
frames. During SfM the view shows the SIFT keypoints, the image-pair match
matrix beside the newest matched pair, and the growing sparse model.

`output/splat.ply` stays in the normalized frame the optimizer used. When the
run has gravity priors, the first view also writes
`output/splat_gravity_up.ply`: the same Gaussians rotated so world down is
+Y, for viewing in other 3DGS tools. Both use the standard 3DGS PLY layout.

## Run layout

Every capture and run gets its own folder, and nothing is overwritten:

```text
files/capture_to_view/
  captures/capture_<time>_<id>/       video.mp4, arcore_poses.txt, sensors.csv, capture.png
  runs/run_<time>_<id>/
    input/capture/                    copy of the capture
    extract/                          images/, frame_priors.txt
    sfm/                              database.db, sparse/
    training/                         images/, sparse/0/
    output/                           bench.json, splat.ply, splat_gravity_up.ply, render/, target/
    screenshots/                      one screenshot per stage
    pipeline.log, telemetry.csv, project_manifest.json
```

`project_manifest.json` records the settings, per-stage times, and SHA-256
hashes of every file. To copy a finished run to the computer and score its
held-out views, run from bash (PowerShell's `>` corrupts binary output):

```sh
adb exec-out run-as org.unclearness.cross3dgs.e2e \
    tar -c -C files/capture_to_view/runs <run id> > run.tar
mkdir -p out/capture_to_view && tar -x -f run.tar -C out/capture_to_view
uv run --with torch --with torchvision --with torchmetrics \
    python scripts/common_evaluator.py --name <run id> \
    --render-dir out/capture_to_view/<run id>/output/render \
    --target-dir out/capture_to_view/<run id>/output/target \
    --output out/capture_to_view/<run id>/common_eval.json
```

## Launch options

```sh
adb shell am start -n org.unclearness.cross3dgs.e2e/org.unclearness.cross3dgs.E2eActivity \
    --es run_id <run id> --es viewer_reference_image frame_0096.jpg
```

opens a finished run in the viewer from a chosen registered camera.
`--ez sfm_preview false` turns off the SfM live preview for new runs.
