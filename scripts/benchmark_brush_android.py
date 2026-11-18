"""Run the Brush baseline on an Android phone over adb.

Uses a Brush Android app built with patches/brush_android_benchmark.patch,
which on launch trains on a dataset ZIP in its app-owned folder and logs its
progress and its own training time to logcat. This script builds that ZIP
(the scene's images, sparse/0, and Brush arguments), pushes it, launches the
app, waits for completion, and pulls the trained splat.ply. With --evaluate it
renders the held-out views from that PLY on this computer with the patched
desktop brush-cli (no further training) and scores them with
scripts/common_evaluator.py.

Example:
  uv run python scripts/benchmark_brush_android.py --serial <adb serial> \
      --apk <brush apk> --scene bicycle --seed 42 \
      --output out/android_brush/bicycle --evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_run  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = "com.splats.app"
ACTIVITY = f"{PACKAGE}/.MainActivity"
DEVICE_ROOT = f"/sdcard/Android/data/{PACKAGE}/files"
# The path the patched app loads at launch.
DEVICE_BUNDLE = f"{DEVICE_ROOT}/brush_bicycle_fixed_common.zip"
COMPLETE = re.compile(
    r"brush_benchmark complete iterations=(\d+) train_seconds=([0-9.]+) splats=(\d+)")


def adb(serial: str, *args: str, check: bool = True) -> str:
    result = subprocess.run(["adb", "-s", serial, *args], capture_output=True)
    if check and result.returncode:
        raise SystemExit(f"adb {' '.join(args)} failed: "
                         f"{result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("utf-8", "replace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(scene_dir: Path, image_dir: str, brush_args: list[str],
                 destination: Path) -> None:
    images = sorted(path for path in (scene_dir / image_dir).iterdir() if path.is_file())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as bundle:
        for image in images:
            bundle.write(image, f"images/{image.name}")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            bundle.write(scene_dir / "sparse" / "0" / name, f"sparse/0/{name}")
        bundle.writestr("args.txt", " ".join(brush_args) + "\n")


def brush_cli(brush_root: Path) -> Path:
    for name in ("brush-cli.exe", "brush-cli"):
        candidate = brush_root / "target" / "release" / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no release brush-cli under {brush_root}; build the patched "
                     "checkout with cargo build --release")


def rerender(brush_root: Path, scene_dir: Path, image_dir: str, ply: Path,
             iterations: int, max_resolution: int, output: Path) -> Path:
    """Render the held-out views of ``ply`` on the host without training."""
    staged = output / "rerender_scene"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(scene_dir / image_dir, staged / "images")
    shutil.copytree(scene_dir / "sparse" / "0", staged / "sparse" / "0")
    shutil.copy2(ply, staged / "init.ply")
    renders = output / "rerender"
    command = [str(brush_cli(brush_root)), str(staged),
               "--start-iter", str(iterations), "--total-train-iters", str(iterations),
               "--sh-degree", "3", "--eval-split-every", "8", "--eval-every", "1",
               "--eval-save-to-disk", "--eval-initial-state-and-exit",
               "--max-resolution", str(max_resolution),
               "--export-path", str(renders), "--export-name", "unused.ply"]
    environment = dict(os.environ, WGPU_BACKEND="vulkan", RUST_LOG="info")
    with open(output / "rerender.log", "w", encoding="utf-8") as log:
        subprocess.run(command, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return renders / f"eval_{iterations}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", required=True)
    parser.add_argument("--apk", type=Path,
                        help="install this Brush APK first (otherwise use the installed one)")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/360_v2"))
    parser.add_argument("--scene", default="bicycle")
    parser.add_argument("--image-dir", default="images_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--max-splats", type=int, default=1_000_000)
    parser.add_argument("--brush-root", type=Path, default=Path("cache/brush"),
                        help="patched Brush checkout, for --evaluate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=600)
    args = parser.parse_args()

    scene_dir = (REPOSITORY / args.dataset_root / args.scene).resolve()
    if not (scene_dir / args.image_dir).is_dir():
        raise SystemExit(f"missing {scene_dir / args.image_dir}; "
                         "run scripts/download_datasets.py first")
    first_image = next(path for path in sorted((scene_dir / args.image_dir).iterdir())
                       if path.is_file())
    max_resolution = max(Image.open(first_image).size)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device_export = f"{DEVICE_ROOT}/export_{args.scene}_seed{args.seed}"
    brush_args = [
        "--total-train-iters", str(args.iterations), "--sh-degree", "3",
        "--ssim-weight", "0.2", "--max-splats", str(args.max_splats),
        "--seed", str(args.seed), "--eval-split-every", "8",
        "--eval-every", str(args.iterations), "--eval-save-to-disk",
        "--defer-final-eval-to-host", "--max-resolution", str(max_resolution),
        "--max-scene-batch-cache-size", "1GiB",
        "--export-every", str(args.iterations + 1),
        "--export-path", device_export, "--export-name", "splat.ply",
    ]
    bundle = output / "bundle.zip"
    build_bundle(scene_dir, args.image_dir, brush_args, bundle)

    if args.apk is not None:
        print(adb(args.serial, "install", "-r", str(args.apk.resolve())).strip())
    adb(args.serial, "shell", "am", "force-stop", PACKAGE)
    adb(args.serial, "shell", "mkdir", "-p", DEVICE_ROOT)
    adb(args.serial, "shell", "rm", "-rf", device_export)
    adb(args.serial, "push", str(bundle.resolve()), DEVICE_BUNDLE)
    if adb(args.serial, "shell", "sha256sum", DEVICE_BUNDLE).split()[0] != sha256_file(bundle):
        raise SystemExit("pushed bundle does not match")
    adb(args.serial, "logcat", "-c")
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    adb(args.serial, "shell", "am", "start", "-n", ACTIVITY)

    complete = None
    last_progress = ""
    while time.monotonic() - started < args.timeout_minutes * 60:
        time.sleep(args.poll_seconds)
        logcat = adb(args.serial, "logcat", "-d", "-v", "threadtime", check=False)
        progress = [line for line in logcat.splitlines()
                    if "brush_benchmark progress" in line]
        if progress and progress[-1] != last_progress:
            last_progress = progress[-1]
            print(last_progress, flush=True)
        complete = COMPLETE.search(logcat)
        if complete or ("panicked at" in logcat):
            break
    (output / "logcat.txt").write_text(
        adb(args.serial, "logcat", "-d", "-v", "threadtime", check=False),
        encoding="utf-8")
    if complete is None:
        adb(args.serial, "shell", "am", "force-stop", PACKAGE, check=False)
        raise SystemExit("Brush did not finish; see logcat.txt")

    pulled = output / "device_export"
    if pulled.exists():
        shutil.rmtree(pulled)
    adb(args.serial, "pull", device_export, str(pulled))
    adb(args.serial, "shell", "am", "force-stop", PACKAGE, check=False)
    plys = sorted(pulled.rglob("splat.ply"))
    if not plys:
        raise SystemExit(f"no splat.ply in {pulled}")
    shutil.copy2(plys[-1], output / "splat.ply")

    bench = {
        "method": "Brush (Android)",
        "scene": args.scene,
        "seed": args.seed,
        "iterations": int(complete.group(1)),
        "final_splats": int(complete.group(3)),
        "timing": {"brush_reported_train_seconds": float(complete.group(2))},
        "wall_seconds_launch_to_complete": time.monotonic() - started,
        "started_utc": started_utc,
        "brush_args": brush_args,
        "device_model": adb(args.serial, "shell", "getprop",
                            "ro.product.model").strip(),
    }
    (output / "bench.json").write_text(json.dumps(bench, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(bench["timing"], indent=2))

    if args.evaluate:
        render_dir = rerender((REPOSITORY / args.brush_root).resolve(), scene_dir,
                              args.image_dir, output / "splat.ply",
                              bench["iterations"], max_resolution, output.resolve())
        print(json.dumps(evaluate_run(
            output, f"android_brush_{args.scene}_seed{args.seed}",
            lpips=not args.no_lpips, render_dir=render_dir, render_glob="*.png",
            dataset=(str(args.dataset_root), args.scene, args.image_dir)), indent=2))


if __name__ == "__main__":
    main()
