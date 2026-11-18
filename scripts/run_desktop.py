"""Train and evaluate scenes on this machine with Cross3DGS, VkSplat, or Brush.

For every scene, method, and seed, trains once with the standard-quality
settings, scores the held-out views (every eighth image) with
scripts/common_evaluator.py, and finally writes ``summary.json`` with the
per-run results and the per-scene, per-method medians.

Cross3DGS uses the ``desktop.arguments`` of the preset config. VkSplat and
Brush run through scripts/benchmark_vksplat.py and scripts/benchmark_brush.py
with the same images, split, update count, capacity, SH degree, and SSIM
weight (see the README for their checkouts).

Examples:
  # Cross3DGS on bicycle, three seeds
  uv run python scripts/run_desktop.py --scenes bicycle \
      --seeds 20260815 20260816 20260817 --output out/desktop

  # The 11-scene comparison with both baselines
  uv run python scripts/run_desktop.py --scenes all \
      --methods cross3dgs vksplat brush \
      --seeds 20260815 20260816 20260817 --output out/desktop
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_run  # noqa: E402
from gpu_cooldown import wait_for_cool_gpu  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
# Scene -> (dataset root, image folder), as fetched by scripts/download_datasets.py.
SCENES = {
    "bicycle": ("data/360_v2", "images_4"),
    "garden": ("data/360_v2", "images_4"),
    "stump": ("data/360_v2", "images_4"),
    "room": ("data/360_v2", "images_2"),
    "counter": ("data/360_v2", "images_2"),
    "kitchen": ("data/360_v2", "images_2"),
    "bonsai": ("data/360_v2", "images_2"),
    "train": ("data/tandt", "images"),
    "truck": ("data/tandt", "images"),
    "drjohnson": ("data/deep_blending", "images"),
    "playroom": ("data/deep_blending", "images"),
}
METHODS = ("cross3dgs", "vksplat", "brush")


def default_executable() -> Path:
    # Build folders of the msvc, linux, and macos CMake presets.
    candidates = [
        REPOSITORY / "build/tools/train_cli/Release/cross_3dgs_train.exe",
        REPOSITORY / "build-linux/tools/train_cli/cross_3dgs_train",
        REPOSITORY / "build-macos/tools/train_cli/cross_3dgs_train",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def link_directory(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` (a junction on Windows, a symlink elsewhere)."""
    target = target.resolve()
    if link.exists():
        if link.resolve() != target:
            raise SystemExit(f"{link} points to {link.resolve()}, not {target}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        link.symlink_to(target, target_is_directory=True)


def brush_scene(scene: str, root: str, image_dir: str) -> Path:
    """Brush reads <scene>/images and <scene>/sparse; link them to the dataset."""
    staged = REPOSITORY / "cache" / "brush_scenes" / scene
    source = REPOSITORY / root / scene
    link_directory(staged / "images", source / image_dir)
    link_directory(staged / "sparse", source / "sparse")
    return staged


def method_command(method: str, args: argparse.Namespace, config: dict, scene: str,
                   root: str, image_dir: str, seed: int, iterations: int,
                   run_dir: Path) -> list[str]:
    if method == "cross3dgs":
        values = {"dataset_root": root, "scene": scene, "image_dir": image_dir,
                  "iterations": str(iterations), "seed": str(seed),
                  "output": str(run_dir)}
        return [str(args.executable)] + [
            argument.format(**values) for argument in config["desktop"]["arguments"]]
    if method == "vksplat":
        return [sys.executable, "scripts/benchmark_vksplat.py",
                "--dataset-root", root, "--scene", scene, "--image-dir", image_dir,
                "--sparse-dir", "sparse/0", "--iterations", str(iterations),
                "--warmup", "20", "--seed", str(seed), "--view-order", "shuffle",
                "--strategy", "mcmc", "--capacity", str(config["gaussian_capacity"]),
                "--ssim-lambda", str(config["ssim_weight"]),
                "--quality-eval-every", "0", "--image-cache-device", "cpu",
                "--output", str(run_dir), "--evaluation-image-dir", str(run_dir),
                "--write-ply"]
    return [sys.executable, "scripts/benchmark_brush.py",
            "--brush-root", str(args.brush_root),
            "--scene-path", str(brush_scene(scene, root, image_dir)),
            "--source-dataset-root", root, "--source-scene", scene,
            "--source-image-dir", image_dir, "--iterations", str(iterations),
            "--seed", str(seed), "--quality-eval-every", str(iterations),
            "--max-resolution", "0",
            "--max-splats", str(config["gaussian_capacity"]),
            "--sh-degree", str(config["sh_degree"]),
            "--ssim-weight", str(config["ssim_weight"]),
            "--backend", "vulkan", "--eval-save-to-disk", "--output", str(run_dir)]


def train_seconds(method: str, bench: dict):
    if method == "brush":
        return bench.get("timing", {}).get("brush_reported_train_seconds")
    return bench.get("total_train_seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=REPOSITORY / "configs/standard_quality.json")
    parser.add_argument("--scenes", nargs="+", default=["bicycle"],
                        help=f"scene names, or 'all' for {', '.join(SCENES)}")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["cross3dgs"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--iterations", type=int,
                        help="override the preset's update count (smoke runs)")
    parser.add_argument("--executable", type=Path, default=default_executable(),
                        help="Cross3DGS trainer (default: the CMake preset build)")
    parser.add_argument("--brush-root", type=Path, default=Path("cache/brush"),
                        help="Brush checkout with a release brush-cli build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cooldown", action="store_true",
                        help="wait for a cool GPU before each run")
    parser.add_argument("--no-lpips", action="store_true",
                        help="skip LPIPS (no torch download)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    iterations = args.iterations or config["iterations"]
    scenes = list(SCENES) if args.scenes == ["all"] else args.scenes
    unknown = [scene for scene in scenes if scene not in SCENES]
    if unknown:
        raise SystemExit(f"unknown scenes {unknown}; choose from {', '.join(SCENES)}")
    for scene in scenes:
        root, image_dir = SCENES[scene]
        if not (REPOSITORY / root / scene / image_dir).is_dir():
            raise SystemExit(f"missing {root}/{scene}/{image_dir}; "
                             "run scripts/download_datasets.py first")
    if "cross3dgs" in args.methods and not args.executable.is_file():
        raise SystemExit(f"trainer not found: {args.executable}; build it first "
                         "(see README) or pass --executable")

    runs = []
    for scene in scenes:
        root, image_dir = SCENES[scene]
        for method in args.methods:
            for seed in args.seeds:
                run_dir = args.output / scene / method / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.cooldown:
                    wait_for_cool_gpu(f"{scene}/{method}/seed{seed}")
                command = method_command(method, args, config, scene, root, image_dir,
                                         seed, iterations, run_dir)
                (run_dir / "command.json").write_text(
                    json.dumps(command, indent=2) + "\n", encoding="utf-8")
                print(subprocess.list2cmdline(command), flush=True)
                with open(run_dir / "console.log", "w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=REPOSITORY, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                bench = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
                if method == "cross3dgs" and bench.get("overflow_count", 0) != 0:
                    raise SystemExit(f"{scene} seed {seed}: tile-Gaussian pair "
                                     f"overflow {bench['overflow_count']}")
                # Brush writes its held-out renders as eval_<updates>/<image>.png.
                render_dir, render_glob = ((run_dir / f"eval_{iterations}", "*.png")
                                           if method == "brush" else (None, None))
                metrics = evaluate_run(run_dir, f"{scene}_{method}_seed{seed}",
                                       lpips=not args.no_lpips, render_dir=render_dir,
                                       render_glob=render_glob,
                                       dataset=(root, scene, image_dir))
                runs.append({"scene": scene, "method": method, "seed": seed,
                             "train_seconds": train_seconds(method, bench),
                             **metrics})
                print(json.dumps(runs[-1], indent=2), flush=True)

    medians = []
    for scene in scenes:
        for method in args.methods:
            cell = [run for run in runs if run["scene"] == scene and run["method"] == method]
            row = {"scene": scene, "method": method}
            for key in ("train_seconds", "mean_view_psnr", "ssim", "lpips_alex"):
                values = [run[key] for run in cell if run.get(key) is not None]
                row[key] = statistics.median(values) if values else None
            medians.append(row)

    summary = {
        "config": str(args.config),
        "iterations": iterations,
        "seeds": args.seeds,
        "host": platform.platform(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "medians": medians,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'scene':<10} {'method':<10} {'seconds':>9} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6}")
    for row in medians:
        def fmt(value, spec):
            return format(value, spec) if value is not None else "-"
        print(f"{row['scene']:<10} {row['method']:<10} "
              f"{fmt(row['train_seconds'], '9.1f')} {fmt(row['mean_view_psnr'], '7.2f')} "
              f"{fmt(row['ssim'], '6.3f')} {fmt(row['lpips_alex'], '6.3f')}")


if __name__ == "__main__":
    main()
