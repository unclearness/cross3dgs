"""Common evaluator: re-score any backend's saved renders with one implementation.

Every quality number (Cross3DGS on any platform, VkSplat, Brush) comes from
this script so PSNR/SSIM/LPIPS aggregation is identical across backends.

Metrics:
  - mean L1 (per-view mean absolute difference, averaged over views)
  - global-MSE PSNR (single PSNR over the pooled squared error)
  - mean-per-view PSNR (Brush-style: average of per-view PSNRs)
  - SSIM (VkSplat-compatible 11x11 Gaussian window, borders cropped)
  - LPIPS (AlexNet, torchmetrics, inputs in [0, 1])

The target set is derived from the dataset split rule used by every backend in
this project: image filenames sorted ascending, validation iff zero-based
index % eval_interval == 0. Alternatively pass --target-dir to compare against
an explicit target dump. Renders must be named view_000.png, view_001.png, ...
in split order, or supply arbitrary sorted PNGs with --render-glob.

The output JSON records a sha256 for every render and target file, the repo
revision, and the exact command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_vksplat import image_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="label for this evaluation (e.g. android_bicycle)")
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--render-glob", default="view_*.png")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-dir", type=Path,
                        help="explicit target PNG dump (paired by sorted name)")
    target.add_argument("--dataset-root", type=Path,
                        help="derive targets from the dataset split rule")
    parser.add_argument("--scene", default="bicycle")
    parser.add_argument("--image-dir", default="images_4")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--split", choices=["validation", "train"],
                        default="validation")
    parser.add_argument("--no-lpips", action="store_true",
                        help="skip LPIPS (avoids the torch dependency)")
    parser.add_argument("--bench-json", type=Path,
                        help="optional backend bench.json to hash and reference")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_rgb01(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def dataset_targets(args: argparse.Namespace) -> list[Path]:
    image_dir = args.dataset_root / args.scene / args.image_dir
    names = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    if not names:
        raise SystemExit(f"no images under {image_dir}")
    selected = [
        image_dir / name
        for index, name in enumerate(names)
        if (index % args.eval_interval == 0) == (args.split == "validation")
    ]
    return selected


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    renders = sorted(args.render_dir.glob(args.render_glob))
    if not renders:
        raise SystemExit(f"no renders match {args.render_dir}/{args.render_glob}")
    if args.target_dir is not None:
        targets = sorted(args.target_dir.glob("*.png"))
    else:
        targets = dataset_targets(args)
    if len(renders) != len(targets):
        raise SystemExit(
            f"split mismatch: {len(renders)} renders vs {len(targets)} targets")

    lpips_metric = None
    if not args.no_lpips:
        import torch
        from torchmetrics.image.lpip import (
            LearnedPerceptualImagePatchSimilarity,
        )
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True).eval()

    views = []
    squared_error = 0.0
    value_count = 0
    for view, (render_path, target_path) in enumerate(zip(renders, targets)):
        rendered = load_rgb01(render_path)
        target = load_rgb01(target_path)
        if rendered.shape != target.shape:
            raise SystemExit(
                f"resolution mismatch at view {view}: "
                f"{rendered.shape} vs {target.shape}")
        difference = rendered - target
        view_sq = float(np.sum(difference * difference))
        squared_error += view_sq
        value_count += difference.size
        entry = {
            "view": view,
            "render": render_path.name,
            "target": target_path.name,
            "render_sha256": sha256_file(render_path),
            "target_sha256": sha256_file(target_path),
            "l1": float(np.mean(np.abs(difference))),
            "psnr": -10.0 * float(np.log10(view_sq / difference.size)),
            "ssim": image_ssim(rendered, target),
        }
        if lpips_metric is not None:
            import torch
            with torch.no_grad():
                entry["lpips_alex"] = float(lpips_metric(
                    torch.from_numpy(rendered).permute(2, 0, 1).unsqueeze(0),
                    torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)))
        views.append(entry)

    result = {
        "schema_version": 1,
        "evaluator": "scripts/common_evaluator.py",
        "name": args.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": git_revision(root),
        "command": sys.argv,
        "render_dir": str(args.render_dir),
        "target_source": (
            str(args.target_dir) if args.target_dir is not None else {
                "dataset_root": str(args.dataset_root),
                "scene": args.scene,
                "image_dir": args.image_dir,
                "eval_interval": args.eval_interval,
                "split": args.split,
                "split_rule": "filename sort ascending; validation iff "
                              "zero-based index % eval_interval == 0",
            }
        ),
        "bench_json": (
            {"path": str(args.bench_json),
             "sha256": sha256_file(args.bench_json)}
            if args.bench_json is not None else None
        ),
        "view_count": len(views),
        "mean_l1": float(np.mean([entry["l1"] for entry in views])),
        "global_mse_psnr": -10.0 * float(np.log10(squared_error / value_count)),
        "mean_view_psnr": float(np.mean([entry["psnr"] for entry in views])),
        "ssim": float(np.mean([entry["ssim"] for entry in views])),
        "lpips_alex": (
            float(np.mean([entry["lpips_alex"] for entry in views]))
            if lpips_metric is not None else None
        ),
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "name", "view_count", "mean_l1", "global_mse_psnr",
        "mean_view_psnr", "ssim", "lpips_alex")}, indent=2))


if __name__ == "__main__":
    main()
