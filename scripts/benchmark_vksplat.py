"""Run the pinned VkSplat checkout on the same scene, split, and budget as Cross3DGS."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def scheduled_mcmc_count(initial: int, completed_iterations: int,
                         capacity: int) -> int:
    count = initial
    for step in range(600, min(completed_iterations, 25000), 100):
        count = min(int(1.05 * count), capacity)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/360_v2")
    parser.add_argument("--scene", default="bicycle")
    parser.add_argument("--image-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-order", choices=("cyclic", "shuffle"),
                        default="cyclic")
    parser.add_argument("--strategy", choices=("fixed", "default", "mcmc"), default="mcmc")
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=0,
        help="Evaluate all validation views every N training steps (0: final only)",
    )
    parser.add_argument(
        "--image-cache-device", choices=("cpu", "gpu"), default="cpu"
    )
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--init-opacity", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--output", default="out/benchmarks/mipnerf360/bicycle/vksplat_smoke"
    )
    parser.add_argument(
        "--evaluation-image-dir",
        default="",
        help="Save target, rendered, and 4x absolute-error PNGs for each evaluated view",
    )
    parser.add_argument(
        "--write-ply",
        action="store_true",
        help="Write the final Gaussian state to OUTPUT/splat.ply",
    )
    parser.add_argument(
        "--save-curve-renders",
        action="store_true",
        help="Save every quality-checkpoint render under "
             "EVALUATION_IMAGE_DIR/curve_render (for offline LPIPS curves)",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not 0 <= args.warmup < args.iterations:
        parser.error("--warmup must be non-negative and less than --iterations")
    if args.quality_eval_every < 0:
        parser.error("--quality-eval-every must be non-negative")
    return args


def image_ssim(image: "np.ndarray", target: "np.ndarray") -> float:
    """VkSplat-compatible 11x11 Gaussian-window SSIM, averaged over RGB."""
    import cv2
    import numpy as np

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu_image = cv2.filter2D(image, -1, window, borderType=cv2.BORDER_CONSTANT)
    mu_target = cv2.filter2D(target, -1, window, borderType=cv2.BORDER_CONSTANT)
    mu_image2 = mu_image * mu_image
    mu_target2 = mu_target * mu_target
    covariance = cv2.filter2D(
        image * target, -1, window, borderType=cv2.BORDER_CONSTANT
    ) - mu_image * mu_target
    variance_image = cv2.filter2D(
        image * image, -1, window, borderType=cv2.BORDER_CONSTANT
    ) - mu_image2
    variance_target = cv2.filter2D(
        target * target, -1, window, borderType=cv2.BORDER_CONSTANT
    ) - mu_target2
    score = ((2.0 * mu_image * mu_target + 0.01**2) *
             (2.0 * covariance + 0.03**2)) / (
        (mu_image2 + mu_target2 + 0.01**2) *
        (variance_image + variance_target + 0.03**2)
    )
    return float(np.mean(score[5:-5, 5:-5, :3]))


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    vksplat_root = repository_root / "external" / "vksplat"
    vksplat_python_root = vksplat_root / "vksplat"
    sys.path.insert(0, str(vksplat_python_root))

    import vksplat  # type: ignore
    import numpy as np
    from simple_trainer import MCMCTrainerConfig, TrainerConfig, join_dir

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (repository_root / dataset_root).resolve()
    scene_path = dataset_root / args.scene
    output = Path(args.output)
    if not output.is_absolute():
        output = repository_root / output
    output.mkdir(parents=True, exist_ok=True)
    evaluation_image_dir = None
    if args.evaluation_image_dir:
        evaluation_image_dir = Path(args.evaluation_image_dir)
        if not evaluation_image_dir.is_absolute():
            evaluation_image_dir = repository_root / evaluation_image_dir

    def save_rgb_png(path: Path, image: np.ndarray) -> None:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        rgb8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write PNG: {path}")

    config = MCMCTrainerConfig() if args.strategy == "mcmc" else TrainerConfig()
    config.dataset_dir = str(scene_path)
    config.image_dir = join_dir(str(scene_path), args.image_dir)
    config.sparse_dir = join_dir(str(scene_path), args.sparse_dir)
    config.mask_dir = ""
    config.output_dir = str(output)
    config.output_ply = str(output / "splat.ply")
    config.train_steps = args.iterations
    config.eval_interval = args.eval_interval
    config.ssim_lambda = args.ssim_lambda
    config.image_cache_device = args.image_cache_device
    config.init_scale = args.init_scale
    config.init_opacity = args.init_opacity
    config.enable_viewer = False
    if args.strategy == "fixed":
        # Keep the initial sparse cloud fixed for primitive/kernel comparisons.
        config.strategy = "default"
        config.refine_start_iter = args.iterations + 1
        config.refine_stop_iter = args.iterations + 1
    elif args.strategy == "mcmc":
        config.cap_max = args.capacity

    module = vksplat.VkSplat()
    shader_dir = vksplat_python_root / "shader"
    module.initialize(str(shader_dir) + os.sep, args.device)
    # Official pinned VkSplat revisions expose no Python native-RNG setter.
    # Do not claim that the outer repeat seed controls its internal MCMC RNG:
    # record its documented fixed native seed while still seeding the adapter's
    # shuffled view order below.
    native_rng_control = hasattr(module, "set_seed")
    native_rng_seed = args.seed if native_rng_control else 42
    if native_rng_control:
        module.set_seed(args.seed)
    metadata = module.set_train_config(asdict(config))
    with (scene_path / args.sparse_dir / "points3D.bin").open("rb") as stream:
        initial_gaussian_count = struct.unpack("<Q", stream.read(8))[0]

    expected_revision = "e26c254938c81ff85998cd357a9e005e255d9b03"
    revision = git_revision(vksplat_root)
    if revision != expected_revision:
        raise RuntimeError(
            f"VkSplat revision {revision} violates pinned fairness revision "
            f"{expected_revision}"
        )

    def evaluate(iteration: int, train_seconds: float) -> dict:
        view_l1: list[float] = []
        squared_error = 0.0
        value_count = 0
        view_ssim: list[float] = []
        for image_index in range(module.num_val):
            module.render_val(image_index)
            rendered = np.asarray(module.pixel_state, dtype=np.float32)[..., :3]
            wanted = np.asarray(
                module.get_val_image(image_index), dtype=np.float32
            )[..., :3] / 255.0
            difference = rendered - wanted
            view_l1.append(float(np.mean(np.abs(difference))))
            view_ssim.append(image_ssim(rendered, wanted))
            squared_error += float(np.sum(difference * difference))
            value_count += difference.size
            if evaluation_image_dir is not None and iteration == args.iterations:
                filename = f"view_{image_index:03d}.png"
                save_rgb_png(evaluation_image_dir / "target" / filename, wanted)
                save_rgb_png(evaluation_image_dir / "render" / filename, rendered)
                save_rgb_png(
                    evaluation_image_dir / "error_x4" / filename,
                    np.abs(difference) * 4.0,
                )
            if args.save_curve_renders and evaluation_image_dir is not None:
                save_rgb_png(
                    evaluation_image_dir / "curve_render" /
                    f"iteration_{iteration:05d}_view_{image_index:03d}.png",
                    rendered,
                )
        mse = squared_error / value_count
        return {
            "iteration": iteration,
            "train_seconds": train_seconds,
            "validation_mean_l1": statistics.mean(view_l1),
            "validation_psnr": -10.0 * float(np.log10(mse)),
            "validation_ssim": statistics.mean(view_ssim),
            "validation_view_l1": view_l1,
            "validation_view_ssim": view_ssim,
            "gaussian_count": scheduled_mcmc_count(
                initial_gaussian_count, iteration, args.capacity
            ) if args.strategy == "mcmc" else len(module.opacities),
            "current_memory_bytes": module.get_vram_usage(),
            "peak_memory_bytes": module.get_peak_vram_usage(),
        }

    random.seed(args.seed)
    view_count = module.num_train
    if args.max_views > 0:
        view_count = min(view_count, args.max_views)
    image_order = list(range(view_count))
    step_ms: list[float] = []
    quality_curve: list[dict] = []
    accumulated_train_seconds = 0.0
    for step in range(args.iterations):
        if (args.view_order == "shuffle" and step > 0 and
                step % len(image_order) == 0):
            random.shuffle(image_order)
        image_index = image_order[step % len(image_order)]
        start = perf_counter()
        module.train_step(image_index, step)
        elapsed = perf_counter() - start
        accumulated_train_seconds += elapsed
        step_ms.append(elapsed * 1000.0)
        completed = step + 1
        if (args.quality_eval_every > 0 and
                completed % args.quality_eval_every == 0):
            quality_curve.append(evaluate(completed, accumulated_train_seconds))
    total_seconds = accumulated_train_seconds
    if not quality_curve or quality_curve[-1]["iteration"] != args.iterations:
        quality_curve.append(evaluate(args.iterations, total_seconds))
    final_quality = quality_curve[-1]

    steady = sorted(step_ms[args.warmup :])
    result = {
        "schema_version": 1,
        "backend": "VkSplat",
        "revision": revision,
        "rng_seed": args.seed,
        "native_rng_control": native_rng_control,
        "native_rng_seed": native_rng_seed,
        "view_order_rng_seed": args.seed,
        "device": metadata.get("device", {}),
        "dataset_root": Path(args.dataset_root).as_posix(),
        "scene": args.scene,
        "image_dir": args.image_dir,
        "sparse_dir": Path(args.sparse_dir).as_posix(),
        "train_image_count": module.num_train,
        "sampled_train_image_count": view_count,
        "validation_image_count": module.num_val,
        "seed": args.seed,
        "view_order": args.view_order,
        "ssim_lambda": args.ssim_lambda,
        "image_cache_device": args.image_cache_device,
        "init_scale": args.init_scale,
        "init_opacity": args.init_opacity,
        "strategy": args.strategy,
        "gaussian_capacity": args.capacity if args.strategy == "mcmc" else None,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "first_iteration_ms": step_ms[0],
        "median_step_ms": statistics.median(steady),
        "p10_step_ms": percentile(steady, 0.10),
        "p50_step_ms": percentile(steady, 0.50),
        "p90_step_ms": percentile(steady, 0.90),
        "p99_step_ms": percentile(steady, 0.99),
        "total_train_seconds": total_seconds,
        "evaluation_split": "validation",
        "evaluation_mean_l1": final_quality["validation_mean_l1"],
        "evaluation_psnr": final_quality["validation_psnr"],
        "evaluation_ssim": final_quality["validation_ssim"],
        "evaluation_view_l1": final_quality["validation_view_l1"],
        "evaluation_view_ssim": final_quality["validation_view_ssim"],
        "final_gaussian_count": final_quality["gaussian_count"],
        "initial_gaussian_count": initial_gaussian_count,
        "current_memory_bytes": module.get_vram_usage(),
        "peak_memory_bytes": module.get_peak_vram_usage(),
        "stage_timings": module.get_timing_breakdown(),
        "vram_breakdown": module.get_vram_breakdown(),
        "quality_curve": quality_curve,
        "train_metadata": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in metadata.items()
        },
        "step_ms": step_ms,
    }
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    (output / "bench.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_ply:
        module.write_ply(str(output / "splat.ply"))
    print(json.dumps({key: result[key] for key in (
        "backend", "revision", "device", "iterations", "median_step_ms",
        "p90_step_ms", "total_train_seconds", "final_gaussian_count",
        "peak_memory_bytes",
    )}, indent=2))
    module.cleanup()


if __name__ == "__main__":
    main()
