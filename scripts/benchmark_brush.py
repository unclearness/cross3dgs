"""Run the pinned Brush CLI on the same scene, split, and budget as Cross3DGS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image


PINNED_BRUSH_REVISION = "3b80985709e2ec04fd6c8622a40e36473647a8e0"
PINNED_ADAPTER_PATCH_SHA256 = (
    "e8522e539c3921482d72a3a7a69824e6a007d5cd34583bab90d7d3cdb6b2006b"
)
PINNED_PATCHED_SOURCE_SHA256 = (
    "ef6567ad2b39f7526517417526192ade3dce89159527b548bc5266bca8862b8b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brush-root", default="cache/brush")
    parser.add_argument("--scene-path", default="cache/brush_bicycle")
    parser.add_argument("--source-dataset-root", default="data/360_v2")
    parser.add_argument("--source-scene", default="bicycle")
    parser.add_argument("--source-image-dir", default="images_4")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--quality-eval-every", type=int, default=1_000)
    parser.add_argument(
        "--max-resolution", type=int, default=0,
        help="maximum source-image dimension (0: infer native maximum)",
    )
    parser.add_argument("--max-splats", type=int, default=1_000_000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument(
        "--cubecl-device",
        default="",
        help=("optional CUBECL_WGPU_DEFAULT_DEVICE override, for example "
              "DiscreteGpu(0)"),
    )
    parser.add_argument(
        "--expected-adapter-substring",
        default="",
        help=("fail unless structured Brush adapter provenance records this "
              "name substring"),
    )
    parser.add_argument(
        "--expected-adapter-vendor-id",
        type=lambda value: int(value, 0),
        default=None,
        help="fail unless Brush reports this vendor ID, e.g. 0x1002",
    )
    parser.add_argument(
        "--expected-adapter-backend",
        default="",
        help="fail unless Brush reports this wgpu backend, e.g. Vulkan",
    )
    parser.add_argument(
        "--expected-adapter-driver-substring",
        default="",
        help=("fail unless Brush reports this driver/driver-info substring, "
              "e.g. radv"),
    )
    parser.add_argument("--eval-save-to-disk", action="store_true",
                        help="save eval renders (Brush writes them under "
                             "the export path)")
    parser.add_argument(
        "--output",
        default="out/benchmarks/mipnerf360/bicycle/brush/nvidia_30k",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.eval_interval <= 0 or args.quality_eval_every <= 0:
        parser.error("evaluation intervals must be positive")
    if args.max_resolution < 0:
        parser.error("--max-resolution must be non-negative")
    return args


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nvidia_gpu() -> dict[str, str] | None:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None
    name, driver, memory_mib = (part.strip() for part in line.split(",", 2))
    return {"name": name, "driver": driver, "memory_mib": memory_mib}


def parse_humantime(text: str) -> float | None:
    match = re.search(r"Done training! Took FormattedDuration\(([^)]+)\)", text)
    if not match:
        return None
    total = 0.0
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3}
    for value, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|h|m|s)", match.group(1)):
        total += float(value) * units[unit]
    return total


def parse_adapter_info(text: str) -> dict[str, str | int] | None:
    prefix = "BRUSH_ADAPTER_INFO|"
    matches = []
    for line in text.splitlines():
        offset = line.find(prefix)
        if offset < 0:
            continue
        fields: dict[str, str] = {}
        for item in line[offset + len(prefix):].split("|"):
            key, separator, value = item.partition("=")
            if separator:
                fields[key] = value
        required = {
            "name", "vendor", "device", "device_type", "backend",
            "driver", "driver_info",
        }
        if not required.issubset(fields):
            raise RuntimeError(
                "malformed structured Brush adapter provenance: " + line
            )
        matches.append({
            **fields,
            "vendor": int(fields["vendor"], 0),
            "device": int(fields["device"], 0),
        })
    if len(matches) > 1:
        raise RuntimeError("multiple Brush adapter provenance records found")
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    brush_root = resolve(repository_root, args.brush_root)
    scene_path = resolve(repository_root, args.scene_path)
    output = resolve(repository_root, args.output)
    output.mkdir(parents=True, exist_ok=True)

    executable = brush_root / "target" / "release" / "brush-cli.exe"
    if not executable.is_file():
        executable = brush_root / "target" / "release" / "brush-cli"
    if not executable.is_file():
        raise FileNotFoundError(f"Brush Release CLI not found: {executable}")

    revision = git_revision(brush_root)
    if revision != PINNED_BRUSH_REVISION:
        raise RuntimeError(
            f"Brush revision {revision} violates pinned fairness revision "
            f"{PINNED_BRUSH_REVISION}"
        )
    expected_adapter_evidence = any((
        args.expected_adapter_substring,
        args.expected_adapter_vendor_id is not None,
        args.expected_adapter_backend,
        args.expected_adapter_driver_substring,
    ))
    adapter_instrumentation = None
    if expected_adapter_evidence:
        patch_path = (
            repository_root / "patches" / "brush_adapter_provenance.patch")
        source_path = brush_root / "crates" / "brush-process" / "src" / "lib.rs"
        patch_sha256 = sha256(patch_path)
        source_sha256 = sha256(source_path)
        if patch_sha256 != PINNED_ADAPTER_PATCH_SHA256:
            raise RuntimeError(
                f"Brush adapter patch hash {patch_sha256} violates pinned "
                f"{PINNED_ADAPTER_PATCH_SHA256}"
            )
        if source_sha256 != PINNED_PATCHED_SOURCE_SHA256:
            raise RuntimeError(
                f"Brush patched source hash {source_sha256} violates pinned "
                f"{PINNED_PATCHED_SOURCE_SHA256}"
            )
        adapter_instrumentation = {
            "patch_sha256": patch_sha256,
            "patched_source": "crates/brush-process/src/lib.rs",
            "patched_source_sha256": source_sha256,
        }
    image_paths = sorted(
        path for path in (scene_path / "images").iterdir() if path.is_file()
    )
    image_count = len(image_paths)
    required_sparse = [
        scene_path / "sparse" / "0" / name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    ]
    if image_count == 0 or not all(path.is_file() for path in required_sparse):
        raise RuntimeError(
            "Brush scene must expose source images as images/ and the original "
            "sparse/0 reconstruction"
        )
    image_sizes = [Image.open(path).size for path in image_paths]
    max_resolution = args.max_resolution or max(max(size) for size in image_sizes)
    with required_sparse[2].open("rb") as stream:
        initial_point_count = struct.unpack("<Q", stream.read(8))[0]
    validation_image_count = (image_count + args.eval_interval - 1) // args.eval_interval
    train_image_count = image_count - validation_image_count

    command = [
        str(executable),
        str(scene_path),
        "--total-train-iters", str(args.iterations),
        "--sh-degree", str(args.sh_degree),
        "--ssim-weight", str(args.ssim_weight),
        "--max-splats", str(args.max_splats),
        "--seed", str(args.seed),
        "--eval-split-every", str(args.eval_interval),
        "--eval-every", str(args.quality_eval_every),
        "--max-resolution", str(max_resolution),
        "--export-every", str(args.iterations + 1),
        "--export-path", str(output),
        "--export-name", "splat.ply",
    ]
    if args.eval_save_to_disk:
        command += ["--eval-save-to-disk"]
    environment = os.environ.copy()
    environment["WGPU_BACKEND"] = args.backend
    if args.cubecl_device:
        environment["CUBECL_WGPU_DEFAULT_DEVICE"] = args.cubecl_device
    environment["RUST_LOG"] = "info"

    log_lines: list[str] = []
    evaluations: list[dict[str, float | int]] = []
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    eval_pattern = re.compile(
        r"Eval iter (\d+): PSNR ([0-9.eE+-]+), ssim ([0-9.eE+-]+)"
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_lines.append(line)
        match = eval_pattern.search(line)
        if match:
            evaluations.append(
                {
                    "iteration": int(match.group(1)),
                    "wall_seconds_at_report": perf_counter() - started,
                    "validation_psnr": float(match.group(2)),
                    "validation_ssim": float(match.group(3)),
                }
            )
    return_code = process.wait()
    wall_seconds = perf_counter() - started
    raw_log = "".join(log_lines)
    (output / "brush.log").write_text(raw_log, encoding="utf-8")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    adapter_info = parse_adapter_info(raw_log)
    adapter_log_verified = False
    if expected_adapter_evidence:
        if adapter_info is None:
            raise RuntimeError(
                "Brush completed but emitted no structured adapter "
                "provenance; refusing measurement promotion"
            )
        checks = [
            (
                not args.expected_adapter_substring or
                args.expected_adapter_substring.casefold() in
                str(adapter_info["name"]).casefold(),
                "name",
            ),
            (
                args.expected_adapter_vendor_id is None or
                args.expected_adapter_vendor_id == adapter_info["vendor"],
                "vendor",
            ),
            (
                not args.expected_adapter_backend or
                args.expected_adapter_backend.casefold() ==
                str(adapter_info["backend"]).casefold(),
                "backend",
            ),
            (
                not args.expected_adapter_driver_substring or
                args.expected_adapter_driver_substring.casefold() in
                (f"{adapter_info['driver']} "
                 f"{adapter_info['driver_info']}").casefold(),
                "driver",
            ),
        ]
        failed = [label for passed, label in checks if not passed]
        if failed:
            raise RuntimeError(
                "Brush adapter provenance violates expected "
                f"{', '.join(failed)}: {adapter_info}"
            )
        adapter_log_verified = True

    result = {
        "schema_version": 1,
        "backend": "Brush native WebGPU-compatible wgpu",
        "brush_revision": revision,
        "wgpu_backend": args.backend,
        "cubecl_device_override": args.cubecl_device or None,
        "expected_adapter_substring": args.expected_adapter_substring or None,
        "expected_adapter_vendor_id": args.expected_adapter_vendor_id,
        "expected_adapter_backend": args.expected_adapter_backend or None,
        "expected_adapter_driver_substring": (
            args.expected_adapter_driver_substring or None
        ),
        "adapter_info": adapter_info,
        "adapter_instrumentation": adapter_instrumentation,
        "adapter_log_verified": adapter_log_verified,
        "build": "release",
        "gpu": nvidia_gpu(),
        "dataset": {
            "source_scene": str(
                Path(args.source_dataset_root) / args.source_scene
            ).replace("\\", "/"),
            "brush_scene_resolved": str(scene_path),
            "image_dir": f"{args.source_image_dir} exposed as images",
            "sparse_dir": "sparse/0",
            "image_count": image_count,
            "train_image_count": train_image_count,
            "validation_image_count": validation_image_count,
            "split_rule": (
                "filename sort, validation iff zero-based index % "
                f"{args.eval_interval} == 0"
            ),
            "initial_point_count": initial_point_count,
            "native_image_dimensions": sorted(set(image_sizes)),
        },
        "config": {
            "iterations": args.iterations,
            "seed": args.seed,
            "sh_degree": args.sh_degree,
            "ssim_weight": args.ssim_weight,
            "max_splats": args.max_splats,
            "max_resolution": max_resolution,
            "quality_eval_every": args.quality_eval_every,
            "brush_optimizer_and_refinement": "Brush revision defaults",
        },
        "timing": {
            "process_wall_seconds": wall_seconds,
            "brush_reported_train_seconds": parse_humantime(raw_log),
            "note": "Brush reported time excludes evaluation but is rounded by its CLI.",
        },
        "quality_curve": evaluations,
        "command": command,
    }
    (output / "bench.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
