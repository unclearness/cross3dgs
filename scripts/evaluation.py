"""Score a finished training run with scripts/common_evaluator.py.

Renders are read from ``<run>/render`` (or another folder and glob), and the
targets either from ``<run>/target``, written by the device runners, or from
the dataset's held-out split (every eighth image by sorted file name). The
scores go to ``<run>/common_eval.json``. LPIPS needs torch, torchvision, and
torchmetrics, which are not project dependencies, so the evaluator runs
through ``uv run --with``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
EVALUATOR = REPOSITORY / "scripts" / "common_evaluator.py"


def evaluator_command(run_dir: Path, name: str, lpips: bool = True,
                      render_dir: Path | None = None, render_glob: str | None = None,
                      dataset: tuple[str, str, str] | None = None) -> list[str]:
    command = ["uv", "run"]
    if lpips:
        command += ["--with", "torch", "--with", "torchvision",
                    "--with", "torchmetrics"]
    command += ["python", str(EVALUATOR), "--name", name,
                "--render-dir", str(render_dir or run_dir / "render"),
                "--output", str(run_dir / "common_eval.json")]
    if render_glob:
        command += ["--render-glob", render_glob]
    if dataset is None:
        command += ["--target-dir", str(run_dir / "target")]
    else:
        root, scene, image_dir = dataset
        command += ["--dataset-root", root, "--scene", scene, "--image-dir", image_dir]
    bench = run_dir / "bench.json"
    if bench.is_file():
        command += ["--bench-json", str(bench)]
    if not lpips:
        command.append("--no-lpips")
    return command


def evaluate_run(run_dir: Path, name: str, lpips: bool = True,
                 render_dir: Path | None = None, render_glob: str | None = None,
                 dataset: tuple[str, str, str] | None = None) -> dict:
    """Score ``run_dir`` and return the summary metrics.

    ``dataset`` is (dataset root, scene, image folder); without it the targets
    are read from ``<run>/target``.
    """
    renders = render_dir or run_dir / "render"
    if not renders.is_dir():
        raise SystemExit(f"{run_dir} has no renders at {renders}")
    if dataset is None and not (run_dir / "target").is_dir():
        raise SystemExit(f"{run_dir} has no target/ to evaluate against")
    subprocess.run(evaluator_command(run_dir, name, lpips, render_dir, render_glob,
                                     dataset),
                   cwd=REPOSITORY, check=True)
    result = json.loads((run_dir / "common_eval.json").read_text(encoding="utf-8"))
    return {key: result[key] for key in (
        "view_count", "mean_view_psnr", "ssim", "lpips_alex")}
