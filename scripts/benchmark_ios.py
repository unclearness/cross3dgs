"""Drive an iOS training benchmark over `xcrun devicectl` (adb-path analogue).

Launches the cross3dgs iOS app with the argument vector from a config JSON
(@BUNDLE@/@DOCUMENTS@ placeholders are substituted by the app), polls
Documents/run_status.json until the in-process training run reports
completion, then copies every artifact (bench.json, candidate_ledger.json,
step_times_ms.csv, renders, targets, splat.ply, checkpoint, caps.json,
console.log, on-device telemetry.csv) out of the app data container and
writes a run manifest. With --evaluate, the pulled renders are scored with
scripts/common_evaluator.py.

Example:
  uv run python scripts/benchmark_ios.py --device <udid> \
    --config configs/standard_quality.json \
    --output out/ios/bicycle --evaluate \
    [--iterations 200 --gaussian-capacity 20000]   # reduced-capacity smoke

  # capability probe only (no training arguments):
  uv run python scripts/benchmark_ios.py --device <udid> \
    --config configs/standard_quality.json --output out/ios/caps --caps-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_run  # noqa: E402


def devicectl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["xcrun", "devicectl", *args]
    return subprocess.run(command, capture_output=True, text=True, check=check)


def copy_from(device: str, bundle_id: str, source: str,
              destination: Path, check: bool = True) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = devicectl(
        "device", "copy", "from", "--device", device,
        "--domain-type", "appDataContainer",
        "--domain-identifier", bundle_id,
        "--source", source, "--destination", str(destination), check=False)
    if result.returncode != 0:
        if check:
            raise SystemExit(
                f"devicectl copy from {source} failed:\n{result.stderr}")
        return False
    return True


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


def read_run_status(device: str, bundle_id: str,
                    scratch: Path) -> dict | None:
    path = scratch / "run_status.json"
    path.unlink(missing_ok=True)
    if not copy_from(device, bundle_id, "Documents/run_status.json", path,
                     check=False):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def override_argument(arguments: list[str], flag: str,
                      value: str) -> list[str]:
    result = list(arguments)
    if flag in result:
        index = result.index(flag)
        result[index + 1] = value
    else:
        result += [flag, value]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True,
                        help="devicectl device identifier or UDID")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/standard_quality.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int,
                        help="override preset --train-iterations (smoke runs)")
    parser.add_argument("--gaussian-capacity", type=int,
                        help="override preset --gaussian-capacity "
                             "(reduced-capacity engineering smoke)")
    parser.add_argument("--extra-arg", action="append", default=[],
                        metavar="ARG",
                        help="append a raw train_cli argument token "
                             "(repeatable); recorded in the manifest")
    parser.add_argument("--caps-only", action="store_true",
                        help="launch with no training arguments: capability "
                             "probe only")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="score the pulled renders with scripts/common_evaluator.py "
             "(writes common_eval.json)")
    parser.add_argument("--no-lpips", action="store_true",
                        help="with --evaluate, skip LPIPS (no torch download)")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout-minutes", type=int, default=180)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle_id = config["ios_launch"]["bundle_id"]
    device = args.device
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / ".poll"
    scratch.mkdir(exist_ok=True)
    host_console_path = output / "host_console.txt"
    host_console_path.write_text("", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        with open(host_console_path, "a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    emit("command=" + subprocess.list2cmdline(sys.argv))

    arguments: list[str] = []
    deviations: dict[str, object] = {}
    if not args.caps_only:
        arguments = list(config["ios_launch"]["arguments"])
        if args.iterations is not None:
            arguments = override_argument(
                arguments, "--train-iterations", str(args.iterations))
            deviations["train_iterations"] = args.iterations
        if args.gaussian_capacity is not None:
            arguments = override_argument(
                arguments, "--gaussian-capacity", str(args.gaussian_capacity))
            deviations["gaussian_capacity"] = args.gaussian_capacity
        if args.extra_arg:
            arguments += args.extra_arg
            deviations["extra_args"] = args.extra_arg

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    launch = devicectl(
        "device", "process", "launch", "--terminate-existing",
        "--device", device, bundle_id, *arguments, check=False)
    if launch.returncode != 0:
        raise SystemExit(f"devicectl launch failed:\n{launch.stderr}")
    emit(launch.stdout.strip())

    terminal_state = "caps_only" if args.caps_only else "finished"
    deadline = started + args.timeout_minutes * 60.0
    status: dict | None = None
    while True:
        time.sleep(args.poll_seconds)
        status = read_run_status(device, bundle_id, scratch)
        if status is not None:
            emit(f"[{time.monotonic() - started:.0f}s] run_status={status}")
            # Ignore stale status files from before this launch.
            if status.get("utc", "") >= started_utc:
                if status.get("state") == terminal_state:
                    break
                if status.get("state") == "finished" and args.caps_only:
                    break
        if time.monotonic() > deadline:
            copy_from(device, bundle_id, "Documents/console.log",
                      output / "console.log", check=False)
            raise SystemExit("timeout waiting for run_status.json completion")

    wall_seconds = time.monotonic() - started
    emit(f"terminal state after {wall_seconds:.0f} s")

    pull_specs = {
        "Documents/caps.json": output / "caps.json",
        "Documents/console.log": output / "console.log",
        "Documents/telemetry.csv": output / "device_telemetry.csv",
        "Documents/run_status.json": output / "run_status.json",
    }
    if not args.caps_only:
        # Derive the on-device artifact locations from the launch arguments
        # (later duplicates win, matching train_cli's parser), so output-path
        # overrides via --extra-arg are pulled from the right place instead
        # of a stale default directory.
        def argument_value(flag: str) -> str | None:
            value = None
            for index, token in enumerate(arguments[:-1]):
                if token == flag:
                    value = arguments[index + 1]
            return value

        def container_path(value: str) -> str | None:
            prefix = "@DOCUMENTS@/"
            if value and value.startswith(prefix):
                return "Documents/" + value[len(prefix):]
            return None

        for flag, name in (("--benchmark-json", "bench.json"),
                           ("--candidate-ledger-json",
                            "candidate_ledger.json"),
                           ("--step-times-csv", "step_times_ms.csv"),
                           ("--output-ply", "splat.ply"),
                           ("--output-checkpoint", "checkpoint.d3gs")):
            value = argument_value(flag)
            source = container_path(value) if value else None
            if source:
                pull_specs[source] = output / name
        eval_dir = argument_value("--evaluation-image-dir")
        eval_source = container_path(eval_dir) if eval_dir else None
        if eval_source:
            pull_specs[f"{eval_source}/render"] = output / "render"
            pull_specs[f"{eval_source}/target"] = output / "target"
    pulled: dict[str, bool] = {}
    for source, destination in pull_specs.items():
        pulled[source] = copy_from(device, bundle_id, source, destination,
                                   check=False)
        emit(f"pull {source}: {'ok' if pulled[source] else 'MISSING'}")

    exit_code = status.get("exit_code") if status else None
    bench_summary = None
    bench_path = output / "bench.json"
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        bench_summary = {key: bench.get(key) for key in (
            "completed_iterations", "median_step_ms", "p10_step_ms",
            "p90_step_ms", "total_train_seconds", "evaluation_psnr",
            "evaluation_mean_view_psnr", "evaluation_mean_l1",
            "overflow_count", "state_non_finite", "peak_memory_bytes")}

    device_info = devicectl("device", "info", "details", "--device", device,
                            check=False)
    (output / "device_info.txt").write_text(
        device_info.stdout + device_info.stderr, encoding="utf-8")

    artifacts = [path for path in output.rglob("*")
                 if path.is_file() and path.name != "manifest.json"
                 and ".poll" not in path.parts]
    manifest = {
        "schema_version": 1,
        "evidence_status": config.get("status"),
        "preset": config["name"],
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "arguments": arguments,
        "deviations_from_preset": deviations,
        "caps_only": args.caps_only,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds_launch_to_terminal": wall_seconds,
        "device": device,
        "bundle_id": bundle_id,
        "run_status": status,
        "app_exit_code": exit_code,
        "pulled": pulled,
        "artifact_sha256": {
            str(path.relative_to(output)).replace("\\", "/"):
                sha256_file(path)
            for path in sorted(artifacts)
        },
        "repository_revision": git_revision(root),
        "command": sys.argv,
        "bench_summary": bench_summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if bench_summary:
        emit(json.dumps(bench_summary, indent=2))
    if exit_code not in (0, None):
        raise SystemExit(f"training exit_code={exit_code}; see console.log")
    if args.evaluate and not args.caps_only:
        print(json.dumps(evaluate_run(output, f"ios_{config['name']}",
                                      lpips=not args.no_lpips), indent=2))


if __name__ == "__main__":
    main()
