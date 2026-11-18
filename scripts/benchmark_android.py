"""Drive an Android training benchmark over adb.

Optionally installs the APK given with --apk, then launches the Cross3DGS
app with the preset extras from a config JSON,
collects host-side telemetry (battery, GPU clock/busy, process PSS) while the
run is in flight, captures the app logcat stream, waits for bench.json to be
rewritten, then pulls every artifact (bench.json, candidate_ledger.json,
step_times_ms.csv, renders, targets, splat.ply, checkpoint.d3gs) and writes a
run manifest with device
properties, separate host-argument and installed APK hashes, checkpoint
hashes, and the repository revision.

Example:
  uv run python scripts/benchmark_android.py \
    --serial <adb serial> --config configs/standard_quality.json \
    --scene bicycle --output out/android/bicycle --evaluate \
    [--iterations 200]   # optional smoke override
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_run  # noqa: E402

PACKAGE = "org.unclearness.cross3dgs"
ACTIVITY = f"{PACKAGE}/.MainActivity"
DEVICE_OUT = "files/out/bicycle"


def adb(serial: str, *args: str, binary: bool = False,
        check: bool = True) -> bytes | str:
    command = ["adb", "-s", serial, *args]
    result = subprocess.run(command, capture_output=True, check=check)
    return result.stdout if binary else result.stdout.decode(
        "utf-8", "replace")


def run_as(serial: str, shell_command: str, check: bool = True) -> str:
    return adb(serial, "shell",
               f"run-as {PACKAGE} sh -c '{shell_command}'", check=check)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_apk_provenance(serial: str) -> list[dict[str, str | None]]:
    """Hash every APK path installed for PACKAGE at experiment start."""
    paths = []
    for line in adb(serial, "shell", "pm", "path", PACKAGE).splitlines():
        line = line.strip()
        if line.startswith("package:"):
            paths.append(line.removeprefix("package:"))
    records = []
    for path in paths:
        output = adb(serial, "shell", "sha256sum", path, check=False).strip()
        fields = output.split()
        digest = fields[0].lower() if fields else None
        if digest is None or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest):
            digest = None
        records.append({"device_path": path, "sha256": digest})
    return records


def third_party_packages(serial: str) -> list[str]:
    packages = []
    for line in adb(
            serial, "shell", "pm", "list", "packages", "-3",
            check=False).splitlines():
        if line.startswith("package:"):
            packages.append(line.removeprefix("package:").strip())
    return sorted(package for package in packages if package)


def force_stop_other_user_apps(
        serial: str, keep_packages: set[str]) -> dict[str, object]:
    packages = [
        package for package in third_party_packages(serial)
        if package not in keep_packages
    ]
    pid_before = {
        package: adb(
            serial, "shell", "pidof", package, check=False).strip()
        for package in packages
    }
    for package in packages:
        adb(serial, "shell", "am", "force-stop", package, check=False)
    time.sleep(2)
    pid_after = {
        package: adb(
            serial, "shell", "pidof", package, check=False).strip()
        for package in packages
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kept_packages": sorted(keep_packages),
        "force_stopped_packages": packages,
        "pid_before": pid_before,
        "pid_after": pid_after,
        "remaining_pids": {
            package: pid for package, pid in pid_after.items() if pid
        },
        "memory_before": adb(
            serial, "shell", "dumpsys", "meminfo", "-c", check=False),
    }


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_snapshot(serial: str) -> dict:
    battery = adb(serial, "shell", "dumpsys battery")
    fields = {}
    for line in battery.splitlines():
        line = line.strip()
        if line.startswith("level:"):
            fields["battery_level"] = int(line.split(":")[1])
        if line.startswith("temperature:"):
            fields["battery_temp_dC"] = int(line.split(":")[1])
    fields["gpu_clock_mhz"] = adb(
        serial, "shell", "cat /sys/class/kgsl/kgsl-3d0/clock_mhz",
        check=False).strip()
    fields["gpubusy"] = adb(
        serial, "shell", "cat /sys/class/kgsl/kgsl-3d0/gpubusy",
        check=False).strip()
    thermal = adb(serial, "shell", "dumpsys thermalservice", check=False)
    status = re.search(r"^Thermal Status:\s*(\d+)", thermal, re.MULTILINE)
    current_thermal = thermal.partition(
        "Current temperatures from HAL:")[2].partition(
            "Current cooling devices from HAL:")[0]

    def temperature(name: str) -> tuple[float | None, int | None]:
        match = re.search(
            rf"Temperature\{{mValue=([0-9.]+), mType=\d+, "
            rf"mName={name}, mStatus=(\d+)\}}", current_thermal,
            re.IGNORECASE)
        return ((float(match.group(1)), int(match.group(2)))
                if match else (None, None))

    gpu_temp, gpu_status = temperature("GPU")
    skin_temp, skin_status = temperature("SKIN")
    fields["thermal_status"] = int(status.group(1)) if status else None
    fields["gpu_temp_C"] = gpu_temp
    fields["gpu_thermal_status"] = gpu_status
    fields["skin_temp_C"] = skin_temp
    fields["skin_thermal_status"] = skin_status
    pid = adb(serial, "shell", f"pidof {PACKAGE}", check=False).strip()
    fields["app_pid"] = pid or None
    if pid:
        meminfo = adb(serial, "shell", f"dumpsys meminfo {PACKAGE}",
                      check=False)
        for line in meminfo.splitlines():
            if "TOTAL PSS:" in line:
                fields["mem_pss_kb"] = int(line.split()[2])
                break
    return fields


def device_snapshot_with_retry(serial: str) -> dict:
    """Retry only the read-only telemetry snapshot after transient ADB loss.

    Launch, force-stop, and artifact-copy commands deliberately remain
    single-shot: replaying those commands could change experiment state.
    Uses the same ~90 s re-enumeration window as bench_mtime; a missed
    telemetry sample during the window costs one row, not the acquisition.
    """
    for delay in POLL_RETRY_DELAYS:
        try:
            return device_snapshot(serial)
        except subprocess.CalledProcessError:
            time.sleep(delay)
    return device_snapshot(serial)


POLL_RETRY_DELAYS = (2.0, 4.0, 8.0, 15.0, 30.0, 30.0)
# A battery run counts the cable as removed only after the device has been
# absent this long, so a transient transport drop is not taken for the unplug.
UNPLUG_CONFIRM_SECONDS = 10.0


def adb_state(serial: str) -> str:
    result = subprocess.run(["adb", "-s", serial, "get-state"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "absent"


def wait_unplugged_run(serial: str, emit, previous_mtime: str,
                       deadline: float, unplug_timeout_seconds: float,
                       poll_seconds: int) -> dict:
    """Battery run: the cable comes out after launch and back after training.

    No host telemetry is sampled; an on-device logger started by the caller
    records it. The record says whether training had finished when the cable
    came back, so a run that charged before its end can be rejected.
    """
    launched = time.monotonic()
    emit("waiting for USB removal")
    absent_since = None
    absent_utc = ""
    while True:
        state = adb_state(serial)
        moment = time.monotonic()
        if state == "device":
            absent_since = None
            if moment - launched > unplug_timeout_seconds:
                adb(serial, "shell", "am", "force-stop", PACKAGE, check=False)
                raise SystemExit("USB was not removed within the unplug "
                                 "timeout; run aborted")
        elif absent_since is None:
            absent_since = moment
            absent_utc = datetime.now(timezone.utc).isoformat()
        elif moment - absent_since >= UNPLUG_CONFIRM_SECONDS:
            break
        time.sleep(1.0)
    record = {
        "unplug_confirm_seconds": UNPLUG_CONFIRM_SECONDS,
        "unplugged_utc": absent_utc,
        "launch_to_unplug_seconds": absent_since - launched,
    }
    emit(f"USB removed {record['launch_to_unplug_seconds']:.0f} s after "
         "launch")
    while adb_state(serial) != "device":
        if time.monotonic() > deadline:
            raise SystemExit("device did not return before the timeout")
        time.sleep(5.0)
    record["replugged_utc"] = datetime.now(timezone.utc).isoformat()
    record["launch_to_replug_seconds"] = time.monotonic() - launched
    emit(f"USB back {record['launch_to_replug_seconds']:.0f} s after launch")
    time.sleep(5.0)
    current = bench_mtime(serial)
    record["bench_ready_at_replug"] = current not in (previous_mtime, "0")
    while current in (previous_mtime, "0"):
        if time.monotonic() > deadline:
            raise SystemExit("timeout waiting for bench.json")
        time.sleep(poll_seconds)
        current = bench_mtime(serial)
    record["launch_to_bench_seconds"] = time.monotonic() - launched
    return record


def bench_mtime(serial: str) -> str:
    """Read the on-device bench.json mtime, retrying transient ADB loss.

    This is the completion poll of a multi-hour cell. It is a read-only stat,
    so replaying it cannot change experiment state -- the same reason
    device_snapshot_with_retry is allowed to retry, and unlike launch,
    force-stop, and artifact-copy commands, which stay single-shot.

    The backoff spans ~90 s because the observed failures are not single
    dropped commands: under heavy host USB traffic the transport drops and
    the device re-enumerates (briefly listing as offline or unauthorized),
    which takes tens of seconds to come back while the on-device training
    continues unharmed. A genuine device loss still fails closed after the
    window.
    """
    command = (f"stat -c %Y {DEVICE_OUT}/bench.json 2>/dev/null || echo 0")
    for index, delay in enumerate(POLL_RETRY_DELAYS):
        try:
            return run_as(serial, command).strip()
        except subprocess.CalledProcessError:
            time.sleep(delay)
    return run_as(serial, command).strip()


def pull_file(serial: str, device_path: str,
              host_path: Path) -> tuple[float, int]:
    started = time.monotonic()
    host_path.parent.mkdir(parents=True, exist_ok=True)
    data = adb(serial, "exec-out", f"run-as {PACKAGE} cat {device_path}",
               binary=True)
    host_path.write_bytes(data)
    return time.monotonic() - started, len(data)


def capture_failure_context(serial: str, output: Path) -> None:
    """Preserve OS-level evidence before a later launch rotates it away."""
    captures = {
        "full_logcat.txt": ("logcat", "-b", "all", "-d", "-v", "threadtime"),
        "activity_exit_info.txt": (
            "shell", "dumpsys", "activity", "exit-info", PACKAGE),
        "thermal_service.txt": ("shell", "dumpsys", "thermalservice"),
        "memory_info.txt": ("shell", "dumpsys", "meminfo"),
    }
    for name, command in captures.items():
        (output / name).write_text(
            adb(serial, *command, check=False), encoding="utf-8")


def main() -> None:
    global PACKAGE, ACTIVITY, DEVICE_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--package", default=PACKAGE,
                        help="installed application ID; use a separate ID for "
                             "experimental APKs")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/standard_quality.json"))
    parser.add_argument("--scene", default="bicycle",
                        help="on-device scene directory to train and collect "
                             "(files/data/<scene>, files/out/<scene>); also "
                             "sets the scene intent extra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int,
                        help="override preset iterations (smoke runs)")
    parser.add_argument("--extra", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override or add an intent extra (int/float/bool "
                             "inferred; repeatable). Deviations from the "
                             "preset are recorded in the manifest.")
    parser.add_argument("--apk", type=Path,
                        help="install this APK first (otherwise use the "
                             "installed one) and record its hash")
    parser.add_argument(
        "--training-view-seed", type=int,
        help="training-view sampling seed. Passed as the training_view_seed "
             "intent extra and verified against the pulled bench.json, so an "
             "old APK that ignores the extra fails the run instead of "
             "silently training at the compile-time 42. Omit to preserve "
             "historical behavior (no extra, no check).")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=180)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="expected hash for an init_checkpoint override; defaults to "
             "the preset initial_state hash")
    parser.add_argument(
        "--preserve-other-apps", action="store_true",
        help="do not force-stop third-party apps; record their PIDs instead")
    parser.add_argument(
        "--min-battery-level", type=int,
        help="abort and force-stop Cross3DGS below this battery percentage")
    parser.add_argument(
        "--abort-on-thermal-status", action="store_true",
        help="abort if overall, GPU, or skin HAL thermal status is nonzero")
    parser.add_argument(
        "--screen-brightness", type=int,
        help="temporarily set Android system brightness (0--255); the "
             "original value is restored when the harness exits")
    parser.add_argument(
        "--screen-off-after-launch", action="store_true",
        help="send KEYCODE_SLEEP one second after launch for controlled "
             "headless thermal runs")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="score the pulled renders with scripts/common_evaluator.py "
             "(writes common_eval.json)")
    parser.add_argument("--no-lpips", action="store_true",
                        help="with --evaluate, skip LPIPS (no torch download)")
    parser.add_argument(
        "--unplugged", action="store_true",
        help="battery run: after launch wait for the USB cable to be removed "
             "and put back, then for bench.json. Host telemetry is not "
             "sampled; an on-device logger must record it.")
    parser.add_argument("--unplug-timeout-minutes", type=float, default=10.0,
                        help="with --unplugged, abort if the cable is still "
                             "in this long after launch")
    parser.add_argument(
        "--keep-package", action="append", default=[],
        help="third-party package to leave running (repeatable; use only for "
             "strictly required measurement infrastructure)")
    args = parser.parse_args()
    PACKAGE = args.package
    ACTIVITY = f"{PACKAGE}/org.unclearness.cross3dgs.MainActivity"
    DEVICE_OUT = f"files/out/{args.scene}"
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    extras = dict(config["android_launch"]["extras"])
    if args.scene != "bicycle":
        extras["scene"] = args.scene
    if args.iterations is not None:
        extras["iterations"] = args.iterations
    if args.training_view_seed is not None:
        extras["training_view_seed"] = args.training_view_seed
    for override in args.extra:
        key, _, raw = override.partition("=")
        if not raw:
            raise SystemExit(f"expected KEY=VALUE, got {override}")
        if raw in ("true", "false"):
            value: object = raw == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        extras[key] = value
    if extras.get("scene", "bicycle") != args.scene:
        raise SystemExit(
            f"scene extra {extras.get('scene')!r} contradicts "
            f"--scene {args.scene!r}; artifacts would be collected from "
            f"{DEVICE_OUT}")
    serial = args.serial
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    host_console_path = output / "host_console.txt"
    host_console_path.write_text("", encoding="utf-8")
    repository_revision_at_start = git_revision(root)
    if args.apk is not None:
        print(adb(serial, "install", "-r", str(args.apk.resolve())).strip(),
              flush=True)
    installed_apks_at_start = installed_apk_provenance(serial)
    original_screen_brightness = adb(
        serial, "shell", "settings", "get", "system", "screen_brightness",
        check=False).strip()
    if args.screen_brightness is not None:
        if not 0 <= args.screen_brightness <= 255:
            raise SystemExit("--screen-brightness must be in [0, 255]")
        adb(serial, "shell", "settings", "put", "system",
            "screen_brightness", str(args.screen_brightness))

        def restore_brightness() -> None:
            if original_screen_brightness.isdigit():
                adb(serial, "shell", "settings", "put", "system",
                    "screen_brightness", original_screen_brightness,
                    check=False)

        atexit.register(restore_brightness)

    def emit(message: str) -> None:
        print(message, flush=True)
        with open(host_console_path, "a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    emit("command=" + subprocess.list2cmdline(sys.argv))

    checkpoint = extras.get("init_checkpoint", "")
    if checkpoint:
        device_hash = run_as(
            serial,
            f"sha256sum files/data/{args.scene}/{checkpoint}").split()[0]
        expected = (args.expected_checkpoint_sha256 or
                    config["initial_state"]["sha256"])
        if device_hash != expected:
            raise SystemExit(
                f"device checkpoint hash {device_hash} != {expected}")

    if args.preserve_other_apps:
        packages = [package for package in third_party_packages(serial)
                    if package != PACKAGE]
        clean_state = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "preserved; only Cross3DGS is force-stopped",
            "observed_pids": {
                package: adb(serial, "shell", "pidof", package,
                             check=False).strip()
                for package in packages
            },
            "force_stopped_packages": [],
            "remaining_pids": {},
        }
    else:
        clean_state = force_stop_other_user_apps(
            serial, {PACKAGE, *args.keep_package})
    (output / "clean_app_state.json").write_text(
        json.dumps(clean_state, indent=2) + "\n", encoding="utf-8")
    emit("force_stopped_third_party_packages={}".format(
        len(clean_state["force_stopped_packages"])))
    if clean_state["remaining_pids"]:
        raise SystemExit(
            "third-party processes remained after force-stop; "
            "see clean_app_state.json")
    adb(serial, "shell", "am", "force-stop", PACKAGE)
    adb(serial, "logcat", "-c")
    # Best effort; some devices refuse `input` from the adb shell uid.
    adb(serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP", check=False)
    previous_mtime = bench_mtime(serial)
    time.sleep(1)

    launch = ["shell", "am", "start", "-n", ACTIVITY]
    for key, value in extras.items():
        if isinstance(value, bool):
            launch += ["--ez", key, "true" if value else "false"]
        elif isinstance(value, int):
            launch += ["--ei", key, str(value)]
        elif isinstance(value, float):
            launch += ["--ef", key, str(value)]
        else:
            launch += ["--es", key, str(value)]
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    launch_started = time.monotonic()
    launch_output = adb(serial, *launch).strip()
    launch_adb_seconds = time.monotonic() - launch_started
    emit(launch_output)
    if args.screen_off_after_launch:
        time.sleep(1)
        adb(serial, "shell", "input", "keyevent", "KEYCODE_SLEEP",
            check=False)

    unplugged_run: dict | None = None
    telemetry_path = output / "telemetry.csv"
    with open(telemetry_path, "w", encoding="utf-8") as telemetry:
        telemetry.write("elapsed_seconds,battery_level,battery_temp_dC,"
                        "gpu_clock_mhz,gpubusy,thermal_status,gpu_temp_C,"
                        "gpu_thermal_status,skin_temp_C,skin_thermal_status,"
                        "mem_pss_kb,app_pid\n")
        deadline = started + args.timeout_minutes * 60.0
        missing_pid_polls = 0
        while True:
            if args.unplugged:
                unplugged_run = wait_unplugged_run(
                    serial, emit, previous_mtime, deadline,
                    args.unplug_timeout_minutes * 60.0, args.poll_seconds)
                emit(f"bench.json updated after "
                     f"{time.monotonic() - started:.0f} s")
                break
            snapshot = device_snapshot_with_retry(serial)
            if (args.min_battery_level is not None and
                    snapshot.get("battery_level", 100) <
                    args.min_battery_level):
                capture_failure_context(serial, output)
                adb(serial, "shell", "am", "force-stop", PACKAGE,
                    check=False)
                raise SystemExit(
                    "battery fell below --min-battery-level; run aborted")
            if args.abort_on_thermal_status and any(
                    snapshot.get(key, 0) not in (None, 0)
                    for key in ("thermal_status", "gpu_thermal_status",
                                "skin_thermal_status")):
                capture_failure_context(serial, output)
                adb(serial, "shell", "am", "force-stop", PACKAGE,
                    check=False)
                raise SystemExit(
                    "HAL thermal status became nonzero; run aborted")
            telemetry.write(
                f"{time.monotonic() - started:.0f},"
                f"{snapshot.get('battery_level', '')},"
                f"{snapshot.get('battery_temp_dC', '')},"
                f"{snapshot.get('gpu_clock_mhz', '')},"
                f"\"{snapshot.get('gpubusy', '')}\","
                f"{snapshot.get('thermal_status', '')},"
                f"{snapshot.get('gpu_temp_C', '')},"
                f"{snapshot.get('gpu_thermal_status', '')},"
                f"{snapshot.get('skin_temp_C', '')},"
                f"{snapshot.get('skin_thermal_status', '')},"
                f"{snapshot.get('mem_pss_kb', '')},"
                f"{snapshot.get('app_pid', '')}\n")
            telemetry.flush()
            current = bench_mtime(serial)
            if current != previous_mtime and current != "0":
                emit(f"bench.json updated after "
                     f"{time.monotonic() - started:.0f} s")
                break
            log = adb(serial, "logcat", "-d", "-s", "cross3dgs:I",
                      check=False)
            if "Optimization failed" in log:
                (output / "logcat.txt").write_text(log, encoding="utf-8")
                capture_failure_context(serial, output)
                raise SystemExit("training reported a failure; see logcat.txt")
            if snapshot.get("app_pid") is None:
                missing_pid_polls += 1
            else:
                missing_pid_polls = 0
            if missing_pid_polls >= 3:
                (output / "logcat.txt").write_text(log, encoding="utf-8")
                capture_failure_context(serial, output)
                raise SystemExit(
                    "app process absent for three consecutive polls before "
                    "bench.json update")
            if time.monotonic() > deadline:
                (output / "logcat.txt").write_text(log, encoding="utf-8")
                capture_failure_context(serial, output)
                raise SystemExit("timeout waiting for bench.json")
            progress = [line for line in log.splitlines()
                        if "progress iteration=" in line]
            if progress:
                emit(progress[-1].split("cross3dgs:")[-1].strip())
            time.sleep(args.poll_seconds)

    launch_to_bench_seconds = time.monotonic() - started

    (output / "logcat.txt").write_text(
        adb(serial, "logcat", "-d", "-s", "cross3dgs:I", check=False),
        encoding="utf-8")

    pull_started = time.monotonic()
    pull_timings: dict[str, dict[str, float | int]] = {}

    def timed_pull(device_path: str, host_path: Path) -> None:
        seconds, byte_count = pull_file(serial, device_path, host_path)
        pull_timings[str(host_path.relative_to(output))] = {
            "seconds": seconds, "bytes": byte_count}

    timed_pull(f"{DEVICE_OUT}/bench.json", output / "bench.json")
    timed_pull(f"{DEVICE_OUT}/candidate_ledger.json",
               output / "candidate_ledger.json")
    timed_pull(f"{DEVICE_OUT}/step_times_ms.csv",
               output / "step_times_ms.csv")
    timed_pull(f"{DEVICE_OUT}/splat.ply", output / "splat.ply")
    timed_pull(f"{DEVICE_OUT}/checkpoint.d3gs",
               output / "checkpoint.d3gs")
    for kind in ("render", "target"):
        names = run_as(serial, f"ls {DEVICE_OUT}/{kind}").split()
        for name in names:
            timed_pull(f"{DEVICE_OUT}/{kind}/{name}",
                       output / kind / name)
    artifact_pull_seconds = time.monotonic() - pull_started

    bench = json.loads((output / "bench.json").read_text(encoding="utf-8"))
    if args.training_view_seed is not None and \
            bench.get("training_view_seed") != args.training_view_seed:
        raise SystemExit(
            f"bench.json reports training_view_seed="
            f"{bench.get('training_view_seed')!r} but "
            f"{args.training_view_seed} was requested; the installed APK "
            f"either ignores the extra or predates the settable seed")
    ledger = json.loads(
        (output / "candidate_ledger.json").read_text(encoding="utf-8"))
    schedule = bench.get("schedule_ir", {})
    if not ledger.get("configured_accepted"):
        raise SystemExit("runtime candidate ledger rejected configured schedule")
    if ledger.get("configured_candidate_id") != schedule.get(
            "schedule_candidate_id"):
        raise SystemExit("bench schedule and candidate ledger disagree")
    if ledger.get("policy_selected_candidate_id") != schedule.get(
            "policy_selected_candidate_id"):
        raise SystemExit("bench policy selection and candidate ledger disagree")
    expected_selection_source = (
        "runtime_candidate_ledger+storage_range_lowering"
        if schedule.get("split_sh_rest_storage") else
        "runtime_candidate_ledger")
    if schedule.get("schedule_selection_source") != \
            expected_selection_source:
        raise SystemExit("runtime schedule was not selected by candidate ledger")
    properties = {
        key: adb(serial, "shell", "getprop", key).strip()
        for key in ("ro.product.model", "ro.product.manufacturer",
                    "ro.build.version.release", "ro.build.version.sdk",
                    "ro.build.fingerprint")
    }
    # Finalize the console before hashing artifacts. No emit() call may follow
    # manifest creation, otherwise its recorded host_console.txt hash is stale.
    emit(json.dumps({key: bench.get(key) for key in (
        "completed_iterations", "median_step_ms", "p10_step_ms",
        "p90_step_ms", "total_train_seconds", "evaluation_psnr",
        "evaluation_mean_view_psnr", "evaluation_mean_l1",
        "overflow_count", "state_non_finite", "peak_memory_bytes")},
        indent=2))
    pulled_artifacts = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "preset": config["name"],
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "extras": extras,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds_launch_to_bench": launch_to_bench_seconds,
        "host_timing": {
            "launch_adb_seconds": launch_adb_seconds,
            "launch_to_bench_detected_seconds": launch_to_bench_seconds,
            "artifact_pull_seconds": artifact_pull_seconds,
            "launch_to_artifacts_pulled_seconds": time.monotonic() - started,
            "artifact_pulls": pull_timings,
        },
        "artifact_sha256": {
            str(path.relative_to(output)).replace("\\", "/"):
                sha256_file(path)
            for path in sorted(pulled_artifacts)
        },
        "serial": serial,
        "device_properties": properties,
        "apk_sha256": sha256_file(args.apk) if args.apk else None,
        "apk_sha256_semantics": (
            "host --apk argument, installed before the run"),
        "installed_apks_at_start": installed_apks_at_start,
        "repository_revision": repository_revision_at_start,
        "repository_revision_at_start": repository_revision_at_start,
        "clean_app_state_file": "clean_app_state.json",
        "display_control": {
            "original_screen_brightness": original_screen_brightness,
            "requested_screen_brightness": args.screen_brightness,
            "screen_off_after_launch": args.screen_off_after_launch,
        },
        "command": sys.argv,
        "unplugged_run": unplugged_run,
        "bench_summary": {key: bench.get(key) for key in (
            "completed_iterations", "median_step_ms", "p10_step_ms",
            "p90_step_ms", "total_train_seconds", "evaluation_psnr",
            "evaluation_mean_view_psnr", "evaluation_mean_l1",
            "overflow_count", "state_non_finite", "peak_memory_bytes")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.evaluate:
        print(json.dumps(evaluate_run(output, f"android_{args.scene}",
                                      lpips=not args.no_lpips), indent=2))


if __name__ == "__main__":
    main()
