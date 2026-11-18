"""GPU cool-down guard for desktop benchmark runners.

NVIDIA hosts use the original temperature-controlled protocol. Some portable
Vulkan hosts (notably Iris Xe) expose no temperature through ``nvidia-smi``;
those hosts instead record an explicit fixed idle proxy. A missing reading is
never interpreted as a cool GPU.
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone

COOL_THRESHOLD_C = 60
POLL_SECONDS = 15
TIMEOUT_SECONDS = 20 * 60
NO_TEMPERATURE_IDLE_SECONDS = 10 * 60


def utc_timestamp(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if epoch is None else epoch, timezone.utc
    ).isoformat()


def gpu_temperature() -> int | None:
    """Return the NVIDIA temperature, or ``None`` when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return int(out.splitlines()[0])


def wait_for_cool_gpu(label: str = "", *, not_before_epoch: float | None = None) -> dict:
    """Wait for the applicable cool-down protocol and record its witness.

    The NVIDIA result intentionally retains the historical two-key schema.
    For hosts without a temperature provider, ``not_before_epoch`` is the end
    of the prior evaluation or failure handling; we wait until 600 seconds
    have elapsed from it and record both endpoints.
    """
    start = time.time()
    temp = gpu_temperature()
    if temp is None:
        previous_end = not_before_epoch if not_before_epoch is not None else start
        remaining = max(0.0, NO_TEMPERATURE_IDLE_SECONDS - (start - previous_end))
        if remaining:
            print(f"[cooldown{label}] no GPU temperature; fixed idle {remaining:.1f}s", flush=True)
            time.sleep(remaining)
        launch = time.time()
        waited = round(launch - start, 1)
        print(f"[cooldown{label}] no GPU temperature; launch after {waited}s", flush=True)
        return {
            "start_temperature_c": None,
            "cooldown_wait_seconds": waited,
            "cooldown_mode": "fixed_wait_no_gpu_temp",
            "previous_cell_finished_utc": utc_timestamp(previous_end),
            "training_launch_utc": utc_timestamp(launch),
        }
    while temp > COOL_THRESHOLD_C:
        if time.time() - start > TIMEOUT_SECONDS:
            print(f"[cooldown{label}] TIMEOUT at {temp} C "
                  f"(threshold {COOL_THRESHOLD_C} C)", flush=True)
            break
        print(f"[cooldown{label}] {temp} C > {COOL_THRESHOLD_C} C, waiting...",
              flush=True)
        time.sleep(POLL_SECONDS)
        temp = gpu_temperature()
        if temp is None:
            return wait_for_cool_gpu(label, not_before_epoch=not_before_epoch)
    waited = round(time.time() - start, 1)
    print(f"[cooldown{label}] start at {temp} C after {waited}s", flush=True)
    return {"start_temperature_c": temp, "cooldown_wait_seconds": waited}
