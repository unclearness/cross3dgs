"""Download and unpack the evaluation datasets into data/.

  mipnerf360  Mip-NeRF 360 (Barron et al.), 360_v2.zip, about 12 GB, into
              data/360_v2/<scene> (bicycle, garden, stump, room, counter,
              kitchen, bonsai).
  tandt_db    Tanks and Temples (train, truck) and Deep Blending (drjohnson,
              playroom) as packaged by the 3D Gaussian Splatting authors,
              tandt_db.zip, about 650 MB, into data/tandt/<scene> and
              data/deep_blending/<scene>.

Existing scene directories are never overwritten; downloads go to
cache/downloads and are reused when present.

Example:
  uv run python scripts/download_datasets.py mipnerf360
  uv run python scripts/download_datasets.py mipnerf360 --scenes bicycle garden
  uv run python scripts/download_datasets.py tandt_db
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = Path(__file__).resolve().parents[1]

DATASETS = {
    "mipnerf360": {
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "archive": "360_v2.zip",
        "sha256": None,
        # archive prefix -> destination below --data-root
        "scenes": {name: (PurePosixPath(name), Path("360_v2") / name) for name in (
            "bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai")},
    },
    "tandt_db": {
        "url": ("https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
                "datasets/input/tandt_db.zip"),
        "archive": "tandt_db.zip",
        "sha256": "816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1",
        "scenes": {
            "train": (PurePosixPath("tandt/train"), Path("tandt/train")),
            "truck": (PurePosixPath("tandt/truck"), Path("tandt/truck")),
            "drjohnson": (PurePosixPath("db/drjohnson"),
                          Path("deep_blending/drjohnson")),
            "playroom": (PurePosixPath("db/playroom"),
                         Path("deep_blending/playroom")),
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"reusing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "cross3dgs-download/1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 22):
            output.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{destination.name}: {done / total:6.1%} of "
                      f"{total / 1e9:.2f} GB", end="", flush=True)
    print()
    if total and partial.stat().st_size != total:
        raise SystemExit(f"incomplete download: {partial}")
    os.replace(partial, destination)


def extract(archive: Path, prefix: PurePosixPath, destination: Path) -> None:
    """Extract every member under ``prefix`` into ``destination``."""
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.parts[:len(prefix.parts)] != prefix.parts:
                continue
            relative = PurePosixPath(*path.parts[len(prefix.parts):])
            if ".." in relative.parts or relative.is_absolute():
                raise SystemExit(f"unsafe archive member: {member.filename}")
            target = staging / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            count += 1
    if count == 0:
        raise SystemExit(f"{archive.name} has no members under {prefix}/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    print(f"{destination}: {count} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--scenes", nargs="+",
                        help="subset of scenes (default: all of the dataset)")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--cache", type=Path,
                        default=REPOSITORY / "cache" / "downloads")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    scenes = args.scenes or list(dataset["scenes"])
    unknown = sorted(set(scenes) - set(dataset["scenes"]))
    if unknown:
        raise SystemExit(f"unknown scenes for {args.dataset}: {unknown}")
    pending = [scene for scene in scenes
               if not (args.data_root / dataset["scenes"][scene][1]).exists()]
    for scene in sorted(set(scenes) - set(pending)):
        print(f"{args.data_root / dataset['scenes'][scene][1]} exists; skipped")
    if not pending:
        return

    archive = args.cache / dataset["archive"]
    download(dataset["url"], archive)
    if dataset["sha256"] is not None and sha256(archive) != dataset["sha256"]:
        raise SystemExit(f"{archive} does not match the expected SHA-256")
    for scene in pending:
        prefix, relative = dataset["scenes"][scene]
        extract(archive, prefix, args.data_root / relative)


if __name__ == "__main__":
    sys.exit(main())
