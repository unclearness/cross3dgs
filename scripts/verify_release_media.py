#!/usr/bin/env python3
"""Verify the reviewed release-media allowlist before publishing the project page."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

MEDIA_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.tif', '.tiff', '.ico', '.mp4', '.webm', '.mov', '.avi', '.mkv', '.pdf', '.ply', '.glb', '.gltf', '.splat', '.ksplat', '.zip', '.tar', '.tgz', '.7z'}

def validate(root: Path, manifest: dict, paths: list[str]) -> list[str]:
    expected = {entry['path']: entry for entry in manifest['files']}
    actual = {p for p in paths if Path(p).suffix.lower() in MEDIA_SUFFIXES}
    errors = [f'Media inventory mismatch: missing={sorted(expected.keys()-actual)}; unexpected={sorted(actual-expected.keys())}'] if actual != expected.keys() else []
    for name, entry in expected.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            errors.append(f'Missing or invalid media path: {name}')
            continue
        data = target.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            errors.append(f'Unreviewed media bytes: {name}')
    return errors

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'project_page/media_manifest.json').read_text(encoding='utf-8'))
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root).decode('utf-8').split('\0')
    errors = validate(root, manifest, [p for p in paths if p])
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'Release media verified: {len(manifest["files"])} reviewed assets; capture subject: teddy bear.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
