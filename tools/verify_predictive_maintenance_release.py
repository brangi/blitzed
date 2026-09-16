#!/usr/bin/env python3
"""Verify a Blitzed predictive-maintenance release directory.

The verifier checks the manifest schema, rejects paths outside the release
folder, and validates every listed artifact's size and SHA-256 checksum before
flashing or deploying a model.

Usage:
    python3 tools/verify_predictive_maintenance_release.py \
        /path/to/release/blitzed_release_manifest.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path):
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path):
    """Load and minimally validate a release manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ValueError("Manifest must be a version 1 object")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("Manifest must contain a non-empty files list")
    return manifest


def verify_manifest(manifest_path):
    """Return verification errors for a manifest and its release artifacts."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as error:
        return [str(error)]

    release_dir = manifest_path.parent
    errors = []
    for index, artifact in enumerate(manifest["files"]):
        prefix = f"files[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        relative_path = artifact.get("path")
        expected_size = artifact.get("size_bytes")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"{prefix}.path must be a non-empty relative path")
            continue
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"{prefix}.path escapes the release directory: {relative_path}")
            continue
        artifact_path = (release_dir / candidate).resolve()
        try:
            artifact_path.relative_to(release_dir)
        except ValueError:
            errors.append(f"{prefix}.path escapes the release directory: {relative_path}")
            continue
        if not artifact_path.is_file():
            errors.append(f"{relative_path}: file is missing")
            continue
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"{relative_path}: invalid size_bytes")
        elif artifact_path.stat().st_size != expected_size:
            errors.append(
                f"{relative_path}: size mismatch "
                f"(expected {expected_size}, got {artifact_path.stat().st_size})"
            )
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            errors.append(f"{relative_path}: invalid sha256")
        else:
            actual_sha256 = sha256_file(artifact_path)
            if actual_sha256 != expected_sha256:
                errors.append(
                    f"{relative_path}: SHA-256 mismatch "
                    f"(expected {expected_sha256}, got {actual_sha256})"
                )
    return errors


def main():
    """Verify a release and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to blitzed_release_manifest.json")
    args = parser.parse_args()

    errors = verify_manifest(args.manifest)
    if errors:
        print("Release verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Release verified: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
