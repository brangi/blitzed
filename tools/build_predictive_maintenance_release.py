#!/usr/bin/env python3
"""Build and verify a customer predictive-maintenance model release.

This command requires a labeled customer CSV and performs the release workflow:

1. Run the dataset quality gate.
2. Refuse unresolved quality warnings unless explicitly overridden.
3. Train and export the INT8 model and reports.
4. Verify the generated release manifest and checksums.

Usage:
    python3 tools/build_predictive_maintenance_release.py \
        --dataset data/machine-readings.csv \
        --output-dir releases/machine-v1
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from train_predictive_maintenance import build_release_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_predictive_maintenance.py"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_predictive_maintenance_release.py"
DEFAULT_MIN_SHUTDOWN_RECALL = 0.90


def shutdown_recall_error(report, minimum_recall):
    """Return an acceptance-gate error, or ``None`` when recall passes."""
    try:
        recall = float(
            report["int8_validation"]["per_class"]["shutdown_required"]
        )
    except (KeyError, TypeError, ValueError):
        return "model report does not contain INT8 shutdown_required recall"
    if not math.isfinite(recall):
        return "INT8 shutdown_required recall is not finite"
    if recall < minimum_recall:
        return (
            f"INT8 shutdown_required recall {recall:.1%} is below the "
            f"required minimum {minimum_recall:.1%}"
        )
    return None


def build_shutdown_recall_gate_report(report, minimum_recall):
    """Return the auditable shutdown-recall acceptance result."""
    observed_recall = report.get("int8_validation", {}).get("per_class", {}).get(
        "shutdown_required"
    )
    error = shutdown_recall_error(report, minimum_recall)
    return {
        "gate": "shutdown_required_int8_recall",
        "minimum_recall": minimum_recall,
        "observed_recall": observed_recall,
        "status": "blocked" if error else "passed",
        "error": error,
    }


def parse_args():
    """Parse release build options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Labeled customer CSV dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Release directory to create")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Training learning rate")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of each class reserved for validation",
    )
    parser.add_argument(
        "--allow-quality-warnings",
        action="store_true",
        help="Build anyway when the quality report contains warnings",
    )
    parser.add_argument(
        "--min-shutdown-recall",
        type=float,
        default=DEFAULT_MIN_SHUTDOWN_RECALL,
        help=(
            "Minimum held-out INT8 recall for shutdown_required "
            f"(default: {DEFAULT_MIN_SHUTDOWN_RECALL * 100:.0f}%%)"
        ),
    )
    return parser.parse_args()


def run(command):
    """Run a child command and return its exit status."""
    print("$ " + " ".join(str(part) for part in command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def main():
    """Build, verify, and report a customer release."""
    args = parse_args()
    if not 0.0 <= args.min_shutdown_recall <= 1.0:
        print("Error: --min-shutdown-recall must be between 0 and 1", file=sys.stderr)
        return 2
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    quality_report = output_dir / "blitzed_dataset_quality.json"
    manifest = output_dir / "blitzed_release_manifest.json"

    if not dataset.is_file():
        print(f"Error: dataset does not exist: {dataset}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        str(dataset),
        "--quality-only",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--validation-split",
        str(args.validation_split),
    ]
    if run(quality_command) != 0:
        return 1

    try:
        quality = json.loads(quality_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Error: could not read quality report: {error}", file=sys.stderr)
        return 1

    warnings = quality.get("warnings", [])
    if warnings and not args.allow_quality_warnings:
        print("Release blocked by dataset quality warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        print("Use --allow-quality-warnings only after reviewing the report.", file=sys.stderr)
        return 2

    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--validation-split",
        str(args.validation_split),
    ]
    if run(train_command) != 0:
        return 1

    model_report_path = output_dir / "blitzed_model_report.json"
    gate_report_path = output_dir / "blitzed_release_gate.json"
    try:
        model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Error: could not read model report: {error}", file=sys.stderr)
        return 1

    gate_report = build_shutdown_recall_gate_report(
        model_report, args.min_shutdown_recall
    )
    gate_error = gate_report["error"]
    gate_report_path.write_text(json.dumps(gate_report, indent=2) + "\n", encoding="utf-8")
    if gate_error:
        print(f"Release blocked by safety acceptance gate: {gate_error}", file=sys.stderr)
        print(
            "Collect more representative shutdown_required data or explicitly "
            "lower --min-shutdown-recall after review.",
            file=sys.stderr,
        )
        return 2

    manifest_payload = {
        "manifest_version": 1,
        "product": "Blitzed Predictive Maintenance Starter Kit",
        "provenance": model_report.get("provenance", {}),
        "files": [],
    }
    manifest_payload = build_release_manifest(
        output_dir,
        [
            output_dir / "blitzed_model_weights.h",
            output_dir / "blitzed_model_weights.bin",
            model_report_path,
            gate_report_path,
        ],
        manifest_payload["provenance"],
    )
    Path(manifest).write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Safety acceptance gate passed: "
        f"INT8 shutdown_required recall >= {args.min_shutdown_recall:.1%}"
    )

    if run([sys.executable, str(VERIFY_SCRIPT), str(manifest)]) != 0:
        return 1

    print(f"Release build complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
