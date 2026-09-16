#!/usr/bin/env python3
"""Collect labeled predictive-maintenance samples from ESP32 monitor output.

The predictive-maintenance firmware emits structured lines like:

    [00012] Temp: 38.2 C | Accel RMS: [0.42, 0.51, 0.39] g | Status: healthy | Latency: 12 us

This utility extracts the sensor readings and applies an operator-supplied
*ground-truth* label for the capture session. The firmware's predicted status
is intentionally ignored because it is not a valid training label.

Examples:
    idf.py -p /dev/cu.usbserial-0001 monitor | \
        python3 tools/collect_predictive_maintenance_data.py \
        --label healthy --output data/machine-readings.csv

    python3 tools/collect_predictive_maintenance_data.py \
        --input capture.log --label warning --output data/machine-readings.csv --append
"""

import argparse
import csv
import re
import sys
from pathlib import Path


CLASS_NAMES = ["healthy", "warning", "critical", "shutdown_required"]
FIELDNAMES = ["temperature", "rms_accel_x", "rms_accel_y", "rms_accel_z", "label"]

SAMPLE_PATTERN = re.compile(
    r"Temp:\s*(?P<temperature>-?\d+(?:\.\d+)?)\s*C\s*\|\s*"
    r"Accel RMS:\s*\[\s*"
    r"(?P<x>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<z>-?\d+(?:\.\d+)?)\s*\]\s*g"
)


def parse_args():
    """Parse collector options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Saved ESP32 monitor log; omit to read monitor output from stdin",
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=CLASS_NAMES,
        help="Ground-truth label for this capture session",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV dataset path to create or append",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing dataset instead of replacing it",
    )
    return parser.parse_args()


def iter_input_lines(input_path):
    """Yield monitor lines from a file or stdin."""
    if input_path is None:
        yield from sys.stdin
        return
    try:
        with input_path.open(encoding="utf-8", errors="replace") as input_file:
            yield from input_file
    except OSError as error:
        raise ValueError(f"Could not read input log {input_path}: {error}") from error


def collect_rows(lines, label):
    """Extract sensor rows and apply the supplied ground-truth label."""
    rows = []
    for line in lines:
        match = SAMPLE_PATTERN.search(line)
        if match is None:
            continue
        rows.append(
            {
                "temperature": match.group("temperature"),
                "rms_accel_x": match.group("x"),
                "rms_accel_y": match.group("y"),
                "rms_accel_z": match.group("z"),
                "label": label,
            }
        )
    return rows


def write_rows(output_path, rows, append):
    """Write rows with the training script's CSV header."""
    if not rows:
        raise ValueError("No predictive-maintenance sensor lines were found in the input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    """Extract and write one labeled capture session."""
    args = parse_args()
    try:
        rows = collect_rows(iter_input_lines(args.input), args.label)
        write_rows(args.output, rows, args.append)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    source = str(args.input) if args.input else "stdin"
    print(f"Collected {len(rows)} {args.label} samples from {source}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
