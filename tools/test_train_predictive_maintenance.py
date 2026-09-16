# Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np

from build_predictive_maintenance_release import (
    build_shutdown_recall_gate_report,
    shutdown_recall_error,
)
from train_predictive_maintenance import CLASS_NAMES, generate_training_data


FIRMWARE_MAIN = (
    Path(__file__).resolve().parents[1]
    / "esp32_demo"
    / "predictive_maintenance"
    / "main"
    / "main.c"
).read_text()


def test_synthetic_data_is_reproducible():
    first_X, first_y = generate_training_data(n_samples=400, seed=7)
    second_X, second_y = generate_training_data(n_samples=400, seed=7)

    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)


def test_synthetic_labels_cover_both_shutdown_failure_modes():
    X, y = generate_training_data(n_samples=4000, seed=42)
    shutdown = X[y == CLASS_NAMES.index("shutdown_required")]

    assert np.all((shutdown[:, 0] >= 85.0) | (shutdown[:, 1:].max(axis=1) >= 7.5))
    assert np.any(shutdown[:, 0] >= 85.0)
    assert np.any(shutdown[:, 1:].max(axis=1) >= 7.5)


def test_synthetic_classes_are_not_labeled_as_healthy_at_warning_levels():
    X, y = generate_training_data(n_samples=4000, seed=42)
    healthy = X[y == CLASS_NAMES.index("healthy")]

    assert np.all(healthy[:, 0] < 42.0)
    assert np.all(healthy[:, 1:].max(axis=1) < 2.2)


def test_shutdown_recall_gate_accepts_safe_int8_model():
    report = {"int8_validation": {"per_class": {"shutdown_required": 0.95}}}

    assert shutdown_recall_error(report, 0.90) is None


def test_shutdown_recall_gate_rejects_unsafe_int8_model():
    report = {"int8_validation": {"per_class": {"shutdown_required": 0.84}}}

    error = shutdown_recall_error(report, 0.90)

    assert error is not None
    assert "84.0%" in error
    assert "90.0%" in error


def test_shutdown_recall_gate_rejects_missing_metric():
    assert shutdown_recall_error({}, 0.90) is not None


def test_shutdown_recall_gate_report_is_auditable():
    report = {"int8_validation": {"per_class": {"shutdown_required": 0.93}}}

    gate_report = build_shutdown_recall_gate_report(report, 0.90)

    assert gate_report == {
        "gate": "shutdown_required_int8_recall",
        "minimum_recall": 0.90,
        "observed_recall": 0.93,
        "status": "passed",
        "error": None,
    }


def test_firmware_does_not_classify_failed_sensor_reads():
    assert "Status: sensor_fault" in FIRMWARE_MAIN
    assert "Fault: %s" in FIRMWARE_MAIN
    assert '"temperature_and_accelerometer"' in FIRMWARE_MAIN
    assert "Status: sensor_recovered" in FIRMWARE_MAIN
    assert "sensor_fault_active = false" in FIRMWARE_MAIN
    assert "temperature = 0.0f;  // safe fallback" not in FIRMWARE_MAIN
    assert "model will still classify" not in FIRMWARE_MAIN
    assert "continue;" in FIRMWARE_MAIN
