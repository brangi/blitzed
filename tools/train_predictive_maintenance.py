#!/usr/bin/env python3
"""
Standalone training script for ESP32 predictive maintenance classifier.

Trains a 2-layer dense network (4->32->4) using only numpy.
Fuses temperature + 3-axis vibration (RMS accel) into a single inference.
Implements INT8 post-training quantization and exports to C header.

Model architecture:
    Dense(4, 32) + ReLU + Dense(32, 4)
    Total parameters: (4*32 + 32) + (32*4 + 4) = 160 + 132 = 292

Classes:
    0: healthy           — normal temp (25-40 C) + low vibration (RMS < 2.0 g per axis)
    1: warning           — elevated temp (40-58 C) OR mildly elevated vibration (one axis 2.5-5.0 g)
    2: critical          — high temp (58-80 C) + elevated multi-axis vibration (3.5-7.0 g)
    3: shutdown_required — extreme temp (85-120 C) OR extreme vibration (any axis 8.0-12.0 g)

Normalisation (must match main.c):
    temperature / 120.0
    acceleration / 12.0

Usage:
    python tools/train_predictive_maintenance.py
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASS_NAMES = ["healthy", "warning", "critical", "shutdown_required"]
FEATURE_COLUMNS = ["temperature", "rms_accel_x", "rms_accel_y", "rms_accel_z"]
LABEL_COLUMN = "label"


def sha256_file(path):
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(project_root):
    """Return the current Git revision, or ``unknown`` outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_release_manifest(output_dir, artifact_paths, provenance):
    """Build a checksummed manifest for an exported model release."""
    output_dir = Path(output_dir).resolve()
    files = []
    for artifact_path in artifact_paths:
        artifact_path = Path(artifact_path).resolve()
        files.append(
            {
                "path": str(artifact_path.relative_to(output_dir)),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            }
        )
    return {
        "manifest_version": 1,
        "product": "Blitzed Predictive Maintenance Starter Kit",
        "provenance": provenance,
        "files": files,
    }


def build_provenance(args, project_root, dataset_path=None):
    """Build audit metadata for a quality report or exported model."""
    script_path = Path(__file__).resolve()
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(project_root),
        "training_script": str(script_path),
        "training_script_sha256": sha256_file(script_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "parameters": {
            "seed": args.seed,
            "samples": args.samples,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "validation_split": args.validation_split,
        },
    }
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
        provenance["dataset_path"] = str(dataset_path)
        provenance["dataset_sha256"] = sha256_file(dataset_path)
    else:
        provenance["dataset_source"] = "synthetic"
    return provenance


# -----------------------------------------------------------------------
# Neural Network (NumPy only)
# -----------------------------------------------------------------------

def xavier_init(input_size, output_size):
    """Xavier/Glorot uniform weight initialisation."""
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Sub-gradient of ReLU (0 at exactly 0 follows convention)."""
    return (x > 0).astype(float)


def softmax(x):
    """Numerically-stable row-wise softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Mean cross-entropy loss over a mini-batch."""
    m = y_true.shape[0]
    # Clip to avoid log(0)
    log_likelihood = -np.log(np.clip(y_pred[range(m), y_true], 1e-10, 1.0))
    return np.sum(log_likelihood) / m


class DenseLayer:
    """Dense (fully-connected) layer with forward and backward pass."""

    def __init__(self, input_size, output_size):
        self.weights = xavier_init(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward(self, grad_output, learning_rate):
        m = self.input.shape[0]
        grad_weights = np.dot(self.input.T, grad_output) / m
        grad_bias = np.sum(grad_output, axis=0, keepdims=True) / m
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input


# -----------------------------------------------------------------------
# Synthetic Data Generation
# -----------------------------------------------------------------------

def generate_training_data(n_samples=3000, seed=42):
    """
    Generate synthetic predictive-maintenance sensor data.

    Features (raw, before normalisation):
        [0] temperature  °C
        [1] rms_accel_x  g
        [2] rms_accel_y  g
        [3] rms_accel_z  g

    Class distribution:
        40% healthy, 25% warning, 20% critical, 15% shutdown_required

    Synthetic labels follow the same OR/AND safety rules described above; they
    are intended to exercise both independent shutdown failure modes.
    """
    np.random.seed(seed)

    n_healthy   = int(n_samples * 0.40)
    n_warning   = int(n_samples * 0.25)
    n_critical  = int(n_samples * 0.20)
    n_shutdown  = n_samples - n_healthy - n_warning - n_critical

    samples = []
    labels  = []

    # ---- Class 0: healthy ----
    # Keep healthy readings below both warning thresholds so labels are not
    # contradictory at the boundary.
    temp_h = np.random.uniform(25.0, 40.0, n_healthy)
    vib_h = np.random.uniform(0.1, 1.8, (n_healthy, 3))
    noise_h = np.random.normal(0, 0.1, (n_healthy, 3))
    vib_h = np.clip(vib_h + noise_h, 0.05, 12.0)
    samples.append(np.column_stack([temp_h, vib_h]))
    labels.append(np.zeros(n_healthy, dtype=int))

    # ---- Class 1: warning ----
    # Warning is an OR condition: elevated temperature with low vibration, or
    # one mildly elevated vibration axis at an otherwise normal temperature.
    warning_temperature = np.random.rand(n_warning) < 0.5
    temp_w = np.where(
        warning_temperature,
        np.random.uniform(40.0, 58.0, n_warning),
        np.random.uniform(30.0, 40.0, n_warning),
    )
    vib_w = np.random.uniform(0.5, 2.2, (n_warning, 3))
    warn_axis = np.random.randint(0, 3, n_warning)
    warn_vib_level = np.random.uniform(2.5, 5.0, n_warning)
    for i in range(n_warning):
        if not warning_temperature[i]:
            vib_w[i, warn_axis[i]] = warn_vib_level[i]
    noise_w = np.random.normal(0, 0.15, (n_warning, 3))
    vib_w = np.clip(vib_w + noise_w, 0.05, 12.0)
    samples.append(np.column_stack([temp_w, vib_w]))
    labels.append(np.ones(n_warning, dtype=int))

    # ---- Class 2: critical ----
    # High temperature and elevated multi-axis vibration, but below the
    # shutdown thresholds.
    temp_c = np.random.uniform(58.0, 80.0, n_critical)
    vib_c = np.random.uniform(3.5, 7.0, (n_critical, 3))
    noise_c = np.random.normal(0, 0.2, (n_critical, 3))
    vib_c = np.clip(vib_c + noise_c, 0.05, 12.0)
    samples.append(np.column_stack([temp_c, vib_c]))
    labels.append(np.full(n_critical, 2, dtype=int))

    # ---- Class 3: shutdown_required ----
    # Shutdown is also an OR condition: extreme temperature OR a severe
    # vibration spike. Generate both failure modes so the model sees each.
    temperature_shutdown = np.random.rand(n_shutdown) < 0.5
    temp_s = np.where(
        temperature_shutdown,
        np.random.uniform(85.0, 120.0, n_shutdown),
        np.random.uniform(30.0, 80.0, n_shutdown),
    )
    vib_s = np.random.uniform(0.5, 3.0, (n_shutdown, 3))
    shutdown_axis = np.random.randint(0, 3, n_shutdown)
    shutdown_vib_level = np.random.uniform(8.0, 12.0, n_shutdown)
    for i in range(n_shutdown):
        if not temperature_shutdown[i]:
            vib_s[i, shutdown_axis[i]] = shutdown_vib_level[i]
    noise_s = np.random.normal(0, 0.3, (n_shutdown, 3))
    vib_s = np.clip(vib_s + noise_s, 0.05, 12.0)
    samples.append(np.column_stack([temp_s, vib_s]))
    labels.append(np.full(n_shutdown, 3, dtype=int))

    # ---- Combine and shuffle ----
    X = np.vstack(samples)
    y = np.concatenate(labels)

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def load_dataset_csv(path):
    """Load labeled sensor readings from the starter kit CSV contract.

    Required columns are ``temperature``, ``rms_accel_x``, ``rms_accel_y``,
    ``rms_accel_z``, and ``label``. Labels may be class names or integer IDs.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")

    samples = []
    labels = []
    label_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV must include a header row")
        missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                samples.append([float(row[column]) for column in FEATURE_COLUMNS])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV row {row_number}") from error

            raw_label_value = row.get(LABEL_COLUMN)
            if raw_label_value is None or not raw_label_value.strip():
                raise ValueError(f"Missing label on CSV row {row_number}")
            raw_label = raw_label_value.strip().lower()
            if raw_label in label_ids:
                labels.append(label_ids[raw_label])
            else:
                try:
                    label_id = int(raw_label)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid label on CSV row {row_number}: {row[LABEL_COLUMN]!r}"
                    ) from error
                if label_id < 0 or label_id >= len(CLASS_NAMES):
                    raise ValueError(
                        f"Label on CSV row {row_number} must be 0-{len(CLASS_NAMES) - 1}"
                    )
                labels.append(label_id)

    if len(samples) < 8:
        raise ValueError("Dataset must contain at least 8 rows")
    X = np.asarray(samples, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains non-finite sensor values")
    return X, y


def dataset_quality_report(X, y):
    """Return dataset quality metrics and actionable warnings."""
    class_counts = np.bincount(y, minlength=len(CLASS_NAMES))
    feature_ranges = {
        "temperature": (0.0, 120.0),
        "rms_accel_x": (0.0, 12.0),
        "rms_accel_y": (0.0, 12.0),
        "rms_accel_z": (0.0, 12.0),
    }
    feature_stats = {}
    out_of_range = {}
    warnings = []

    for index, feature_name in enumerate(FEATURE_COLUMNS):
        values = X[:, index]
        minimum, maximum = feature_ranges[feature_name]
        out_of_range[feature_name] = int(np.count_nonzero((values < minimum) | (values > maximum)))
        feature_stats[feature_name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        if out_of_range[feature_name]:
            warnings.append(
                f"{feature_name} has {out_of_range[feature_name]} readings outside "
                f"the expected range [{minimum}, {maximum}]"
            )

    missing_classes = [
        CLASS_NAMES[class_id]
        for class_id, count in enumerate(class_counts)
        if count == 0
    ]
    if missing_classes:
        warnings.append("Missing classes: " + ", ".join(missing_classes))
    if any(count < 2 for count in class_counts if count > 0):
        warnings.append("Some classes have fewer than two samples and cannot be split for validation")

    nonzero_counts = class_counts[class_counts > 0]
    if len(nonzero_counts) > 1 and nonzero_counts.max() / nonzero_counts.min() > 5:
        warnings.append("Class imbalance exceeds 5:1; collect more samples for minority classes")

    duplicate_count = int(len(X) - len(np.unique(X, axis=0)))
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate sensor rows")

    return {
        "rows": int(len(X)),
        "class_counts": {
            CLASS_NAMES[class_id]: int(count)
            for class_id, count in enumerate(class_counts)
        },
        "feature_stats": feature_stats,
        "out_of_expected_range": out_of_range,
        "duplicate_rows": duplicate_count,
        "missing_classes": missing_classes,
        "warnings": warnings,
    }


def split_dataset(X, y, validation_split, seed):
    """Create a deterministic stratified train/validation split."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation-split must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < 2:
            train_indices.extend(class_indices.tolist())
            continue
        class_indices = class_indices[rng.permutation(len(class_indices))]
        validation_count = max(1, int(round(len(class_indices) * validation_split)))
        validation_count = min(validation_count, len(class_indices) - 1)
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    if not validation_indices or not train_indices:
        raise ValueError("Dataset must contain enough samples for train and validation splits")
    missing_training_classes = [
        CLASS_NAMES[class_id] for class_id in range(len(CLASS_NAMES))
        if not any(y[index] == class_id for index in train_indices)
    ]
    if missing_training_classes:
        raise ValueError(
            "Training data is missing classes: " + ", ".join(missing_training_classes)
        )
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    train_indices = train_indices[rng.permutation(len(train_indices))]
    validation_indices = validation_indices[rng.permutation(len(validation_indices))]
    return X[train_indices], y[train_indices], X[validation_indices], y[validation_indices]


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_model(X_norm, y, epochs=500, learning_rate=0.1):
    """
    Train the Dense(4,32)+ReLU+Dense(32,4) network.

    X_norm must already be normalised:  temp/120.0, accel/12.0
    """
    layer1 = DenseLayer(4, 32)
    layer2 = DenseLayer(32, 4)

    print("Training model...")
    print(f"Layer 1: {layer1.weights.shape} weights + {layer1.bias.shape[1]} biases")
    print(f"Layer 2: {layer2.weights.shape} weights + {layer2.bias.shape[1]} biases")

    total_params = (4 * 32 + 32) + (32 * 4 + 4)
    print(f"Total parameters: {total_params}")

    accuracy = 0.0  # initialised here so it's visible after the loop
    for epoch in range(epochs):
        # ---- Forward pass ----
        z1     = layer1.forward(X_norm)
        a1     = relu(z1)
        z2     = layer2.forward(a1)
        y_pred = softmax(z2)

        loss       = cross_entropy_loss(y_pred, y)
        predictions = np.argmax(y_pred, axis=1)
        accuracy    = np.mean(predictions == y)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f} — Accuracy: {accuracy:.4f}")

        # ---- Backward pass ----
        m       = X_norm.shape[0]
        grad_z2 = y_pred.copy()
        grad_z2[range(m), y] -= 1

        grad_a1 = layer2.backward(grad_z2, learning_rate)
        grad_z1 = grad_a1 * relu_derivative(z1)
        layer1.backward(grad_z1, learning_rate)

    print(f"\nFinal training accuracy: {accuracy:.4f}")
    return layer1, layer2


# -----------------------------------------------------------------------
# INT8 Quantization
# -----------------------------------------------------------------------

def quantize_weights(weights):
    """
    Quantize a float32 weight array to INT8 using symmetric (zero_point=0) quantization.

    Symmetric quantization is required so that the edge INT8 matmul kernel
    (which computes sum(x_q * w_q) without a zero-point correction term) gives
    correct results.  The dequantization is simply: float_val = w_q * scale.

    Returns:
        quantized  — INT8 numpy array (same shape)
        scale      — float32 scale factor  (w_abs_max / 127)
        zero_point — always 0
    """
    w_abs_max = float(np.max(np.abs(weights)))
    if w_abs_max == 0.0:
        return np.zeros_like(weights, dtype=np.int8), 1.0, 0

    scale     = w_abs_max / 127.0
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)

    return quantized, scale, 0


def quantize_bias(bias, input_scale, weight_scale):
    """
    Quantize bias to INT32.

    bias_scale = input_scale * weight_scale
    q_bias     = round(bias / bias_scale)
    """
    bias_scale = input_scale * weight_scale
    quantized  = np.round(bias / bias_scale).astype(np.int32)
    return quantized, bias_scale


def quantize_model(layer1, layer2, X_calibration, input_scale=1.0 / 255.0):
    """
    Apply INT8 post-training quantization with activation calibration.

    Uses calibration data to determine the actual activation range after
    layer 1, which produces a much better output_scale than the naive
    input_scale * weight_scale propagation. This is critical for multi-input
    models where the accumulator range is wide.
    """
    print("\nQuantizing model to INT8 (with activation calibration)...")

    # --- Run calibration pass to find real activation ranges ---
    z1_cal = np.dot(X_calibration, layer1.weights) + layer1.bias
    a1_cal = relu(z1_cal)
    a1_abs_max = float(np.max(np.abs(a1_cal)))
    if a1_abs_max == 0.0:
        a1_abs_max = 1.0
    # Calibrated output1_scale: maps the real activation range to INT8
    output1_scale = a1_abs_max / 127.0
    print(f"Calibrated layer1 activation range: [0, {a1_abs_max:.4f}], output_scale={output1_scale:.8f}")

    # Layer 1 weights
    q_w1, w1_scale, w1_zp = quantize_weights(layer1.weights)
    # Bias scale for accumulator: input_scale * w1_scale
    # (bias is added to the raw accumulator, not the requantized output)
    acc1_scale = input_scale * w1_scale
    q_b1 = np.round(layer1.bias.flatten() / acc1_scale).astype(np.int32)

    # Layer 2 weights
    q_w2, w2_scale, w2_zp = quantize_weights(layer2.weights)
    # Layer 2 accumulator scale: output1_scale * w2_scale
    acc2_scale = output1_scale * w2_scale
    q_b2 = np.round(layer2.bias.flatten() / acc2_scale).astype(np.int32)
    output2_scale = output1_scale * w2_scale

    quantized = {
        'layer1': {
            'weights':           q_w1,
            'bias':              q_b1,
            'weight_scale':      w1_scale,
            'weight_zero_point': w1_zp,
            'output_scale':      output1_scale,
            'output_zero_point': 0,
            'acc_scale':         acc1_scale,
        },
        'layer2': {
            'weights':           q_w2,
            'bias':              q_b2,
            'weight_scale':      w2_scale,
            'weight_zero_point': w2_zp,
            'output_scale':      output2_scale,
            'output_zero_point': 0,
            'acc_scale':         acc2_scale,
        },
        'input_scale':      input_scale,
        'input_zero_point': 0,
    }

    print(f"Layer 1 weights: scale={w1_scale:.6f}, zero_point={w1_zp}")
    print(f"Layer 2 weights: scale={w2_scale:.6f}, zero_point={w2_zp}")

    return quantized


def test_quantized_model(quantized, X_norm, y, verbose=True):
    """
    Evaluate the quantized model using the same pipeline as the ESP32 kernel.

    Mirrors the exact INT8 pipeline in blitzed_inference.c:
      1. Quantize inputs using input_scale
      2. INT32 matmul + INT32 bias (no zero-point correction; weights are symmetric)
      3. Dequantize to float, apply ReLU
      4. Requantize to inter-layer INT8 using layer1_output_scale
      5. INT32 matmul for layer 2 + INT32 bias
      6. Dequantize, softmax, argmax

    Weight zero_point is always 0 (symmetric quantization) so no correction
    term is needed in the matmul — this matches the C kernel exactly.
    """
    input_scale = quantized['input_scale']
    input_zp    = quantized['input_zero_point']

    l1 = quantized['layer1']
    l2 = quantized['layer2']

    # Step 1 — quantize inputs (mirrors blitzed_quantize_input)
    X_q = np.clip(
        np.round(X_norm / input_scale) + input_zp,
        -128, 127
    ).astype(np.int32)

    # Step 2 — Layer 1 matmul (symmetric weights: w_zp=0 so no correction)
    # acc = bias + sum_i( x_q[i] * w_q[i,j] )
    # Accumulator is in scale = input_scale * w1_scale (= acc_scale)
    w1   = l1['weights'].astype(np.int32)   # (4, 32)
    b1   = l1['bias'].astype(np.int32)      # (32,)
    acc1 = np.dot(X_q, w1) + b1             # (N, 32)

    # Step 3 — dequantize accumulator to float using acc_scale, apply ReLU
    acc_scale = l1['acc_scale']  # = input_scale * w1_scale
    z1 = acc1.astype(np.float32) * acc_scale
    a1 = relu(z1)

    # Step 4 — requantize for layer 2 using calibrated output_scale
    out1_scale = l1['output_scale']  # calibrated from activation range
    a1_q = np.clip(
        np.round(a1 / out1_scale) + l1['output_zero_point'],
        -128, 127
    ).astype(np.int32)

    # Step 5 — Layer 2 matmul (symmetric weights)
    w2   = l2['weights'].astype(np.int32)   # (32, 4)
    b2   = l2['bias'].astype(np.int32)      # (4,)
    acc2 = np.dot(a1_q, w2) + b2            # (N, 4)

    # Step 6 — dequantize and classify
    # Accumulator scale = output1_scale * w2_scale = l2['acc_scale']
    z2 = acc2.astype(np.float32) * l2['acc_scale']

    y_pred      = softmax(z2)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = float(np.mean(predictions == y))

    if verbose:
        print(f"Quantized model accuracy: {accuracy:.4f}")
        for cls in range(4):
            mask = y == cls
            cls_acc = np.mean(predictions[mask] == y[mask]) if mask.any() else 0.0
            print(f"  Class {cls} accuracy: {cls_acc:.4f}  ({mask.sum()} samples)")

    return accuracy, predictions


def predict_float_model(layer1, layer2, X_norm):
    """Return float-model class predictions for normalized inputs."""
    hidden = relu(layer1.forward(X_norm))
    logits = layer2.forward(hidden)
    return np.argmax(softmax(logits), axis=1)


def accuracy_by_class(predictions, labels, class_names):
    """Return overall and per-class accuracy metrics."""
    metrics = {"overall": float(np.mean(predictions == labels))}
    metrics["per_class"] = {}
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        metrics["per_class"][class_name] = (
            float(np.mean(predictions[mask] == labels[mask])) if mask.any() else 0.0
        )
    return metrics


# -----------------------------------------------------------------------
# C Header / Binary Export
# -----------------------------------------------------------------------

def format_array_as_c(arr, values_per_line=12):
    """Format a flat numpy array as a C array body (no braces)."""
    flat  = arr.flatten()
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk  = flat[i:i + values_per_line]
        values = ', '.join(str(int(v)) for v in chunk)
        lines.append(f"    {values}")
    return ',\n'.join(lines)


def export_to_c_header(quantized, output_path):
    """Export quantized model to a self-contained C header."""

    layer1 = quantized['layer1']
    layer2 = quantized['layer2']

    size_bytes = (
        layer1['weights'].nbytes + layer1['bias'].nbytes +
        layer2['weights'].nbytes + layer2['bias'].nbytes
    )

    total_params = layer1['weights'].size + layer1['bias'].size + \
                   layer2['weights'].size + layer2['bias'].size

    header = f"""// Auto-generated by tools/train_predictive_maintenance.py
// Model: 2-layer dense network for ESP32 predictive maintenance
// Architecture: Dense(4, 32) + ReLU + Dense(32, 4)
// Total parameters: {total_params}
// Quantized model size: {size_bytes} bytes
//
// Input features (normalised before quantization):
//   [0] temperature   / 120.0  (raw °C)
//   [1] rms_accel_x   / 12.0   (raw g)
//   [2] rms_accel_y   / 12.0   (raw g)
//   [3] rms_accel_z   / 12.0   (raw g)

#ifndef BLITZED_MODEL_WEIGHTS_H
#define BLITZED_MODEL_WEIGHTS_H

#include <stdint.h>

// ---- Layer 1: Dense(4, 32) + ReLU ----
#define LAYER1_INPUT_SIZE  4
#define LAYER1_OUTPUT_SIZE 32

// Weight layout: layer1_weights[input_idx * LAYER1_OUTPUT_SIZE + output_idx]
static const int8_t layer1_weights[{layer1['weights'].size}] = {{
{format_array_as_c(layer1['weights'])}
}};

static const int32_t layer1_bias[{layer1['bias'].size}] = {{
{format_array_as_c(layer1['bias'])}
}};

static const float layer1_weight_scale = {layer1['weight_scale']:.10f}f;
static const int32_t layer1_weight_zero_point = {layer1['weight_zero_point']};
static const float layer1_output_scale = {layer1['output_scale']:.10f}f;
static const int32_t layer1_output_zero_point = {layer1['output_zero_point']};

// ---- Layer 2: Dense(32, 4) ----
#define LAYER2_INPUT_SIZE  32
#define LAYER2_OUTPUT_SIZE 4

// Weight layout: layer2_weights[input_idx * LAYER2_OUTPUT_SIZE + output_idx]
static const int8_t layer2_weights[{layer2['weights'].size}] = {{
{format_array_as_c(layer2['weights'])}
}};

static const int32_t layer2_bias[{layer2['bias'].size}] = {{
{format_array_as_c(layer2['bias'])}
}};

static const float layer2_weight_scale = {layer2['weight_scale']:.10f}f;
static const int32_t layer2_weight_zero_point = {layer2['weight_zero_point']};
static const float layer2_output_scale = {layer2['output_scale']:.10f}f;
static const int32_t layer2_output_zero_point = {layer2['output_zero_point']};

// ---- Input quantization parameters ----
// input_scale = 1/255 — normalised inputs in [-1, 1] map to INT8
#define INPUT_SCALE {quantized['input_scale']:.10f}f
#define INPUT_ZERO_POINT {quantized['input_zero_point']}

// ---- Class labels ----
#define NUM_CLASSES 4
static const char* class_labels[NUM_CLASSES] = {{
    "healthy",
    "warning",
    "critical",
    "shutdown_required"
}};

#endif // BLITZED_MODEL_WEIGHTS_H
"""

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"\nExported C header to: {output_path}")
    print(f"Model size: {size_bytes} bytes ({total_params} total parameters)")

    return size_bytes


def export_binary_weights(quantized, output_path):
    """Export raw INT8/INT32 weight arrays to a binary file for the Rust pipeline."""

    with open(output_path, 'wb') as f:
        quantized['layer1']['weights'].tofile(f)
        quantized['layer1']['bias'].tofile(f)
        quantized['layer2']['weights'].tofile(f)
        quantized['layer2']['bias'].tofile(f)

    print(f"Exported binary weights to: {output_path}")


def print_weight_statistics(layer1, layer2, quantized):
    """Print float and quantized weight statistics for debugging."""

    print("\n=== Weight Statistics ===")

    print(f"\nLayer 1 (float32):")
    print(f"  Weights: min={layer1.weights.min():.4f}, max={layer1.weights.max():.4f}, "
          f"mean={layer1.weights.mean():.4f}, std={layer1.weights.std():.4f}")
    print(f"  Bias:    min={layer1.bias.min():.4f}, max={layer1.bias.max():.4f}")

    print(f"\nLayer 2 (float32):")
    print(f"  Weights: min={layer2.weights.min():.4f}, max={layer2.weights.max():.4f}, "
          f"mean={layer2.weights.mean():.4f}, std={layer2.weights.std():.4f}")
    print(f"  Bias:    min={layer2.bias.min():.4f}, max={layer2.bias.max():.4f}")

    print(f"\nLayer 1 (int8):")
    w1q = quantized['layer1']['weights']
    b1q = quantized['layer1']['bias']
    print(f"  Weights: min={w1q.min()}, max={w1q.max()}, mean={w1q.mean():.2f}, "
          f"nonzero={np.count_nonzero(w1q)}/{w1q.size}")
    print(f"  Bias:    min={b1q.min()}, max={b1q.max()}, mean={b1q.mean():.2f}")

    print(f"\nLayer 2 (int8):")
    w2q = quantized['layer2']['weights']
    b2q = quantized['layer2']['bias']
    print(f"  Weights: min={w2q.min()}, max={w2q.max()}, mean={w2q.mean():.2f}, "
          f"nonzero={np.count_nonzero(w2q)}/{w2q.size}")
    print(f"  Bias:    min={b2q.min()}, max={b2q.max()}, mean={b2q.mean():.2f}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args():
    """Parse reproducible training and export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000, help="Synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient descent learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Labeled CSV dataset; synthetic data is used when omitted",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of each class reserved for validation",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Write dataset quality JSON and exit without training (requires --dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for model artifacts (defaults to the demo main directory)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("ESP32 Predictive Maintenance Classifier Training")
    print("=" * 60)

    if args.samples < 8 or args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("samples must be >= 8, epochs must be >= 1, and learning-rate must be positive")
    if args.quality_only and not args.dataset:
        raise ValueError("--quality-only requires --dataset")

    # ---- Load or generate data ----
    if args.dataset:
        print(f"\nLoading labeled dataset: {args.dataset}")
        X_raw, y = load_dataset_csv(args.dataset)
        dataset_source = str(args.dataset)
    else:
        print("\nGenerating synthetic training data...")
        X_raw, y = generate_training_data(n_samples=args.samples, seed=args.seed)
        dataset_source = "synthetic"
    print(f"Loaded {len(X_raw)} samples")
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    for cls, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  Class {cls} ({name}): {count} samples ({100*count/len(y):.1f}%)")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    provenance = build_provenance(args, project_root, args.dataset)
    quality = dataset_quality_report(X_raw, y)
    print(f"Duplicate sensor rows: {quality['duplicate_rows']}")
    if quality["warnings"]:
        print("Dataset quality warnings:")
        for warning in quality["warnings"]:
            print(f"  - {warning}")

    if args.quality_only:
        output_dir = args.output_dir or project_root / "esp32_demo" / "predictive_maintenance" / "main"
        output_dir.mkdir(parents=True, exist_ok=True)
        quality_path = output_dir / "blitzed_dataset_quality.json"
        quality_path.write_text(
            json.dumps(
                {"dataset_source": dataset_source, "provenance": provenance, **quality},
                indent=2,
            )
            + "\n"
        )
        print(f"Dataset quality report: {quality_path}")
        return

    # ---- Deterministic stratified train/validation split ----
    # Keep the validation set separate from both training and activation calibration.
    X_train_raw, y_train, X_validation_raw, y_validation = split_dataset(
        X_raw, y, args.validation_split, args.seed + 1
    )

    # ---- Normalise inputs ----
    # Normalisation mirrors the on-device code in main.c:
    #   temp  / 120.0  maps the 0-120 °C range to [0, 1]
    #   accel / 12.0   maps the 0-12 g  range to [0, 1]
    # This keeps all 4 features on the same scale, preventing any single
    # feature from dominating the gradient during training.
    TEMP_NORM  = 120.0
    ACCEL_NORM = 12.0

    def normalize_inputs(values):
        normalized = values.copy()
        normalized[:, 0] /= TEMP_NORM
        normalized[:, 1] /= ACCEL_NORM
        normalized[:, 2] /= ACCEL_NORM
        normalized[:, 3] /= ACCEL_NORM
        return normalized

    X_train_norm = normalize_inputs(X_train_raw)
    X_validation_norm = normalize_inputs(X_validation_raw)

    print(f"\nNormalised input ranges:")
    feature_names = ["temp/120", "rms_x/12", "rms_y/12", "rms_z/12"]
    for i, name in enumerate(feature_names):
        print(f"  [{i}] {name}: [{X_train_norm[:, i].min():.3f}, {X_train_norm[:, i].max():.3f}]")
    print(f"Training samples: {len(X_train_norm)} | Validation samples: {len(X_validation_norm)}")

    # ---- Train ----
    layer1, layer2 = train_model(
        X_train_norm,
        y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # ---- Quantize ----
    # input_scale = max_abs / 127.0 so that the full normalised range [-1, 1]
    # maps to INT8 [-127, 127]. Using 1/255 would clip at ±0.498 — destroying
    # the ability to distinguish classes with higher feature values.
    input_scale = 1.0 / 127.0
    quantized = quantize_model(layer1, layer2, X_train_norm, input_scale=input_scale)

    # ---- Evaluate on held-out data ----
    float_predictions = predict_float_model(layer1, layer2, X_validation_norm)
    quantized_accuracy, quantized_predictions = test_quantized_model(
        quantized, X_validation_norm, y_validation
    )
    float_metrics = accuracy_by_class(float_predictions, y_validation, CLASS_NAMES)
    quantized_metrics = accuracy_by_class(quantized_predictions, y_validation, CLASS_NAMES)
    print(f"Float validation accuracy: {float_metrics['overall']:.4f}")
    print(f"INT8 validation accuracy: {quantized_accuracy:.4f}")

    # ---- Print statistics ----
    print_weight_statistics(layer1, layer2, quantized)

    # ---- Export ----
    default_output_dir = project_root / "esp32_demo" / "predictive_maintenance" / "main"
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    header_path = output_dir / "blitzed_model_weights.h"
    binary_path = output_dir / "blitzed_model_weights.bin"
    report_path = output_dir / "blitzed_model_report.json"
    manifest_path = output_dir / "blitzed_release_manifest.json"

    model_size = export_to_c_header(quantized, str(header_path))
    export_binary_weights(quantized, str(binary_path))
    report = {
        "product": "Blitzed Predictive Maintenance Starter Kit",
        "dataset_source": dataset_source,
        "seed": args.seed,
        "samples": len(X_raw),
        "training_samples": len(X_train_norm),
        "validation_samples": len(X_validation_norm),
        "validation_split": args.validation_split,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "architecture": "Dense(4,32)+ReLU->Dense(32,4)",
        "classes": CLASS_NAMES,
        "dataset_quality": quality,
        "provenance": provenance,
        "float_validation": float_metrics,
        "int8_validation": quantized_metrics,
        "int8_accuracy_loss_percentage": (float_metrics["overall"] - quantized_metrics["overall"]) * 100.0,
        "quantized_model_bytes": model_size,
        "artifacts": {
            "weights_header": str(header_path),
            "weights_binary": str(binary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = build_release_manifest(
        output_dir,
        [header_path, binary_path, report_path],
        provenance,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validation report: {report_path}")
    print(f"Release manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Flash the project:  cd esp32_demo/predictive_maintenance && idf.py flash monitor")
    print(f"  2. Connect MPU6050 to GPIO 21 (SDA) / GPIO 22 (SCL)")
    print(f"  3. Observe real-time predictions in the serial monitor")
    print(f"\nModel files:")
    print(f"  {header_path}")
    print(f"  {binary_path}")
    print(f"  {report_path}")
    print(f"  {manifest_path}")


if __name__ == '__main__':
    main()
