#!/usr/bin/env python3
"""
Standalone training script for ESP32 predictive maintenance classifier.

Trains a 2-layer dense network (4->32->4) using only numpy.
Fuses temperature + 3-axis vibration (RMS accel) into a single inference.
Implements INT8 post-training quantization and exports to C header.

Model architecture:
    Dense(4, 32) + ReLU + Dense(32, 4)
    Total parameters: (4*32 + 32) + (32*4 + 4) = 160 + 128 + 4 = 164 weights + 36 bias = 196

Classes:
    0: healthy           — normal temp (25-45 C) + low vibration (RMS < 2.0 g per axis)
    1: warning           — elevated temp (40-60 C) OR mildly elevated vibration (one axis 2.5-5.0 g)
    2: critical          — high temp (55-80 C) AND elevated vibration (multi-axis 3.0-8.0 g)
    3: shutdown_required — extreme temp (75-120 C) OR extreme vibration (any axis > 8.0 g)

Normalisation (must match main.c):
    temperature / 120.0
    acceleration / 12.0

Usage:
    python tools/train_predictive_maintenance.py
"""

import numpy as np
import os


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
    """
    np.random.seed(seed)

    n_healthy   = int(n_samples * 0.40)
    n_warning   = int(n_samples * 0.25)
    n_critical  = int(n_samples * 0.20)
    n_shutdown  = n_samples - n_healthy - n_warning - n_critical

    samples = []
    labels  = []

    # ---- Class 0: healthy ----
    # Temperature: 25-45 °C, vibration RMS < 2.0 g per axis
    temp_h  = np.random.uniform(25.0, 45.0, n_healthy)
    vib_h   = np.random.uniform(0.1, 2.0, (n_healthy, 3))
    noise_h = np.random.normal(0, 0.1, (n_healthy, 3))
    vib_h   = np.clip(vib_h + noise_h, 0.05, 12.0)
    samples.append(np.column_stack([temp_h, vib_h]))
    labels.append(np.zeros(n_healthy, dtype=int))

    # ---- Class 1: warning ----
    # Either slightly elevated temperature OR mildly elevated vibration on one axis.
    # We randomly pick which condition triggers for each sample.
    temp_w = np.random.uniform(40.0, 60.0, n_warning)
    vib_w  = np.random.uniform(0.5, 3.0, (n_warning, 3))

    # Elevate exactly one vibration axis to warning level (2.5-5.0 g) for ~half
    warn_axis = np.random.randint(0, 3, n_warning)
    warn_vib_level = np.random.uniform(2.5, 5.0, n_warning)
    for i in range(n_warning):
        if np.random.rand() > 0.5:
            vib_w[i, warn_axis[i]] = warn_vib_level[i]

    noise_w = np.random.normal(0, 0.15, (n_warning, 3))
    vib_w   = np.clip(vib_w + noise_w, 0.05, 12.0)
    samples.append(np.column_stack([temp_w, vib_w]))
    labels.append(np.ones(n_warning, dtype=int))

    # ---- Class 2: critical ----
    # High temperature (60-80 °C) AND elevated multi-axis vibration (4.0-8.0 g).
    temp_c = np.random.uniform(60.0, 80.0, n_critical)
    vib_c  = np.random.uniform(4.0, 8.0, (n_critical, 3))
    noise_c = np.random.normal(0, 0.2, (n_critical, 3))
    vib_c   = np.clip(vib_c + noise_c, 0.05, 12.0)
    samples.append(np.column_stack([temp_c, vib_c]))
    labels.append(np.full(n_critical, 2, dtype=int))

    # ---- Class 3: shutdown_required ----
    # Extreme temperature (> 85 °C) AND extreme vibration (all axes > 7.0 g).
    # Both conditions present simultaneously — the worst case scenario.
    temp_s = np.random.uniform(85.0, 120.0, n_shutdown)
    vib_s  = np.random.uniform(7.0, 12.0, (n_shutdown, 3))
    noise_s = np.random.normal(0, 0.3, (n_shutdown, 3))
    vib_s   = np.clip(vib_s + noise_s, 0.05, 12.0)
    samples.append(np.column_stack([temp_s, vib_s]))
    labels.append(np.full(n_shutdown, 3, dtype=int))

    # ---- Combine and shuffle ----
    X = np.vstack(samples)
    y = np.concatenate(labels)

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


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


def test_quantized_model(quantized, X_norm, y):
    """
    Evaluate quantized model accuracy on the training set.

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
    accuracy    = np.mean(predictions == y)
    print(f"Quantized model accuracy: {accuracy:.4f}")

    for cls in range(4):
        mask    = (y == cls)
        cls_acc = np.mean(predictions[mask] == y[mask]) if mask.any() else 0.0
        print(f"  Class {cls} accuracy: {cls_acc:.4f}  ({mask.sum()} samples)")

    return accuracy


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

def main():
    print("=" * 60)
    print("ESP32 Predictive Maintenance Classifier Training")
    print("=" * 60)

    # ---- Generate data ----
    print("\nGenerating training data...")
    X_raw, y = generate_training_data(n_samples=3000)
    print(f"Generated {len(X_raw)} samples")
    class_names = ["healthy", "warning", "critical", "shutdown_required"]
    counts = np.bincount(y, minlength=4)
    for cls, (name, count) in enumerate(zip(class_names, counts)):
        print(f"  Class {cls} ({name}): {count} samples ({100*count/len(y):.1f}%)")

    # ---- Normalise inputs ----
    # Normalisation mirrors the on-device code in main.c:
    #   temp  / 120.0  maps the 0-120 °C range to [0, 1]
    #   accel / 12.0   maps the 0-12 g  range to [0, 1]
    # This keeps all 4 features on the same scale, preventing any single
    # feature from dominating the gradient during training.
    TEMP_NORM  = 120.0
    ACCEL_NORM = 12.0

    X_norm = X_raw.copy()
    X_norm[:, 0] /= TEMP_NORM   # temperature
    X_norm[:, 1] /= ACCEL_NORM  # rms_x
    X_norm[:, 2] /= ACCEL_NORM  # rms_y
    X_norm[:, 3] /= ACCEL_NORM  # rms_z

    print(f"\nNormalised input ranges:")
    feature_names = ["temp/120", "rms_x/12", "rms_y/12", "rms_z/12"]
    for i, name in enumerate(feature_names):
        print(f"  [{i}] {name}: [{X_norm[:, i].min():.3f}, {X_norm[:, i].max():.3f}]")

    # ---- Train ----
    layer1, layer2 = train_model(X_norm, y, epochs=1000, learning_rate=0.1)

    # ---- Quantize ----
    # input_scale = max_abs / 127.0 so that the full normalised range [-1, 1]
    # maps to INT8 [-127, 127]. Using 1/255 would clip at ±0.498 — destroying
    # the ability to distinguish classes with higher feature values.
    input_scale = 1.0 / 127.0
    quantized = quantize_model(layer1, layer2, X_norm, input_scale=input_scale)

    # ---- Evaluate quantized accuracy ----
    test_quantized_model(quantized, X_norm, y)

    # ---- Print statistics ----
    print_weight_statistics(layer1, layer2, quantized)

    # ---- Export ----
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    header_path = os.path.join(project_root, 'esp32_demo', 'predictive_maintenance',
                               'main', 'blitzed_model_weights.h')
    binary_path = os.path.join(project_root, 'esp32_demo', 'predictive_maintenance',
                               'main', 'blitzed_model_weights.bin')

    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    model_size = export_to_c_header(quantized, header_path)
    export_binary_weights(quantized, binary_path)

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


if __name__ == '__main__':
    main()
