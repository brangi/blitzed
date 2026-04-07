#!/usr/bin/env python3
"""
Standalone training script for ESP32 vibration pattern classifier.

Trains a 2-layer dense network (3->32->4) using only numpy.
Implements calibrated INT8 post-training quantization and exports to C header.

Input features: RMS acceleration on X, Y, Z axes (from MPU6050)
Classes:
    0: normal       - balanced rotation, low vibration
    1: imbalance    - one axis dominates
    2: misalignment - two axes elevated
    3: bearing_fault - high-frequency noise across all axes

Training lessons:
    - Normalize inputs by 12.0 (max RMS range) BEFORE training to prevent
      gradient explosion.  lr=0.1 on unnormalized data causes inf loss.
    - Use calibrated output scales (derived from activation statistics) rather
      than the naive accumulated scale (input_scale * weight_scale).  The
      naive scheme causes saturation in the INT32 accumulator for multi-input
      layers and collapses quantized accuracy to near-random.

Usage:
    python tools/train_vibration_classifier.py
"""

import numpy as np
import os


# --- Neural Network Implementation (NumPy only) ---

def xavier_init(input_size, output_size):
    """Xavier/Glorot initialization for weights."""
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax activation for output layer."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss with epsilon for numerical stability."""
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)
    return np.sum(log_likelihood) / m


class DenseLayer:
    """Dense layer with forward and backward pass."""

    def __init__(self, input_size, output_size):
        self.weights = xavier_init(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, x):
        """Forward pass."""
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward(self, grad_output, learning_rate):
        """Backward pass with gradient descent."""
        m = self.input.shape[0]

        # Gradients
        grad_weights = np.dot(self.input.T, grad_output) / m
        grad_bias = np.sum(grad_output, axis=0, keepdims=True) / m
        grad_input = np.dot(grad_output, self.weights.T)

        # Update weights
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


# --- Data Generation ---

def generate_training_data(n_samples=2000, seed=42):
    """
    Generate synthetic vibration sensor data from MPU6050 accelerometer.

    Features: [rms_x, rms_y, rms_z] — RMS acceleration per axis in g units.

    Classes:
    - 0: normal       — balanced rotation, all axes low RMS (0.0–2.0 g)
    - 1: imbalance    — one axis dominates (3.0–8.0 g), others low (0.5–2.0 g)
    - 2: misalignment — two axes elevated (2.5–6.0 g), one low (0.5–1.5 g)
    - 3: bearing_fault — all axes high RMS (4.0–12.0 g), high-frequency noise
    """
    np.random.seed(seed)

    # Class distribution: 40% normal, 20% imbalance, 20% misalignment, 20% bearing_fault
    n_normal = int(n_samples * 0.40)
    n_imbalance = int(n_samples * 0.20)
    n_misalignment = int(n_samples * 0.20)
    n_bearing = n_samples - n_normal - n_imbalance - n_misalignment

    samples = []
    labels = []

    # Class 0: normal — all axes low and uniform (<1.2g)
    for _ in range(n_normal):
        rms = np.random.uniform(0.1, 1.2, 3)
        noise = np.random.normal(0, 0.05, 3)
        samples.append(np.clip(rms + noise, 0.0, None))
        labels.append(0)

    # Class 1: imbalance — exactly ONE axis moderate-high (3-6g), others very low (<0.5g)
    # Distinguished from bearing_fault (ALL high) and misalignment (TWO high)
    for _ in range(n_imbalance):
        dominant_axis = np.random.randint(0, 3)
        rms = np.random.uniform(0.05, 0.5, 3)
        rms[dominant_axis] = np.random.uniform(3.0, 6.0)
        noise = np.random.normal(0, 0.05, 3)
        samples.append(np.clip(rms + noise, 0.0, None))
        labels.append(1)

    # Class 2: misalignment — TWO axes elevated (3.5-7g), one low (<0.8g)
    for _ in range(n_misalignment):
        low_axis = np.random.randint(0, 3)
        rms = np.random.uniform(3.5, 7.0, 3)
        rms[low_axis] = np.random.uniform(0.1, 0.8)
        noise = np.random.normal(0, 0.15, 3)
        samples.append(np.clip(rms + noise, 0.0, None))
        labels.append(2)

    # Class 3: bearing_fault — ALL axes uniformly high (>5g)
    for _ in range(n_bearing):
        rms = np.random.uniform(5.0, 12.0, 3)
        noise = np.random.normal(0, 0.3, 3)
        samples.append(np.clip(rms + noise, 0.0, None))
        labels.append(3)

    X = np.array(samples, dtype=np.float32)
    y = np.array(labels, dtype=int)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


# --- Training ---

def train_model(X, y, epochs=500, learning_rate=0.1):
    """Train the 2-layer network: Dense(3, 32) + ReLU + Dense(32, 4)."""

    # Initialize layers
    layer1 = DenseLayer(3, 32)
    layer2 = DenseLayer(32, 4)

    print("Training model...")
    print(f"Layer 1: {layer1.weights.shape} weights + {layer1.bias.shape} bias")
    print(f"Layer 2: {layer2.weights.shape} weights + {layer2.bias.shape} bias")

    # Dense(3,32): 3*32 + 32 = 128 params
    # Dense(32,4): 32*4 + 4 = 132 params
    total_params = (3 * 32 + 32) + (32 * 4 + 4)
    print(f"Total parameters: {total_params}")

    accuracy = 0.0

    for epoch in range(epochs):
        # Forward pass
        z1 = layer1.forward(X)
        a1 = relu(z1)
        z2 = layer2.forward(a1)
        y_pred = softmax(z2)

        # Compute loss
        loss = cross_entropy_loss(y_pred, y)

        # Compute accuracy
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

        # Backward pass
        m = X.shape[0]
        grad_z2 = y_pred.copy()
        grad_z2[range(m), y] -= 1

        grad_a1 = layer2.backward(grad_z2, learning_rate)
        grad_z1 = grad_a1 * relu_derivative(z1)
        layer1.backward(grad_z1, learning_rate)

    print(f"\nFinal training accuracy: {accuracy:.4f}")

    return layer1, layer2


# --- Quantization ---

def quantize_weights(weights):
    """
    Quantize weights to INT8 using the full symmetric scale.

    Returns:
        quantized_weights: INT8 array
        scale: float32 scale factor
        zero_point: int zero point
    """
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))

    # INT8 range: -128 to 127
    scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
    zero_point = int(np.round(-w_min / scale)) - 128
    zero_point = int(np.clip(zero_point, -128, 127))

    quantized = np.round(weights / scale) + zero_point
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    return quantized, scale, zero_point


def calibrate_output_scales(layer1, layer2, X_norm):
    """
    Compute calibrated output scales by collecting activation statistics
    from the training data.  This is required for multi-input layers where
    the naive accumulated scale (input_scale * weight_scale) causes INT8
    saturation.

    Returns:
        o1_scale: scale that maps layer1 post-ReLU activations into [0, 127]
        o2_scale: scale for layer2 output covering the full INT8 range
        o2_zp:    zero point for layer2 output
    """
    # Collect layer 1 activations
    a1 = relu(layer1.forward(X_norm))
    a1_max = float(np.max(a1))
    # Map [0, a1_max] to INT8 [0, 127]
    o1_scale = a1_max / 127.0

    # Collect layer 2 pre-softmax logits
    z2 = layer2.forward(a1)
    z2_min = float(np.min(z2))
    z2_max = float(np.max(z2))
    # Map [z2_min, z2_max] to INT8 [-128, 127]
    o2_scale = (z2_max - z2_min) / 255.0 if z2_max != z2_min else 1.0
    o2_zp = int(np.round(-z2_min / o2_scale)) - 128
    o2_zp = int(np.clip(o2_zp, -128, 127))

    print(f"Layer 1 activation range: [0.0, {a1_max:.4f}] -> scale={o1_scale:.8f}")
    print(f"Layer 2 output range: [{z2_min:.4f}, {z2_max:.4f}] -> scale={o2_scale:.8f}, zp={o2_zp}")

    return o1_scale, o2_scale, o2_zp


def quantize_model(layer1, layer2, X_norm, input_scale=1.0/255.0):
    """
    Apply calibrated INT8 post-training quantization to the model.

    Uses activation statistics from X_norm to compute output scales that
    prevent INT8 saturation in the accumulated layer outputs.

    Returns dictionary with quantized parameters and scales.
    """
    print("\nQuantizing model to INT8 (calibrated)...")

    # Weight quantization
    q_w1, w1_scale, w1_zp = quantize_weights(layer1.weights)
    q_w2, w2_scale, w2_zp = quantize_weights(layer2.weights)

    # Calibrated output scales from activation statistics
    o1_scale, o2_scale, o2_zp = calibrate_output_scales(layer1, layer2, X_norm)

    # Bias quantization: bias_scale = accumulator_input_scale * weight_scale
    # Layer 1 bias: accumulated scale is input_scale * w1_scale
    b1_scale = input_scale * w1_scale
    q_b1 = np.round(layer1.bias / b1_scale).astype(np.int32)

    # Layer 2 bias: accumulated scale is o1_scale * w2_scale
    b2_scale = o1_scale * w2_scale
    q_b2 = np.round(layer2.bias / b2_scale).astype(np.int32)

    print(f"Layer 1 weights: scale={w1_scale:.6f}, zero_point={w1_zp}")
    print(f"Layer 2 weights: scale={w2_scale:.6f}, zero_point={w2_zp}")

    quantized = {
        'layer1': {
            'weights': q_w1,
            'bias': q_b1,
            'weight_scale': w1_scale,
            'weight_zero_point': w1_zp,
            'output_scale': o1_scale,
            'output_zero_point': 0   # ReLU output is always >= 0, zp=0
        },
        'layer2': {
            'weights': q_w2,
            'bias': q_b2,
            'weight_scale': w2_scale,
            'weight_zero_point': w2_zp,
            'output_scale': o2_scale,
            'output_zero_point': o2_zp
        },
        'input_scale': input_scale,
        'input_zero_point': 0
    }

    return quantized


def test_quantized_model(quantized, X_norm, y):
    """
    Test accuracy by simulating the exact C kernel arithmetic.

    This mirrors the INT32 accumulate -> float requantize -> INT8 clamp path
    used in blitzed_inference.c so the reported accuracy is what the ESP32
    will actually achieve.
    """
    input_scale = quantized['input_scale']
    q_w1 = quantized['layer1']['weights']
    q_b1 = quantized['layer1']['bias']
    w1_scale = quantized['layer1']['weight_scale']
    o1_scale = quantized['layer1']['output_scale']
    o1_zp = quantized['layer1']['output_zero_point']

    q_w2 = quantized['layer2']['weights']
    q_b2 = quantized['layer2']['bias']
    w2_scale = quantized['layer2']['weight_scale']
    o2_scale = quantized['layer2']['output_scale']
    o2_zp = quantized['layer2']['output_zero_point']

    correct = 0
    class_correct = np.zeros(4, dtype=int)
    class_total = np.zeros(4, dtype=int)

    for xi in range(len(X_norm)):
        # Quantize 3 normalized input features
        inp = np.clip(np.round(X_norm[xi] / input_scale), -128, 127).astype(np.int8)

        # Layer 1: Dense(3, 32) + ReLU
        hidden = np.zeros(32, dtype=np.int32)
        for j in range(32):
            acc = int(q_b1[0, j])
            for i in range(3):
                acc += int(inp[i]) * int(q_w1[i, j])
            float_val = float(acc) * input_scale * w1_scale
            req = int(round(float_val / o1_scale)) + o1_zp
            req = max(o1_zp, req)  # ReLU in quantized space
            req = min(127, max(-128, req))
            hidden[j] = req

        # Layer 2: Dense(32, 4)
        out = np.zeros(4, dtype=np.int8)
        for j in range(4):
            acc = int(q_b2[0, j])
            for i in range(32):
                acc += int(hidden[i]) * int(q_w2[i, j])
            float_val = float(acc) * o1_scale * w2_scale
            req = int(round(float_val / o2_scale)) + o2_zp
            req = min(127, max(-128, req))
            out[j] = np.int8(req)

        pred = int(np.argmax(out))
        class_total[y[xi]] += 1
        if pred == y[xi]:
            correct += 1
            class_correct[y[xi]] += 1

    accuracy = correct / len(X_norm)
    print(f"Quantized model accuracy: {accuracy:.4f}")

    class_names = ['normal', 'imbalance', 'misalignment', 'bearing_fault']
    for cls_idx, cls_name in enumerate(class_names):
        if class_total[cls_idx] > 0:
            cls_acc = class_correct[cls_idx] / class_total[cls_idx]
            print(f"  Class {cls_idx} ({cls_name}): {cls_acc:.4f} ({class_total[cls_idx]} samples)")

    return accuracy


# --- Export to C Header ---

def format_array_as_c(arr, values_per_line=12):
    """Format numpy array as C array initializer."""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i+values_per_line]
        values = ', '.join(str(int(v)) for v in chunk)
        lines.append(f"    {values}")
    return ',\n'.join(lines)


def export_to_c_header(quantized, output_path):
    """Export quantized model to C header file."""

    layer1 = quantized['layer1']
    layer2 = quantized['layer2']

    size_bytes = (
        layer1['weights'].nbytes + layer1['bias'].nbytes +
        layer2['weights'].nbytes + layer2['bias'].nbytes
    )

    header = f"""// Auto-generated by tools/train_vibration_classifier.py
// Model: 2-layer dense network for ESP32 vibration pattern classification
// Architecture: Dense(3, 32) + ReLU + Dense(32, 4)
// Input: [rms_x, rms_y, rms_z] — RMS acceleration per axis from MPU6050
// Total parameters: 260
// Quantized model size: {size_bytes} bytes
// Quantization: calibrated INT8 PTQ (output scales from activation statistics)

#ifndef BLITZED_MODEL_WEIGHTS_H
#define BLITZED_MODEL_WEIGHTS_H

#include <stdint.h>

// Layer 1: Dense(3, 32) + ReLU
#define LAYER1_INPUT_SIZE 3
#define LAYER1_OUTPUT_SIZE 32

// Weights stored row-major: layer1_weights[input_idx * LAYER1_OUTPUT_SIZE + output_idx]
static const int8_t layer1_weights[{layer1['weights'].size}] = {{
{format_array_as_c(layer1['weights'])}
}};

static const int32_t layer1_bias[{layer1['bias'].size}] = {{
{format_array_as_c(layer1['bias'])}
}};

static const float layer1_weight_scale = {layer1['weight_scale']:.10f}f;
static const int32_t layer1_weight_zero_point = {layer1['weight_zero_point']};
// Calibrated output scale: maps post-ReLU activations to INT8 [0, 127]
static const float layer1_output_scale = {layer1['output_scale']:.10f}f;
static const int32_t layer1_output_zero_point = {layer1['output_zero_point']};

// Layer 2: Dense(32, 4)
#define LAYER2_INPUT_SIZE 32
#define LAYER2_OUTPUT_SIZE 4

// Weights stored row-major: layer2_weights[input_idx * LAYER2_OUTPUT_SIZE + output_idx]
static const int8_t layer2_weights[{layer2['weights'].size}] = {{
{format_array_as_c(layer2['weights'])}
}};

static const int32_t layer2_bias[{layer2['bias'].size}] = {{
{format_array_as_c(layer2['bias'])}
}};

static const float layer2_weight_scale = {layer2['weight_scale']:.10f}f;
static const int32_t layer2_weight_zero_point = {layer2['weight_zero_point']};
// Calibrated output scale: maps output logits to INT8 [-128, 127]
static const float layer2_output_scale = {layer2['output_scale']:.10f}f;
static const int32_t layer2_output_zero_point = {layer2['output_zero_point']};

// Input quantization parameters
// Normalize each RMS value by INPUT_NORM_FACTOR (12.0g) before quantizing
#define INPUT_SCALE {quantized['input_scale']:.10f}f
#define INPUT_ZERO_POINT {quantized['input_zero_point']}
#define INPUT_NORM_FACTOR 12.0f

// Class labels
#define NUM_CLASSES 4
static const char* class_labels[NUM_CLASSES] = {{
    "normal",
    "imbalance",
    "misalignment",
    "bearing_fault"
}};

#endif // BLITZED_MODEL_WEIGHTS_H
"""

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"\nExported C header to: {output_path}")
    print(f"Model size: {size_bytes} bytes")

    return size_bytes


def export_binary_weights(quantized, output_path):
    """Export raw quantized weights to binary file for Rust pipeline."""

    with open(output_path, 'wb') as f:
        # Write layer 1 weights and bias
        quantized['layer1']['weights'].tofile(f)
        quantized['layer1']['bias'].tofile(f)

        # Write layer 2 weights and bias
        quantized['layer2']['weights'].tofile(f)
        quantized['layer2']['bias'].tofile(f)

    print(f"Exported binary weights to: {output_path}")


def print_weight_statistics(layer1, layer2, quantized):
    """Print weight statistics for debugging."""

    print("\n=== Weight Statistics ===")
    print(f"\nLayer 1 (float32):")
    print(f"  Weights: min={layer1.weights.min():.4f}, max={layer1.weights.max():.4f}, mean={layer1.weights.mean():.4f}")
    print(f"  Bias: min={layer1.bias.min():.4f}, max={layer1.bias.max():.4f}")

    print(f"\nLayer 2 (float32):")
    print(f"  Weights: min={layer2.weights.min():.4f}, max={layer2.weights.max():.4f}, mean={layer2.weights.mean():.4f}")
    print(f"  Bias: min={layer2.bias.min():.4f}, max={layer2.bias.max():.4f}")

    print(f"\nLayer 1 (int8):")
    print(f"  Weights: min={quantized['layer1']['weights'].min()}, max={quantized['layer1']['weights'].max()}, mean={quantized['layer1']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer1']['bias'].min()}, max={quantized['layer1']['bias'].max()}")

    print(f"\nLayer 2 (int8):")
    print(f"  Weights: min={quantized['layer2']['weights'].min()}, max={quantized['layer2']['weights'].max()}, mean={quantized['layer2']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer2']['bias'].min()}, max={quantized['layer2']['bias'].max()}")


# --- Main ---

def main():
    """Main training pipeline."""

    print("=" * 60)
    print("ESP32 Vibration Pattern Classifier Training")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    X, y = generate_training_data(n_samples=2000)
    print(f"Generated {len(X)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"  0=normal: {np.sum(y == 0)}, 1=imbalance: {np.sum(y == 1)}, "
          f"2=misalignment: {np.sum(y == 2)}, 3=bearing_fault: {np.sum(y == 3)}")

    # Train model
    # Normalize inputs: divide by 12.0 (max RMS range) for stable training.
    # Critical: without normalization, lr=0.1 causes gradient explosion (inf loss).
    X_normalized = X / 12.0
    layer1, layer2 = train_model(X_normalized, y, epochs=1000, learning_rate=0.1)

    # Quantize model with calibrated output scales
    # input_scale = 1.0/127.0 so the full normalized range [0, 1.0]
    # maps to INT8 [0, 127]. Using 1/255 clips at ~0.498 and destroys
    # high-vibration classes.
    input_scale = 1.0 / 127.0
    quantized = quantize_model(layer1, layer2, X_normalized, input_scale=input_scale)

    # Test quantized model by simulating the exact C kernel
    print("\nTesting quantized model (C-kernel simulation)...")
    test_quantized_model(quantized, X_normalized, y)

    # Print statistics
    print_weight_statistics(layer1, layer2, quantized)

    # Export to C header
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    header_path = os.path.join(
        project_root, 'esp32_demo', 'vibration_classifier', 'main',
        'blitzed_model_weights.h'
    )
    binary_path = os.path.join(
        project_root, 'esp32_demo', 'vibration_classifier', 'main',
        'blitzed_model_weights.bin'
    )

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    model_size = export_to_c_header(quantized, header_path)
    export_binary_weights(quantized, binary_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Connect MPU6050 to ESP32: SDA=GPIO21, SCL=GPIO22")
    print(f"  2. Build and flash: cd esp32_demo/vibration_classifier && idf.py build flash")
    print(f"  3. Monitor output: idf.py monitor")
    print(f"\nModel files:")
    print(f"  - {header_path}")
    print(f"  - {binary_path}")


if __name__ == '__main__':
    main()
