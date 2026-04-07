#!/usr/bin/env python3
"""
Standalone training script for ESP32 temperature anomaly detector.

Trains a 2-layer dense network (1->16->4) using only numpy.
Implements INT8 post-training quantization and exports to C header.

Usage:
    python tools/train_temp_anomaly_detector.py
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
    """Cross-entropy loss."""
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-12)
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
    Generate synthetic temperature anomaly data.

    Classes:
    - 0: normal         — typical operating temp (20°C to 45°C)
    - 1: cold_warning   — below normal, potential freezing/shutdown risk (-10°C to 15°C)
    - 2: hot_warning    — above normal, needs attention (50°C to 75°C)
    - 3: critical_overheat — danger, requires immediate action (80°C to 120°C)
    """
    np.random.seed(seed)

    # Class distribution: 50% normal, 20% cold_warning, 15% hot_warning, 15% critical_overheat
    n_normal = int(n_samples * 0.50)
    n_cold = int(n_samples * 0.20)
    n_hot = int(n_samples * 0.15)
    n_critical = n_samples - n_normal - n_cold - n_hot

    # Generate samples with noise within each range
    normal_samples = np.random.uniform(20.0, 45.0, n_normal)
    cold_samples = np.random.uniform(-10.0, 15.0, n_cold)
    hot_samples = np.random.uniform(50.0, 75.0, n_hot)
    critical_samples = np.random.uniform(80.0, 120.0, n_critical)

    # Combine
    X = np.concatenate([normal_samples, cold_samples, hot_samples, critical_samples])
    y = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_cold, dtype=int),
        np.full(n_hot, 2, dtype=int),
        np.full(n_critical, 3, dtype=int)
    ])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices].reshape(-1, 1)
    y = y[indices]

    return X, y


# --- Training ---

def train_model(X, y, epochs=500, learning_rate=0.1):
    """Train the 2-layer network."""

    # Initialize layers
    layer1 = DenseLayer(1, 16)
    layer2 = DenseLayer(16, 4)

    print("Training model...")
    print(f"Layer 1: {layer1.weights.shape} weights + {layer1.bias.shape} bias")
    print(f"Layer 2: {layer2.weights.shape} weights + {layer2.bias.shape} bias")

    total_params = (1 * 16 + 16) + (16 * 4 + 4)
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

def quantize_weights(weights, dtype='int8'):
    """
    Quantize weights to INT8.

    Returns:
        quantized_weights: INT8 array
        scale: float32 scale factor
        zero_point: int8 zero point
    """
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))

    if dtype == 'int8':
        # INT8 range: -128 to 127
        scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
        zero_point = int(np.round(-w_min / scale)) - 128
        zero_point = np.clip(zero_point, -128, 127)

        quantized = np.round(weights / scale) + zero_point
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return quantized, scale, zero_point


def quantize_bias(bias, input_scale, weight_scale):
    """
    Quantize bias to INT32.

    bias_scale = input_scale * weight_scale
    quantized_bias = round(bias / bias_scale)
    """
    bias_scale = input_scale * weight_scale
    quantized = np.round(bias / bias_scale).astype(np.int32)
    return quantized, bias_scale


def quantize_model(layer1, layer2, X_normalized, input_scale=1.0/255.0):
    """
    Apply INT8 post-training quantization to the model.

    Uses calibrated activation statistics (collected over the training set) to
    determine the output scale for layer 1. This prevents hidden-layer saturation
    that occurs when using the naive output_scale = input_scale * weight_scale formula.

    Returns dictionary with quantized parameters and scales.
    """
    print("\nQuantizing model to INT8...")

    # Layer 1 weights
    q_w1, w1_scale, w1_zp = quantize_weights(layer1.weights)

    # Calibrate layer 1 output scale from actual activation statistics.
    # Run a float forward pass and measure the real activation range after ReLU.
    z1_cal = np.dot(X_normalized, layer1.weights) + layer1.bias
    a1_cal = relu(z1_cal)
    a1_max = float(np.max(np.abs(a1_cal)))
    if a1_max == 0.0:
        a1_max = 1.0
    # Map observed activation range [-a1_max, a1_max] -> INT8 [-128, 127]
    output1_scale = a1_max / 127.0

    # Bias for layer 1: quantized at scale = input_scale * w1_scale
    q_b1, b1_scale = quantize_bias(layer1.bias, input_scale, w1_scale)

    # Layer 2 weights
    q_w2, w2_scale, w2_zp = quantize_weights(layer2.weights)
    q_b2, b2_scale = quantize_bias(layer2.bias, output1_scale, w2_scale)

    # Final output scale: calibrate from layer 2 output statistics
    z2_cal = np.dot(a1_cal, layer2.weights) + layer2.bias
    z2_max = float(np.max(np.abs(z2_cal)))
    if z2_max == 0.0:
        z2_max = 1.0
    output2_scale = z2_max / 127.0

    quantized = {
        'layer1': {
            'weights': q_w1,
            'bias': q_b1,
            'weight_scale': w1_scale,
            'weight_zero_point': w1_zp,
            'output_scale': output1_scale,
            'output_zero_point': 0
        },
        'layer2': {
            'weights': q_w2,
            'bias': q_b2,
            'weight_scale': w2_scale,
            'weight_zero_point': w2_zp,
            'output_scale': output2_scale,
            'output_zero_point': 0
        },
        'input_scale': input_scale,
        'input_zero_point': 0
    }

    print(f"Layer 1 weights: scale={w1_scale:.6f}, zero_point={w1_zp}")
    print(f"Layer 1 output:  calibrated scale={output1_scale:.6f} (max activation={a1_max:.4f})")
    print(f"Layer 2 weights: scale={w2_scale:.6f}, zero_point={w2_zp}")
    print(f"Layer 2 output:  calibrated scale={output2_scale:.6f} (max logit={z2_max:.4f})")

    return quantized


def test_quantized_model(quantized, X, y):
    """
    Test accuracy of quantized model.

    Mirrors the exact INT8 forward pass in blitzed_inference.c:
      1. Quantize input using input_scale
      2. Layer 1: INT32 accumulation -> float requantize -> ReLU -> INT32 hidden
      3. Layer 2: INT32 accumulation -> float requantize -> INT8 output
    """
    n = X.shape[0]
    input_scale = quantized['input_scale']
    l1 = quantized['layer1']
    l2 = quantized['layer2']

    l1_w = l1['weights'].astype(np.int32)   # shape (1, 16)
    l1_b = l1['bias'].astype(np.int32)       # shape (1, 16)
    l2_w = l2['weights'].astype(np.int32)   # shape (16, 4)
    l2_b = l2['bias'].astype(np.int32)       # shape (1, 4)

    # Quantize inputs: q = round(x / input_scale) + zero_point
    X_q = np.clip(np.round(X / input_scale) + quantized['input_zero_point'],
                  -128, 127).astype(np.int32)

    # Layer 1: Dense(1, 16) — acc = bias + X_q @ W1
    # acc is in units of (input_scale * w1_scale)
    acc1 = X_q @ l1_w + l1_b                # shape (n, 16)

    # Requantize: convert from accumulator scale to calibrated layer1_output_scale
    #   float_val = acc * input_scale * w1_scale
    #   requantized = round(float_val / calibrated_output_scale) + out_zp
    float1 = acc1.astype(np.float64) * input_scale * l1['weight_scale']
    req1 = np.round(float1 / l1['output_scale']) + l1['output_zero_point']

    # ReLU in quantized space: max(output_zero_point, req)
    req1 = np.maximum(req1, l1['output_zero_point'])
    hidden = np.clip(req1, -128, 127).astype(np.int32)  # shape (n, 16)

    # Layer 2: Dense(16, 4) — acc = bias + hidden @ W2
    # bias was quantized at scale = layer1_output_scale * w2_scale
    acc2 = hidden @ l2_w + l2_b             # shape (n, 4)

    # Requantize: convert from accumulator scale to calibrated layer2_output_scale
    float2 = acc2.astype(np.float64) * l1['output_scale'] * l2['weight_scale']
    req2 = np.round(float2 / l2['output_scale']) + l2['output_zero_point']
    output = np.clip(req2, -128, 127).astype(np.int8)   # shape (n, 4)

    # Argmax on INT8 logits (same as C inference: no softmax needed for classification)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == y)

    print(f"Quantized model accuracy: {accuracy:.4f}")

    return accuracy


# --- Export to C Header ---

def format_array_as_c(arr, name, dtype, values_per_line=12):
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

    # Calculate model size
    size_bytes = (
        layer1['weights'].nbytes + layer1['bias'].nbytes +
        layer2['weights'].nbytes + layer2['bias'].nbytes
    )

    header = f"""// Auto-generated by tools/train_temp_anomaly_detector.py
// Model: 2-layer dense network for ESP32 temperature anomaly detection
// Architecture: Dense(1, 16) + ReLU + Dense(16, 4)
// Total parameters: 84
// Quantized model size: {size_bytes} bytes

#ifndef BLITZED_MODEL_WEIGHTS_H
#define BLITZED_MODEL_WEIGHTS_H

#include <stdint.h>

// Layer 1: Dense(1, 16) + ReLU
#define LAYER1_INPUT_SIZE 1
#define LAYER1_OUTPUT_SIZE 16

static const int8_t layer1_weights[{layer1['weights'].size}] = {{
{format_array_as_c(layer1['weights'], 'layer1_weights', 'int8_t')}
}};

static const int32_t layer1_bias[{layer1['bias'].size}] = {{
{format_array_as_c(layer1['bias'], 'layer1_bias', 'int32_t')}
}};

static const float layer1_weight_scale = {layer1['weight_scale']:.10f}f;
static const int32_t layer1_weight_zero_point = {layer1['weight_zero_point']};
static const float layer1_output_scale = {layer1['output_scale']:.10f}f;
static const int32_t layer1_output_zero_point = {layer1['output_zero_point']};

// Layer 2: Dense(16, 4)
#define LAYER2_INPUT_SIZE 16
#define LAYER2_OUTPUT_SIZE 4

static const int8_t layer2_weights[{layer2['weights'].size}] = {{
{format_array_as_c(layer2['weights'], 'layer2_weights', 'int8_t')}
}};

static const int32_t layer2_bias[{layer2['bias'].size}] = {{
{format_array_as_c(layer2['bias'], 'layer2_bias', 'int32_t')}
}};

static const float layer2_weight_scale = {layer2['weight_scale']:.10f}f;
static const int32_t layer2_weight_zero_point = {layer2['weight_zero_point']};
static const float layer2_output_scale = {layer2['output_scale']:.10f}f;
static const int32_t layer2_output_zero_point = {layer2['output_zero_point']};

// Input quantization parameters
// Temperatures are normalized by dividing by INPUT_MAX (120.0) before quantization.
// input_scale covers the full normalized range [INPUT_MIN/120, 1.0] mapped into INT8.
#define INPUT_SCALE {quantized['input_scale']:.10f}f
#define INPUT_ZERO_POINT {quantized['input_zero_point']}
#define INPUT_MIN -10.0f
#define INPUT_MAX 120.0f

// Class labels
#define NUM_CLASSES 4
static const char* class_labels[NUM_CLASSES] = {{
    "normal",
    "cold_warning",
    "hot_warning",
    "critical_overheat"
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
    print(f"  Bias: min={layer1.bias.min():.4f}, max={layer1.bias.max():.4f}, mean={layer1.bias.mean():.4f}")

    print(f"\nLayer 2 (float32):")
    print(f"  Weights: min={layer2.weights.min():.4f}, max={layer2.weights.max():.4f}, mean={layer2.weights.mean():.4f}")
    print(f"  Bias: min={layer2.bias.min():.4f}, max={layer2.bias.max():.4f}, mean={layer2.bias.mean():.4f}")

    print(f"\nLayer 1 (int8):")
    print(f"  Weights: min={quantized['layer1']['weights'].min()}, max={quantized['layer1']['weights'].max()}, mean={quantized['layer1']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer1']['bias'].min()}, max={quantized['layer1']['bias'].max()}, mean={quantized['layer1']['bias'].mean():.2f}")

    print(f"\nLayer 2 (int8):")
    print(f"  Weights: min={quantized['layer2']['weights'].min()}, max={quantized['layer2']['weights'].max()}, mean={quantized['layer2']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer2']['bias'].min()}, max={quantized['layer2']['bias'].max()}, mean={quantized['layer2']['bias'].mean():.2f}")


# --- Main ---

def main():
    """Main training pipeline."""

    print("=" * 60)
    print("ESP32 Temperature Anomaly Detector Training")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    X, y = generate_training_data(n_samples=2000)
    print(f"Generated {len(X)} samples")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"  0=normal: {np.sum(y==0)}, 1=cold_warning: {np.sum(y==1)}, "
          f"2=hot_warning: {np.sum(y==2)}, 3=critical_overheat: {np.sum(y==3)}")

    # Normalize inputs: divide by 120.0 (max temp range) for stable training
    # Training lesson: normalization is CRITICAL — unnormalized data causes gradient explosion
    X_normalized = X / 120.0
    layer1, layer2 = train_model(X_normalized, y, epochs=500, learning_rate=0.1)

    # Quantize model
    # Input range after normalization: temps in [-10, 120] / 120.0 = [-0.083, 1.0]
    # Map to INT8 range [-128, 127] using scale = range / 255:
    #   full range = 1.0 - (-0.083) = 1.083
    #   scale = 1.083 / 255.0 = 0.004247
    # This ensures the full temperature range maps within [-128, 127] without overflow.
    # input_scale must be large enough so that the full normalized range
    # [-0.083, 1.0] maps within INT8 [-128, 127]. Using /255 clips at ±0.54
    # which destroys high-temp classes. Use /127 so 1.0 → 127.
    input_range = 1.0 - ((-10.0) / 120.0)  # = 1.0 - (-0.0833) = 1.0833
    input_scale = input_range / 127.0
    quantized = quantize_model(layer1, layer2, X_normalized, input_scale=input_scale)

    # Test quantized model accuracy (use normalized input)
    test_quantized_model(quantized, X_normalized, y)

    # Print statistics
    print_weight_statistics(layer1, layer2, quantized)

    # Export to C header
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    header_path = os.path.join(project_root, 'esp32_demo', 'temp_anomaly', 'main', 'blitzed_model_weights.h')
    binary_path = os.path.join(project_root, 'esp32_demo', 'temp_anomaly', 'main', 'blitzed_model_weights.bin')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    model_size = export_to_c_header(quantized, header_path)
    export_binary_weights(quantized, binary_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Include the header in your ESP32 code: #include \"blitzed_model_weights.h\"")
    print(f"  2. Implement INT8 inference using the quantized weights")
    print(f"  3. Flash to ESP32 and test with internal temperature sensor data")
    print(f"\nModel files:")
    print(f"  - {header_path}")
    print(f"  - {binary_path}")


if __name__ == '__main__':
    main()
