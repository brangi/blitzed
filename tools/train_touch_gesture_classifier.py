#!/usr/bin/env python3
"""
Standalone training script for ESP32 touch gesture classifier.

Trains a 2-layer dense network (20->32->5) using only numpy.
Implements calibrated INT8 post-training quantization and exports to C header.

Input features: 20 floats extracted from a 1.5s touch window (75 steps at 50Hz)
across 4 capacitive touch pads (GPIO 4, 12, 14, 27).

Classes:
    0: swipe_right  - pads activate left-to-right in ascending time order
    1: swipe_left   - pads activate right-to-left in descending time order
    2: single_tap   - one pad touched briefly once
    3: double_tap   - same pad tapped twice with a gap
    4: long_press   - one or two pads held for 600-1400ms

Feature layout (20 features):
    Per-pad features (4 pads x 4 features = 16):
        [pad*4 + 0]  touch_duration   : fraction of window with pad below threshold [0, 1]
        [pad*4 + 1]  touch_centroid   : weighted average time of touch activity [0, 1]
        [pad*4 + 2]  mean_intensity   : avg (baseline - reading) / baseline during touch [0, 1]
        [pad*4 + 3]  touch_count      : distinct touch-release transitions / 5  [0, 1]
    Global features (4):
        [16]  active_pin_count   : how many pads were touched at all, / 4.0
        [17]  spatial_sweep      : correlation(pad_index, touch_centroid), mapped [0, 1] via (x+1)/2
        [18]  total_duration     : sum of per-pad durations / 4.0
        [19]  inter_onset_interval : mean time between pad onset times / 1.0

Training lessons inherited from vibration_classifier:
    - Features are already in [0, 1] so INPUT_NORM_FACTOR = 1.0.
    - Use INPUT_SCALE = 1/127 so the full [0, 1] float range maps to INT8 [0, 127].
      Using 1/255 would clip at 0.498 and destroy high-value features.
    - Use calibrated output scales (derived from activation statistics) rather than
      the naive accumulated scale. The naive scheme causes INT8 saturation in the
      accumulator for multi-input (20-input) layers and collapses accuracy to near-random.

Usage:
    python tools/train_touch_gesture_classifier.py
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# Neural Network Implementation (NumPy only)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

# Touch pad configuration (mirrors touch_sensor.h)
N_PADS = 4
N_STEPS = 75        # 1.5 seconds at 50Hz
N_FEATURES = 20     # 4 pads * 4 per-pad features + 4 global features
BASELINE_NOMINAL = 1600.0
THRESHOLD_RATIO = 0.70   # threshold = 70% of baseline


def extract_features(window, n_steps, baselines, thresholds):
    """
    Extract 20 float features from a raw touch window.

    All output features are designed to fall in [0, 1].  This function is
    intentionally written in plain Python/NumPy arithmetic so it can be
    ported line-by-line to C for the ESP32 firmware.

    Args:
        window:     numpy array of shape (n_steps, 4), raw capacitive readings
        n_steps:    number of valid time steps (<= window.shape[0])
        baselines:  array of 4 float baseline (untouched) values per pad
        thresholds: array of 4 float threshold values per pad

    Returns:
        features: numpy array of shape (20,) in [0, 1]
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)

    # Build boolean touched matrix: True where pad is actively touched
    touched = np.zeros((n_steps, N_PADS), dtype=bool)
    for p in range(N_PADS):
        touched[:, p] = window[:n_steps, p] < thresholds[p]

    # Time axis: [0, 1] for steps 0 .. n_steps-1
    time_axis = np.arange(n_steps, dtype=np.float32) / max(n_steps - 1, 1)

    # Per-pad features (features 0..15)
    onset_times = []   # first touch time per pad (in [0, 1]), for inter_onset_interval

    for p in range(N_PADS):
        pad_touched = touched[:, p]          # shape (n_steps,)
        pad_raw = window[:n_steps, p].astype(np.float32)
        baseline = float(baselines[p])

        # --- touch_duration: fraction of steps that are touched ---
        duration = float(np.sum(pad_touched)) / n_steps
        features[p * 4 + 0] = duration

        # --- touch_centroid: weighted average time position of touch activity ---
        touch_weight = pad_touched.astype(np.float32)
        total_weight = float(np.sum(touch_weight))
        if total_weight > 0.0:
            centroid = float(np.sum(time_axis * touch_weight)) / total_weight
        else:
            centroid = 0.0
        features[p * 4 + 1] = centroid

        # --- mean_intensity: avg (baseline - reading) / baseline during active steps ---
        if total_weight > 0.0:
            delta = np.clip(baseline - pad_raw, 0.0, baseline)
            intensity = float(np.sum(delta * touch_weight)) / (baseline * total_weight)
            intensity = float(np.clip(intensity, 0.0, 1.0))
        else:
            intensity = 0.0
        features[p * 4 + 2] = intensity

        # --- touch_count: number of distinct touch-release transitions / 5 ---
        count = 0
        in_touch = False
        first_onset = -1.0
        for t in range(n_steps):
            if pad_touched[t] and not in_touch:
                in_touch = True
                count += 1
                if first_onset < 0.0:
                    first_onset = time_axis[t]
            elif not pad_touched[t] and in_touch:
                in_touch = False
        features[p * 4 + 3] = float(np.clip(count / 5.0, 0.0, 1.0))

        # Record first onset time for global inter-onset feature
        if first_onset >= 0.0:
            onset_times.append(first_onset)

    # Global features (features 16..19)

    # --- active_pin_count: how many pads were touched at all / 4 ---
    active_count = float(np.sum(np.any(touched, axis=0)))
    features[16] = active_count / 4.0

    # --- spatial_sweep: correlation between pad index and touch centroid ---
    # Positive means left-to-right swipe (pad 0 first), negative = right-to-left
    pad_indices = np.array([0.0, 1.0, 2.0, 3.0])
    centroids = np.array([features[p * 4 + 1] for p in range(N_PADS)], dtype=np.float32)
    active_mask = np.any(touched, axis=0)

    if np.sum(active_mask) >= 2:
        pi = pad_indices[active_mask]
        ci = centroids[active_mask]
        # Pearson correlation between pad index and centroid time
        pi_mean = np.mean(pi)
        ci_mean = np.mean(ci)
        pi_dev = pi - pi_mean
        ci_dev = ci - ci_mean
        denom = np.sqrt(np.sum(pi_dev ** 2) * np.sum(ci_dev ** 2))
        if denom > 1e-6:
            corr = float(np.sum(pi_dev * ci_dev) / denom)
        else:
            corr = 0.0
        corr = float(np.clip(corr, -1.0, 1.0))
    else:
        corr = 0.0
    # Map [-1, 1] to [0, 1]
    features[17] = (corr + 1.0) / 2.0

    # --- total_duration: sum of per-pad durations / 4 ---
    features[18] = float(np.sum(features[0:16:4])) / 4.0

    # --- inter_onset_interval: mean time between successive pad onset times ---
    if len(onset_times) >= 2:
        onset_sorted = sorted(onset_times)
        gaps = [onset_sorted[i + 1] - onset_sorted[i] for i in range(len(onset_sorted) - 1)]
        ioi = float(np.mean(gaps))
        ioi = float(np.clip(ioi, 0.0, 1.0))
    else:
        ioi = 0.0
    features[19] = ioi

    return features


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def _make_window(onset_ms, duration_ms, pad_indices, n_steps, sample_rate_hz,
                 baseline, noise_std=20.0, touch_value_range=(300, 800)):
    """
    Build a raw touch window (n_steps x 4) for a set of pad activations.

    Args:
        onset_ms:         list of onset times in milliseconds (one per pad in pad_indices)
        duration_ms:      list of active durations in milliseconds (one per pad in pad_indices)
        pad_indices:      list of pad indices (0-3) that are activated
        n_steps:          total number of time steps
        sample_rate_hz:   sampling rate (50 Hz)
        baseline:         untouched baseline reading (~1600)
        noise_std:        standard deviation of baseline noise
        touch_value_range: (min, max) raw reading when touched

    Returns:
        window: numpy array of shape (n_steps, 4), dtype float32
    """
    step_ms = 1000.0 / sample_rate_hz   # 20 ms per step
    window = np.full((n_steps, N_PADS), baseline, dtype=np.float32)

    for p, onset, dur in zip(pad_indices, onset_ms, duration_ms):
        touch_start = int(onset / step_ms)
        touch_end = int((onset + dur) / step_ms)
        touch_start = max(0, min(touch_start, n_steps))
        touch_end = max(touch_start, min(touch_end, n_steps))
        if touch_start < touch_end:
            touch_val = np.random.uniform(touch_value_range[0], touch_value_range[1])
            window[touch_start:touch_end, p] = touch_val

    # Add sensor noise everywhere
    window += np.random.normal(0, noise_std, window.shape).astype(np.float32)
    window = np.clip(window, 50, 2200).astype(np.float32)
    return window


def generate_training_data(n_samples_per_class=2000, seed=42):
    """
    Generate synthetic touch gesture data for 5 classes.

    Each sample is a raw touch window of shape (N_STEPS, 4) from which
    20 features are extracted.  The window simulates ESP32 capacitive readings:
    baseline ~1600, touched ~300-800.

    Classes:
        0: swipe_right  — pads 0,1,2,3 activate in ascending time order
        1: swipe_left   — pads 3,2,1,0 activate in ascending time order
        2: single_tap   — one pad, one brief contact
        3: double_tap   — same pad tapped twice with a gap
        4: long_press   — one or two pads held continuously

    Returns:
        X: float32 array of shape (N, 20) — extracted features in [0, 1]
        y: int array of shape (N,) — class labels 0-4
    """
    np.random.seed(seed)
    sample_rate_hz = 50
    step_ms = 1000.0 / sample_rate_hz
    window_ms = N_STEPS * step_ms   # 1500 ms

    baseline = BASELINE_NOMINAL
    thresholds = np.full(N_PADS, baseline * THRESHOLD_RATIO)
    baselines_arr = np.full(N_PADS, baseline)

    X_list = []
    y_list = []

    # --- Class 0: swipe_right (pad 0 -> 1 -> 2 -> 3) ---
    for _ in range(n_samples_per_class):
        # Use 3 or 4 pads (80% chance all 4)
        n_active = 4 if np.random.random() < 0.80 else 3
        if n_active == 4:
            active_pads = [0, 1, 2, 3]
        else:
            # Skip one of the middle pads randomly
            skip = np.random.randint(1, 3)
            active_pads = [p for p in [0, 1, 2, 3] if p != skip]

        # Stagger: 50-200ms between each successive pad onset
        stagger = np.random.uniform(50, 200)
        duration_each = np.random.uniform(50, 150)
        speed_var = np.random.uniform(0.8, 1.2)

        onset_ms = [i * stagger * speed_var for i in range(len(active_pads))]
        duration_ms = [np.random.uniform(50, 150) for _ in active_pads]

        # Keep within window
        max_end = max(o + d for o, d in zip(onset_ms, duration_ms))
        if max_end > window_ms - step_ms:
            scale = (window_ms - step_ms) / max_end
            onset_ms = [o * scale for o in onset_ms]
            duration_ms = [d * scale for d in duration_ms]

        window = _make_window(onset_ms, duration_ms, active_pads,
                              N_STEPS, sample_rate_hz, baseline)
        feats = extract_features(window, N_STEPS, baselines_arr, thresholds)
        X_list.append(feats)
        y_list.append(0)

    # --- Class 1: swipe_left (pad 3 -> 2 -> 1 -> 0) ---
    for _ in range(n_samples_per_class):
        n_active = 4 if np.random.random() < 0.80 else 3
        if n_active == 4:
            active_pads = [3, 2, 1, 0]
        else:
            skip = np.random.randint(1, 3)
            all_pads = [3, 2, 1, 0]
            active_pads = [p for i, p in enumerate(all_pads) if i != skip]

        stagger = np.random.uniform(50, 200)
        speed_var = np.random.uniform(0.8, 1.2)
        onset_ms = [i * stagger * speed_var for i in range(len(active_pads))]
        duration_ms = [np.random.uniform(50, 150) for _ in active_pads]

        max_end = max(o + d for o, d in zip(onset_ms, duration_ms))
        if max_end > window_ms - step_ms:
            scale = (window_ms - step_ms) / max_end
            onset_ms = [o * scale for o in onset_ms]
            duration_ms = [d * scale for d in duration_ms]

        window = _make_window(onset_ms, duration_ms, active_pads,
                              N_STEPS, sample_rate_hz, baseline)
        feats = extract_features(window, N_STEPS, baselines_arr, thresholds)
        X_list.append(feats)
        y_list.append(1)

    # --- Class 2: single_tap (one pad, one contact, 50-250ms) ---
    for _ in range(n_samples_per_class):
        pad = np.random.randint(0, N_PADS)
        onset = np.random.uniform(100, window_ms - 350)
        dur = np.random.uniform(50, 250)

        window = _make_window([onset], [dur], [pad],
                              N_STEPS, sample_rate_hz, baseline)
        feats = extract_features(window, N_STEPS, baselines_arr, thresholds)
        X_list.append(feats)
        y_list.append(2)

    # --- Class 3: double_tap (same pad, two taps, gap 100-300ms) ---
    for _ in range(n_samples_per_class):
        pad = np.random.randint(0, N_PADS)
        tap1_dur = np.random.uniform(50, 150)
        gap = np.random.uniform(100, 300)
        tap2_dur = np.random.uniform(50, 150)

        tap1_onset = np.random.uniform(50, 300)
        tap2_onset = tap1_onset + tap1_dur + gap

        # Clamp second tap to fit in window
        if tap2_onset + tap2_dur > window_ms - step_ms:
            tap2_onset = window_ms - step_ms - tap2_dur
        if tap2_onset <= tap1_onset + tap1_dur:
            tap2_onset = tap1_onset + tap1_dur + 50

        # Generate two separate windows and OR the touch activity
        window = np.full((N_STEPS, N_PADS), baseline, dtype=np.float32)
        window += np.random.normal(0, 20.0, window.shape).astype(np.float32)

        step_ms_f = 1000.0 / sample_rate_hz
        for onset_t, dur_t in [(tap1_onset, tap1_dur), (tap2_onset, tap2_dur)]:
            ts = int(onset_t / step_ms_f)
            te = int((onset_t + dur_t) / step_ms_f)
            ts = max(0, min(ts, N_STEPS))
            te = max(ts, min(te, N_STEPS))
            if ts < te:
                touch_val = np.random.uniform(300, 800)
                window[ts:te, pad] = touch_val

        window = np.clip(window, 50, 2200).astype(np.float32)
        feats = extract_features(window, N_STEPS, baselines_arr, thresholds)
        X_list.append(feats)
        y_list.append(3)

    # --- Class 4: long_press (1-2 pads held 600-1400ms) ---
    for _ in range(n_samples_per_class):
        n_pads = np.random.choice([1, 2], p=[0.70, 0.30])
        pads = np.random.choice(N_PADS, size=n_pads, replace=False).tolist()

        dur = np.random.uniform(600, 1400)
        onset = np.random.uniform(20, max(40, window_ms - dur - 20))

        onset_ms_list = [onset] * n_pads
        duration_ms_list = [dur] * n_pads

        window = _make_window(onset_ms_list, duration_ms_list, pads,
                              N_STEPS, sample_rate_hz, baseline)
        feats = extract_features(window, N_STEPS, baselines_arr, thresholds)
        X_list.append(feats)
        y_list.append(4)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X, y, epochs=1000, learning_rate=0.1):
    """Train the 2-layer network: Dense(20, 32) + ReLU + Dense(32, 5)."""

    layer1 = DenseLayer(20, 32)
    layer2 = DenseLayer(32, 5)

    print("Training model...")
    print(f"Layer 1: {layer1.weights.shape} weights + {layer1.bias.shape} bias")
    print(f"Layer 2: {layer2.weights.shape} weights + {layer2.bias.shape} bias")

    # Dense(20,32): 20*32 + 32 = 672 params
    # Dense(32,5):  32*5  + 5  = 165 params
    total_params = (20 * 32 + 32) + (32 * 5 + 5)
    print(f"Total parameters: {total_params}")

    n_classes = 5
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

        if (epoch + 1) % 100 == 0:
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


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

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
    from the training data.  Required for multi-input layers (20 inputs)
    where the naive accumulated scale (input_scale * weight_scale) causes
    INT8 saturation.

    Returns:
        o1_scale: scale that maps layer1 post-ReLU activations into [0, 127]
        o2_scale: scale for layer2 output covering the full INT8 range
        o2_zp:    zero point for layer2 output
    """
    # Collect layer 1 activations
    a1 = relu(layer1.forward(X_norm))
    a1_max = float(np.max(a1))
    # Map [0, a1_max] to INT8 [0, 127]
    o1_scale = a1_max / 127.0 if a1_max > 0.0 else 1.0

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


def quantize_model(layer1, layer2, X_norm, input_scale=1.0 / 127.0):
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

    n_classes = 5
    n_hidden = 32
    n_inputs = 20

    correct = 0
    class_correct = np.zeros(n_classes, dtype=int)
    class_total = np.zeros(n_classes, dtype=int)

    for xi in range(len(X_norm)):
        # Quantize 20 normalized input features to INT8
        inp = np.clip(np.round(X_norm[xi] / input_scale), -128, 127).astype(np.int8)

        # Layer 1: Dense(20, 32) + ReLU
        hidden = np.zeros(n_hidden, dtype=np.int32)
        for j in range(n_hidden):
            acc = int(q_b1[0, j])
            for i in range(n_inputs):
                acc += int(inp[i]) * int(q_w1[i, j])
            float_val = float(acc) * input_scale * w1_scale
            req = int(round(float_val / o1_scale)) + o1_zp
            req = max(o1_zp, req)   # ReLU in quantized space
            req = min(127, max(-128, req))
            hidden[j] = req

        # Layer 2: Dense(32, 5)
        out = np.zeros(n_classes, dtype=np.int8)
        for j in range(n_classes):
            acc = int(q_b2[0, j])
            for i in range(n_hidden):
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

    class_names = ['swipe_right', 'swipe_left', 'single_tap', 'double_tap', 'long_press']
    for cls_idx, cls_name in enumerate(class_names):
        if class_total[cls_idx] > 0:
            cls_acc = class_correct[cls_idx] / class_total[cls_idx]
            print(f"  Class {cls_idx} ({cls_name}): {cls_acc:.4f} ({class_total[cls_idx]} samples)")

    return accuracy


# ---------------------------------------------------------------------------
# Export to C Header
# ---------------------------------------------------------------------------

def format_array_as_c(arr, values_per_line=12):
    """Format numpy array as C array initializer."""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i + values_per_line]
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

    header = f"""// Auto-generated by tools/train_touch_gesture_classifier.py
// Model: 2-layer dense network for ESP32 touch gesture classification
// Architecture: Dense(20, 32) + ReLU + Dense(32, 5)
// Input: 20 float features from 1.5s touch window (75 steps at 50Hz, 4 pads)
// Touch pads: GPIO 4 (pad0), GPIO 12 (pad1), GPIO 14 (pad2), GPIO 27 (pad3)
// Total parameters: {(20 * 32 + 32) + (32 * 5 + 5)}
// Quantized model size: {size_bytes} bytes
// Quantization: calibrated INT8 PTQ (output scales from activation statistics)

#ifndef BLITZED_MODEL_WEIGHTS_H
#define BLITZED_MODEL_WEIGHTS_H

#include <stdint.h>

// Layer 1: Dense(20, 32) + ReLU
#define LAYER1_INPUT_SIZE  20
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

// Layer 2: Dense(32, 5)
#define LAYER2_INPUT_SIZE  32
#define LAYER2_OUTPUT_SIZE 5

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
// Features are pre-normalized to [0, 1]; divide by INPUT_NORM_FACTOR (1.0) before quantizing
#define INPUT_SCALE        {quantized['input_scale']:.10f}f
#define INPUT_ZERO_POINT   {quantized['input_zero_point']}
#define INPUT_NORM_FACTOR  1.0f

// Model dimensions
#define NUM_FEATURES  20
#define NUM_CLASSES   5

// Feature layout (per-pad features repeat for pads 0-3):
//   pad*4+0: touch_duration   (fraction of window touched, [0,1])
//   pad*4+1: touch_centroid   (weighted avg time position of touch, [0,1])
//   pad*4+2: mean_intensity   (avg (baseline-reading)/baseline during touch, [0,1])
//   pad*4+3: touch_count      (distinct touch-release transitions / 5, [0,1])
// Global features:
//   [16]: active_pin_count    (pads touched at all / 4, [0,1])
//   [17]: spatial_sweep       (pad_idx vs centroid correlation mapped to [0,1])
//   [18]: total_duration      (sum of per-pad durations / 4, [0,1])
//   [19]: inter_onset_interval (mean gap between pad onset times / 1.0, [0,1])

// Class labels
static const char* class_labels[NUM_CLASSES] = {{
    "swipe_right",
    "swipe_left",
    "single_tap",
    "double_tap",
    "long_press"
}};

// Touch pad GPIO mapping
#define TOUCH_PAD_GPIO_0   4
#define TOUCH_PAD_GPIO_1  12
#define TOUCH_PAD_GPIO_2  14
#define TOUCH_PAD_GPIO_3  27

// Calibration: threshold = 70% of baseline (touch causes reading to DROP)
#define TOUCH_THRESHOLD_RATIO 0.70f

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
    print(f"  Weights: min={layer1.weights.min():.4f}, max={layer1.weights.max():.4f}, "
          f"mean={layer1.weights.mean():.4f}")
    print(f"  Bias: min={layer1.bias.min():.4f}, max={layer1.bias.max():.4f}")

    print(f"\nLayer 2 (float32):")
    print(f"  Weights: min={layer2.weights.min():.4f}, max={layer2.weights.max():.4f}, "
          f"mean={layer2.weights.mean():.4f}")
    print(f"  Bias: min={layer2.bias.min():.4f}, max={layer2.bias.max():.4f}")

    print(f"\nLayer 1 (int8):")
    print(f"  Weights: min={quantized['layer1']['weights'].min()}, "
          f"max={quantized['layer1']['weights'].max()}, "
          f"mean={quantized['layer1']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer1']['bias'].min()}, "
          f"max={quantized['layer1']['bias'].max()}")

    print(f"\nLayer 2 (int8):")
    print(f"  Weights: min={quantized['layer2']['weights'].min()}, "
          f"max={quantized['layer2']['weights'].max()}, "
          f"mean={quantized['layer2']['weights'].mean():.2f}")
    print(f"  Bias: min={quantized['layer2']['bias'].min()}, "
          f"max={quantized['layer2']['bias'].max()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main training pipeline."""

    print("=" * 60)
    print("ESP32 Touch Gesture Classifier Training")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    X, y = generate_training_data(n_samples_per_class=2000)
    print(f"Generated {len(X)} samples ({len(np.unique(y))} classes)")
    class_names = ['swipe_right', 'swipe_left', 'single_tap', 'double_tap', 'long_press']
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_idx}={cls_name}: {np.sum(y == cls_idx)}")

    print(f"\nFeature shape: {X.shape}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Features already in [0, 1] — INPUT_NORM_FACTOR = 1.0")

    # Train model
    # Features are already in [0, 1] so no normalization factor is needed.
    # Critical: INPUT_SCALE = 1/127 maps the full [0, 1] float range to
    # INT8 [0, 127].  Using 1/255 would clip at ~0.498 and destroy
    # high-value features (e.g. long_press durations near 1.0).
    X_normalized = X   # INPUT_NORM_FACTOR = 1.0
    layer1, layer2 = train_model(X_normalized, y, epochs=1000, learning_rate=0.1)

    # Quantize model with calibrated output scales
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
        project_root, 'esp32_demo', 'touch_gesture', 'main',
        'blitzed_model_weights.h'
    )
    binary_path = os.path.join(
        project_root, 'esp32_demo', 'touch_gesture', 'main',
        'blitzed_model_weights.bin'
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    model_size = export_to_c_header(quantized, header_path)
    export_binary_weights(quantized, binary_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Wire 4 touch pads to ESP32: GPIO 4, 12, 14, 27")
    print(f"     (bare wire or copper foil ~2-5cm works as touch pad)")
    print(f"  2. Flash data-collection mode to collect real gesture data:")
    print(f"     cd esp32_demo/touch_gesture && idf.py build flash monitor")
    print(f"     (set MODE_DATA_COLLECTION 1 in main/main.c)")
    print(f"  3. Retrain on real data by replacing synthetic samples")
    print(f"  4. Switch to inference mode (set MODE_DATA_COLLECTION 0) and reflash")
    print(f"\nModel files:")
    print(f"  - {header_path}")
    print(f"  - {binary_path}")


if __name__ == '__main__':
    main()
