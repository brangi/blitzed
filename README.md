# Blitzed

<div align="center">

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/brangi/blitzed)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/brangi/blitzed/workflows/CI/badge.svg)](https://github.com/brangi/blitzed/actions)
[![codecov](https://codecov.io/gh/brangi/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangi/blitzed)

**Train a model. Quantize to INT8. Generate C code. Flash to ESP32. Done.**

</div>

---

Blitzed takes trained neural networks, quantizes them to INT8, and generates complete ESP-IDF projects that run real inference on ESP32 microcontrollers. No TensorFlow Lite, no runtime interpreters — just C code with embedded weights.

## Measured Hardware Results

Tested on ESP32-WROOM-32 (240MHz, 512KB SRAM) with ESP-IDF v5.3:

| Demo | Sensor | Inference | Throughput | Model Size | Heap Used |
|------|--------|-----------|------------|------------|-----------|
| Hall classifier | Built-in hall effect | **17 us** | 58,823/sec | 140 bytes | 448 bytes |
| Touch gesture | Built-in capacitive touch | **157 us** | 6,369/sec | 948 bytes | ~1 KB |

These are real numbers from a real ESP32, not estimates.

## ESP32 Demos

### Touch Gesture Recognition (verified on hardware)

Classifies 5 gesture types from temporal patterns across 4 capacitive touch pins. This demo genuinely requires ML — timing patterns, spatial order, and per-person variation create a feature space that can't be replicated with if/else rules.

- **Gestures**: swipe right, swipe left, single tap, double tap, long press
- **Architecture**: Dense(20, 32) + ReLU + Dense(32, 5) — 837 parameters
- **Accuracy**: 94.5% INT8 quantized (96.0% float)
- **Touch pins**: GPIO 4, 12, 14, 27 (built-in capacitive, no external hardware)
- **Feature extraction**: 20 temporal/spatial features from 1.5s window at 50Hz
- **Includes data collection mode** for capturing real gesture training data

```bash
# Train the model (NumPy only, no GPU)
python tools/train_touch_gesture_classifier.py

# Build and flash
source ~/esp/esp-idf-v5.3/export.sh
cd esp32_demo/touch_gesture
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash monitor
```

### Hall Sensor Classifier (verified on hardware)

Classifies magnetic field readings from the ESP32's built-in hall effect sensor.

- **Architecture**: Dense(1, 16) + ReLU + Dense(16, 3) — 83 parameters
- **Inference**: 17us mean, 113us max
- **No external hardware needed**

```bash
cd esp32_demo/hall_classifier
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash monitor
```

### Vibration Classifier (needs MPU6050)

Classifies vibration patterns from an MPU6050 accelerometer: normal, imbalance, misalignment, bearing fault.

- **Architecture**: Dense(3, 32) + ReLU + Dense(32, 4) — 164 parameters, 368 bytes
- **Wiring**: MPU6050 I2C on GPIO21 (SDA) / GPIO22 (SCL)
- **Not yet tested on hardware** — needs MPU6050 sensor

### Predictive Maintenance (needs MPU6050)

Multi-sensor fusion: temperature + accelerometer data for equipment health classification.

- **Architecture**: Dense(4, 32) + ReLU + Dense(32, 4) — 292 parameters, 400 bytes
- **Not yet tested on hardware**

### Temperature Anomaly (not yet flashed)

Classifies ESP32 internal die temperature into normal/cold/hot/critical ranges. Note: the internal sensor measures chip temperature, not ambient.

- **Architecture**: Dense(1, 16) + ReLU + Dense(16, 4) — 84 parameters, 160 bytes

## How It Works

1. **Train** a tiny classifier with NumPy (no PyTorch/TF needed)
2. **Quantize** weights to INT8 with calibrated output scales
3. **Export** to a C header with weights baked in as const arrays
4. **Build** a standalone ESP-IDF project with the inference kernel
5. **Flash** to ESP32 and run real-time inference

The training scripts in `tools/` handle steps 1-3. Each produces a `blitzed_model_weights.h` that plugs directly into the ESP-IDF project.

## Building

```bash
# Rust core
cargo build -p blitzed-core --no-default-features --features "quantization,hardware-targets"

# Run tests (413 passing)
cargo test -p blitzed-core --no-default-features --features "quantization,hardware-targets"

# ESP32 demos (requires ESP-IDF v5.x)
source ~/esp/esp-idf-v5.3/export.sh
cd esp32_demo/hall_classifier
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash monitor
```

> `onnx` and `pytorch` default features require native libraries that may not be available on all platforms. Use `--no-default-features` for reliable builds.

## Project Structure

```
blitzed-core/          Rust optimization engine
  src/
    optimization/      Quantization, pruning, distillation
    targets/           Hardware constraint definitions
    converters/        Model format loading (ONNX, PyTorch, TF)
    codegen/           C code generation for embedded targets
    model.rs           Weight extraction and quantization
    tensor_ops.rs      Matrix math, convolutions, activations
esp32_demo/
    hall_classifier/   Built-in hall sensor, 3 classes (verified)
    touch_gesture/     Capacitive touch, 5 gestures (verified)
    temp_anomaly/      Internal temp sensor, 4 classes
    vibration_classifier/  MPU6050 accelerometer, 4 classes
    predictive_maintenance/  Multi-sensor fusion, 4 classes
tools/                 NumPy training scripts (one per demo)
python/                Python package (CLI, converters)
blitzed-py/            PyO3 Rust-Python bindings
```

## What's Not Done

- STM32, Arduino, Raspberry Pi codegen (hardware constraints defined, no code generation yet)
- Generic C codegen produces structural templates, not runnable inference
- Deployment artifact generators output structural descriptions (`build_ready: false`)
- Vibration and predictive maintenance demos need hardware testing with MPU6050
- No Conv1D/LSTM/temporal layers in inference kernel yet (dense layers only)

## INT8 Quantization

All models use calibrated post-training quantization:
- Weights quantized to INT8 with per-layer symmetric scales
- INT32 accumulators prevent overflow during inference
- Output scales calibrated from training data activation statistics (not naive scale multiplication)
- Input normalization to [0, 1] before quantization (critical for training stability)

## Requirements

- **Rust 1.70+** for the core library
- **Python 3.8+** with NumPy for training scripts
- **ESP-IDF v5.x** for building and flashing ESP32 demos
- **ESP32-WROOM-32** dev board (tested with 30-pin variant)

## License

Apache License 2.0 or MIT License, at your option.

## Author

**Gibran Rodriguez** - [brangi000@gmail.com](mailto:brangi000@gmail.com)
