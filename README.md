# Blitzed

<div align="center">

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/brangi/blitzed)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/brangi/blitzed/workflows/CI/badge.svg)](https://github.com/brangi/blitzed/actions)
[![codecov](https://codecov.io/gh/brangi/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangi/blitzed)

**Edge AI optimization framework for deploying neural networks on microcontrollers.**

</div>

---

## What It Does

Blitzed takes trained neural networks, quantizes them to INT8, and generates complete, compilable ESP-IDF projects that run real inference on ESP32 microcontrollers. No TensorFlow Lite, no runtime interpreters — just C code with embedded weights.

## What Works Today

- **Weight extraction and quantization**: Extract weights from trained models, quantize to INT8 with calibrated scales
- **ESP32 code generation**: `Esp32CodeGen::generate_from_weights()` produces complete ESP-IDF projects with real inference kernels
- **4 complete ESP32 demo projects** (trained models with INT8 weights baked in):
  - **Hall sensor classifier** — Built-in hall effect sensor, 3 classes, 83 params, 140 bytes
  - **Temperature anomaly detector** — Internal temp sensor, 4 classes, 84 params, 160 bytes
  - **Vibration classifier** — MPU6050 I2C accelerometer, 4 classes, 164 params, 368 bytes
  - **Predictive maintenance** — Temp + accelerometer fusion, 4 classes, 292 params, 400 bytes
- **Tensor operations**: Matrix multiply, Conv2D, pooling, batch norm, activations (ReLU, sigmoid, tanh)
- **Quantization math**: INT8/INT4 scale and zero-point calculation, calibration, per-channel support
- **Training scripts**: NumPy-only (no PyTorch/TF dependency) scripts to train tiny classifiers
- **413 tests passing** across all modules
- **CI pipeline**: Multi-platform (Linux, Windows, macOS), clippy, fmt, coverage, security audit

## What's Not Done Yet

- **On-device verification**: Demos need to be flashed to real ESP32 hardware
- **Measured benchmarks**: No real latency/memory numbers yet (need hardware)
- **Generic C codegen**: `CCodeGen` generates structural templates, not runnable inference
- **Deployment artifacts**: `build_ready: false` — structural descriptions only

## ESP32 Demos

Each demo in `esp32_demo/` is a standalone ESP-IDF project with:
- Pre-trained INT8 weights in C header files
- Real inference kernels (not placeholders)
- Sensor reading and classification logic
- Serial output for results

```bash
# To build and flash (requires ESP-IDF v5.x and ESP32 board):
cd esp32_demo/hall_classifier
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

The hall classifier and temp anomaly demos work with the ESP32's built-in sensors. The vibration and predictive maintenance demos require an MPU6050 wired to GPIO21 (SDA) / GPIO22 (SCL).

## Quick Start

```bash
# Build Rust core
cargo build -p blitzed-core --no-default-features --features "quantization,hardware-targets"

# Run tests
cargo test -p blitzed-core --no-default-features --features "quantization,hardware-targets"

# Build with all features
cargo build -p blitzed-core --no-default-features --features "quantization,pruning,distillation,hardware-targets"

# Train a model (NumPy only, no GPU needed)
python tools/train_hall_sensor_classifier.py
```

## Architecture

```
blitzed-core/          Rust optimization engine
  src/
    optimization/      Quantization, pruning, distillation
    targets/           ESP32, STM32, Arduino, Raspberry Pi constraints
    converters/        ONNX, PyTorch, TensorFlow model loading
    codegen/           C/C++ code generation for embedded targets
    tensor_ops.rs      Matrix math, convolutions, activations
    inference.rs       Host-side inference graph execution
    model.rs           Model representation, weight extraction, quantization
    config.rs          Configuration and target presets
esp32_demo/            4 standalone ESP-IDF demo projects
tools/                 NumPy training scripts for demo models
python/                Python package (CLI, high-level API)
blitzed-py/            PyO3 Rust-Python bindings
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `onnx` | yes | ONNX model format support |
| `pytorch` | yes | PyTorch model format support (requires libtorch) |
| `quantization` | yes | INT8/INT4/binary quantization |
| `pruning` | no | Network pruning |
| `distillation` | no | Knowledge distillation |
| `hardware-targets` | no | Hardware-specific code generation |

> **Note**: `onnx` and `pytorch` default features require native libraries. For most development, use `--no-default-features --features "quantization,hardware-targets"`.

## Supported Platforms

| Target | Status | Notes |
|--------|--------|-------|
| ESP32 | Working | Real codegen with `generate_from_weights()`, 4 demo projects |
| STM32 | Specs defined | Hardware constraints defined, codegen not yet implemented |
| Arduino | Specs defined | Hardware constraints defined, codegen not yet implemented |
| Raspberry Pi | Specs defined | Hardware constraints defined, codegen not yet implemented |

## Quantization Notes

Lessons learned from training and quantizing tiny models for ESP32:

- **Input scale matters**: `1/255` clips at ±0.498 and destroys features. Use `1/127` for full [0,1] range.
- **Output scale calibration**: Naive `input_scale * weight_scale` fails for multi-input models. Calibrate from actual activation ranges.
- **Normalize inputs before training**: Divide by sensor max range to prevent gradient explosion.
- **Tiny INT8 models have limits**: ~86% accuracy on vibration classification with axis-rotation invariance is realistic for sub-200 parameter INT8 models.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

## Author

**Gibran Rodriguez** - [brangi000@gmail.com](mailto:brangi000@gmail.com)
