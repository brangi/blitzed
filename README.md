# Blitzed

<div align="center">

[![CI](https://github.com/brangirod/blitzed/workflows/CI/badge.svg)](https://github.com/brangirod/blitzed/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/brangirod/blitzed#license)
[![Rust Version](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://forge.rust-lang.org/)
[![codecov](https://codecov.io/gh/brangirod/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangirod/blitzed)

**High-performance edge AI optimization framework for deploying machine learning models on edge devices.**

[Features](#features) •
[Supported Platforms](#supported-platforms) •
[Quick Start](#quick-start)

</div>

---

## Features

- **Hardware-Aware Optimization**: Tailored optimization strategies for ESP32, Raspberry Pi, Arduino, and STM32
- **Advanced Quantization**: INT8/INT4 quantization with minimal accuracy loss
- **ONNX Model Support**: Comprehensive analysis and optimization of ONNX models
- **Performance Profiling**: Real-time benchmarking and optimization tracking
- **Production Ready**: Enterprise-grade code quality with comprehensive test suite
- **Python Bindings**: Python API in development (see blitzed-py directory)

## Supported Platforms

| Platform | Status | Variants | Optimization Features |
|----------|---------|----------|----------------------|
| **ESP32** | Supported | ESP32, S2, S3, C3 | Hardware-specific tuning, power optimization |
| **Raspberry Pi** | Supported | Zero, Zero 2, 3B, 4B, Pi 5 | Multi-core optimization, GPU acceleration |
| **Arduino** | Supported | Uno, Nano, Mega | Memory-constrained optimization |
| **STM32** | Supported | F4, F7, H7 series | Real-time optimization |

## Quick Start

```bash
# Build from source
git clone https://github.com/brangirod/blitzed
cd blitzed
cargo build --release

# Run tests
cargo test
```

```python
# Python bindings are in development
# See blitzed-py/ directory for current implementation status
```

## CI Status & Test Results

| Build Status | Coverage | Tests | Security Audit |
|--------------|----------|--------|----------------|
| [![CI](https://github.com/brangirod/blitzed/workflows/CI/badge.svg)](https://github.com/brangirod/blitzed/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/brangirod/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangirod/blitzed) | ![Tests](https://img.shields.io/github/actions/workflow/status/brangirod/blitzed/ci.yml?label=36%20tests) | ![Security](https://img.shields.io/github/actions/workflow/status/brangirod/blitzed/ci.yml?label=security%20audit) |

**Test Run:**
[View detailed test results →](https://github.com/brangirod/blitzed/actions/workflows/ci.yml)

---

**Author:** Gibran Rodriguez <brangi000@gmail.com>