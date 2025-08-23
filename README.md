# Blitzed

<div align="center">

[![CI](https://github.com/brangi/blitzed/actions/workflows/ci.yml/badge.svg)](https://github.com/brangi/blitzed/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/brangi/blitzed)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

**High-performance edge AI optimization framework for deploying machine learning models on edge devices.**

[Features](#features) •
[Quick Start](#quick-start)

</div>

---

## Features

- **Quantization**: INT8 quantization algorithms
- **Hardware Targets**: ESP32 and Raspberry Pi optimization strategies
- **Rust Core**: Core optimization engine written in Rust
- **Python Bindings**: Python API in development (see blitzed-py directory)

## Quick Start

```bash
# Build from source
git clone https://github.com/brangi/blitzed
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
| [![CI](https://github.com/brangi/blitzed/actions/workflows/ci.yml/badge.svg)](https://github.com/brangi/blitzed/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/brangi/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangi/blitzed) | [![Tests](https://github.com/brangi/blitzed/actions/workflows/ci.yml/badge.svg?label=tests)](https://github.com/brangi/blitzed/actions/workflows/ci.yml) | [![Security](https://github.com/brangi/blitzed/actions/workflows/ci.yml/badge.svg?label=security)](https://github.com/brangi/blitzed/actions/workflows/ci.yml) |

**Test Run:** [View detailed test results →](https://github.com/brangi/blitzed/actions/workflows/ci.yml)

---

**Author:** Gibran Rodriguez <brangi000@gmail.com>