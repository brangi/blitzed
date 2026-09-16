# AGENTS.md

## Project overview

Blitzed is an edge-AI optimization and deployment framework. The main implementation is a Rust workspace that quantizes small neural networks and generates embedded inference code. The repository also contains Python APIs/tools and standalone ESP-IDF v5.x demos.

## Repository map

- `blitzed-core/` — primary Rust crate.
  - `src/model.rs` — model metadata, extracted weights, quantized weight structures, and weight quantization.
  - `src/inference.rs` — general-purpose native inference graph/runtime and memory pool.
  - `src/tensor_ops.rs` — tensor operations and activations.
  - `src/optimization/` — quantization, pruning, distillation, and optimizer orchestration.
  - `src/converters/` — ONNX, PyTorch, TensorFlow, and base converter paths.
  - `src/codegen/` — C and target-specific generators; ESP32 generation can emit a complete ESP-IDF project from quantized weights.
  - `src/targets/` — hardware constraints and optimization strategies for ESP32, STM32, Arduino, Raspberry Pi, and mobile targets.
  - `src/deployment.rs`, `src/validation.rs`, `src/simulation/`, `src/benchmarking/` — deployment validation, cross-format checks, simulation, and performance tooling.
- `blitzed-py/` — PyO3 bindings crate.
- `python/blitzed/` — pure-Python package, CLI, optimization helpers, and converters.
- `tools/` — NumPy-only training scripts for the ESP32 demo models.
- `esp32_demo/` — standalone ESP-IDF projects: `hall_classifier`, `touch_gesture`, `temp_anomaly`, `vibration_classifier`, and `predictive_maintenance`.
- `tests/` and `blitzed-core/tests/` — Rust integration tests; Python tests are under `python/blitzed/tests/`.
- `README.md` — user-facing status, measured hardware results, and demo instructions.
- `CLAUDE.md` — existing general build/test/convention notes; keep this file and this document consistent when changing workflows.

## Current implementation boundaries

- ESP32 hall-classifier and touch-gesture demos are documented as hardware-verified in `README.md`.
- Vibration and predictive-maintenance demos require an MPU6050 and are not hardware-verified.
- Temperature anomaly demo is not yet flashed/verified.
- The ESP32 deployment path is the most complete code-generation path and uses dense INT8 inference with embedded weights.
- Generic C code generation and deployment artifact generation remain structural/template-oriented unless explicitly verified otherwise.
- The current embedded inference kernel primarily supports dense layers; Conv2d is represented in the model types but is not implemented in the ESP32 generated inference path.
- Do not claim hardware support or measured performance without checking the README and the relevant demo source.

## Build and test

Use the repository's feature-light commands for reliable local Rust builds because default features include native ONNX and PyTorch dependencies:

```bash
cargo build -p blitzed-core --no-default-features --features "quantization,hardware-targets"
cargo test -p blitzed-core --no-default-features --features "quantization,hardware-targets"
```

The CI-equivalent full core validation uses:

```bash
cargo fmt --all -- --check
cargo clippy -p blitzed-core --all-targets --no-default-features --features quantization,pruning,distillation,hardware-targets -- -D warnings
cargo test --no-default-features --features quantization,pruning,distillation,hardware-targets
cargo test --doc -p blitzed-core --no-default-features --features quantization,pruning,distillation,hardware-targets
```

Python checks and tests:

```bash
pytest python/
black python/ --check
isort python/ --check-only --profile black
mypy python/
```

ESP32 builds require ESP-IDF v5.x to be sourced first:

```bash
source ~/esp/esp-idf-v5.3/export.sh
cd esp32_demo/<demo>
idf.py build
idf.py -p <serial-device> flash monitor
```

Do not assume ESP-IDF, a board, sensors, or optional native model libraries are available. When hardware is unavailable, validate generated files and host-side tests instead.

## Rust conventions

- Rust edition 2021; use `rustfmt.toml` and four-space indentation.
- Preserve the Apache 2.0 license header used by Rust source files.
- Use `snake_case` for functions/variables and `PascalCase` for types.
- Use `BlitzedError` and the crate `Result<T>` alias for fallible APIs; propagate errors with `?`.
- Keep feature-gated native integrations behind the existing Cargo features (`onnx`, `pytorch`, `quantization`, `pruning`, `distillation`, `hardware-targets`).
- Prefer extending existing modules and public types over introducing parallel abstractions.
- Add or update unit/integration tests for behavior changes, especially quantization scales, generated C text, model metadata, and target constraints.

## Python conventions

- Support Python 3.8+.
- Format with Black (88 columns) and organize imports with isort's Black profile.
- Keep type hints and Google-style docstrings consistent with existing modules.
- Training scripts should remain lightweight and NumPy-based unless a dependency change is intentional and documented.

## ESP32 and generated-code conventions

- Treat files such as `esp32_demo/*/main/blitzed_model_weights.h` as generated artifacts from the corresponding `tools/train_*.py` script; update the source/training flow when changing generated content.
- Keep quantization behavior consistent across Rust quantization, generated C, and checked-in demo inference kernels: INT8 weights, INT32 accumulators, explicit scales/zero points, requantization, saturation, and quantized ReLU.
- Keep ESP-IDF component registration and project layout valid (`CMakeLists.txt`, `main/CMakeLists.txt`, `sdkconfig.defaults`, and `main/` sources).
- Be careful with sensor-specific input normalization and feature extraction; these are part of each model's contract, not generic inference logic.

## Change workflow

1. Read `README.md`, `CLAUDE.md`, and the relevant module/demo before editing.
2. Make the smallest change that addresses the request and preserve existing conventions.
3. Run focused tests first, then the feature-light Rust test command and relevant Python checks when practical.
4. If hardware or optional native dependencies are unavailable, report that explicitly rather than treating an unrun check as passing.
5. Update documentation when behavior, measured results, supported targets, or current limitations change.
