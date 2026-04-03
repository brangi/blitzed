# Blitzed

<div align="center">

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/brangi/blitzed)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/brangi/blitzed/workflows/CI/badge.svg)](https://github.com/brangi/blitzed/actions)
[![codecov](https://codecov.io/gh/brangi/blitzed/branch/main/graph/badge.svg)](https://codecov.io/gh/brangi/blitzed)

**Ultra-fast edge AI optimization framework for deploying neural networks on microcontrollers.**

</div>

---

## Performance First

Blitzed delivers **measurably superior performance** on resource-constrained microcontrollers:

| Framework | Inference Time | Performance Gain |
|-----------|---------------|-----------------|
| **Blitzed** | **7.0 μs** | **Baseline** |
| TensorFlow Lite | 71.7 μs | 10.2x slower |

*Benchmarked on ESP32-D0WDQ6 with identical neural network models*

## Key Features

- **Ultra-fast inference**: Sub-10 microsecond neural network execution
- **Advanced optimization**: INT8/INT4/binary quantization techniques  
- **Multi-platform**: ESP32, STM32, Arduino, Raspberry Pi deployment
- **Python API**: High-level interface for rapid prototyping
- **Rust core**: Zero-cost abstractions and memory safety
- **Code generation**: Automatic C/C++ deployment code generation
- **Built-in profiling**: Real-time memory and performance analysis

## Supported Platforms

| Target | Status | Testing Status | Deployment |
|--------|--------|---------------|------------|
| **ESP32** | Verified | Tested on ESP32-D0WDQ6 | PlatformIO/ESP-IDF |
| **STM32** | In Development | Untested | STM32CubeIDE |  
| **Arduino** | In Development | Untested | Arduino IDE |
| **Raspberry Pi** | In Development | Untested | Native Linux |

## Prerequisites

- **Rust 1.70+**
- **Python 3.8+** (for Python bindings)
- **ESP-IDF** or **PlatformIO** (for ESP32 deployment)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/brangi/blitzed.git
cd blitzed

# Build Rust core
cargo build --release

# Run tests to verify installation
cargo test --workspace
```

**Python bindings (experimental):**
```bash
# Install Python bindings (development stage)
cd blitzed-py
pip install maturin
maturin develop --release
```

*Note: Python bindings are under active development and may have limited functionality.*

### 2. Python API Usage

```python
import blitzed

# Initialize the library
blitzed.init()

# Check library version
print(f"Blitzed version: {blitzed.VERSION}")

# List supported targets
targets = blitzed.list_targets()
print(f"Supported targets: {targets}")
```

*Note: The Python API is under active development. More functionality will be added as the framework matures.*

### 3. Rust API Usage

```rust
use blitzed_core::{Config, Model, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Blitzed
    blitzed_core::init()?;
    
    // Load model
    let model = Model::load("path/to/model.onnx")?;
    
    // Create ESP32-optimized configuration
    let config = Config::preset("esp32")?;
    
    // Optimize model
    let optimizer = Optimizer::new(config);
    let result = optimizer.optimize(&model)?;
    
    println!("Compression ratio: {:.2}x", result.compression_ratio);
    println!("Estimated speedup: {:.2}x", result.estimated_speedup);
    
    Ok(())
}
```

## ESP32 Deployment

### Hardware Setup

1. **Connect ESP32** to your computer via USB
2. **Install PlatformIO** (recommended) or ESP-IDF
3. **Identify serial port** (usually `/dev/cu.usbserial-*` on macOS)

### Step-by-Step Deployment

```bash
# 1. Generate deployment code
python -c "
import blitzed
result = blitzed.generate_deployment_code(
    'optimized_model.blz',
    'esp32_project/',
    'esp32'
)
print('Generated:', result['implementation_file'])
"

# 2. Create PlatformIO project
cd esp32_project
pio project init --board esp32dev

# 3. Flash to ESP32
pio run --target upload --upload-port /dev/cu.usbserial-0001

# 4. Monitor output
pio device monitor --port /dev/cu.usbserial-0001
```

### Example ESP32 Output
```
I (298) BLITZED: Model loaded: 15.2KB
I (302) BLITZED: Inference time: 7.0μs
I (306) BLITZED: Memory usage: 89.3KB
I (310) BLITZED: Prediction: [0.95, 0.03, 0.02]
```

## Python API Reference

### Core Functions

#### `blitzed.init()`
Initialize the Blitzed library. Call before using other functions.

#### `blitzed.load_model(path: str) -> dict`
Load and analyze a model file. Returns model information including size, shapes, and memory estimates.

**Parameters:**
- `path`: Path to model file (ONNX, TensorFlow, PyTorch)

**Returns:**
```python
{
    "format": "ONNX",
    "model_size_bytes": 1048576,
    "parameter_count": 25000,
    "operations_count": 150,
    "input_shapes": [[1, 3, 224, 224]],
    "output_shapes": [[1, 1000]],
    "estimated_memory_usage": 2097152
}
```

#### `blitzed.optimize_model(input_path: str, output_path: str, config: dict) -> dict`
Apply full optimization pipeline to a model.

**Configuration options:**
```python
config = {
    "target": "esp32",                    # Target platform
    "quantization_type": "int8",          # int8, int4, binary, mixed
    "calibration_dataset_size": 100,      # Samples for calibration
    "symmetric": True,                    # Symmetric quantization
    "per_channel": True,                  # Per-channel quantization  
    "accuracy_threshold": 5.0,            # Max accuracy loss (%)
    "skip_sensitive_layers": True         # Preserve critical layers
}
```

#### `blitzed.profile_model(model_path: str, config: dict) -> dict`
Profile model performance characteristics.

#### `blitzed.generate_deployment_code(model_path: str, output_dir: str, target: str) -> dict`
Generate optimized C/C++ code for target platform.

#### `blitzed.list_targets() -> List[str]`
Get list of supported deployment targets.

### Quantization Functions

#### `blitzed.quantize_model(input_path: str, output_path: str, config: dict) -> str`
Apply quantization-only optimization.

#### `blitzed.estimate_quantization_impact(model_path: str, config: dict) -> dict`
Estimate quantization effects without applying changes.

## Project Structure

```
blitzed/
├── blitzed-core/          # Rust optimization engine
├── blitzed-py/           # Python bindings (PyO3)
├── python/               # Python package structure  
├── tests/                # Rust tests
└── README.md
```

## Performance Benchmarks

### Verified Performance Results (ESP32-D0WDQ6)

Based on actual hardware testing with identical neural network models:

| Framework | Inference Time | Performance Improvement |
|-----------|---------------|------------------------|
| **Blitzed** | **7.0 μs** | **Baseline** |
| TensorFlow Lite | 71.7 μs | 10.2x slower |

### Optimization Capabilities

The framework provides multiple optimization techniques:
- **INT8 Quantization**: Reduces model size and improves inference speed
- **INT4 Quantization**: Further size reduction for extreme resource constraints
- **Pruning**: Removes redundant network connections
- **Knowledge Distillation**: Creates smaller models that maintain accuracy

*Specific compression ratios vary by model architecture and require individual testing.*

## Use Cases

### Industrial IoT
- **Predictive maintenance**: Vibration analysis, temperature monitoring
- **Quality control**: Real-time defect detection 
- **Process optimization**: Sensor fusion and control systems

### Smart Agriculture  
- **Crop monitoring**: Disease detection, growth analysis
- **Environmental control**: Climate optimization, irrigation control
- **Livestock tracking**: Health monitoring, behavior analysis

### Edge Computing
- **Autonomous vehicles**: Object detection, path planning
- **Security systems**: Intrusion detection, facial recognition  
- **Wearables**: Health monitoring, gesture recognition

## Examples

### Basic Quantization
```python
# Quantize model to INT8 for ESP32
config = {
    "quantization_type": "int8",
    "target": "esp32"
}

blitzed.quantize_model(
    "model.onnx", 
    "quantized_model.blz", 
    config
)
```

### Performance Profiling
```python
# Profile model before optimization
metrics = blitzed.profile_model("model.onnx", {"target": "esp32"})
print(f"Estimated inference time: {metrics['estimated_inference_time_ms']}ms")
print(f"Memory usage: {metrics['estimated_memory_usage']} bytes")
```

### Multi-Platform Deployment
```python
# Generate code for multiple targets
targets = blitzed.list_targets()
for target in ["esp32", "stm32", "arduino"]:
    blitzed.generate_deployment_code(
        "optimized_model.blz",
        f"deployment_{target}/", 
        target
    )
```

## Troubleshooting

### Common Issues

**"Model loading failed"**
- Verify model format is supported (ONNX, TensorFlow, PyTorch)
- Check file path and permissions
- Ensure model is not corrupted

**"ESP32 deployment fails"**
- Verify ESP32 is connected and recognized by system
- Check serial port permissions (`sudo usermod -a -G dialout $USER`)
- Ensure sufficient flash memory (model + firmware < 4MB)

**"Quantization accuracy loss too high"**
- Increase calibration dataset size
- Try mixed precision quantization
- Enable `skip_sensitive_layers` option

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/brangi/blitzed.git
cd blitzed

# Install development dependencies
cargo install cargo-watch
pip install maturin pytest

# Run tests
cargo test --workspace
cd blitzed-py && python -m pytest
```

### Testing
```bash
# Run all tests
make test

# Test specific components  
cargo test optimization
python -m pytest tests/test_quantization.py
```

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

## Author

**Gibran Rodriguez** - [brangi000@gmail.com](mailto:brangi000@gmail.com)
