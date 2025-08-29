// Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Blitzed Core - High-performance edge AI optimization engine
//!
//! This crate provides the core functionality for optimizing and deploying
//! machine learning models on edge devices. It includes implementations for:
//!
//! - Model compression techniques (quantization, pruning, knowledge distillation)
//! - Hardware-specific optimizations and code generation
//! - Cross-platform model conversion and deployment
//! - Performance profiling and accuracy validation

pub mod benchmarking;
pub mod codegen;
pub mod config;
pub mod converters;
pub mod deployment;
pub mod error;
pub mod model;
pub mod onnx_analyzer;
pub mod optimization;
pub mod profiler;
pub mod targets;
pub mod validation;

pub use benchmarking::suite::BenchmarkSuite;
pub use config::Config;
pub use deployment::{DeploymentValidationConfig, HardwareDeploymentValidator};
pub use error::{BlitzedError, Result};
pub use model::{Model, ModelFormat, ModelInfo};
pub use optimization::{OptimizationConfig, Optimizer};
pub use validation::{CrossFormatValidator, ValidationConfig};

/// Core version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the Blitzed core library with logging
pub fn init() -> Result<()> {
    // Try to initialize logger, but don't fail if already initialized
    let _ = env_logger::try_init();
    log::info!("Blitzed Core v{} initialized", VERSION);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
