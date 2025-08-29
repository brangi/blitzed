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

//! Error handling for Blitzed Core

use thiserror::Error;

/// Result type alias for Blitzed operations
pub type Result<T> = std::result::Result<T, BlitzedError>;

/// Comprehensive error types for Blitzed operations
#[derive(Error, Debug)]
pub enum BlitzedError {
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Model format not supported: {format}")]
    UnsupportedFormat { format: String },

    #[error("Optimization failed: {reason}")]
    OptimizationFailed { reason: String },

    #[error("Tensor operation error: {message}")]
    TensorError { message: String },

    #[error("Target hardware not supported: {target}")]
    UnsupportedTarget { target: String },

    #[error("Code generation failed: {0}")]
    CodeGeneration(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Hardware constraint violation: {constraint}")]
    HardwareConstraint { constraint: String },

    #[error("Accuracy threshold exceeded: expected {threshold}%, got {actual}%")]
    AccuracyThreshold { threshold: f32, actual: f32 },

    #[error("Memory limit exceeded: {used} bytes > {limit} bytes")]
    MemoryLimit { used: usize, limit: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    #[error("Internal error: {0}")]
    Internal(String),

    // Benchmarking-specific errors
    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Framework not supported: {framework}")]
    UnsupportedFramework { framework: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Conversion failed from {from} to {to}: {error}")]
    ConversionFailed {
        from: String,
        to: String,
        error: String,
    },

    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("System error: {message}")]
    SystemError { message: String },

    #[error("Report generation error: {message}")]
    ReportGenerationError { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for BlitzedError {
    fn from(err: ort::Error) -> Self {
        BlitzedError::OnnxRuntime(err.to_string())
    }
}

impl From<candle_core::Error> for BlitzedError {
    fn from(err: candle_core::Error) -> Self {
        BlitzedError::Internal(format!("Candle error: {}", err))
    }
}
