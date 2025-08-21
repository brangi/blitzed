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

//! Model format converters and importers

pub mod onnx;
pub mod pytorch;
pub mod tensorflow;

use crate::{BlitzedError, Model, Result};
use std::path::Path;

/// Trait for model format converters
pub trait ModelConverter {
    /// Load model from file path
    fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<Model>;
    
    /// Save model to file path
    fn save_model<P: AsRef<Path>>(&self, model: &Model, path: P) -> Result<()>;
    
    /// Get supported file extensions
    fn supported_extensions(&self) -> &'static [&'static str];
}

/// Universal model converter that delegates to specific converters
pub struct UniversalConverter {
    onnx: onnx::OnnxConverter,
    pytorch: pytorch::PyTorchConverter,
    tensorflow: tensorflow::TensorFlowConverter,
}

impl UniversalConverter {
    pub fn new() -> Self {
        Self {
            onnx: onnx::OnnxConverter::new(),
            pytorch: pytorch::PyTorchConverter::new(),
            tensorflow: tensorflow::TensorFlowConverter::new(),
        }
    }

    pub fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<Model> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| BlitzedError::UnsupportedFormat {
                format: "unknown".to_string(),
            })?;

        match extension {
            "onnx" => self.onnx.load_model(path),
            "pt" | "pth" => self.pytorch.load_model(path),
            "pb" => self.tensorflow.load_model(path),
            _ => Err(BlitzedError::UnsupportedFormat {
                format: extension.to_string(),
            }),
        }
    }
}

impl Default for UniversalConverter {
    fn default() -> Self {
        Self::new()
    }
}