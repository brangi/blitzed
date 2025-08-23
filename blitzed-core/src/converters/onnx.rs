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

//! ONNX model converter and loader

use super::ModelConverter;
use crate::{BlitzedError, Model, Result};
use std::path::Path;

/// ONNX model converter
pub struct OnnxConverter;

impl OnnxConverter {
    pub fn new() -> Self {
        Self
    }
}

impl ModelConverter for OnnxConverter {
    fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<Model> {
        Model::load(path)
    }

    fn save_model<P: AsRef<Path>>(&self, _model: &Model, _path: P) -> Result<()> {
        // TODO: Implement ONNX model saving
        Err(BlitzedError::Internal(
            "ONNX saving not implemented".to_string(),
        ))
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["onnx"]
    }
}

impl Default for OnnxConverter {
    fn default() -> Self {
        Self::new()
    }
}
