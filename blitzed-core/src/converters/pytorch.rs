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

//! PyTorch model converter

use super::ModelConverter;
use crate::{BlitzedError, Model, Result};
use std::path::Path;

/// PyTorch model converter
pub struct PyTorchConverter;

impl PyTorchConverter {
    pub fn new() -> Self {
        Self
    }
}

impl ModelConverter for PyTorchConverter {
    fn load_model<P: AsRef<Path>>(&self, _path: P) -> Result<Model> {
        // TODO: Implement PyTorch model loading
        Err(BlitzedError::UnsupportedFormat {
            format: "PyTorch".to_string(),
        })
    }

    fn save_model<P: AsRef<Path>>(&self, _model: &Model, _path: P) -> Result<()> {
        // TODO: Implement PyTorch model saving
        Err(BlitzedError::Internal(
            "PyTorch saving not implemented".to_string(),
        ))
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["pt", "pth"]
    }
}

impl Default for PyTorchConverter {
    fn default() -> Self {
        Self::new()
    }
}
