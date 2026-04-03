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

//! TensorFlow model converter

use super::ModelConverter;
use crate::{BlitzedError, Model, Result};
use std::path::Path;

/// TensorFlow model converter
pub struct TensorFlowConverter;

impl TensorFlowConverter {
    pub fn new() -> Self {
        Self
    }
}

impl ModelConverter for TensorFlowConverter {
    fn load_model<P: AsRef<Path>>(&self, _path: P) -> Result<Model> {
        // TODO: Implement TensorFlow model loading
        Err(BlitzedError::UnsupportedFormat {
            format: "TensorFlow".to_string(),
        })
    }

    fn save_model<P: AsRef<Path>>(&self, _model: &Model, _path: P) -> Result<()> {
        // TODO: Implement TensorFlow model saving
        Err(BlitzedError::Internal(
            "TensorFlow saving not implemented".to_string(),
        ))
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["pb"]
    }
}

impl Default for TensorFlowConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::ModelConverter;
    use crate::model::{ModelData, ModelFormat, ModelInfo};

    #[test]
    fn test_tensorflow_converter_new() {
        let _converter = TensorFlowConverter::new();
    }

    #[test]
    fn test_tensorflow_converter_default() {
        let _converter: TensorFlowConverter = Default::default();
    }

    #[test]
    fn test_tensorflow_supported_extensions() {
        let converter = TensorFlowConverter::new();
        assert_eq!(converter.supported_extensions(), &["pb"]);
    }

    #[test]
    fn test_tensorflow_load_returns_error() {
        let converter = TensorFlowConverter::new();
        let result = converter.load_model("test.pb");
        assert!(result.is_err());
        match result {
            Err(BlitzedError::UnsupportedFormat { format }) => {
                assert_eq!(format, "TensorFlow");
            }
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    #[test]
    fn test_tensorflow_save_returns_error() {
        let converter = TensorFlowConverter::new();
        let model = Model {
            info: ModelInfo {
                format: ModelFormat::Onnx,
                input_shapes: vec![vec![1, 3, 224, 224]],
                output_shapes: vec![vec![1, 1000]],
                parameter_count: 1000,
                model_size_bytes: 4000,
                operations_count: 2000,
                layers: vec![],
            },
            data: ModelData::Raw(vec![0u8; 100]),
        };
        let result = converter.save_model(&model, "test.pb");
        assert!(result.is_err());
        match result {
            Err(BlitzedError::Internal(msg)) => {
                assert!(msg.contains("TensorFlow saving not implemented"));
            }
            _ => panic!("Expected Internal error"),
        }
    }
}
