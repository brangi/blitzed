//! ONNX model converter and loader

use crate::{BlitzedError, Model, Result};
use super::ModelConverter;
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
        Err(BlitzedError::Internal("ONNX saving not implemented".to_string()))
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