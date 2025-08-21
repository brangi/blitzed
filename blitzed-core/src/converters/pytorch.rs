//! PyTorch model converter

use crate::{BlitzedError, Model, Result};
use super::ModelConverter;
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
        Err(BlitzedError::Internal("PyTorch saving not implemented".to_string()))
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