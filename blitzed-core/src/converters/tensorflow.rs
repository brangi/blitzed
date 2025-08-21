//! TensorFlow model converter

use crate::{BlitzedError, Model, Result};
use super::ModelConverter;
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
        Err(BlitzedError::Internal("TensorFlow saving not implemented".to_string()))
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