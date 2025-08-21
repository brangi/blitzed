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

//! Quantization algorithms for model compression

use crate::{BlitzedError, Model, Result};
use super::{OptimizationTechnique, OptimizationImpact};
use serde::{Deserialize, Serialize};

/// Quantization types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// Binary quantization (1-bit)
    Binary,
    /// Mixed precision quantization
    Mixed,
}

/// Configuration for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub calibration_dataset_size: usize,
    pub symmetric: bool,
    pub per_channel: bool,
    pub skip_sensitive_layers: bool,
    pub accuracy_threshold: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Int8,
            calibration_dataset_size: 100,
            symmetric: true,
            per_channel: true,
            skip_sensitive_layers: true,
            accuracy_threshold: 5.0,
        }
    }
}

/// Quantization algorithm implementation
pub struct Quantizer {
    config: QuantizationConfig,
}

impl Quantizer {
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Perform post-training quantization
    pub fn quantize_post_training(&self, model: &Model) -> Result<QuantizedModel> {
        log::info!("Starting post-training quantization with {:?}", self.config.quantization_type);
        
        match self.config.quantization_type {
            QuantizationType::Int8 => self.quantize_int8(model),
            QuantizationType::Int4 => self.quantize_int4(model),
            QuantizationType::Binary => self.quantize_binary(model),
            QuantizationType::Mixed => self.quantize_mixed(model),
        }
    }

    /// INT8 quantization implementation
    fn quantize_int8(&self, model: &Model) -> Result<QuantizedModel> {
        // Simplified INT8 quantization
        let original_size = model.info().model_size_bytes;
        let quantized_size = (original_size as f32 * 0.25) as usize; // Rough estimate
        
        let quantization_params = QuantizationParams {
            scale: vec![0.1; 10], // Placeholder scales
            zero_point: vec![128; 10], // Placeholder zero points
            quantization_type: QuantizationType::Int8,
        };

        Ok(QuantizedModel {
            original_model_info: model.info().clone(),
            quantized_size,
            quantization_params,
            accuracy_loss: 2.0, // Estimated
        })
    }

    /// INT4 quantization implementation
    fn quantize_int4(&self, _model: &Model) -> Result<QuantizedModel> {
        // TODO: Implement INT4 quantization
        Err(BlitzedError::OptimizationFailed {
            reason: "INT4 quantization not yet implemented".to_string(),
        })
    }

    /// Binary quantization implementation
    fn quantize_binary(&self, _model: &Model) -> Result<QuantizedModel> {
        // TODO: Implement binary quantization
        Err(BlitzedError::OptimizationFailed {
            reason: "Binary quantization not yet implemented".to_string(),
        })
    }

    /// Mixed precision quantization implementation
    fn quantize_mixed(&self, _model: &Model) -> Result<QuantizedModel> {
        // TODO: Implement mixed precision quantization
        Err(BlitzedError::OptimizationFailed {
            reason: "Mixed precision quantization not yet implemented".to_string(),
        })
    }

    /// Calculate quantization parameters for a layer
    fn calculate_quantization_params(&self, values: &[f32]) -> Result<(f32, i32)> {
        if values.is_empty() {
            return Err(BlitzedError::OptimizationFailed {
                reason: "Cannot calculate quantization params for empty values".to_string(),
            });
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            return Ok((1.0, 0));
        }

        let scale = match self.config.quantization_type {
            QuantizationType::Int8 => (max_val - min_val) / 255.0,
            QuantizationType::Int4 => (max_val - min_val) / 15.0,
            _ => (max_val - min_val) / 255.0,
        };

        let zero_point = if self.config.symmetric {
            0
        } else {
            (-min_val / scale).round() as i32
        };

        Ok((scale, zero_point))
    }
}

impl OptimizationTechnique for Quantizer {
    type Config = QuantizationConfig;
    type Output = QuantizedModel;

    fn optimize(&self, model: &Model, config: &Self::Config) -> Result<Self::Output> {
        let quantizer = Self::new(config.clone());
        quantizer.quantize_post_training(model)
    }

    fn estimate_impact(&self, model: &Model, _config: &Self::Config) -> Result<OptimizationImpact> {
        let size_reduction = match self.config.quantization_type {
            QuantizationType::Int8 => 0.75,    // 75% size reduction
            QuantizationType::Int4 => 0.875,   // 87.5% size reduction
            QuantizationType::Binary => 0.96875, // 96.875% size reduction
            QuantizationType::Mixed => 0.6,    // 60% size reduction
        };

        let speed_improvement = match self.config.quantization_type {
            QuantizationType::Int8 => 2.0,
            QuantizationType::Int4 => 3.0,
            QuantizationType::Binary => 8.0,
            QuantizationType::Mixed => 1.5,
        };

        let accuracy_loss = match self.config.quantization_type {
            QuantizationType::Int8 => 2.0,
            QuantizationType::Int4 => 5.0,
            QuantizationType::Binary => 15.0,
            QuantizationType::Mixed => 3.0,
        };

        Ok(OptimizationImpact {
            size_reduction,
            speed_improvement,
            accuracy_loss,
            memory_reduction: size_reduction,
        })
    }
}

/// Quantization parameters for a quantized model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub quantization_type: QuantizationType,
}

/// Result of quantization optimization
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    pub original_model_info: crate::model::ModelInfo,
    pub quantized_size: usize,
    pub quantization_params: QuantizationParams,
    pub accuracy_loss: f32,
}

impl QuantizedModel {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        1.0 - (self.quantized_size as f32 / self.original_model_info.model_size_bytes as f32)
    }

    /// Check if accuracy loss is within threshold
    pub fn check_accuracy_threshold(&self, threshold: f32) -> Result<()> {
        if self.accuracy_loss > threshold {
            return Err(BlitzedError::AccuracyThreshold {
                threshold,
                actual: self.accuracy_loss,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelInfo, ModelFormat};

    fn create_test_model_info() -> ModelInfo {
        ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 1000000,
            model_size_bytes: 4000000, // 4MB
            operations_count: 500000,
        }
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type, QuantizationType::Int8);
        assert!(config.symmetric);
        assert!(config.per_channel);
    }

    #[test]
    fn test_estimate_impact() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config.clone());
        
        // Create a mock model
        let model_info = create_test_model_info();
        let model_data = crate::model::ModelData::Raw(vec![0u8; 1000]);
        let model = Model {
            info: model_info,
            data: model_data,
        };

        let impact = quantizer.estimate_impact(&model, &config).unwrap();
        assert_eq!(impact.size_reduction, 0.75);
        assert_eq!(impact.speed_improvement, 2.0);
        assert_eq!(impact.accuracy_loss, 2.0);
    }

    #[test]
    fn test_quantization_params_calculation() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config);
        
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (scale, zero_point) = quantizer.calculate_quantization_params(&values).unwrap();
        
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Symmetric quantization
    }

    #[test]
    fn test_compression_ratio() {
        let quantized = QuantizedModel {
            original_model_info: create_test_model_info(),
            quantized_size: 1000000, // 1MB from 4MB
            quantization_params: QuantizationParams {
                scale: vec![0.1],
                zero_point: vec![0],
                quantization_type: QuantizationType::Int8,
            },
            accuracy_loss: 2.0,
        };

        assert!((quantized.compression_ratio() - 0.75).abs() < 0.001);
    }
}