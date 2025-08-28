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

use super::{OptimizationImpact, OptimizationTechnique};
use crate::{BlitzedError, Model, Result};
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
        log::info!(
            "Starting post-training quantization with {:?}",
            self.config.quantization_type
        );

        match self.config.quantization_type {
            QuantizationType::Int8 => self.quantize_int8(model),
            QuantizationType::Int4 => self.quantize_int4(model),
            QuantizationType::Binary => self.quantize_binary(model),
            QuantizationType::Mixed => self.quantize_mixed(model),
        }
    }

    /// INT8 quantization implementation
    fn quantize_int8(&self, model: &Model) -> Result<QuantizedModel> {
        log::info!(
            "Performing INT8 quantization (symmetric: {}, per_channel: {})",
            self.config.symmetric,
            self.config.per_channel
        );

        // Process all model weights
        let quantized_layers = self.process_model_weights(model)?;

        // Calculate total sizes
        let original_size: usize = quantized_layers.iter().map(|l| l.original_size).sum();
        let quantized_size: usize = quantized_layers.iter().map(|l| l.quantized_size).sum();

        // Calculate compression metrics
        let compression_ratio = 1.0 - (quantized_size as f32 / original_size as f32);
        let size_reduction_mb = (original_size - quantized_size) as f32 / (1024.0 * 1024.0);

        log::info!(
            "Quantization complete: {:.1}% size reduction ({:.2} MB saved)",
            compression_ratio * 100.0,
            size_reduction_mb
        );

        // Estimate accuracy loss based on quantization parameters
        let avg_accuracy_loss = self.estimate_accuracy_loss(&quantized_layers);

        // Build legacy quantization params for compatibility
        let legacy_params = QuantizationParams {
            scale: quantized_layers.iter().map(|l| l.param.scale).collect(),
            zero_point: quantized_layers
                .iter()
                .map(|l| l.param.zero_point)
                .collect(),
            quantization_type: QuantizationType::Int8,
        };

        Ok(QuantizedModel {
            original_model_info: model.info().clone(),
            quantized_size,
            quantization_params: legacy_params,
            accuracy_loss: avg_accuracy_loss,
            layers: quantized_layers,
        })
    }

    /// Estimate accuracy loss from quantization
    fn estimate_accuracy_loss(&self, layers: &[QuantizedLayer]) -> f32 {
        // Calculate weighted average based on layer sizes
        let total_params: usize = layers.iter().map(|l| l.weight_count).sum();

        let weighted_loss: f32 = layers
            .iter()
            .map(|layer| {
                let weight = layer.weight_count as f32 / total_params as f32;
                let layer_loss = if layer.param.scale < 0.001 {
                    0.5 // Very fine quantization, low loss
                } else if layer.param.scale > 0.1 {
                    5.0 // Coarse quantization, higher loss
                } else {
                    2.0 // Medium quantization
                };
                weight * layer_loss
            })
            .sum();

        // Add base accuracy loss for INT8 quantization
        let base_loss = if self.config.symmetric { 1.5 } else { 2.0 };

        (base_loss + weighted_loss).min(10.0) // Cap at 10% loss
    }

    /// INT4 quantization implementation
    pub fn quantize_int4(&self, model: &Model) -> Result<QuantizedModel> {
        log::info!(
            "Performing INT4 quantization (symmetric: {}, per_channel: {})",
            self.config.symmetric,
            self.config.per_channel
        );
        log::warn!("INT4 quantization is aggressive - expect potential accuracy loss");

        // Process all model weights with INT4 precision
        let quantized_layers = self.process_model_weights_int4(model)?;

        // Calculate total sizes
        let original_size: usize = quantized_layers.iter().map(|l| l.original_size).sum();
        let quantized_size: usize = quantized_layers.iter().map(|l| l.quantized_size).sum();

        // Calculate compression metrics
        let compression_ratio = 1.0 - (quantized_size as f32 / original_size as f32);
        let size_reduction_mb = (original_size - quantized_size) as f32 / (1024.0 * 1024.0);

        log::info!(
            "INT4 Quantization complete: {:.1}% size reduction ({:.2} MB saved)",
            compression_ratio * 100.0,
            size_reduction_mb
        );

        // Estimate accuracy loss (higher for INT4)
        let avg_accuracy_loss = self.estimate_accuracy_loss_int4(&quantized_layers);

        // Build quantization params
        let params = QuantizationParams {
            scale: quantized_layers.iter().map(|l| l.param.scale).collect(),
            zero_point: quantized_layers
                .iter()
                .map(|l| l.param.zero_point)
                .collect(),
            quantization_type: QuantizationType::Int4,
        };

        Ok(QuantizedModel {
            original_model_info: model.info().clone(),
            quantized_size,
            quantization_params: params,
            accuracy_loss: avg_accuracy_loss,
            layers: quantized_layers,
        })
    }

    /// Binary quantization implementation (1-bit weights: -1 or +1)
    pub fn quantize_binary(&self, model: &Model) -> Result<QuantizedModel> {
        log::info!("Performing Binary quantization (1-bit weights: -1/+1)");
        log::warn!(
            "Binary quantization is extremely aggressive - expect significant accuracy loss"
        );

        // Process all model weights with binary precision
        let quantized_layers = self.process_model_weights_binary(model)?;

        // Calculate total sizes
        let original_size: usize = quantized_layers.iter().map(|l| l.original_size).sum();
        let quantized_size: usize = quantized_layers.iter().map(|l| l.quantized_size).sum();

        // Calculate compression metrics
        let compression_ratio = 1.0 - (quantized_size as f32 / original_size as f32);
        let size_reduction_mb = (original_size - quantized_size) as f32 / (1024.0 * 1024.0);

        log::info!(
            "Binary Quantization complete: {:.1}% size reduction ({:.2} MB saved)",
            compression_ratio * 100.0,
            size_reduction_mb
        );

        // Estimate accuracy loss (highest for binary)
        let avg_accuracy_loss = self.estimate_accuracy_loss_binary(&quantized_layers);

        // Build quantization params
        let params = QuantizationParams {
            scale: quantized_layers.iter().map(|l| l.param.scale).collect(),
            zero_point: quantized_layers
                .iter()
                .map(|l| l.param.zero_point)
                .collect(),
            quantization_type: QuantizationType::Binary,
        };

        Ok(QuantizedModel {
            original_model_info: model.info().clone(),
            quantized_size,
            quantization_params: params,
            accuracy_loss: avg_accuracy_loss,
            layers: quantized_layers,
        })
    }

    /// Mixed precision quantization implementation (layer-wise precision optimization)
    pub fn quantize_mixed(&self, model: &Model) -> Result<QuantizedModel> {
        log::info!("Performing Mixed Precision quantization (layer-wise optimization)");
        log::info!(
            "Using INT8 for most layers, FP16 for sensitive layers, INT4 for insensitive layers"
        );

        // Process all model weights with mixed precision
        let quantized_layers = self.process_model_weights_mixed(model)?;

        // Calculate total sizes
        let original_size: usize = quantized_layers.iter().map(|l| l.original_size).sum();
        let quantized_size: usize = quantized_layers.iter().map(|l| l.quantized_size).sum();

        // Calculate compression metrics
        let compression_ratio = 1.0 - (quantized_size as f32 / original_size as f32);
        let size_reduction_mb = (original_size - quantized_size) as f32 / (1024.0 * 1024.0);

        log::info!(
            "Mixed Precision Quantization complete: {:.1}% size reduction ({:.2} MB saved)",
            compression_ratio * 100.0,
            size_reduction_mb
        );

        // Estimate accuracy loss (moderate - balanced approach)
        let avg_accuracy_loss = self.estimate_accuracy_loss_mixed(&quantized_layers);

        // Build quantization params
        let params = QuantizationParams {
            scale: quantized_layers.iter().map(|l| l.param.scale).collect(),
            zero_point: quantized_layers
                .iter()
                .map(|l| l.param.zero_point)
                .collect(),
            quantization_type: QuantizationType::Mixed,
        };

        Ok(QuantizedModel {
            original_model_info: model.info().clone(),
            quantized_size,
            quantization_params: params,
            accuracy_loss: avg_accuracy_loss,
            layers: quantized_layers,
        })
    }

    /// Calculate quantization parameters for a tensor
    fn calculate_quantization_params(&self, values: &[f32]) -> Result<QuantizationParam> {
        if values.is_empty() {
            return Err(BlitzedError::OptimizationFailed {
                reason: "Cannot calculate quantization params for empty values".to_string(),
            });
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Handle constant tensors
        if (max_val - min_val).abs() < f32::EPSILON {
            return Ok(QuantizationParam {
                scale: 1.0,
                zero_point: 0,
            });
        }

        let (scale, zero_point) = match self.config.quantization_type {
            QuantizationType::Int8 => self.calculate_int8_params(min_val, max_val),
            QuantizationType::Int4 => self.calculate_int4_params(min_val, max_val),
            _ => self.calculate_int8_params(min_val, max_val),
        };

        Ok(QuantizationParam { scale, zero_point })
    }

    /// Calculate INT8 quantization parameters
    fn calculate_int8_params(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        if self.config.symmetric {
            // Symmetric quantization: range [-127, 127], zero_point = 0
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 127.0;
            (scale, 0)
        } else {
            // Asymmetric quantization: range [0, 255]
            let scale = (max_val - min_val) / 255.0;
            let zero_point = ((-min_val / scale).round() as i32).clamp(0, 255);
            (scale, zero_point)
        }
    }

    /// Calculate INT4 quantization parameters
    fn calculate_int4_params(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        if self.config.symmetric {
            // Symmetric quantization: range [-7, 7], zero_point = 0
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 7.0;
            (scale, 0)
        } else {
            // Asymmetric quantization: range [0, 15]
            let scale = (max_val - min_val) / 15.0;
            let zero_point = ((-min_val / scale).round() as i32).clamp(0, 15);
            (scale, zero_point)
        }
    }

    /// Quantize FP32 values to INT8
    fn quantize_values_int8(&self, values: &[f32], param: &QuantizationParam) -> Vec<i8> {
        values
            .iter()
            .map(|&x| {
                let quantized = (x / param.scale).round() + param.zero_point as f32;
                if self.config.symmetric {
                    quantized.clamp(-127.0, 127.0) as i8
                } else {
                    quantized.clamp(0.0, 255.0) as u8 as i8
                }
            })
            .collect()
    }

    /// Dequantize INT8 values back to FP32 (for validation)
    #[allow(dead_code)]
    fn dequantize_values_int8(&self, quantized: &[i8], param: &QuantizationParam) -> Vec<f32> {
        quantized
            .iter()
            .map(|&q| (q as f32 - param.zero_point as f32) * param.scale)
            .collect()
    }

    /// Process model weights for INT4 quantization
    fn process_model_weights_int4(&self, model: &Model) -> Result<Vec<QuantizedLayer>> {
        // For now, simulate processing model layers for INT4
        // In a real implementation, this would extract actual weights from the model
        let mut quantized_layers = Vec::new();

        // Create realistic layer simulation based on actual model parameter count
        let total_params = model.info.parameter_count;
        let layer_configs = vec![
            ("backbone", total_params * 8 / 10, (-0.5, 0.5)), // 80% of parameters in backbone
            ("classifier", total_params * 2 / 10, (-0.3, 0.3)), // 20% of parameters in classifier
        ];

        for (layer_name, weight_count, (min_val, max_val)) in layer_configs {
            // Generate simulated weights for this layer
            let weights = self.generate_weights_with_range(weight_count, min_val, max_val);

            // Calculate INT4 quantization parameters (more aggressive)
            let param = self.calculate_quantization_params_int4(&weights)?;

            // Quantize the weights to INT4
            let quantized_weights = self.quantize_values_int4(&weights, &param);

            // Calculate sizes (INT4 = 4 bits = 0.5 bytes per weight)
            let original_size = weight_count * 4; // FP32 = 4 bytes per weight
            let quantized_size = (weight_count + 1) / 2; // INT4 = 0.5 bytes per weight (rounded up)

            let layer = QuantizedLayer {
                name: layer_name.to_string(),
                param,
                quantized_weights,
                weight_count,
                original_size,
                quantized_size,
            };

            quantized_layers.push(layer);

            log::debug!(
                "Layer {}: {} weights, {:.2} KB -> {:.2} KB ({:.1}% reduction)",
                layer_name,
                weight_count,
                original_size as f32 / 1024.0,
                quantized_size as f32 / 1024.0,
                (1.0 - quantized_size as f32 / original_size as f32) * 100.0
            );
        }

        Ok(quantized_layers)
    }

    /// Calculate quantization parameters for INT4 (more aggressive)
    fn calculate_quantization_params_int4(&self, values: &[f32]) -> Result<QuantizationParam> {
        if values.is_empty() {
            return Err(BlitzedError::OptimizationFailed {
                reason: "Cannot calculate INT4 quantization params for empty values".to_string(),
            });
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if self.config.symmetric {
            // Symmetric quantization: range is -7 to 7 for INT4
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 7.0; // 4-bit signed: -7 to 7
            Ok(QuantizationParam {
                scale,
                zero_point: 0, // Symmetric
            })
        } else {
            // Asymmetric quantization: range is 0 to 15 for INT4
            let scale = (max_val - min_val) / 15.0; // 4-bit unsigned: 0 to 15
            let zero_point = (-min_val / scale).round() as i32;
            Ok(QuantizationParam { scale, zero_point })
        }
    }

    /// Quantize values to INT4 (4-bit integers)
    fn quantize_values_int4(&self, values: &[f32], param: &QuantizationParam) -> Vec<i8> {
        values
            .iter()
            .map(|&x| {
                let quantized = (x / param.scale).round() + param.zero_point as f32;
                if self.config.symmetric {
                    quantized.clamp(-7.0, 7.0) as i8 // 4-bit signed range
                } else {
                    quantized.clamp(0.0, 15.0) as i8 // 4-bit unsigned range
                }
            })
            .collect()
    }

    /// Estimate accuracy loss for INT4 quantization (higher than INT8)
    fn estimate_accuracy_loss_int4(&self, layers: &[QuantizedLayer]) -> f32 {
        let total_params: usize = layers.iter().map(|l| l.weight_count).sum();

        let weighted_loss: f32 = layers
            .iter()
            .map(|layer| {
                let weight = layer.weight_count as f32 / total_params as f32;
                // INT4 has higher quantization error than INT8
                let layer_loss = if layer.param.scale < 0.005 {
                    3.0 // Very fine quantization, but still higher loss than INT8
                } else if layer.param.scale > 0.2 {
                    12.0 // Coarse quantization, significant loss
                } else {
                    7.0 // Medium quantization
                };
                weight * layer_loss
            })
            .sum();

        // Higher base loss for INT4 quantization
        let base_loss = if self.config.symmetric { 4.0 } else { 5.0 };

        (base_loss + weighted_loss).min(20.0) // Cap at 20% loss for INT4
    }

    /// Process model weights for binary quantization (1-bit)
    fn process_model_weights_binary(&self, model: &Model) -> Result<Vec<QuantizedLayer>> {
        // For now, simulate processing model layers for binary quantization
        // In a real implementation, this would extract actual weights from the model
        let mut quantized_layers = Vec::new();

        // Create realistic layer simulation based on actual model parameter count
        let total_params = model.info.parameter_count;
        let layer_configs = vec![
            ("early_layers", total_params * 2 / 10, (-0.5, 0.5), false), // 20% kept as FP32
            ("backbone", total_params * 6 / 10, (-0.4, 0.4), true),      // 60% binary
            ("classifier", total_params * 2 / 10, (-0.2, 0.2), false),   // 20% kept as FP32
        ];

        for (layer_name, weight_count, (min_val, max_val), use_binary) in layer_configs {
            // Generate simulated weights for this layer
            let weights = self.generate_weights_with_range(weight_count, min_val, max_val);

            if use_binary {
                // Binary quantization: weights become -1 or +1
                let param = self.calculate_binary_quantization_params(&weights)?;
                let quantized_weights = self.quantize_values_binary(&weights, &param);

                // Calculate sizes (Binary = 1 bit = 0.125 bytes per weight)
                let original_size = weight_count * 4; // FP32 = 4 bytes per weight
                let quantized_size = (weight_count + 7) / 8; // 1 bit per weight (rounded up to bytes)

                let layer = QuantizedLayer {
                    name: layer_name.to_string(),
                    param,
                    quantized_weights,
                    weight_count,
                    original_size,
                    quantized_size,
                };

                quantized_layers.push(layer);

                log::debug!(
                    "Layer {} (Binary): {} weights, {:.2} KB -> {:.2} KB ({:.1}% reduction)",
                    layer_name,
                    weight_count,
                    original_size as f32 / 1024.0,
                    quantized_size as f32 / 1024.0,
                    (1.0 - quantized_size as f32 / original_size as f32) * 100.0
                );
            } else {
                // Keep as FP32 for sensitive layers
                let param = QuantizationParam {
                    scale: 1.0,
                    zero_point: 0,
                };
                let original_size = weight_count * 4;

                let layer = QuantizedLayer {
                    name: layer_name.to_string(),
                    param,
                    quantized_weights: weights.into_iter().map(|w| (w * 127.0) as i8).collect(), // Scale to i8 range
                    weight_count,
                    original_size,
                    quantized_size: original_size, // No compression for FP32 layers
                };

                quantized_layers.push(layer);

                log::debug!(
                    "Layer {} (FP32): {} weights, kept at full precision",
                    layer_name,
                    weight_count
                );
            }
        }

        Ok(quantized_layers)
    }

    /// Calculate quantization parameters for binary quantization
    fn calculate_binary_quantization_params(&self, values: &[f32]) -> Result<QuantizationParam> {
        if values.is_empty() {
            return Err(BlitzedError::OptimizationFailed {
                reason: "Cannot calculate binary quantization params for empty values".to_string(),
            });
        }

        // For binary quantization, we use the mean of absolute values as the scaling factor
        let mean_abs = values.iter().map(|&x| x.abs()).sum::<f32>() / values.len() as f32;

        // Scale factor is the mean absolute value (this becomes the magnitude of ±1)
        let scale = mean_abs;

        Ok(QuantizationParam {
            scale,
            zero_point: 0, // Binary is always symmetric around 0
        })
    }

    /// Quantize values to binary (1-bit: -1 or +1)
    fn quantize_values_binary(&self, values: &[f32], _param: &QuantizationParam) -> Vec<i8> {
        values
            .iter()
            .map(|&x| {
                // Binary quantization: sign of the weight determines -1 or +1
                if x >= 0.0 {
                    1 // Positive weights become +1
                } else {
                    -1 // Negative weights become -1
                }
            })
            .collect()
    }

    /// Estimate accuracy loss for binary quantization (highest loss)
    fn estimate_accuracy_loss_binary(&self, layers: &[QuantizedLayer]) -> f32 {
        let total_params: usize = layers.iter().map(|l| l.weight_count).sum();
        let binary_params: usize = layers
            .iter()
            .filter(|l| l.quantized_size < l.original_size / 2) // Identify binary layers by compression
            .map(|l| l.weight_count)
            .sum();

        let binary_ratio = binary_params as f32 / total_params as f32;

        // Base loss increases with percentage of binary weights
        let base_loss = 10.0 + (binary_ratio * 15.0); // 10-25% base loss

        // Additional loss based on layer characteristics
        let layer_loss: f32 = layers
            .iter()
            .map(|layer| {
                let weight = layer.weight_count as f32 / total_params as f32;
                if layer.quantized_size < layer.original_size / 2 {
                    weight * 20.0 // High loss for binary layers
                } else {
                    weight * 2.0 // Low loss for FP32 layers
                }
            })
            .sum();

        (base_loss + layer_loss).min(40.0) // Cap at 40% loss for binary
    }

    /// Process model weights for mixed precision quantization
    fn process_model_weights_mixed(&self, model: &Model) -> Result<Vec<QuantizedLayer>> {
        // For now, simulate processing model layers with mixed precision
        // In a real implementation, this would analyze layer sensitivity and choose optimal precision
        let mut quantized_layers = Vec::new();

        // Mixed precision strategy based on actual model parameter count:
        // - First/last layers: FP16 (sensitive to accuracy)
        // - Large middle layers: INT4 (less sensitive, more compression)
        // - Some layers: INT8 (balanced)
        let total_params = model.info.parameter_count;
        let layer_configs = vec![
            ("early_layers", total_params * 2 / 10, (-0.6, 0.6), "fp16"), // 20% FP16 - high precision
            ("backbone_int4", total_params * 4 / 10, (-0.4, 0.4), "int4"), // 40% INT4 - aggressive compression
            ("backbone_int8", total_params * 3 / 10, (-0.3, 0.3), "int8"), // 30% INT8 - balanced
            ("classifier", total_params / 10, (-0.2, 0.2), "fp16"), // 10% FP16 - high precision
        ];

        for (layer_name, weight_count, (min_val, max_val), precision) in layer_configs {
            // Generate simulated weights for this layer
            let weights = self.generate_weights_with_range(weight_count, min_val, max_val);
            let original_size = weight_count * 4; // FP32 = 4 bytes per weight

            let (param, quantized_weights, quantized_size, precision_name) = match precision {
                "fp16" => {
                    // FP16: 2 bytes per weight, simulate by scaling to i8 range
                    let param = QuantizationParam {
                        scale: 1.0,
                        zero_point: 0,
                    };
                    let quantized: Vec<i8> = weights
                        .into_iter()
                        .map(|w| (w * 100.0).clamp(-127.0, 127.0) as i8)
                        .collect();
                    (param, quantized, weight_count * 2, "FP16")
                }
                "int8" => {
                    // INT8: Use existing INT8 quantization
                    let param = self.calculate_quantization_params(&weights)?;
                    let quantized = self.quantize_values_int8(&weights, &param);
                    (param, quantized, weight_count, "INT8")
                }
                "int4" => {
                    // INT4: Use existing INT4 quantization
                    let param = self.calculate_quantization_params_int4(&weights)?;
                    let quantized = self.quantize_values_int4(&weights, &param);
                    (param, quantized, (weight_count + 1) / 2, "INT4")
                }
                _ => {
                    return Err(BlitzedError::OptimizationFailed {
                        reason: format!("Unknown precision type: {}", precision),
                    });
                }
            };

            let layer = QuantizedLayer {
                name: layer_name.to_string(),
                param,
                quantized_weights,
                weight_count,
                original_size,
                quantized_size,
            };

            quantized_layers.push(layer);

            log::debug!(
                "Layer {} ({}): {} weights, {:.2} KB -> {:.2} KB ({:.1}% reduction)",
                layer_name,
                precision_name,
                weight_count,
                original_size as f32 / 1024.0,
                quantized_size as f32 / 1024.0,
                (1.0 - quantized_size as f32 / original_size as f32) * 100.0
            );
        }

        Ok(quantized_layers)
    }

    /// Estimate accuracy loss for mixed precision quantization
    fn estimate_accuracy_loss_mixed(&self, layers: &[QuantizedLayer]) -> f32 {
        let total_params: usize = layers.iter().map(|l| l.weight_count).sum();

        let weighted_loss: f32 = layers
            .iter()
            .map(|layer| {
                let weight = layer.weight_count as f32 / total_params as f32;

                // Determine precision type by compression ratio
                let compression_ratio =
                    1.0 - (layer.quantized_size as f32 / layer.original_size as f32);

                let layer_loss = if compression_ratio < 0.1 {
                    1.5 // FP16 layers - minimal loss
                } else if compression_ratio < 0.6 {
                    3.0 // INT8 layers - low loss
                } else {
                    6.0 // INT4 layers - higher loss
                };

                weight * layer_loss
            })
            .sum();

        // Base loss for mixed precision (moderate)
        let base_loss = 2.5;

        (base_loss + weighted_loss).min(15.0) // Cap at 15% loss for mixed precision
    }

    /// Process model weights for quantization
    fn process_model_weights(&self, _model: &Model) -> Result<Vec<QuantizedLayer>> {
        // For now, simulate processing model layers
        // In a real implementation, this would extract actual weights from the model
        let mut quantized_layers = Vec::new();

        // Simulate different layer types with different weight distributions
        let layer_configs = vec![
            ("conv1", self.generate_conv_weights(64, 3, 3, 3)), // Conv layer
            ("conv2", self.generate_conv_weights(128, 64, 3, 3)), // Conv layer
            ("fc1", self.generate_fc_weights(512, 1024)),       // FC layer
            ("fc2", self.generate_fc_weights(1024, 1000)),      // FC layer
        ];

        for (name, weights) in layer_configs {
            let quantized_layer = self.quantize_layer(name, &weights)?;
            quantized_layers.push(quantized_layer);
        }

        Ok(quantized_layers)
    }

    /// Quantize a single layer's weights
    fn quantize_layer(&self, name: &str, weights: &[f32]) -> Result<QuantizedLayer> {
        let param = self.calculate_quantization_params(weights)?;
        let quantized_weights = self.quantize_values_int8(weights, &param);

        // Calculate size reduction
        let original_size = weights.len() * 4; // FP32 = 4 bytes
        let quantized_size = quantized_weights.len(); // INT8 = 1 byte

        Ok(QuantizedLayer {
            name: name.to_string(),
            original_size,
            quantized_size,
            param,
            quantized_weights,
            weight_count: weights.len(),
        })
    }

    /// Generate simulated convolutional layer weights
    fn generate_conv_weights(
        &self,
        out_channels: usize,
        in_channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let total_weights = out_channels * in_channels * height * width;
        (0..total_weights)
            .map(|i| {
                // Simulate Xavier/Glorot initialization
                let fan_in = in_channels * height * width;
                let fan_out = out_channels * height * width;
                let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let x = (i as f32 * 0.01) % std::f32::consts::TAU; // Simple pseudo-random
                scale * x.sin()
            })
            .collect()
    }

    /// Generate simulated weights within a specific range
    fn generate_weights_with_range(&self, count: usize, min_val: f32, max_val: f32) -> Vec<f32> {
        (0..count)
            .map(|i| {
                // Simple pseudo-random generation within range
                let x = (i as f32 * 0.017) % 1.0; // 0.0 to 1.0
                min_val + x * (max_val - min_val)
            })
            .collect()
    }

    /// Generate simulated fully connected layer weights
    fn generate_fc_weights(&self, input_size: usize, output_size: usize) -> Vec<f32> {
        let total_weights = input_size * output_size;
        (0..total_weights)
            .map(|i| {
                // Simulate Xavier initialization
                let scale = (2.0 / (input_size + output_size) as f32).sqrt();
                let x = (i as f32 * 0.013) % std::f32::consts::TAU; // Simple pseudo-random
                scale * x.cos()
            })
            .collect()
    }
}

impl OptimizationTechnique for Quantizer {
    type Config = QuantizationConfig;
    type Output = QuantizedModel;

    fn optimize(&self, model: &Model, config: &Self::Config) -> Result<Self::Output> {
        let quantizer = Self::new(config.clone());
        quantizer.quantize_post_training(model)
    }

    fn estimate_impact(
        &self,
        _model: &Model,
        _config: &Self::Config,
    ) -> Result<OptimizationImpact> {
        let size_reduction = match self.config.quantization_type {
            QuantizationType::Int8 => 0.75,      // 75% size reduction
            QuantizationType::Int4 => 0.875,     // 87.5% size reduction
            QuantizationType::Binary => 0.96875, // 96.875% size reduction
            QuantizationType::Mixed => 0.6,      // 60% size reduction
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

/// Single quantization parameter set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParam {
    pub scale: f32,
    pub zero_point: i32,
}

/// Quantization parameters for a quantized model (legacy compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub quantization_type: QuantizationType,
}

/// Quantized layer information
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    pub name: String,
    pub original_size: usize,
    pub quantized_size: usize,
    pub param: QuantizationParam,
    pub quantized_weights: Vec<i8>,
    pub weight_count: usize,
}

/// Result of quantization optimization
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    pub original_model_info: crate::model::ModelInfo,
    pub quantized_size: usize,
    pub quantization_params: QuantizationParams,
    pub accuracy_loss: f32,
    pub layers: Vec<QuantizedLayer>,
}

impl QuantizedModel {
    /// Get compression ratio (original_size / quantized_size)
    pub fn compression_ratio(&self) -> f32 {
        if self.quantized_size == 0 {
            1.0 // No compression
        } else {
            self.original_model_info.model_size_bytes as f32 / self.quantized_size as f32
        }
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

    /// Get detailed quantization statistics
    pub fn get_stats(&self) -> QuantizationStats {
        let total_original_params: usize = self.layers.iter().map(|l| l.weight_count).sum();
        let avg_scale: f32 =
            self.layers.iter().map(|l| l.param.scale).sum::<f32>() / self.layers.len() as f32;

        let size_reduction_mb = (self.original_model_info.model_size_bytes - self.quantized_size)
            as f32
            / (1024.0 * 1024.0);

        QuantizationStats {
            compression_ratio: self.compression_ratio(),
            size_reduction_mb,
            accuracy_loss: self.accuracy_loss,
            total_parameters: total_original_params,
            layers_quantized: self.layers.len(),
            average_scale: avg_scale,
        }
    }

    /// Get layer-by-layer breakdown
    pub fn get_layer_breakdown(&self) -> Vec<LayerQuantizationInfo> {
        self.layers
            .iter()
            .map(|layer| LayerQuantizationInfo {
                name: layer.name.clone(),
                compression_ratio: 1.0 - (layer.quantized_size as f32 / layer.original_size as f32),
                scale: layer.param.scale,
                zero_point: layer.param.zero_point,
                parameter_count: layer.weight_count,
            })
            .collect()
    }
}

/// Quantization statistics summary
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    pub compression_ratio: f32,
    pub size_reduction_mb: f32,
    pub accuracy_loss: f32,
    pub total_parameters: usize,
    pub layers_quantized: usize,
    pub average_scale: f32,
}

/// Per-layer quantization information
#[derive(Debug, Clone)]
pub struct LayerQuantizationInfo {
    pub name: String,
    pub compression_ratio: f32,
    pub scale: f32,
    pub zero_point: i32,
    pub parameter_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelFormat, ModelInfo};

    fn create_test_model_info() -> ModelInfo {
        use crate::model::LayerInfo;

        ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 1_000_000,
            model_size_bytes: 4_000_000, // 4MB
            operations_count: 500_000,
            layers: vec![
                LayerInfo {
                    name: "conv1".to_string(),
                    layer_type: "Conv2d".to_string(),
                    input_shape: vec![1, 3, 224, 224],
                    output_shape: vec![1, 64, 112, 112],
                    parameter_count: 9408, // 3*64*7*7
                    flops: 118_013_952,
                },
                LayerInfo {
                    name: "fc".to_string(),
                    layer_type: "Linear".to_string(),
                    input_shape: vec![1, 2048],
                    output_shape: vec![1, 1000],
                    parameter_count: 2_048_000,
                    flops: 2_048_000,
                },
            ],
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
        let param = quantizer.calculate_quantization_params(&values).unwrap();

        assert!(param.scale > 0.0);
        assert_eq!(param.zero_point, 0); // Symmetric quantization
    }

    #[test]
    fn test_symmetric_vs_asymmetric_quantization() {
        // Test symmetric quantization
        let symmetric_config = QuantizationConfig {
            symmetric: true,
            ..Default::default()
        };
        let symmetric_quantizer = Quantizer::new(symmetric_config);

        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let symmetric_param = symmetric_quantizer
            .calculate_quantization_params(&values)
            .unwrap();
        assert_eq!(symmetric_param.zero_point, 0);

        // Test asymmetric quantization
        let asymmetric_config = QuantizationConfig {
            symmetric: false,
            ..Default::default()
        };
        let asymmetric_quantizer = Quantizer::new(asymmetric_config);

        let asymmetric_param = asymmetric_quantizer
            .calculate_quantization_params(&values)
            .unwrap();
        assert!(asymmetric_param.zero_point >= 0 && asymmetric_param.zero_point <= 255);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config);

        let original_values = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let param = quantizer
            .calculate_quantization_params(&original_values)
            .unwrap();

        let quantized_values = quantizer.quantize_values_int8(&original_values, &param);
        let dequantized = quantizer.dequantize_values_int8(&quantized_values, &param);

        // Check that dequantized values are reasonably close to originals
        for (orig, deq) in original_values.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Original: {}, Dequantized: {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_full_int8_quantization() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config);

        // Create a test model
        let model_info = create_test_model_info();
        let model_data = crate::model::ModelData::Raw(vec![0u8; 1000]);
        let model = Model {
            info: model_info,
            data: model_data,
        };

        let quantized_model = quantizer.quantize_int8(&model).unwrap();

        // Verify quantization results
        assert!(!quantized_model.layers.is_empty());
        assert!(quantized_model.compression_ratio() > 0.5); // Should achieve significant compression
        assert!(quantized_model.accuracy_loss < 10.0); // Should be reasonable accuracy loss

        // Test statistics
        let stats = quantized_model.get_stats();
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.layers_quantized > 0);

        // Test layer breakdown
        let breakdown = quantized_model.get_layer_breakdown();
        assert_eq!(breakdown.len(), quantized_model.layers.len());
    }

    #[test]
    fn test_compression_ratio() {
        let quantized = QuantizedModel {
            original_model_info: create_test_model_info(),
            quantized_size: 1_000_000, // 1MB from 4MB
            quantization_params: QuantizationParams {
                scale: vec![0.1],
                zero_point: vec![0],
                quantization_type: QuantizationType::Int8,
            },
            accuracy_loss: 2.0,
            layers: vec![], // Empty for this test
        };

        assert!((quantized.compression_ratio() - 4.0).abs() < 0.001); // 4MB / 1MB = 4.0x
    }
}
