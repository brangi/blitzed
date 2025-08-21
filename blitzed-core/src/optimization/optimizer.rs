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

//! Main optimizer orchestrating multiple optimization techniques

use crate::{BlitzedError, Model, Result, Config};
use super::{OptimizationTechnique, OptimizationImpact, Quantizer, QuantizationConfig};
use serde::{Deserialize, Serialize};

/// Configuration for the optimization pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub quantization: Option<QuantizationConfig>,
    pub pruning_enabled: bool,
    pub distillation_enabled: bool,
    pub target_compression_ratio: f32,
    pub max_accuracy_loss: f32,
    pub optimization_passes: u32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            quantization: Some(QuantizationConfig::default()),
            pruning_enabled: false,
            distillation_enabled: false,
            target_compression_ratio: 0.75,
            max_accuracy_loss: 5.0,
            optimization_passes: 1,
        }
    }
}

/// Result of the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub original_size: usize,
    pub optimized_size: usize,
    pub compression_ratio: f32,
    pub estimated_accuracy_loss: f32,
    pub estimated_speedup: f32,
    pub optimization_time_ms: u64,
    pub techniques_applied: Vec<String>,
}

impl OptimizationResult {
    /// Check if optimization meets the target criteria
    pub fn meets_criteria(&self, config: &OptimizationConfig) -> bool {
        self.compression_ratio >= config.target_compression_ratio
            && self.estimated_accuracy_loss <= config.max_accuracy_loss
    }

    /// Generate optimization summary
    pub fn summary(&self) -> String {
        format!(
            "Optimization Summary:\n\
             - Size: {:.1} MB → {:.1} MB ({:.1}% reduction)\n\
             - Estimated accuracy loss: {:.2}%\n\
             - Estimated speedup: {:.1}x\n\
             - Optimization time: {} ms\n\
             - Techniques: {}",
            self.original_size as f32 / (1024.0 * 1024.0),
            self.optimized_size as f32 / (1024.0 * 1024.0),
            self.compression_ratio * 100.0,
            self.estimated_accuracy_loss,
            self.estimated_speedup,
            self.optimization_time_ms,
            self.techniques_applied.join(", ")
        )
    }
}

/// Main optimizer orchestrating different optimization techniques
pub struct Optimizer {
    config: Config,
}

impl Optimizer {
    /// Create a new optimizer with configuration
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Optimize a model using configured techniques
    pub fn optimize(&self, model: &Model) -> Result<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let original_size = model.info().model_size_bytes;
        
        log::info!("Starting optimization for model with size {} bytes", original_size);

        // Check hardware constraints
        model.check_memory_constraints(self.config.hardware.memory_limit)?;

        // Build optimization pipeline
        let optimization_config = self.build_optimization_config();
        let mut techniques_applied = Vec::new();
        let mut current_size = original_size;
        let mut total_accuracy_loss = 0.0;
        let mut total_speedup = 1.0;

        // Apply quantization if enabled
        if let Some(quant_config) = &optimization_config.quantization {
            log::info!("Applying quantization optimization");
            let quantizer = Quantizer::new(quant_config.clone());
            let impact = quantizer.estimate_impact(model, quant_config)?;
            
            // Check if this optimization is beneficial
            if impact.accuracy_loss <= optimization_config.max_accuracy_loss {
                current_size = (current_size as f32 * (1.0 - impact.size_reduction)) as usize;
                total_accuracy_loss += impact.accuracy_loss;
                total_speedup *= impact.speed_improvement;
                techniques_applied.push("INT8 Quantization".to_string());
                
                log::info!("Quantization applied: {}% size reduction, {:.1}% accuracy loss", 
                          impact.size_reduction * 100.0, impact.accuracy_loss);
            } else {
                log::warn!("Skipping quantization: accuracy loss too high ({:.1}%)", impact.accuracy_loss);
            }
        }

        // Apply pruning if enabled (placeholder)
        if optimization_config.pruning_enabled {
            log::info!("Pruning optimization requested but not yet implemented");
            // TODO: Implement pruning
        }

        // Apply distillation if enabled (placeholder)
        if optimization_config.distillation_enabled {
            log::info!("Distillation optimization requested but not yet implemented");
            // TODO: Implement knowledge distillation
        }

        let optimization_time = start_time.elapsed().as_millis() as u64;
        let compression_ratio = 1.0 - (current_size as f32 / original_size as f32);

        let result = OptimizationResult {
            original_size,
            optimized_size: current_size,
            compression_ratio,
            estimated_accuracy_loss: total_accuracy_loss,
            estimated_speedup: total_speedup,
            optimization_time_ms: optimization_time,
            techniques_applied,
        };

        // Validate results against criteria
        if !result.meets_criteria(&optimization_config) {
            log::warn!("Optimization did not meet target criteria");
            if result.compression_ratio < optimization_config.target_compression_ratio {
                return Err(BlitzedError::OptimizationFailed {
                    reason: format!(
                        "Target compression ratio {:.1}% not achieved (got {:.1}%)",
                        optimization_config.target_compression_ratio * 100.0,
                        result.compression_ratio * 100.0
                    ),
                });
            }
        }

        log::info!("Optimization completed successfully");
        log::info!("{}", result.summary());

        Ok(result)
    }

    /// Build optimization configuration from global config
    fn build_optimization_config(&self) -> OptimizationConfig {
        OptimizationConfig {
            quantization: if self.config.optimization.enable_quantization {
                Some(QuantizationConfig::default())
            } else {
                None
            },
            pruning_enabled: self.config.optimization.enable_pruning,
            distillation_enabled: self.config.optimization.enable_distillation,
            target_compression_ratio: self.config.optimization.target_compression,
            max_accuracy_loss: self.config.optimization.max_accuracy_loss,
            optimization_passes: 1,
        }
    }

    /// Estimate optimization impact without applying changes
    pub fn estimate_impact(&self, model: &Model) -> Result<OptimizationImpact> {
        let optimization_config = self.build_optimization_config();
        let mut impacts = Vec::new();

        // Estimate quantization impact
        if let Some(quant_config) = &optimization_config.quantization {
            let quantizer = Quantizer::new(quant_config.clone());
            let impact = quantizer.estimate_impact(model, quant_config)?;
            impacts.push(impact);
        }

        // TODO: Add estimates for other optimization techniques

        if impacts.is_empty() {
            Ok(OptimizationImpact {
                size_reduction: 0.0,
                speed_improvement: 1.0,
                accuracy_loss: 0.0,
                memory_reduction: 0.0,
            })
        } else {
            Ok(OptimizationImpact::combine(&impacts))
        }
    }

    /// Get optimization recommendations for the model
    pub fn recommend(&self, model: &Model) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        let memory_usage = model.estimate_memory_usage();

        // Memory-based recommendations
        if memory_usage > self.config.hardware.memory_limit {
            recommendations.push(format!(
                "Model requires {}KB but target has {}KB - consider aggressive quantization",
                memory_usage / 1024,
                self.config.hardware.memory_limit / 1024
            ));
        }

        // Size-based recommendations
        let size_mb = model.info().model_size_bytes as f32 / (1024.0 * 1024.0);
        if size_mb > 10.0 {
            recommendations.push("Large model detected - consider quantization and pruning".to_string());
        }

        // Hardware-specific recommendations
        match self.config.hardware.target.as_str() {
            "arduino" => {
                recommendations.push("Arduino target: Use INT8 quantization and aggressive pruning".to_string());
            }
            "esp32" => {
                recommendations.push("ESP32 target: INT8 quantization recommended for optimal performance".to_string());
            }
            "mobile" => {
                recommendations.push("Mobile target: Consider mixed precision for best accuracy/performance balance".to_string());
            }
            _ => {}
        }

        if recommendations.is_empty() {
            recommendations.push("Model appears suitable for target hardware".to_string());
        }

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelInfo, ModelFormat, ModelData};

    fn create_test_model() -> Model {
        let info = ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 1000000,
            model_size_bytes: 4000000, // 4MB
            operations_count: 500000,
        };
        
        Model {
            info,
            data: ModelData::Raw(vec![0u8; 1000]),
        }
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.quantization.is_some());
        assert_eq!(config.target_compression_ratio, 0.75);
        assert_eq!(config.max_accuracy_loss, 5.0);
    }

    #[test]
    fn test_optimization_result_criteria() {
        let result = OptimizationResult {
            original_size: 4000000,
            optimized_size: 1000000,
            compression_ratio: 0.75,
            estimated_accuracy_loss: 2.0,
            estimated_speedup: 2.0,
            optimization_time_ms: 1000,
            techniques_applied: vec!["Quantization".to_string()],
        };

        let config = OptimizationConfig::default();
        assert!(result.meets_criteria(&config));
    }

    #[test]
    fn test_optimizer_creation() {
        let config = Config::default();
        let optimizer = Optimizer::new(config);
        
        // Should create without errors
        assert_eq!(optimizer.config.optimization.max_accuracy_loss, 5.0);
    }

    #[test]
    fn test_estimate_impact() {
        let config = Config::default();
        let optimizer = Optimizer::new(config);
        let model = create_test_model();

        let impact = optimizer.estimate_impact(&model).unwrap();
        assert!(impact.size_reduction > 0.0);
        assert!(impact.speed_improvement >= 1.0);
    }

    #[test]
    fn test_recommendations() {
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        
        let optimizer = Optimizer::new(config);
        let model = create_test_model();

        let recommendations = optimizer.recommend(&model).unwrap();
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("ESP32")));
    }
}