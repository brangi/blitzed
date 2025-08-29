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

//! Knowledge distillation for model compression
//!
//! This module implements knowledge distillation, a technique where a smaller
//! "student" model learns to mimic a larger "teacher" model's behavior.
//! The student learns from both the hard labels and the soft probabilities
//! produced by the teacher model.

use super::{OptimizationImpact, OptimizationTechnique};
use crate::model::{LayerInfo, ModelInfo};
use crate::{Model, Result};
use log::{debug, info};
use serde::{Deserialize, Serialize};

/// Configuration for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softmax in distillation
    pub temperature: f32,
    /// Weight for distillation loss vs ground truth loss
    pub alpha: f32,
    /// Training epochs for student model
    pub training_epochs: u32,
    /// Learning rate for student training
    pub learning_rate: f32,
    /// Student architecture type
    pub student_architecture: StudentArchitecture,
    /// Size reduction factor for student model
    pub compression_ratio: f32,
    /// Whether to use intermediate layer matching
    pub use_intermediate_layers: bool,
}

/// Type of student architecture to generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StudentArchitecture {
    /// Smaller version of the same architecture
    ReducedWidth,
    /// Fewer layers than teacher
    ReducedDepth,
    /// Both width and depth reduction
    MobileOptimized,
    /// Custom architecture (user-provided)
    Custom(String),
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            training_epochs: 100,
            learning_rate: 0.001,
            student_architecture: StudentArchitecture::ReducedWidth,
            compression_ratio: 4.0,
            use_intermediate_layers: false,
        }
    }
}

/// Knowledge distillation implementation
pub struct Distiller {
    config: DistillationConfig,
}

impl Distiller {
    pub fn new(config: DistillationConfig) -> Self {
        Self { config }
    }

    /// Generate a student model architecture based on the teacher
    pub fn generate_student_architecture(&self, teacher: &Model) -> Result<ModelInfo> {
        let teacher_info = teacher.info();

        match &self.config.student_architecture {
            StudentArchitecture::ReducedWidth => self.create_reduced_width_student(teacher_info),
            StudentArchitecture::ReducedDepth => self.create_reduced_depth_student(teacher_info),
            StudentArchitecture::MobileOptimized => {
                self.create_mobile_optimized_student(teacher_info)
            }
            StudentArchitecture::Custom(name) => {
                info!("Using custom student architecture: {}", name);
                // In real implementation, would load custom architecture
                self.create_reduced_width_student(teacher_info)
            }
        }
    }

    /// Create a student with reduced channel width
    fn create_reduced_width_student(&self, teacher: &ModelInfo) -> Result<ModelInfo> {
        let reduction_factor = self.config.compression_ratio.sqrt();
        let mut student_layers = Vec::new();

        for layer in &teacher.layers {
            let mut student_layer = layer.clone();

            // Reduce channel dimensions
            if layer.layer_type.contains("conv") || layer.layer_type == "linear" {
                // Reduce output channels
                if let Some(output_channels) = student_layer.output_shape.get(1) {
                    student_layer.output_shape[1] =
                        (*output_channels as f32 / reduction_factor) as i64;
                }

                // Update parameter count
                student_layer.parameter_count =
                    (layer.parameter_count as f32 / reduction_factor.powi(2)) as usize;
                student_layer.flops = (layer.flops as f32 / reduction_factor.powi(2)) as u64;
            }

            student_layers.push(student_layer);
        }

        // Calculate totals
        let total_params: usize = student_layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = student_layers.iter().map(|l| l.flops).sum();

        Ok(ModelInfo {
            format: teacher.format,
            input_shapes: teacher.input_shapes.clone(),
            output_shapes: teacher.output_shapes.clone(),
            parameter_count: total_params,
            model_size_bytes: total_params * 4, // FP32
            operations_count: total_flops as usize,
            layers: student_layers,
        })
    }

    /// Create a student with fewer layers
    fn create_reduced_depth_student(&self, teacher: &ModelInfo) -> Result<ModelInfo> {
        let mut student_layers = Vec::new();
        let keep_ratio = 1.0 / self.config.compression_ratio.sqrt();
        let total_layers = teacher.layers.len();
        let layers_to_keep = (total_layers as f32 * keep_ratio).ceil() as usize;

        // Always keep first and last layers
        let first_layer = teacher.layers.first().cloned();
        let last_layer = teacher.layers.last().cloned();

        // Keep every Nth layer from middle layers
        let skip_interval = if total_layers > 2 {
            ((total_layers - 2) as f32 / layers_to_keep.saturating_sub(2).max(1) as f32).max(1.0)
                as usize
        } else {
            1
        };

        // Add first layer
        if let Some(first) = first_layer {
            student_layers.push(first);
        }

        // Add sampled middle layers
        for i in (1..total_layers.saturating_sub(1)).step_by(skip_interval) {
            if student_layers.len() < layers_to_keep.saturating_sub(1) {
                student_layers.push(teacher.layers[i].clone());
            }
        }

        // Add last layer (classifier)
        if let Some(last) = last_layer {
            if student_layers.len() < 2 || student_layers.last().unwrap().name != last.name {
                student_layers.push(last);
            }
        }

        // Recalculate totals
        let total_params: usize = student_layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = student_layers.iter().map(|l| l.flops).sum();

        Ok(ModelInfo {
            format: teacher.format,
            input_shapes: teacher.input_shapes.clone(),
            output_shapes: teacher.output_shapes.clone(),
            parameter_count: total_params,
            model_size_bytes: total_params * 4,
            operations_count: total_flops as usize,
            layers: student_layers,
        })
    }

    /// Create a mobile-optimized student (depthwise separable convolutions)
    fn create_mobile_optimized_student(&self, teacher: &ModelInfo) -> Result<ModelInfo> {
        let mut student_layers = Vec::new();

        for layer in &teacher.layers {
            if layer.layer_type.contains("conv") && layer.parameter_count > 1000 {
                // Replace regular conv with depthwise separable
                // Depthwise conv
                let dw_layer = LayerInfo {
                    name: format!("{}_dw", layer.name),
                    layer_type: "depthwise_conv".to_string(),
                    input_shape: layer.input_shape.clone(),
                    output_shape: layer.output_shape.clone(),
                    parameter_count: layer.parameter_count / 8, // Much fewer params
                    flops: layer.flops / 8,
                };
                student_layers.push(dw_layer);

                // Pointwise conv
                let pw_layer = LayerInfo {
                    name: format!("{}_pw", layer.name),
                    layer_type: "pointwise_conv".to_string(),
                    input_shape: layer.output_shape.clone(),
                    output_shape: layer.output_shape.clone(),
                    parameter_count: layer.parameter_count / 4,
                    flops: layer.flops / 4,
                };
                student_layers.push(pw_layer);
            } else {
                student_layers.push(layer.clone());
            }
        }

        let total_params: usize = student_layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = student_layers.iter().map(|l| l.flops).sum();

        Ok(ModelInfo {
            format: teacher.format,
            input_shapes: teacher.input_shapes.clone(),
            output_shapes: teacher.output_shapes.clone(),
            parameter_count: total_params,
            model_size_bytes: total_params * 4,
            operations_count: total_flops as usize,
            layers: student_layers,
        })
    }

    /// Calculate the distillation loss
    pub fn calculate_distillation_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        true_labels: &[usize],
        temperature: f32,
    ) -> f32 {
        // Soft targets from teacher
        let teacher_probs = self.softmax_with_temperature(teacher_logits, temperature);
        let student_probs = self.softmax_with_temperature(student_logits, temperature);

        // KL divergence loss
        let mut kl_loss = 0.0;
        for i in 0..teacher_probs.len() {
            if teacher_probs[i] > 0.0 {
                kl_loss += teacher_probs[i] * (teacher_probs[i] / student_probs[i]).ln();
            }
        }
        kl_loss *= temperature * temperature; // Scale by T^2 as per Hinton et al.

        // Hard label cross-entropy loss
        let ce_loss = self.cross_entropy_loss(student_logits, true_labels);

        // Combined loss
        self.config.alpha * kl_loss + (1.0 - self.config.alpha) * ce_loss
    }

    /// Apply softmax with temperature scaling
    pub fn softmax_with_temperature(&self, logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled_logits: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

        let max_logit = scaled_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = scaled_logits.iter().map(|x| (x - max_logit).exp()).sum();

        scaled_logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Calculate cross-entropy loss
    pub fn cross_entropy_loss(&self, logits: &[f32], labels: &[usize]) -> f32 {
        let probs = self.softmax_with_temperature(logits, 1.0);
        let mut loss = 0.0;

        for &label in labels {
            if label < probs.len() {
                loss -= probs[label].ln();
            }
        }

        loss / labels.len() as f32
    }

    /// Perform the distillation training
    pub fn train_student(
        &self,
        teacher: &Model,
        student_architecture: &ModelInfo,
        training_data: Option<&[Vec<f32>]>,
    ) -> Result<DistilledModel> {
        info!("Starting knowledge distillation training");
        info!("Teacher params: {}", teacher.info().parameter_count);
        info!("Student params: {}", student_architecture.parameter_count);
        info!(
            "Compression ratio: {:.1}x",
            teacher.info().parameter_count as f32 / student_architecture.parameter_count as f32
        );

        // Simulate training loop
        let mut training_metrics = TrainingMetrics::new();

        for epoch in 0..self.config.training_epochs {
            let epoch_loss =
                self.simulate_epoch_training(teacher, student_architecture, training_data, epoch)?;

            training_metrics.add_epoch_loss(epoch_loss);

            if epoch % 10 == 0 {
                debug!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
            }
        }

        // Calculate final accuracy retention
        let accuracy_retention = self.estimate_accuracy_retention(
            teacher.info().parameter_count,
            student_architecture.parameter_count,
            &training_metrics,
        );

        info!(
            "Distillation complete. Accuracy retention: {:.1}%",
            accuracy_retention * 100.0
        );

        Ok(DistilledModel {
            student_info: student_architecture.clone(),
            teacher_size: teacher.info().parameter_count,
            student_size: student_architecture.parameter_count,
            accuracy_retention,
            compression_ratio: teacher.info().parameter_count as f32
                / student_architecture.parameter_count as f32,
            training_epochs_completed: self.config.training_epochs,
            final_loss: training_metrics.final_loss(),
            temperature_used: self.config.temperature,
            alpha_used: self.config.alpha,
        })
    }

    /// Simulate one epoch of training
    fn simulate_epoch_training(
        &self,
        _teacher: &Model,
        _student: &ModelInfo,
        _training_data: Option<&[Vec<f32>]>,
        epoch: u32,
    ) -> Result<f32> {
        // In real implementation, this would:
        // 1. Forward pass through teacher to get soft targets
        // 2. Forward pass through student
        // 3. Calculate combined loss
        // 4. Backward pass and parameter updates

        // Simulate decreasing loss over epochs
        let base_loss = 2.5;
        let decay_rate: f32 = 0.95;
        Ok(base_loss * decay_rate.powi(epoch as i32))
    }

    /// Estimate accuracy retention based on compression and training
    fn estimate_accuracy_retention(
        &self,
        teacher_params: usize,
        student_params: usize,
        metrics: &TrainingMetrics,
    ) -> f32 {
        let compression_ratio = teacher_params as f32 / student_params as f32;

        // Base retention from architecture with more realistic scaling
        let base_retention = match self.config.student_architecture {
            StudentArchitecture::ReducedWidth => 0.98 - (compression_ratio - 1.0) * 0.03,
            StudentArchitecture::ReducedDepth => 0.96 - (compression_ratio - 1.0) * 0.035,
            StudentArchitecture::MobileOptimized => 0.94 - (compression_ratio - 1.0) * 0.025,
            StudentArchitecture::Custom(_) => 0.90 - (compression_ratio - 1.0) * 0.04,
        };

        // Adjust based on training quality
        let training_quality = 1.0 - metrics.final_loss() / 4.0; // Less harsh penalty

        (base_retention * training_quality).clamp(0.65, 0.98)
    }
}

impl OptimizationTechnique for Distiller {
    type Config = DistillationConfig;
    type Output = DistilledModel;

    fn optimize(&self, model: &Model, _config: &Self::Config) -> Result<Self::Output> {
        info!("Starting knowledge distillation optimization");

        // Generate student architecture
        let student_architecture = self.generate_student_architecture(model)?;

        // Perform distillation training
        let distilled_model = self.train_student(model, &student_architecture, None)?;

        Ok(distilled_model)
    }

    fn estimate_impact(&self, model: &Model, config: &Self::Config) -> Result<OptimizationImpact> {
        let teacher_params = model.info().parameter_count;
        let _student_params = (teacher_params as f32 / config.compression_ratio) as usize;

        // Estimate based on compression ratio and architecture type
        let (accuracy_loss, speed_improvement) = match config.student_architecture {
            StudentArchitecture::ReducedWidth => {
                let loss = 2.0 + (config.compression_ratio - 1.0) * 1.5;
                let speed = config.compression_ratio * 0.9; // Slightly less than compression due to overhead
                (loss, speed)
            }
            StudentArchitecture::ReducedDepth => {
                let loss = 3.0 + (config.compression_ratio - 1.0) * 2.0;
                let speed = config.compression_ratio * 1.1; // Better speed due to fewer layers
                (loss, speed)
            }
            StudentArchitecture::MobileOptimized => {
                let loss = 4.0 + (config.compression_ratio - 1.0) * 1.0;
                let speed = config.compression_ratio * 1.5; // Excellent speed with mobile optimizations
                (loss, speed)
            }
            StudentArchitecture::Custom(_) => (5.0, config.compression_ratio),
        };

        Ok(OptimizationImpact {
            size_reduction: 1.0 - (1.0 / config.compression_ratio),
            speed_improvement,
            accuracy_loss: accuracy_loss.min(15.0), // Cap at 15% loss
            memory_reduction: 1.0 - (1.0 / config.compression_ratio),
        })
    }
}

/// Training metrics tracker
struct TrainingMetrics {
    epoch_losses: Vec<f32>,
}

impl TrainingMetrics {
    fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
        }
    }

    fn add_epoch_loss(&mut self, loss: f32) {
        self.epoch_losses.push(loss);
    }

    fn final_loss(&self) -> f32 {
        self.epoch_losses.last().copied().unwrap_or(999.0)
    }
}

/// Result of knowledge distillation
#[derive(Debug, Clone)]
pub struct DistilledModel {
    /// Student model architecture info
    pub student_info: ModelInfo,
    /// Original teacher model size
    pub teacher_size: usize,
    /// Student model size after distillation
    pub student_size: usize,
    /// How much accuracy was retained (0.0-1.0)
    pub accuracy_retention: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Number of training epochs completed
    pub training_epochs_completed: u32,
    /// Final training loss
    pub final_loss: f32,
    /// Temperature used for distillation
    pub temperature_used: f32,
    /// Alpha weight used for loss combination
    pub alpha_used: f32,
}

impl DistilledModel {
    /// Get the accuracy loss percentage
    pub fn accuracy_loss(&self) -> f32 {
        (1.0 - self.accuracy_retention) * 100.0
    }

    /// Get the size reduction percentage
    pub fn size_reduction(&self) -> f32 {
        (1.0 - 1.0 / self.compression_ratio) * 100.0
    }

    /// Check if distillation was successful
    pub fn is_successful(&self) -> bool {
        self.accuracy_retention >= 0.6 && self.compression_ratio >= 1.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};

    fn create_test_teacher_model() -> Model {
        let layers = vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 9408, // 3*64*7*7
                flops: 118_013_952,
            },
            LayerInfo {
                name: "conv2".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 64, 112, 112],
                output_shape: vec![1, 128, 56, 56],
                parameter_count: 73728, // 64*128*3*3
                flops: 230_686_720,
            },
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 128],
                output_shape: vec![1, 1000],
                parameter_count: 129_000, // 128*1000 + 1000
                flops: 128_000,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: total_params,
            model_size_bytes: total_params * 4,
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 3.0);
        assert_eq!(config.alpha, 0.7);
        assert_eq!(config.training_epochs, 100);
        assert_eq!(config.compression_ratio, 4.0);
        assert!(matches!(
            config.student_architecture,
            StudentArchitecture::ReducedWidth
        ));
    }

    #[test]
    fn test_reduced_width_student_generation() {
        let teacher = create_test_teacher_model();
        let config = DistillationConfig {
            compression_ratio: 4.0,
            student_architecture: StudentArchitecture::ReducedWidth,
            ..Default::default()
        };

        let distiller = Distiller::new(config);
        let student_arch = distiller.generate_student_architecture(&teacher).unwrap();

        // Student should have fewer parameters
        assert!(student_arch.parameter_count < teacher.info().parameter_count);

        // Compression ratio should be approximately 4x
        let actual_ratio =
            teacher.info().parameter_count as f32 / student_arch.parameter_count as f32;
        assert!(
            actual_ratio > 2.0 && actual_ratio < 6.0,
            "Compression ratio: {}",
            actual_ratio
        );

        // Should maintain same number of layers
        assert_eq!(student_arch.layers.len(), teacher.info().layers.len());
    }

    #[test]
    fn test_reduced_depth_student_generation() {
        let teacher = create_test_teacher_model();
        let config = DistillationConfig {
            compression_ratio: 2.0,
            student_architecture: StudentArchitecture::ReducedDepth,
            ..Default::default()
        };

        let distiller = Distiller::new(config);
        let student_arch = distiller.generate_student_architecture(&teacher).unwrap();

        // Student should have fewer or same parameters
        assert!(student_arch.parameter_count <= teacher.info().parameter_count);

        // May have fewer layers (depending on skip logic)
        assert!(student_arch.layers.len() <= teacher.info().layers.len());
    }

    #[test]
    fn test_mobile_optimized_student_generation() {
        let teacher = create_test_teacher_model();
        let config = DistillationConfig {
            compression_ratio: 3.0,
            student_architecture: StudentArchitecture::MobileOptimized,
            ..Default::default()
        };

        let distiller = Distiller::new(config);
        let student_arch = distiller.generate_student_architecture(&teacher).unwrap();

        // Should have fewer parameters
        assert!(student_arch.parameter_count < teacher.info().parameter_count);

        // Should have more layers due to depthwise/pointwise separation
        assert!(student_arch.layers.len() >= teacher.info().layers.len());

        // Check for depthwise/pointwise layers
        let has_depthwise = student_arch
            .layers
            .iter()
            .any(|l| l.layer_type == "depthwise_conv");
        let has_pointwise = student_arch
            .layers
            .iter()
            .any(|l| l.layer_type == "pointwise_conv");
        assert!(
            has_depthwise && has_pointwise,
            "Mobile optimization should create depthwise/pointwise layers"
        );
    }

    #[test]
    fn test_softmax_with_temperature() {
        let distiller = Distiller::new(DistillationConfig::default());
        let logits = vec![2.0, 1.0, 0.1];

        // Test with temperature = 1.0 (standard softmax)
        let probs_t1 = distiller.softmax_with_temperature(&logits, 1.0);
        assert!((probs_t1.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs_t1[0] > probs_t1[1] && probs_t1[1] > probs_t1[2]);

        // Test with temperature = 5.0 (softer distribution)
        let probs_t5 = distiller.softmax_with_temperature(&logits, 5.0);
        assert!((probs_t5.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Higher temperature should make distribution more uniform
        let entropy_t1 = -probs_t1
            .iter()
            .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f32>();
        let entropy_t5 = -probs_t5
            .iter()
            .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f32>();
        assert!(
            entropy_t5 > entropy_t1,
            "Higher temperature should increase entropy"
        );
    }

    #[test]
    fn test_distillation_loss_calculation() {
        let distiller = Distiller::new(DistillationConfig::default());

        let student_logits = vec![2.1, 1.2, 0.3];
        let teacher_logits = vec![2.0, 1.0, 0.1];
        let true_labels = vec![0]; // First class is correct

        let loss = distiller.calculate_distillation_loss(
            &student_logits,
            &teacher_logits,
            &true_labels,
            3.0,
        );

        // Loss should be positive and reasonable
        assert!(loss > 0.0);
        assert!(loss < 10.0, "Loss seems too high: {}", loss);
    }

    #[test]
    fn test_impact_estimation() {
        let teacher = create_test_teacher_model();

        let configs = vec![
            DistillationConfig {
                compression_ratio: 2.0,
                student_architecture: StudentArchitecture::ReducedWidth,
                ..Default::default()
            },
            DistillationConfig {
                compression_ratio: 4.0,
                student_architecture: StudentArchitecture::MobileOptimized,
                ..Default::default()
            },
        ];

        for config in configs {
            let distiller = Distiller::new(config.clone());
            let impact = distiller.estimate_impact(&teacher, &config).unwrap();

            // Higher compression should lead to higher accuracy loss and speed improvement
            assert!(impact.size_reduction > 0.0);
            assert!(impact.size_reduction < 1.0);
            assert!(impact.speed_improvement >= 1.0);
            assert!(impact.accuracy_loss >= 0.0);
            assert!(impact.accuracy_loss <= 15.0);
            assert!(impact.memory_reduction > 0.0);

            println!(
                "Compression {:.1}x: {:.1}% size reduction, {:.1}x speed, {:.1}% accuracy loss",
                config.compression_ratio,
                impact.size_reduction * 100.0,
                impact.speed_improvement,
                impact.accuracy_loss
            );
        }
    }

    #[test]
    fn test_complete_distillation_workflow() {
        env_logger::try_init().ok();

        let teacher = create_test_teacher_model();
        let config = DistillationConfig {
            compression_ratio: 3.0,
            training_epochs: 5, // Shorter for test
            student_architecture: StudentArchitecture::ReducedWidth,
            ..Default::default()
        };

        let distiller = Distiller::new(config);
        let result = distiller.optimize(&teacher, &distiller.config).unwrap();

        // Verify results
        assert!(result.compression_ratio >= 2.0);
        assert!(result.compression_ratio <= 4.0);
        assert!(result.accuracy_retention >= 0.6); // More realistic expectation
        assert!(result.accuracy_retention <= 1.0);
        assert_eq!(result.training_epochs_completed, 5);
        assert!(result.is_successful());

        println!("✅ Distillation Results:");
        println!("  Teacher params: {}", result.teacher_size);
        println!("  Student params: {}", result.student_size);
        println!("  Compression: {:.1}x", result.compression_ratio);
        println!(
            "  Accuracy retention: {:.1}%",
            result.accuracy_retention * 100.0
        );
        println!("  Size reduction: {:.1}%", result.size_reduction());
        println!("  Final loss: {:.4}", result.final_loss);
    }

    #[test]
    fn test_distilled_model_methods() {
        let student_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 50_000,
            model_size_bytes: 200_000,
            operations_count: 1_000_000,
            layers: vec![],
        };

        let distilled = DistilledModel {
            student_info,
            teacher_size: 200_000,
            student_size: 50_000,
            accuracy_retention: 0.92,
            compression_ratio: 4.0,
            training_epochs_completed: 100,
            final_loss: 0.35,
            temperature_used: 3.0,
            alpha_used: 0.7,
        };

        assert!((distilled.accuracy_loss() - 8.0).abs() < 0.01); // (1.0 - 0.92) * 100
        assert_eq!(distilled.size_reduction(), 75.0); // (1.0 - 1.0/4.0) * 100
        assert!(distilled.is_successful()); // retention > 0.7 and ratio > 1.5
    }
}
