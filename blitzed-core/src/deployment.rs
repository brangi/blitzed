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

//! Hardware deployment validation and testing
//!
//! This module provides functionality to validate that optimized models can be
//! successfully deployed to edge hardware targets with proper performance characteristics.

use crate::optimization::Optimizer;
use crate::targets::{HardwareTarget, TargetRegistry};
use crate::{BlitzedError, Model, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

/// Results from hardware deployment validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentValidationResult {
    /// Target hardware name
    pub target_name: String,
    /// Model information before optimization
    pub original_model_info: ModelDeploymentInfo,
    /// Model information after optimization
    pub optimized_model_info: ModelDeploymentInfo,
    /// Whether deployment passed hardware constraints
    pub deployment_successful: bool,
    /// Measured performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Generated deployment artifacts
    pub deployment_artifacts: DeploymentArtifacts,
    /// Validation timestamp
    pub validation_timestamp: String,
    /// Any deployment warnings or notes
    pub warnings: Vec<String>,
}

/// Model deployment information for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeploymentInfo {
    /// Model size in bytes
    pub model_size: usize,
    /// Estimated memory usage including activations
    pub memory_usage: usize,
    /// Parameter count
    pub parameter_count: usize,
    /// Operations count (FLOPs)
    pub operations_count: usize,
    /// Whether model fits in target constraints
    pub fits_constraints: bool,
    /// Applied optimization techniques
    pub optimizations_applied: Vec<String>,
}

/// Performance metrics measured during deployment validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Estimated inference time in milliseconds
    pub estimated_inference_time_ms: f64,
    /// Memory efficiency score (0.0-1.0)
    pub memory_efficiency: f64,
    /// Power efficiency estimation (arbitrary units)
    pub power_efficiency: f64,
    /// Latency vs accuracy trade-off score
    pub latency_accuracy_score: f64,
    /// Code generation time in milliseconds
    pub code_generation_time_ms: u64,
}

/// Deployment artifacts generated for hardware target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentArtifacts {
    /// Generated source code files
    pub source_files: Vec<PathBuf>,
    /// Generated header files
    pub header_files: Vec<PathBuf>,
    /// Build configuration files (Makefiles, CMakeLists.txt, etc.)
    pub build_files: Vec<PathBuf>,
    /// Example/demo files
    pub example_files: Vec<PathBuf>,
    /// Total artifact size in bytes
    pub total_size_bytes: usize,
    /// Whether artifacts are build-ready
    pub build_ready: bool,
}

/// Configuration for deployment validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentValidationConfig {
    /// Maximum acceptable inference time in milliseconds
    pub max_inference_time_ms: f64,
    /// Minimum acceptable memory efficiency (0.0-1.0)
    pub min_memory_efficiency: f64,
    /// Whether to generate optimized code variants
    pub generate_optimized_variants: bool,
    /// Whether to validate hardware constraints strictly
    pub strict_constraint_validation: bool,
    /// Output directory for generated artifacts
    pub output_directory: Option<PathBuf>,
    /// Whether to clean up artifacts after validation
    pub cleanup_artifacts: bool,
}

impl Default for DeploymentValidationConfig {
    fn default() -> Self {
        Self {
            max_inference_time_ms: 1000.0, // 1 second max
            min_memory_efficiency: 0.6,    // 60% minimum efficiency
            generate_optimized_variants: true,
            strict_constraint_validation: true,
            output_directory: None,
            cleanup_artifacts: false, // Keep artifacts for inspection
        }
    }
}

/// Hardware deployment validator
pub struct HardwareDeploymentValidator {
    target_registry: TargetRegistry,
    config: DeploymentValidationConfig,
}

impl HardwareDeploymentValidator {
    /// Create new deployment validator with default configuration
    pub fn new() -> Self {
        Self {
            target_registry: TargetRegistry::new(),
            config: DeploymentValidationConfig::default(),
        }
    }

    /// Create validator with custom configuration
    pub fn with_config(config: DeploymentValidationConfig) -> Self {
        Self {
            target_registry: TargetRegistry::new(),
            config,
        }
    }

    /// Validate deployment for ESP32 with comprehensive testing
    pub fn validate_esp32_deployment(
        &self,
        model: &Model,
        optimizer: &Optimizer,
    ) -> Result<DeploymentValidationResult> {
        log::info!("🔥 Starting ESP32 deployment validation");

        let target_name = "ESP32";
        let target = self.target_registry.get_target("esp32")?;

        // Get original model info
        let original_info = self.analyze_model_deployment_info(model, target, vec![])?;

        // Apply optimizations
        let optimization_result = optimizer.optimize(model)?;
        log::info!(
            "   Applied {} optimization techniques",
            optimization_result.techniques_applied.len()
        );

        // Analyze optimized model
        let optimized_info = self.analyze_model_deployment_info(
            model,
            target,
            optimization_result.techniques_applied.clone(),
        )?;

        // Check if deployment is feasible
        let deployment_successful = optimized_info.fits_constraints;

        // Generate deployment artifacts for ESP32
        let artifacts = self.generate_esp32_artifacts(model, &optimization_result)?;

        // Measure performance metrics
        let performance_metrics =
            self.measure_esp32_performance(model, &optimization_result, target)?;

        // Collect warnings
        let mut warnings = Vec::new();
        if !deployment_successful {
            warnings.push(
                "Model does not fit ESP32 memory constraints even after optimization".to_string(),
            );
        }
        if performance_metrics.estimated_inference_time_ms > self.config.max_inference_time_ms {
            warnings.push(format!(
                "Estimated inference time {:.1}ms exceeds target {:.1}ms",
                performance_metrics.estimated_inference_time_ms, self.config.max_inference_time_ms
            ));
        }

        let result = DeploymentValidationResult {
            target_name: target_name.to_string(),
            original_model_info: original_info,
            optimized_model_info: optimized_info,
            deployment_successful,
            performance_metrics,
            deployment_artifacts: artifacts,
            validation_timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            warnings,
        };

        log::info!(
            "✅ ESP32 deployment validation completed: success={}",
            deployment_successful
        );
        Ok(result)
    }

    /// Validate deployment for Arduino Nano 33 BLE (ultra-low-power)
    pub fn validate_arduino_deployment(
        &self,
        model: &Model,
        optimizer: &Optimizer,
    ) -> Result<DeploymentValidationResult> {
        log::info!("🔋 Starting Arduino Nano 33 BLE deployment validation");

        let target_name = "Arduino Nano 33 BLE";
        let target = self.target_registry.get_target("arduino")?;

        // Arduino requires extremely aggressive optimization
        let original_info = self.analyze_model_deployment_info(model, target, vec![])?;

        // Apply optimizations
        let optimization_result = optimizer.optimize(model)?;

        let optimized_info = self.analyze_model_deployment_info(
            model,
            target,
            optimization_result.techniques_applied.clone(),
        )?;

        // Arduino has very strict constraints (< 32KB for model)
        let deployment_successful =
            optimized_info.fits_constraints && optimized_info.model_size < 32 * 1024;

        let artifacts = self.generate_arduino_artifacts(model, &optimization_result)?;
        let performance_metrics =
            self.measure_arduino_performance(model, &optimization_result, target)?;

        let mut warnings = Vec::new();
        if optimized_info.model_size >= 32 * 1024 {
            warnings.push(format!(
                "Model size {}KB exceeds 32KB limit for Arduino deployment",
                optimized_info.model_size / 1024
            ));
        }
        if !deployment_successful {
            warnings.push("Model requires further optimization for Arduino deployment".to_string());
        }

        let result = DeploymentValidationResult {
            target_name: target_name.to_string(),
            original_model_info: original_info,
            optimized_model_info: optimized_info,
            deployment_successful,
            performance_metrics,
            deployment_artifacts: artifacts,
            validation_timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            warnings,
        };

        log::info!(
            "✅ Arduino deployment validation completed: success={}",
            deployment_successful
        );
        Ok(result)
    }

    /// Validate deployment for STM32 with FPU optimization
    pub fn validate_stm32_deployment(
        &self,
        model: &Model,
        optimizer: &Optimizer,
    ) -> Result<DeploymentValidationResult> {
        log::info!("⚡ Starting STM32 deployment validation");

        let target_name = "STM32F4";
        let target = self.target_registry.get_target("stm32")?;

        let original_info = self.analyze_model_deployment_info(model, target, vec![])?;

        // Apply optimizations
        let optimization_result = optimizer.optimize(model)?;

        let optimized_info = self.analyze_model_deployment_info(
            model,
            target,
            optimization_result.techniques_applied.clone(),
        )?;

        let deployment_successful = optimized_info.fits_constraints;

        let artifacts = self.generate_stm32_artifacts(model, &optimization_result)?;
        let performance_metrics =
            self.measure_stm32_performance(model, &optimization_result, target)?;

        let mut warnings = Vec::new();
        if !deployment_successful {
            warnings.push("Model exceeds STM32 memory constraints".to_string());
        }

        // STM32 should leverage FPU for better performance
        if !artifacts.build_ready {
            warnings.push("Generated artifacts may need manual build configuration".to_string());
        }

        let result = DeploymentValidationResult {
            target_name: target_name.to_string(),
            original_model_info: original_info,
            optimized_model_info: optimized_info,
            deployment_successful,
            performance_metrics,
            deployment_artifacts: artifacts,
            validation_timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            warnings,
        };

        log::info!(
            "✅ STM32 deployment validation completed: success={}",
            deployment_successful
        );
        Ok(result)
    }

    /// Analyze model deployment characteristics for a target
    fn analyze_model_deployment_info(
        &self,
        model: &Model,
        target: &dyn HardwareTarget,
        optimizations_applied: Vec<String>,
    ) -> Result<ModelDeploymentInfo> {
        let info = model.info();
        let constraints = target.constraints();

        // Base model size
        let model_size = info.model_size_bytes;

        // Estimate memory usage including activations and intermediate buffers
        let activation_memory = self.estimate_activation_memory(model);
        let memory_usage = model_size + activation_memory + (activation_memory / 2); // Add buffer overhead

        // Check if model fits constraints
        let fits_constraints =
            model_size <= constraints.storage_limit && memory_usage <= constraints.memory_limit;

        Ok(ModelDeploymentInfo {
            model_size,
            memory_usage,
            parameter_count: info.parameter_count,
            operations_count: info.operations_count,
            fits_constraints,
            optimizations_applied,
        })
    }

    /// Estimate activation memory requirements
    fn estimate_activation_memory(&self, model: &Model) -> usize {
        let info = model.info();

        // Estimate based on input/output shapes and layer complexity
        let input_memory: usize = info
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<i64>() as usize * 4) // FP32
            .sum();

        let output_memory: usize = info
            .output_shapes
            .iter()
            .map(|shape| shape.iter().product::<i64>() as usize * 4) // FP32
            .sum();

        // Add intermediate layer activations (rough estimate)
        let intermediate_memory = (input_memory + output_memory) * info.layers.len() / 4;

        input_memory + output_memory + intermediate_memory
    }

    /// Generate ESP32-specific deployment artifacts
    fn generate_esp32_artifacts(
        &self,
        _model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
    ) -> Result<DeploymentArtifacts> {
        log::debug!("Generating ESP32 deployment artifacts");

        // Simulate artifact generation (would use actual code generator)
        let mut artifacts = DeploymentArtifacts {
            source_files: vec![
                PathBuf::from("esp32_model.c"),
                PathBuf::from("esp32_inference.c"),
            ],
            header_files: vec![
                PathBuf::from("esp32_model.h"),
                PathBuf::from("esp32_config.h"),
            ],
            build_files: vec![PathBuf::from("CMakeLists.txt"), PathBuf::from("sdkconfig")],
            example_files: vec![
                PathBuf::from("main.c"),
                PathBuf::from("partition_table.csv"),
            ],
            total_size_bytes: 0,
            build_ready: true,
        };

        // Estimate artifact sizes based on model complexity
        let base_size = 15000; // Base ESP32 code size
        let model_size = optimization_result.optimized_size;
        artifacts.total_size_bytes = base_size + model_size + 5000; // Add overhead

        Ok(artifacts)
    }

    /// Generate Arduino-specific deployment artifacts
    fn generate_arduino_artifacts(
        &self,
        _model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
    ) -> Result<DeploymentArtifacts> {
        log::debug!("Generating Arduino deployment artifacts");

        let mut artifacts = DeploymentArtifacts {
            source_files: vec![PathBuf::from("arduino_model.cpp")],
            header_files: vec![PathBuf::from("arduino_model.h")],
            build_files: vec![], // Arduino uses .ino files
            example_files: vec![PathBuf::from("inference_example.ino")],
            total_size_bytes: 0,
            build_ready: true,
        };

        // Arduino artifacts are typically smaller
        let base_size = 8000; // Base Arduino code size
        let model_size = optimization_result.optimized_size;
        artifacts.total_size_bytes = base_size + model_size + 2000; // Add minimal overhead

        Ok(artifacts)
    }

    /// Generate STM32-specific deployment artifacts
    fn generate_stm32_artifacts(
        &self,
        _model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
    ) -> Result<DeploymentArtifacts> {
        log::debug!("Generating STM32 deployment artifacts");

        let mut artifacts = DeploymentArtifacts {
            source_files: vec![
                PathBuf::from("stm32_model.c"),
                PathBuf::from("stm32_inference.c"),
                PathBuf::from("stm32_hal_config.c"),
            ],
            header_files: vec![
                PathBuf::from("stm32_model.h"),
                PathBuf::from("stm32_config.h"),
            ],
            build_files: vec![PathBuf::from("Makefile"), PathBuf::from("linker_script.ld")],
            example_files: vec![PathBuf::from("main.c"), PathBuf::from("system_config.c")],
            total_size_bytes: 0,
            build_ready: true,
        };

        // STM32 artifacts include HAL and FPU optimization
        let base_size = 20000; // Base STM32 HAL code size
        let model_size = optimization_result.optimized_size;
        artifacts.total_size_bytes = base_size + model_size + 8000; // Add HAL overhead

        Ok(artifacts)
    }

    /// Measure ESP32-specific performance metrics
    fn measure_esp32_performance(
        &self,
        model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
        target: &dyn HardwareTarget,
    ) -> Result<PerformanceMetrics> {
        let start_time = Instant::now();

        // Simulate performance measurement based on ESP32 characteristics
        let info = model.info();
        let constraints = target.constraints();

        // ESP32 dual-core @ 240MHz with WiFi capabilities
        let cpu_factor = constraints.cpu_frequency as f64 / 100.0; // Normalize to 100MHz base
        let operations_per_ms = cpu_factor * 10000.0; // Operations per millisecond

        let estimated_inference_time_ms = info.operations_count as f64 / operations_per_ms;
        let memory_usage =
            optimization_result.optimized_size + self.estimate_activation_memory(model);
        let memory_efficiency = 1.0 - (memory_usage as f64 / constraints.memory_limit as f64);

        // ESP32 power efficiency is moderate (WiFi/BT consume power)
        let power_efficiency = 0.7 - (estimated_inference_time_ms / 1000.0) * 0.1;
        let power_efficiency = power_efficiency.clamp(0.1, 1.0);

        let latency_accuracy_score =
            1.0 - (optimization_result.estimated_accuracy_loss as f64 / 100.0);

        let code_generation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(PerformanceMetrics {
            estimated_inference_time_ms: estimated_inference_time_ms.max(1.0),
            memory_efficiency: memory_efficiency.clamp(0.0, 1.0),
            power_efficiency,
            latency_accuracy_score: latency_accuracy_score.clamp(0.0, 1.0),
            code_generation_time_ms,
        })
    }

    /// Measure Arduino-specific performance metrics
    fn measure_arduino_performance(
        &self,
        model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
        target: &dyn HardwareTarget,
    ) -> Result<PerformanceMetrics> {
        let start_time = Instant::now();

        let info = model.info();
        let constraints = target.constraints();

        // Arduino Nano 33 BLE: ARM Cortex-M4 @ 64MHz
        let cpu_factor = constraints.cpu_frequency as f64 / 100.0;
        let operations_per_ms = cpu_factor * 5000.0; // More conservative for low-power ARM

        let estimated_inference_time_ms = info.operations_count as f64 / operations_per_ms;
        let memory_usage =
            optimization_result.optimized_size + self.estimate_activation_memory(model);
        let memory_efficiency = 1.0 - (memory_usage as f64 / constraints.memory_limit as f64);

        // Arduino optimized for power efficiency
        let power_efficiency = 0.95 - (estimated_inference_time_ms / 5000.0) * 0.2;
        let power_efficiency = power_efficiency.clamp(0.3, 1.0);

        let latency_accuracy_score =
            1.0 - (optimization_result.estimated_accuracy_loss as f64 / 100.0);

        let code_generation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(PerformanceMetrics {
            estimated_inference_time_ms: estimated_inference_time_ms.max(10.0),
            memory_efficiency: memory_efficiency.clamp(0.0, 1.0),
            power_efficiency,
            latency_accuracy_score: latency_accuracy_score.clamp(0.0, 1.0),
            code_generation_time_ms,
        })
    }

    /// Measure STM32-specific performance metrics
    fn measure_stm32_performance(
        &self,
        model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
        target: &dyn HardwareTarget,
    ) -> Result<PerformanceMetrics> {
        let start_time = Instant::now();

        let info = model.info();
        let constraints = target.constraints();

        // STM32F4: ARM Cortex-M4 @ 168MHz with FPU
        let cpu_factor = constraints.cpu_frequency as f64 / 100.0;
        let fpu_boost = if constraints.has_fpu { 1.5 } else { 1.0 }; // FPU acceleration
        let operations_per_ms = cpu_factor * fpu_boost * 15000.0; // Higher with FPU

        let estimated_inference_time_ms = info.operations_count as f64 / operations_per_ms;
        let memory_usage =
            optimization_result.optimized_size + self.estimate_activation_memory(model);
        let memory_efficiency = 1.0 - (memory_usage as f64 / constraints.memory_limit as f64);

        // STM32 balanced performance and efficiency
        let power_efficiency = 0.8 - (estimated_inference_time_ms / 2000.0) * 0.15;
        let power_efficiency = power_efficiency.clamp(0.2, 1.0);

        let latency_accuracy_score =
            1.0 - (optimization_result.estimated_accuracy_loss as f64 / 100.0);

        let code_generation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(PerformanceMetrics {
            estimated_inference_time_ms: estimated_inference_time_ms.max(5.0),
            memory_efficiency: memory_efficiency.clamp(0.0, 1.0),
            power_efficiency,
            latency_accuracy_score: latency_accuracy_score.clamp(0.0, 1.0),
            code_generation_time_ms,
        })
    }

    /// Generate comprehensive deployment report
    pub fn generate_deployment_report(&self, results: &[DeploymentValidationResult]) -> String {
        let mut report = String::new();

        report.push_str("# Hardware Deployment Validation Report\n\n");
        report.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for result in results {
            report.push_str(&format!("## {} Deployment\n\n", result.target_name));

            // Deployment status
            let status = if result.deployment_successful {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            };
            report.push_str(&format!("**Status**: {}\n\n", status));

            // Model comparison
            report.push_str("### Model Analysis\n\n");
            report.push_str("| Metric | Original | Optimized | Change |\n");
            report.push_str("|--------|----------|-----------|--------|\n");

            let size_change = (result.optimized_model_info.model_size as f64
                / result.original_model_info.model_size as f64
                - 1.0)
                * 100.0;
            report.push_str(&format!(
                "| Model Size | {:.1} KB | {:.1} KB | {:.1}% |\n",
                result.original_model_info.model_size as f64 / 1024.0,
                result.optimized_model_info.model_size as f64 / 1024.0,
                size_change
            ));

            let mem_change = (result.optimized_model_info.memory_usage as f64
                / result.original_model_info.memory_usage as f64
                - 1.0)
                * 100.0;
            report.push_str(&format!(
                "| Memory Usage | {:.1} KB | {:.1} KB | {:.1}% |\n",
                result.original_model_info.memory_usage as f64 / 1024.0,
                result.optimized_model_info.memory_usage as f64 / 1024.0,
                mem_change
            ));

            // Performance metrics
            report.push_str("\n### Performance Metrics\n\n");
            report.push_str(&format!(
                "- **Inference Time**: {:.1} ms\n",
                result.performance_metrics.estimated_inference_time_ms
            ));
            report.push_str(&format!(
                "- **Memory Efficiency**: {:.1}%\n",
                result.performance_metrics.memory_efficiency * 100.0
            ));
            report.push_str(&format!(
                "- **Power Efficiency**: {:.1}%\n",
                result.performance_metrics.power_efficiency * 100.0
            ));
            report.push_str(&format!(
                "- **Latency/Accuracy Score**: {:.1}%\n",
                result.performance_metrics.latency_accuracy_score * 100.0
            ));

            // Optimizations applied
            if !result.optimized_model_info.optimizations_applied.is_empty() {
                report.push_str("\n### Optimizations Applied\n\n");
                for opt in &result.optimized_model_info.optimizations_applied {
                    report.push_str(&format!("- {}\n", opt));
                }
            }

            // Warnings
            if !result.warnings.is_empty() {
                report.push_str("\n### Warnings\n\n");
                for warning in &result.warnings {
                    report.push_str(&format!("⚠️ {}\n", warning));
                }
            }

            report.push_str("\n---\n\n");
        }

        report
    }

    /// Enhanced deployment validation using simulation framework
    pub async fn validate_with_enhanced_simulation(
        &self,
        model: &Model,
        optimizer: &Optimizer,
        targets: &[&str],
    ) -> Result<
        Vec<(
            String,
            DeploymentValidationResult,
            crate::simulation::SimulationResult,
        )>,
    > {
        use crate::simulation::{SimulationConfig, SimulationFramework};

        log::info!("🚀 Starting enhanced deployment validation with simulation framework");

        // Initialize simulation framework
        let sim_config = SimulationConfig {
            target_names: targets.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let mut simulation_framework = SimulationFramework::new(sim_config)?;

        let mut results = Vec::new();

        // Validate each target
        for target_name in targets {
            log::info!("🎯 Validating deployment for target: {}", target_name);

            // Run traditional deployment validation
            let deployment_result = match target_name.to_lowercase().as_str() {
                "esp32" => self.validate_esp32_deployment(model, optimizer)?,
                "arduino" => self.validate_arduino_deployment(model, optimizer)?,
                "stm32" => self.validate_stm32_deployment(model, optimizer)?,
                "raspberry_pi" | "raspi" => {
                    // Create a generic deployment validation for Raspberry Pi
                    let target = self.target_registry.get_target("raspberry_pi")?;
                    let original_info =
                        self.analyze_model_deployment_info(model, target, vec![])?;
                    let optimization_result = optimizer.optimize(model)?;
                    let optimized_info = self.analyze_model_deployment_info(
                        model,
                        target,
                        optimization_result.techniques_applied.clone(),
                    )?;

                    DeploymentValidationResult {
                        target_name: "Raspberry Pi".to_string(),
                        original_model_info: original_info,
                        optimized_model_info: optimized_info.clone(),
                        deployment_successful: optimized_info.fits_constraints,
                        performance_metrics: self.measure_raspberry_pi_performance(
                            model,
                            &optimization_result,
                            target,
                        )?,
                        deployment_artifacts: self
                            .generate_raspberry_pi_artifacts(model, &optimization_result)?,
                        validation_timestamp: chrono::Utc::now()
                            .format("%Y-%m-%d %H:%M:%S UTC")
                            .to_string(),
                        warnings: Vec::new(),
                    }
                }
                _ => {
                    return Err(BlitzedError::UnsupportedTarget {
                        target: target_name.to_string(),
                    })
                }
            };

            // Run enhanced simulation
            let simulation_result = simulation_framework
                .simulate_deployment(model, target_name, &deployment_result.deployment_artifacts)
                .await?;

            log::info!(
                "✅ {}: Deployment {} | Simulation {} (confidence: {:.1}%)",
                target_name,
                if deployment_result.deployment_successful {
                    "✓"
                } else {
                    "✗"
                },
                if simulation_result.simulation_successful {
                    "✓"
                } else {
                    "✗"
                },
                simulation_result.confidence_score * 100.0
            );

            results.push((
                target_name.to_string(),
                deployment_result,
                simulation_result,
            ));
        }

        log::info!(
            "🎉 Enhanced deployment validation completed for {} targets",
            targets.len()
        );
        Ok(results)
    }

    /// Generate Raspberry Pi deployment artifacts
    fn generate_raspberry_pi_artifacts(
        &self,
        model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
    ) -> Result<DeploymentArtifacts> {
        use crate::codegen::raspberry_pi::RaspberryPiCodeGen;
        use crate::codegen::CodeGenerator;

        let codegen = RaspberryPiCodeGen::new();
        let output_dir = self
            .config
            .output_directory
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("blitzed_raspberry_pi"));

        let _generated_code = codegen.generate(model, &output_dir)?;

        // Calculate deployment size
        let deployment_size = optimization_result.optimized_size + 50000; // Add Pi runtime overhead

        Ok(DeploymentArtifacts {
            source_files: vec!["main.c".into(), "blitzed_model.c".into()],
            header_files: vec!["blitzed_model.h".into()],
            build_files: vec!["Makefile".into()],
            example_files: vec![output_dir.join("blitzed_benchmark")],
            total_size_bytes: deployment_size,
            build_ready: true,
        })
    }

    /// Measure Raspberry Pi performance metrics
    fn measure_raspberry_pi_performance(
        &self,
        model: &Model,
        optimization_result: &crate::optimization::OptimizationResult,
        target: &dyn HardwareTarget,
    ) -> Result<PerformanceMetrics> {
        let start_time = Instant::now();
        let info = model.info();
        let constraints = target.constraints();

        // Raspberry Pi 4B: ARM Cortex-A72 quad-core @ 1.5GHz
        let cpu_factor = constraints.cpu_frequency as f64 / 100.0;
        let operations_per_ms = cpu_factor * 50000.0; // High-performance ARM with good IPC

        let estimated_inference_time_ms = info.operations_count as f64 / operations_per_ms;
        let memory_usage =
            optimization_result.optimized_size + self.estimate_activation_memory(model);
        let memory_efficiency = 1.0 - (memory_usage as f64 / constraints.memory_limit as f64);

        // Raspberry Pi has good sustained performance
        let power_efficiency = 0.8 - (estimated_inference_time_ms / 10000.0) * 0.1;
        let power_efficiency = power_efficiency.clamp(0.3, 1.0);

        let latency_accuracy_score =
            1.0 - (optimization_result.estimated_accuracy_loss as f64 / 100.0);
        let code_generation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(PerformanceMetrics {
            estimated_inference_time_ms: estimated_inference_time_ms.max(0.1),
            memory_efficiency: memory_efficiency.clamp(0.0, 1.0),
            power_efficiency,
            latency_accuracy_score: latency_accuracy_score.clamp(0.0, 1.0),
            code_generation_time_ms,
        })
    }

    /// Get validation configuration
    pub fn config(&self) -> &DeploymentValidationConfig {
        &self.config
    }

    /// Update validation configuration
    pub fn set_config(&mut self, config: DeploymentValidationConfig) {
        self.config = config;
    }
}

impl Default for HardwareDeploymentValidator {
    fn default() -> Self {
        Self::new()
    }
}
