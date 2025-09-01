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

//! Enhanced Simulation Framework
//!
//! This module provides QEMU-based hardware simulation for realistic testing
//! of edge AI models without requiring physical hardware.

pub mod calibration;
pub mod constraints;
pub mod performance;
pub mod qemu;

pub use calibration::{CalibrationData, PerformanceCalibrator};
pub use constraints::{ConstraintMonitor, ConstraintViolation};
pub use performance::{PerformanceModel, SimulationMetrics};
pub use qemu::{QemuConfig, QemuEmulator};

use crate::Result;
use serde::{Deserialize, Serialize};

/// Enhanced simulation framework for hardware validation
pub struct SimulationFramework {
    /// QEMU emulator instances
    emulators: Vec<QemuEmulator>,
    /// Performance modeling system
    performance_model: PerformanceModel,
    /// Constraint monitoring system
    constraint_monitor: ConstraintMonitor,
    /// Performance calibration system
    calibrator: PerformanceCalibrator,
}

/// Configuration for enhanced simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Enable QEMU emulation
    pub enable_qemu_emulation: bool,
    /// QEMU timeout in seconds
    pub qemu_timeout_seconds: u32,
    /// Enable real-time constraint monitoring
    pub enable_constraint_monitoring: bool,
    /// Performance calibration accuracy threshold
    pub calibration_accuracy_threshold: f32,
    /// Hardware targets to simulate
    pub target_names: Vec<String>,
}

/// Results from enhanced simulation validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Target hardware that was simulated
    pub target_name: String,
    /// Whether simulation completed successfully
    pub simulation_successful: bool,
    /// Measured performance metrics
    pub metrics: SimulationMetrics,
    /// Any constraint violations detected
    pub constraint_violations: Vec<ConstraintViolation>,
    /// Simulation confidence score (0.0-1.0)
    pub confidence_score: f32,
    /// Simulation execution time
    pub simulation_time_ms: u64,
    /// Any warnings or notes
    pub warnings: Vec<String>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enable_qemu_emulation: true,
            qemu_timeout_seconds: 300, // 5 minutes
            enable_constraint_monitoring: true,
            calibration_accuracy_threshold: 0.9,
            target_names: vec![
                "esp32".to_string(),
                "arduino".to_string(),
                "stm32".to_string(),
                "raspberry_pi".to_string(),
            ],
        }
    }
}

impl SimulationFramework {
    /// Create new simulation framework
    pub fn new(config: SimulationConfig) -> Result<Self> {
        let emulators = Self::create_emulators(&config)?;
        let performance_model = PerformanceModel::new();
        let constraint_monitor = ConstraintMonitor::new(config.enable_constraint_monitoring);
        let calibrator = PerformanceCalibrator::new(config.calibration_accuracy_threshold);

        Ok(Self {
            emulators,
            performance_model,
            constraint_monitor,
            calibrator,
        })
    }

    /// Create QEMU emulators for configured targets
    fn create_emulators(config: &SimulationConfig) -> Result<Vec<QemuEmulator>> {
        if !config.enable_qemu_emulation {
            return Ok(Vec::new());
        }

        let mut emulators = Vec::new();
        for target_name in &config.target_names {
            match QemuConfig::for_target(target_name) {
                Ok(qemu_config) => {
                    match QemuEmulator::new(qemu_config) {
                        Ok(emulator) => emulators.push(emulator),
                        Err(e) => {
                            log::warn!("Failed to create emulator for {}: {}", target_name, e);
                            // Continue with other targets
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Unsupported target {}: {}", target_name, e);
                    // Continue with other targets - framework can still work without all emulators
                }
            }
        }
        Ok(emulators)
    }

    /// Run enhanced simulation for a model on specified target
    pub async fn simulate_deployment(
        &mut self,
        _model: &crate::Model,
        target_name: &str,
        deployment_artifacts: &crate::deployment::DeploymentArtifacts,
    ) -> Result<SimulationResult> {
        let start_time = std::time::Instant::now();

        // Start constraint monitoring
        self.constraint_monitor.start_monitoring(target_name)?;

        // Run QEMU simulation if emulator available, otherwise fallback to pure simulation
        let qemu_result = if let Some(emulator) = self
            .emulators
            .iter_mut()
            .find(|e| e.target_name() == target_name)
        {
            // Use actual QEMU emulation
            emulator.run_simulation(deployment_artifacts).await?
        } else {
            // Fallback to pure simulation mode
            let qemu_config = qemu::QemuConfig::for_target(target_name)?;
            let dummy_emulator = qemu::QemuEmulator::new(qemu_config)?;
            dummy_emulator
                .simulate_qemu_execution(deployment_artifacts)
                .await?
        };

        // Measure performance metrics
        let metrics = self.performance_model.analyze_execution(&qemu_result)?;

        // Check for constraint violations
        let violations = self.constraint_monitor.check_violations(&metrics)?;

        // Calculate confidence score based on calibration
        let confidence_score = self
            .calibrator
            .calculate_confidence(&metrics, target_name)?;

        let simulation_time = start_time.elapsed().as_millis() as u64;

        // Collect any warnings
        let mut warnings = Vec::new();
        if confidence_score < 0.7 {
            warnings
                .push("Simulation confidence below 70% - consider physical validation".to_string());
        }
        if !violations.is_empty() {
            warnings.push(format!(
                "Detected {} constraint violations",
                violations.len()
            ));
        }

        Ok(SimulationResult {
            target_name: target_name.to_string(),
            simulation_successful: violations.is_empty(),
            metrics,
            constraint_violations: violations,
            confidence_score,
            simulation_time_ms: simulation_time,
            warnings,
        })
    }

    /// Validate multiple targets in parallel
    pub async fn simulate_multi_target(
        &mut self,
        model: &crate::Model,
        deployment_results: &[(String, crate::deployment::DeploymentValidationResult)],
    ) -> Result<Vec<SimulationResult>> {
        let mut results = Vec::new();

        // Run simulations sequentially to avoid borrowing issues
        // In a real-world scenario, you'd want to refactor for parallel execution
        for (target_name, deployment_result) in deployment_results {
            let result = self
                .simulate_deployment(model, target_name, &deployment_result.deployment_artifacts)
                .await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Calibrate performance model against known measurements
    pub fn calibrate_performance_model(
        &mut self,
        calibration_data: Vec<CalibrationData>,
    ) -> Result<f32> {
        self.calibrator
            .calibrate(&mut self.performance_model, calibration_data)
    }

    /// Get simulation statistics
    pub fn get_statistics(&self) -> SimulationStatistics {
        SimulationStatistics {
            total_emulators: self.emulators.len(),
            calibration_accuracy: self.calibrator.current_accuracy(),
            supported_targets: self
                .emulators
                .iter()
                .map(|e| e.target_name().to_string())
                .collect(),
        }
    }
}

/// Statistics about the simulation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStatistics {
    pub total_emulators: usize,
    pub calibration_accuracy: f32,
    pub supported_targets: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_config_default() {
        let config = SimulationConfig::default();
        assert!(config.enable_qemu_emulation);
        assert!(config.enable_constraint_monitoring);
        assert_eq!(config.qemu_timeout_seconds, 300);
        assert!(config.target_names.contains(&"esp32".to_string()));
        assert!(config.target_names.contains(&"raspberry_pi".to_string()));
    }

    #[tokio::test]
    async fn test_simulation_framework_creation() {
        let config = SimulationConfig {
            enable_qemu_emulation: false, // Disable for test
            ..Default::default()
        };

        let framework = SimulationFramework::new(config);
        assert!(framework.is_ok());

        let framework = framework.unwrap();
        assert_eq!(framework.emulators.len(), 0); // No emulators when disabled
    }
}
