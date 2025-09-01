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

//! Real-time Constraint Monitoring and Validation
//!
//! This module provides real-time monitoring of hardware constraints
//! during simulation to detect violations and ensure deployment viability.

use crate::simulation::performance::SimulationMetrics;
use crate::targets::{HardwareConstraints, TargetRegistry};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Constraint monitoring system for hardware validation
pub struct ConstraintMonitor {
    /// Whether monitoring is enabled
    enabled: bool,
    /// Active monitoring sessions by target
    active_sessions: HashMap<String, MonitoringSession>,
    /// Violation history for analysis
    violation_history: Vec<ConstraintViolation>,
    /// Target registry for constraint lookup
    target_registry: TargetRegistry,
}

/// Active monitoring session for a target
#[derive(Debug)]
struct MonitoringSession {
    target_name: String,
    start_time: Instant,
    constraints: HardwareConstraints,
    real_time_metrics: Vec<RealTimeMetric>,
    violation_count: usize,
}

/// Real-time metric sample during monitoring
#[derive(Debug, Clone)]
struct RealTimeMetric {
    timestamp: Instant,
    memory_usage: u64,
    cpu_utilization: f32,
    power_consumption: f32,
    temperature: f32,
}

/// Constraint violation detected during simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Target where violation occurred
    pub target_name: String,
    /// Type of constraint violated
    pub violation_type: ViolationType,
    /// Severity of the violation
    pub severity: ViolationSeverity,
    /// Measured value that caused violation
    pub measured_value: f64,
    /// Constraint limit that was exceeded
    pub constraint_limit: f64,
    /// Timestamp when violation was detected
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration of the violation
    pub duration_ms: u64,
    /// Suggested remediation actions
    pub remediation_suggestions: Vec<String>,
}

/// Types of constraint violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Memory usage exceeded limit
    MemoryOverage,
    /// Storage/flash usage exceeded limit
    StorageOverage,
    /// CPU utilization too high for sustained operation
    CpuOverutilization,
    /// Power consumption exceeded budget
    PowerOverage,
    /// Temperature exceeded safe operating range
    ThermalOverage,
    /// Inference time exceeded real-time requirements
    LatencyOverage,
    /// Memory bandwidth saturation
    BandwidthSaturation,
    /// Cache miss rate too high
    CacheInefficiency,
}

/// Severity levels for constraint violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ViolationSeverity {
    /// Minor violation, system can still operate
    Warning,
    /// Significant violation, may impact performance
    Critical,
    /// Severe violation, system likely to fail
    Fatal,
}

/// Constraint monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConfig {
    /// Monitoring sample interval in milliseconds
    pub sample_interval_ms: u64,
    /// Memory usage warning threshold (percentage of limit)
    pub memory_warning_threshold: f32,
    /// CPU utilization warning threshold (percentage)
    pub cpu_warning_threshold: f32,
    /// Power consumption warning threshold (percentage of limit)
    pub power_warning_threshold: f32,
    /// Temperature warning threshold (degrees Celsius above ambient)
    pub temperature_warning_threshold: f32,
    /// Enable predictive constraint analysis
    pub enable_predictive_analysis: bool,
}

/// Results from constraint validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationResult {
    /// Target that was validated
    pub target_name: String,
    /// Whether all constraints were satisfied
    pub all_constraints_satisfied: bool,
    /// List of detected violations
    pub violations: Vec<ConstraintViolation>,
    /// Overall constraint health score (0.0-1.0)
    pub health_score: f32,
    /// Predicted time until constraint failure (if applicable)
    pub time_to_failure_ms: Option<u64>,
    /// Resource utilization summary
    pub resource_utilization: ResourceUtilizationSummary,
}

/// Summary of resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationSummary {
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f32,
    /// Storage utilization (0.0-1.0)  
    pub storage_utilization: f32,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// Power utilization (0.0-1.0)
    pub power_utilization: f32,
    /// Thermal utilization (0.0-1.0)
    pub thermal_utilization: f32,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            sample_interval_ms: 100,             // 100ms sampling
            memory_warning_threshold: 0.8,       // 80% of memory limit
            cpu_warning_threshold: 0.85,         // 85% CPU utilization
            power_warning_threshold: 0.9,        // 90% of power budget
            temperature_warning_threshold: 15.0, // 15°C above ambient
            enable_predictive_analysis: true,
        }
    }
}

impl ConstraintMonitor {
    /// Create new constraint monitor
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            active_sessions: HashMap::new(),
            violation_history: Vec::new(),
            target_registry: TargetRegistry::new(),
        }
    }

    /// Start monitoring constraints for a target
    pub fn start_monitoring(&mut self, target_name: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Get hardware constraints for target
        let target = self.target_registry.get_target(target_name)?;

        let constraints = target.constraints().clone();

        // Create monitoring session
        let session = MonitoringSession {
            target_name: target_name.to_string(),
            start_time: Instant::now(),
            constraints,
            real_time_metrics: Vec::new(),
            violation_count: 0,
        };

        self.active_sessions
            .insert(target_name.to_string(), session);

        log::debug!("Started constraint monitoring for target: {}", target_name);
        Ok(())
    }

    /// Check for constraint violations in simulation metrics
    pub fn check_violations(
        &mut self,
        metrics: &SimulationMetrics,
    ) -> Result<Vec<ConstraintViolation>> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        // Ensure monitoring session exists
        if !self.active_sessions.contains_key(&metrics.target_name) {
            self.start_monitoring(&metrics.target_name)?;
        }

        let config = ConstraintConfig::default();
        let now = Instant::now();

        // Get session data (avoid concurrent borrows)
        let constraints = {
            let session = self.active_sessions.get(&metrics.target_name).unwrap();
            session.constraints.clone()
        };

        // Record current metrics
        let current_metric = RealTimeMetric {
            timestamp: now,
            memory_usage: metrics.memory_usage_bytes,
            cpu_utilization: metrics.cpu_utilization_percent,
            power_consumption: metrics.power_consumption_mw,
            temperature: 25.0 + metrics.thermal_metrics.temperature_rise_celsius, // Assume 25°C ambient
        };

        // Update session
        if let Some(session) = self.active_sessions.get_mut(&metrics.target_name) {
            session.real_time_metrics.push(current_metric);

            // Keep metrics history manageable
            if session.real_time_metrics.len() > 1000 {
                session.real_time_metrics.remove(0);
            }
        }

        let mut violations = Vec::new();

        // Check memory constraints
        if let Some(violation) = self.check_memory_constraint(metrics, &constraints, &config) {
            violations.push(violation);
        }

        // Check CPU utilization constraints
        if let Some(violation) = self.check_cpu_constraint(metrics, &constraints, &config) {
            violations.push(violation);
        }

        // Check power constraints
        if let Some(violation) = self.check_power_constraint(metrics, &constraints, &config) {
            violations.push(violation);
        }

        // Check thermal constraints
        if let Some(violation) = self.check_thermal_constraint(metrics, &config) {
            violations.push(violation);
        }

        // Check latency constraints (if real-time requirements exist)
        if let Some(violation) = self.check_latency_constraint(metrics, &constraints) {
            violations.push(violation);
        }

        // Update violation count and history
        if let Some(session) = self.active_sessions.get_mut(&metrics.target_name) {
            session.violation_count += violations.len();
        }

        // Store violations in history
        for violation in &violations {
            self.violation_history.push(violation.clone());
        }

        // Keep violation history manageable
        if self.violation_history.len() > 5000 {
            self.violation_history.remove(0);
        }

        Ok(violations)
    }

    /// Check memory usage against constraints
    fn check_memory_constraint(
        &self,
        metrics: &SimulationMetrics,
        constraints: &HardwareConstraints,
        config: &ConstraintConfig,
    ) -> Option<ConstraintViolation> {
        let memory_usage_ratio =
            metrics.memory_usage_bytes as f64 / constraints.memory_limit as f64;

        if memory_usage_ratio > 1.0 {
            // Fatal: Memory usage exceeds available memory
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::MemoryOverage,
                severity: ViolationSeverity::Fatal,
                measured_value: metrics.memory_usage_bytes as f64,
                constraint_limit: constraints.memory_limit as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0, // Instantaneous detection
                remediation_suggestions: vec![
                    "Reduce model size through quantization".to_string(),
                    "Apply pruning to reduce memory footprint".to_string(),
                    "Use knowledge distillation to create smaller model".to_string(),
                ],
            });
        } else if memory_usage_ratio > config.memory_warning_threshold as f64 {
            // Warning: Memory usage approaching limit
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::MemoryOverage,
                severity: ViolationSeverity::Warning,
                measured_value: metrics.memory_usage_bytes as f64,
                constraint_limit: constraints.memory_limit as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Consider optimizing model to reduce memory usage".to_string(),
                    "Monitor memory usage closely during deployment".to_string(),
                ],
            });
        }

        None
    }

    /// Check CPU utilization constraints
    fn check_cpu_constraint(
        &self,
        metrics: &SimulationMetrics,
        _constraints: &HardwareConstraints,
        config: &ConstraintConfig,
    ) -> Option<ConstraintViolation> {
        if metrics.cpu_utilization_percent > 100.0 {
            // Fatal: CPU utilization over 100% (impossible in practice)
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::CpuOverutilization,
                severity: ViolationSeverity::Fatal,
                measured_value: metrics.cpu_utilization_percent as f64,
                constraint_limit: 100.0,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Model too complex for target hardware".to_string(),
                    "Consider using a less powerful model".to_string(),
                    "Apply aggressive quantization and pruning".to_string(),
                ],
            });
        } else if metrics.cpu_utilization_percent > config.cpu_warning_threshold * 100.0 {
            // Critical: High CPU utilization may impact real-time performance
            let severity = if metrics.cpu_utilization_percent > 95.0 {
                ViolationSeverity::Critical
            } else {
                ViolationSeverity::Warning
            };

            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::CpuOverutilization,
                severity,
                measured_value: metrics.cpu_utilization_percent as f64,
                constraint_limit: (config.cpu_warning_threshold * 100.0) as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Optimize model for better CPU efficiency".to_string(),
                    "Consider using hardware-specific optimizations".to_string(),
                    "Reduce inference frequency if possible".to_string(),
                ],
            });
        }

        None
    }

    /// Check power consumption constraints
    fn check_power_constraint(
        &self,
        metrics: &SimulationMetrics,
        _constraints: &HardwareConstraints,
        config: &ConstraintConfig,
    ) -> Option<ConstraintViolation> {
        // Estimate power budget based on target type
        let power_budget = match metrics.target_name.as_str() {
            "esp32" => 300.0,         // 300mW budget
            "arduino" => 50.0,        // 50mW budget
            "stm32" => 150.0,         // 150mW budget
            "raspberry_pi" => 5000.0, // 5W budget
            _ => 200.0,               // Default budget
        };

        let power_ratio = metrics.power_consumption_mw / power_budget;

        if power_ratio > 1.0 {
            // Critical: Power consumption exceeds budget
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::PowerOverage,
                severity: ViolationSeverity::Critical,
                measured_value: metrics.power_consumption_mw as f64,
                constraint_limit: power_budget as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Reduce inference frequency to save power".to_string(),
                    "Apply quantization to reduce computational load".to_string(),
                    "Consider power-optimized model architecture".to_string(),
                ],
            });
        } else if power_ratio > config.power_warning_threshold {
            // Warning: Power consumption approaching budget
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::PowerOverage,
                severity: ViolationSeverity::Warning,
                measured_value: metrics.power_consumption_mw as f64,
                constraint_limit: (power_budget * config.power_warning_threshold) as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Monitor power consumption during deployment".to_string(),
                    "Consider battery life impact".to_string(),
                ],
            });
        }

        None
    }

    /// Check thermal constraints
    fn check_thermal_constraint(
        &self,
        metrics: &SimulationMetrics,
        config: &ConstraintConfig,
    ) -> Option<ConstraintViolation> {
        let temperature_rise = metrics.thermal_metrics.temperature_rise_celsius;

        if temperature_rise > 50.0 {
            // Fatal: Excessive temperature rise
            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::ThermalOverage,
                severity: ViolationSeverity::Fatal,
                measured_value: temperature_rise as f64,
                constraint_limit: 50.0,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Reduce computational load to prevent overheating".to_string(),
                    "Implement thermal throttling in deployment".to_string(),
                    "Consider adding cooling solutions".to_string(),
                ],
            });
        } else if temperature_rise > config.temperature_warning_threshold {
            // Warning: Temperature approaching concerning levels
            let severity = if temperature_rise > 30.0 {
                ViolationSeverity::Critical
            } else {
                ViolationSeverity::Warning
            };

            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::ThermalOverage,
                severity,
                measured_value: temperature_rise as f64,
                constraint_limit: config.temperature_warning_threshold as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Monitor temperature during extended operation".to_string(),
                    "Consider reducing inference frequency".to_string(),
                    "Implement thermal monitoring in deployment".to_string(),
                ],
            });
        }

        None
    }

    /// Check latency constraints for real-time requirements
    fn check_latency_constraint(
        &self,
        metrics: &SimulationMetrics,
        _constraints: &HardwareConstraints,
    ) -> Option<ConstraintViolation> {
        // Define real-time requirements based on target
        let latency_requirement_us = match metrics.target_name.as_str() {
            "esp32" => 500_000,        // 500ms for IoT applications
            "arduino" => 1_000_000,    // 1s for simple sensor applications
            "stm32" => 200_000,        // 200ms for control applications
            "raspberry_pi" => 100_000, // 100ms for interactive applications
            _ => 1_000_000,            // 1s default
        };

        if metrics.inference_time_us > latency_requirement_us {
            let severity = if metrics.inference_time_us > latency_requirement_us * 2 {
                ViolationSeverity::Critical
            } else {
                ViolationSeverity::Warning
            };

            return Some(ConstraintViolation {
                target_name: metrics.target_name.clone(),
                violation_type: ViolationType::LatencyOverage,
                severity,
                measured_value: metrics.inference_time_us as f64,
                constraint_limit: latency_requirement_us as f64,
                timestamp: chrono::Utc::now(),
                duration_ms: 0,
                remediation_suggestions: vec![
                    "Optimize model for faster inference".to_string(),
                    "Apply quantization to reduce computation time".to_string(),
                    "Consider using hardware-specific optimizations".to_string(),
                    "Evaluate if real-time requirements can be relaxed".to_string(),
                ],
            });
        }

        None
    }

    /// Generate comprehensive constraint validation result
    pub fn generate_validation_result(
        &self,
        target_name: &str,
        violations: &[ConstraintViolation],
    ) -> Result<ConstraintValidationResult> {
        let session = self.active_sessions.get(target_name);

        let all_satisfied = violations.is_empty();

        // Calculate health score based on violations
        let health_score = if violations.is_empty() {
            1.0
        } else {
            let fatal_count = violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Fatal)
                .count();
            let critical_count = violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .count();
            let _warning_count = violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Warning)
                .count();

            // Weighted scoring: Fatal=0.0, Critical=0.3, Warning=0.7
            let base_score = if fatal_count > 0 {
                0.0
            } else if critical_count > 0 {
                0.3
            } else {
                0.7
            };

            // Adjust for number of violations
            let violation_penalty = (violations.len() as f32 * 0.1).min(0.5);
            (base_score - violation_penalty).max(0.0)
        };

        // Calculate resource utilization if session exists
        let resource_utilization = if let Some(session) = session {
            if let Some(latest_metric) = session.real_time_metrics.last() {
                ResourceUtilizationSummary {
                    memory_utilization: (latest_metric.memory_usage as f32
                        / session.constraints.memory_limit as f32)
                        .min(1.0),
                    storage_utilization: 0.5, // Placeholder - would need storage tracking
                    cpu_utilization: latest_metric.cpu_utilization / 100.0,
                    power_utilization: latest_metric.power_consumption / 300.0, // Rough estimate
                    thermal_utilization: (latest_metric.temperature - 25.0) / 50.0, // 25°C base, 50°C range
                }
            } else {
                ResourceUtilizationSummary {
                    memory_utilization: 0.0,
                    storage_utilization: 0.0,
                    cpu_utilization: 0.0,
                    power_utilization: 0.0,
                    thermal_utilization: 0.0,
                }
            }
        } else {
            ResourceUtilizationSummary {
                memory_utilization: 0.0,
                storage_utilization: 0.0,
                cpu_utilization: 0.0,
                power_utilization: 0.0,
                thermal_utilization: 0.0,
            }
        };

        // Predictive analysis for time to failure (simplified)
        let time_to_failure_ms = if !violations.is_empty() && health_score < 0.5 {
            Some(300_000) // 5 minutes estimated time to failure
        } else {
            None
        };

        Ok(ConstraintValidationResult {
            target_name: target_name.to_string(),
            all_constraints_satisfied: all_satisfied,
            violations: violations.to_vec(),
            health_score,
            time_to_failure_ms,
            resource_utilization,
        })
    }

    /// Stop monitoring for a target
    pub fn stop_monitoring(&mut self, target_name: &str) {
        self.active_sessions.remove(target_name);
        log::debug!("Stopped constraint monitoring for target: {}", target_name);
    }

    /// Get monitoring statistics
    pub fn get_statistics(&self) -> ConstraintMonitoringStatistics {
        let total_violations = self.violation_history.len();
        let active_sessions = self.active_sessions.len();

        let violations_by_severity = {
            let mut fatal = 0;
            let mut critical = 0;
            let mut warning = 0;

            for violation in &self.violation_history {
                match violation.severity {
                    ViolationSeverity::Fatal => fatal += 1,
                    ViolationSeverity::Critical => critical += 1,
                    ViolationSeverity::Warning => warning += 1,
                }
            }

            (fatal, critical, warning)
        };

        ConstraintMonitoringStatistics {
            enabled: self.enabled,
            active_sessions,
            total_violations,
            violations_by_severity,
        }
    }
}

/// Statistics about constraint monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintMonitoringStatistics {
    pub enabled: bool,
    pub active_sessions: usize,
    pub total_violations: usize,
    pub violations_by_severity: (usize, usize, usize), // (fatal, critical, warning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::performance::{SimulationMetrics, ThermalMetrics};

    #[test]
    fn test_constraint_monitor_creation() {
        let monitor = ConstraintMonitor::new(true);
        assert!(monitor.enabled);
        assert_eq!(monitor.active_sessions.len(), 0);
        assert_eq!(monitor.violation_history.len(), 0);
    }

    #[test]
    fn test_memory_violation_detection() {
        let mut monitor = ConstraintMonitor::new(true);

        let metrics = SimulationMetrics {
            target_name: "esp32".to_string(),
            inference_time_us: 100_000,
            memory_usage_bytes: 600_000, // Exceeds ESP32 520KB limit
            power_consumption_mw: 120.0,
            cpu_utilization_percent: 75.0,
            memory_bandwidth_utilization_percent: 30.0,
            cache_hit_rate: 0.85,
            efficiency_score: 0.82,
            thermal_metrics: ThermalMetrics {
                temperature_rise_celsius: 8.0,
                throttling_likelihood: 0.1,
                sustained_performance: 0.95,
            },
        };

        let violations = monitor.check_violations(&metrics).unwrap();
        assert!(!violations.is_empty());

        // Should detect memory overage
        let memory_violation = violations
            .iter()
            .find(|v| matches!(v.violation_type, ViolationType::MemoryOverage));
        assert!(memory_violation.is_some());
        assert_eq!(memory_violation.unwrap().severity, ViolationSeverity::Fatal);
    }

    #[test]
    fn test_constraint_validation_result() {
        let monitor = ConstraintMonitor::new(true);

        let violations = vec![ConstraintViolation {
            target_name: "esp32".to_string(),
            violation_type: ViolationType::MemoryOverage,
            severity: ViolationSeverity::Warning,
            measured_value: 400_000.0,
            constraint_limit: 520_000.0,
            timestamp: chrono::Utc::now(),
            duration_ms: 0,
            remediation_suggestions: vec!["Optimize model".to_string()],
        }];

        let result = monitor
            .generate_validation_result("esp32", &violations)
            .unwrap();
        assert_eq!(result.target_name, "esp32");
        assert!(!result.all_constraints_satisfied);
        assert_eq!(result.violations.len(), 1);
        assert!(result.health_score > 0.0 && result.health_score < 1.0);
    }

    #[test]
    fn test_no_violations_healthy() {
        let monitor = ConstraintMonitor::new(true);

        let violations = Vec::new();
        let result = monitor
            .generate_validation_result("esp32", &violations)
            .unwrap();

        assert!(result.all_constraints_satisfied);
        assert_eq!(result.violations.len(), 0);
        assert_eq!(result.health_score, 1.0);
        assert!(result.time_to_failure_ms.is_none());
    }
}
