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

//! Performance Modeling System for Enhanced Simulation
//!
//! This module provides sophisticated performance modeling that leverages
//! QEMU execution data to provide accurate hardware performance predictions.

use crate::simulation::qemu::QemuExecutionResult;
use crate::{BlitzedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance model for hardware simulation
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// Target-specific performance profiles
    target_profiles: HashMap<String, TargetPerformanceProfile>,
    /// Historical performance data for calibration
    performance_history: Vec<PerformanceDataPoint>,
}

/// Performance profile for a specific hardware target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetPerformanceProfile {
    /// Target hardware name
    pub target_name: String,
    /// CPU performance characteristics
    pub cpu_profile: CpuProfile,
    /// Memory performance characteristics
    pub memory_profile: MemoryProfile,
    /// Power consumption model
    pub power_profile: PowerProfile,
    /// Typical performance ranges
    pub performance_ranges: PerformanceRanges,
}

/// CPU performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    /// Base CPU frequency in MHz
    pub base_frequency_mhz: u32,
    /// Instructions per cycle (IPC) typical range
    pub ipc_typical: f32,
    /// Branch prediction accuracy
    pub branch_prediction_accuracy: f32,
    /// Pipeline depth
    pub pipeline_depth: u32,
    /// Cache hierarchy information
    pub cache_info: CacheInfo,
}

/// Memory performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Memory bandwidth in MB/s
    pub bandwidth_mb_per_sec: u32,
    /// Memory access latency in cycles
    pub access_latency_cycles: u32,
    /// Total available memory in bytes
    pub total_memory_bytes: u64,
    /// Memory hierarchy access patterns
    pub access_patterns: MemoryAccessPatterns,
}

/// Power consumption modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerProfile {
    /// Idle power consumption in mW
    pub idle_power_mw: f32,
    /// Active power consumption in mW
    pub active_power_mw: f32,
    /// Peak power consumption in mW
    pub peak_power_mw: f32,
    /// Power scaling with frequency
    pub power_frequency_scaling: f32,
}

/// Cache hierarchy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    /// L1 instruction cache size in KB
    pub l1i_size_kb: u32,
    /// L1 data cache size in KB
    pub l1d_size_kb: u32,
    /// L2 cache size in KB (0 if not present)
    pub l2_size_kb: u32,
    /// Cache line size in bytes
    pub cache_line_size_bytes: u32,
    /// Cache associativity
    pub associativity: u32,
}

/// Memory access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessPatterns {
    /// Sequential access efficiency multiplier
    pub sequential_efficiency: f32,
    /// Random access penalty multiplier
    pub random_access_penalty: f32,
    /// Typical working set size in KB
    pub working_set_size_kb: u32,
}

/// Performance ranges for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRanges {
    /// Typical inference time range in microseconds
    pub inference_time_us: (u32, u32),
    /// Typical memory usage range in bytes
    pub memory_usage_bytes: (u64, u64),
    /// Typical power consumption range in mW
    pub power_consumption_mw: (f32, f32),
    /// Instructions per inference range
    pub instructions_per_inference: (u64, u64),
}

/// Simulation metrics derived from performance modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetrics {
    /// Target hardware name
    pub target_name: String,
    /// Estimated inference time in microseconds
    pub inference_time_us: u32,
    /// Estimated memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Estimated power consumption in mW
    pub power_consumption_mw: f32,
    /// CPU utilization percentage
    pub cpu_utilization_percent: f32,
    /// Memory bandwidth utilization percentage
    pub memory_bandwidth_utilization_percent: f32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Performance efficiency score (0.0-1.0)
    pub efficiency_score: f32,
    /// Thermal characteristics
    pub thermal_metrics: ThermalMetrics,
}

/// Thermal performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// Estimated temperature rise in Celsius
    pub temperature_rise_celsius: f32,
    /// Thermal throttling likelihood (0.0-1.0)
    pub throttling_likelihood: f32,
    /// Sustained performance capability (0.0-1.0)
    pub sustained_performance: f32,
}

/// Historical performance data point for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// Target hardware
    pub target_name: String,
    /// Timestamp of measurement
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Measured performance metrics
    pub measured_metrics: SimulationMetrics,
    /// QEMU execution data that generated these metrics
    pub qemu_data: Option<QemuExecutionDataSummary>,
    /// Source of measurement (simulation/physical)
    pub measurement_source: MeasurementSource,
}

/// Summary of QEMU execution data for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QemuExecutionDataSummary {
    pub instructions_executed: u64,
    pub cpu_cycles: u64,
    pub memory_accesses: u64,
    pub execution_time_ms: u64,
}

/// Source of performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementSource {
    QemuSimulation,
    PhysicalHardware,
    ModelPrediction,
}

impl PerformanceModel {
    /// Create new performance model
    pub fn new() -> Self {
        let mut target_profiles = HashMap::new();

        // Initialize with default profiles for all supported targets
        target_profiles.insert("esp32".to_string(), Self::esp32_profile());
        target_profiles.insert("arduino".to_string(), Self::arduino_profile());
        target_profiles.insert("stm32".to_string(), Self::stm32_profile());
        target_profiles.insert("raspberry_pi".to_string(), Self::raspberry_pi_profile());

        Self {
            target_profiles,
            performance_history: Vec::new(),
        }
    }

    /// Analyze QEMU execution results and generate performance metrics
    pub fn analyze_execution(
        &self,
        qemu_result: &QemuExecutionResult,
    ) -> Result<SimulationMetrics> {
        let profile = self.get_target_profile(qemu_result)?;

        // Calculate inference time based on execution data and CPU profile
        let inference_time_us = self.calculate_inference_time(qemu_result, profile)?;

        // Estimate memory usage from QEMU memory statistics
        let memory_usage_bytes = qemu_result.memory_stats.peak_memory_usage;

        // Calculate power consumption based on execution characteristics
        let power_consumption_mw = self.calculate_power_consumption(qemu_result, profile)?;

        // Calculate CPU utilization
        let cpu_utilization_percent = self.calculate_cpu_utilization(qemu_result, profile)?;

        // Calculate memory bandwidth utilization
        let memory_bandwidth_utilization =
            self.calculate_memory_bandwidth_utilization(qemu_result, profile)?;

        // Use QEMU-reported cache hit rate
        let cache_hit_rate = qemu_result.memory_stats.cache_hit_rate;

        // Calculate efficiency score
        let efficiency_score = self.calculate_efficiency_score(qemu_result, profile)?;

        // Calculate thermal metrics
        let thermal_metrics = self.calculate_thermal_metrics(power_consumption_mw, profile)?;

        Ok(SimulationMetrics {
            target_name: profile.target_name.clone(),
            inference_time_us,
            memory_usage_bytes,
            power_consumption_mw,
            cpu_utilization_percent,
            memory_bandwidth_utilization_percent: memory_bandwidth_utilization,
            cache_hit_rate,
            efficiency_score,
            thermal_metrics,
        })
    }

    /// Get target profile, with fallback to similar profile
    fn get_target_profile(
        &self,
        qemu_result: &QemuExecutionResult,
    ) -> Result<&TargetPerformanceProfile> {
        // Try to extract target name from QEMU result or use fallback logic
        let target_name = if qemu_result.stdout.contains("esp32") {
            "esp32"
        } else if qemu_result.stdout.contains("cortex-m0") {
            "arduino"
        } else if qemu_result.stdout.contains("cortex-m4") {
            "stm32"
        } else if qemu_result.stdout.contains("cortex-a72") || qemu_result.stdout.contains("raspi")
        {
            "raspberry_pi"
        } else {
            // Default fallback
            "esp32"
        };

        self.target_profiles
            .get(target_name)
            .ok_or_else(|| BlitzedError::UnsupportedTarget {
                target: target_name.to_string(),
            })
    }

    /// Calculate inference time from QEMU execution data
    fn calculate_inference_time(
        &self,
        qemu_result: &QemuExecutionResult,
        profile: &TargetPerformanceProfile,
    ) -> Result<u32> {
        // Base calculation on QEMU execution time, adjusted for CPU profile
        let base_time_us = (qemu_result.execution_time_ms * 1000) as f32;

        // Adjust based on CPU performance characteristics
        let ipc_adjustment = 2.0 / (profile.cpu_profile.ipc_typical + 1.0); // Lower IPC = longer time
        let frequency_adjustment = 1000.0 / profile.cpu_profile.base_frequency_mhz as f32; // Lower freq = longer time

        let adjusted_time = base_time_us * ipc_adjustment * frequency_adjustment;

        // Apply performance range validation
        let (min_time, max_time) = profile.performance_ranges.inference_time_us;
        let clamped_time = adjusted_time.max(min_time as f32).min(max_time as f32);

        Ok(clamped_time as u32)
    }

    /// Calculate power consumption from execution characteristics
    fn calculate_power_consumption(
        &self,
        qemu_result: &QemuExecutionResult,
        profile: &TargetPerformanceProfile,
    ) -> Result<f32> {
        let power_profile = &profile.power_profile;

        // Base power consumption calculation
        let _execution_duration_sec = qemu_result.execution_time_ms as f32 / 1000.0;

        // Calculate dynamic power based on CPU utilization
        let cpu_activity = (qemu_result.performance_counters.instructions_executed as f32)
            / (profile.cpu_profile.base_frequency_mhz as f32
                * _execution_duration_sec
                * 1_000_000.0);
        let cpu_activity_clamped = cpu_activity.clamp(0.1, 1.0); // 10-100% activity

        // Power consumption model: idle + (active - idle) * utilization
        let power_consumption = power_profile.idle_power_mw
            + (power_profile.active_power_mw - power_profile.idle_power_mw) * cpu_activity_clamped;

        // Add memory access power cost
        let memory_power = (qemu_result.memory_stats.memory_accesses as f32) * 0.001; // Rough estimate

        let total_power = power_consumption + memory_power;

        // Validate against power ranges
        let (min_power, max_power) = profile.performance_ranges.power_consumption_mw;
        Ok(total_power.max(min_power).min(max_power))
    }

    /// Calculate CPU utilization percentage
    fn calculate_cpu_utilization(
        &self,
        qemu_result: &QemuExecutionResult,
        profile: &TargetPerformanceProfile,
    ) -> Result<f32> {
        let _execution_duration_sec = qemu_result.execution_time_ms as f32 / 1000.0;
        let max_instructions_per_sec = profile.cpu_profile.base_frequency_mhz as f32
            * profile.cpu_profile.ipc_typical
            * 1_000_000.0;

        let actual_instructions_per_sec =
            qemu_result.performance_counters.instructions_executed as f32 / _execution_duration_sec;

        let utilization = (actual_instructions_per_sec / max_instructions_per_sec) * 100.0;
        Ok(utilization.clamp(0.0, 100.0))
    }

    /// Calculate memory bandwidth utilization
    fn calculate_memory_bandwidth_utilization(
        &self,
        qemu_result: &QemuExecutionResult,
        profile: &TargetPerformanceProfile,
    ) -> Result<f32> {
        let _execution_duration_sec = qemu_result.execution_time_ms as f32 / 1000.0;
        let memory_accesses_per_sec =
            qemu_result.memory_stats.memory_accesses as f32 / _execution_duration_sec;

        // Assume each memory access transfers cache line size
        let bytes_per_sec =
            memory_accesses_per_sec * profile.cpu_profile.cache_info.cache_line_size_bytes as f32;
        let mb_per_sec = bytes_per_sec / (1024.0 * 1024.0);

        let utilization = (mb_per_sec / profile.memory_profile.bandwidth_mb_per_sec as f32) * 100.0;
        Ok(utilization.clamp(0.0, 100.0))
    }

    /// Calculate overall efficiency score
    fn calculate_efficiency_score(
        &self,
        qemu_result: &QemuExecutionResult,
        profile: &TargetPerformanceProfile,
    ) -> Result<f32> {
        // Efficiency is based on multiple factors
        let cache_efficiency = qemu_result.memory_stats.cache_hit_rate;
        let branch_efficiency = qemu_result.performance_counters.branch_prediction_accuracy;

        // IPC efficiency (actual vs theoretical)
        let _execution_duration_sec = qemu_result.execution_time_ms as f32 / 1000.0;
        let actual_ipc = qemu_result.performance_counters.instructions_executed as f32
            / qemu_result.performance_counters.cpu_cycles as f32;
        let ipc_efficiency = (actual_ipc / profile.cpu_profile.ipc_typical).min(1.0);

        // Weighted combination
        let efficiency =
            (cache_efficiency * 0.3) + (branch_efficiency * 0.3) + (ipc_efficiency * 0.4);

        Ok(efficiency.clamp(0.0, 1.0))
    }

    /// Calculate thermal performance metrics
    fn calculate_thermal_metrics(
        &self,
        power_consumption_mw: f32,
        profile: &TargetPerformanceProfile,
    ) -> Result<ThermalMetrics> {
        // Simple thermal model based on power consumption
        let power_ratio = power_consumption_mw / profile.power_profile.peak_power_mw;

        // Temperature rise estimation (very simplified)
        let temperature_rise = power_ratio * 20.0; // Up to 20°C rise at peak power

        // Throttling likelihood increases with power consumption
        let throttling_likelihood = if power_ratio > 0.8 {
            (power_ratio - 0.8) * 5.0 // Rapid increase above 80% power
        } else {
            0.0
        }
        .min(1.0);

        // Sustained performance decreases with thermal load
        let sustained_performance = (1.0 - throttling_likelihood * 0.3).max(0.7);

        Ok(ThermalMetrics {
            temperature_rise_celsius: temperature_rise,
            throttling_likelihood,
            sustained_performance,
        })
    }

    /// Add performance data point for calibration
    pub fn add_performance_data(&mut self, data_point: PerformanceDataPoint) {
        self.performance_history.push(data_point);

        // Keep history manageable (last 1000 points)
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    /// Get performance statistics for a target
    pub fn get_target_statistics(&self, target_name: &str) -> Option<TargetStatistics> {
        let profile = self.target_profiles.get(target_name)?;
        let history_points: Vec<_> = self
            .performance_history
            .iter()
            .filter(|p| p.target_name == target_name)
            .collect();

        Some(TargetStatistics {
            target_name: target_name.to_string(),
            total_measurements: history_points.len(),
            performance_profile: profile.clone(),
            recent_accuracy: self.calculate_recent_accuracy(target_name),
        })
    }

    /// Calculate recent prediction accuracy for calibration
    fn calculate_recent_accuracy(&self, target_name: &str) -> f32 {
        let recent_points: Vec<_> = self
            .performance_history
            .iter()
            .filter(|p| p.target_name == target_name)
            .rev()
            .take(10) // Last 10 measurements
            .collect();

        if recent_points.is_empty() {
            return 0.5; // Default uncertainty
        }

        // Calculate accuracy based on prediction vs measurement variance
        // This is a simplified accuracy calculation
        let accuracy_scores: Vec<f32> = recent_points
            .iter()
            .filter_map(|p| {
                match p.measurement_source {
                    MeasurementSource::PhysicalHardware => {
                        // Compare against model prediction
                        Some(0.85) // Placeholder accuracy score
                    }
                    _ => None,
                }
            })
            .collect();

        if accuracy_scores.is_empty() {
            0.75 // Default for simulation-only data
        } else {
            accuracy_scores.iter().sum::<f32>() / accuracy_scores.len() as f32
        }
    }

    // Target-specific performance profiles
    fn esp32_profile() -> TargetPerformanceProfile {
        TargetPerformanceProfile {
            target_name: "esp32".to_string(),
            cpu_profile: CpuProfile {
                base_frequency_mhz: 240,
                ipc_typical: 0.8,
                branch_prediction_accuracy: 0.85,
                pipeline_depth: 5,
                cache_info: CacheInfo {
                    l1i_size_kb: 32,
                    l1d_size_kb: 32,
                    l2_size_kb: 0,
                    cache_line_size_bytes: 32,
                    associativity: 2,
                },
            },
            memory_profile: MemoryProfile {
                bandwidth_mb_per_sec: 200,
                access_latency_cycles: 3,
                total_memory_bytes: 520 * 1024, // 520KB RAM
                access_patterns: MemoryAccessPatterns {
                    sequential_efficiency: 1.2,
                    random_access_penalty: 1.8,
                    working_set_size_kb: 64,
                },
            },
            power_profile: PowerProfile {
                idle_power_mw: 80.0,
                active_power_mw: 160.0,
                peak_power_mw: 240.0,
                power_frequency_scaling: 1.2,
            },
            performance_ranges: PerformanceRanges {
                inference_time_us: (50_000, 2_000_000), // 50ms - 2s
                memory_usage_bytes: (50_000, 500_000),  // 50KB - 500KB
                power_consumption_mw: (80.0, 240.0),
                instructions_per_inference: (10_000, 1_000_000),
            },
        }
    }

    fn arduino_profile() -> TargetPerformanceProfile {
        TargetPerformanceProfile {
            target_name: "arduino".to_string(),
            cpu_profile: CpuProfile {
                base_frequency_mhz: 16,
                ipc_typical: 0.6,
                branch_prediction_accuracy: 0.75,
                pipeline_depth: 3,
                cache_info: CacheInfo {
                    l1i_size_kb: 8,
                    l1d_size_kb: 8,
                    l2_size_kb: 0,
                    cache_line_size_bytes: 16,
                    associativity: 1,
                },
            },
            memory_profile: MemoryProfile {
                bandwidth_mb_per_sec: 32,
                access_latency_cycles: 2,
                total_memory_bytes: 32 * 1024, // 32KB RAM
                access_patterns: MemoryAccessPatterns {
                    sequential_efficiency: 1.1,
                    random_access_penalty: 2.0,
                    working_set_size_kb: 8,
                },
            },
            power_profile: PowerProfile {
                idle_power_mw: 5.0,
                active_power_mw: 20.0,
                peak_power_mw: 40.0,
                power_frequency_scaling: 0.8,
            },
            performance_ranges: PerformanceRanges {
                inference_time_us: (100_000, 5_000_000), // 100ms - 5s
                memory_usage_bytes: (2_000, 30_000),     // 2KB - 30KB
                power_consumption_mw: (5.0, 40.0),
                instructions_per_inference: (5_000, 100_000),
            },
        }
    }

    fn stm32_profile() -> TargetPerformanceProfile {
        TargetPerformanceProfile {
            target_name: "stm32".to_string(),
            cpu_profile: CpuProfile {
                base_frequency_mhz: 72,
                ipc_typical: 1.0,
                branch_prediction_accuracy: 0.88,
                pipeline_depth: 3,
                cache_info: CacheInfo {
                    l1i_size_kb: 16,
                    l1d_size_kb: 16,
                    l2_size_kb: 0,
                    cache_line_size_bytes: 32,
                    associativity: 2,
                },
            },
            memory_profile: MemoryProfile {
                bandwidth_mb_per_sec: 150,
                access_latency_cycles: 2,
                total_memory_bytes: 192 * 1024, // 192KB RAM
                access_patterns: MemoryAccessPatterns {
                    sequential_efficiency: 1.3,
                    random_access_penalty: 1.6,
                    working_set_size_kb: 32,
                },
            },
            power_profile: PowerProfile {
                idle_power_mw: 15.0,
                active_power_mw: 80.0,
                peak_power_mw: 120.0,
                power_frequency_scaling: 1.0,
            },
            performance_ranges: PerformanceRanges {
                inference_time_us: (30_000, 1_500_000), // 30ms - 1.5s
                memory_usage_bytes: (20_000, 180_000),  // 20KB - 180KB
                power_consumption_mw: (15.0, 120.0),
                instructions_per_inference: (20_000, 800_000),
            },
        }
    }

    fn raspberry_pi_profile() -> TargetPerformanceProfile {
        TargetPerformanceProfile {
            target_name: "raspberry_pi".to_string(),
            cpu_profile: CpuProfile {
                base_frequency_mhz: 1500,
                ipc_typical: 2.0,
                branch_prediction_accuracy: 0.95,
                pipeline_depth: 8,
                cache_info: CacheInfo {
                    l1i_size_kb: 32,
                    l1d_size_kb: 32,
                    l2_size_kb: 512,
                    cache_line_size_bytes: 64,
                    associativity: 4,
                },
            },
            memory_profile: MemoryProfile {
                bandwidth_mb_per_sec: 3200,
                access_latency_cycles: 100,
                total_memory_bytes: 1024 * 1024 * 1024, // 1GB RAM
                access_patterns: MemoryAccessPatterns {
                    sequential_efficiency: 1.5,
                    random_access_penalty: 1.2,
                    working_set_size_kb: 1024,
                },
            },
            power_profile: PowerProfile {
                idle_power_mw: 500.0,
                active_power_mw: 3000.0,
                peak_power_mw: 7500.0,
                power_frequency_scaling: 1.5,
            },
            performance_ranges: PerformanceRanges {
                inference_time_us: (1_000, 500_000),          // 1ms - 500ms
                memory_usage_bytes: (1_000_000, 100_000_000), // 1MB - 100MB
                power_consumption_mw: (500.0, 7500.0),
                instructions_per_inference: (100_000, 10_000_000),
            },
        }
    }
}

impl Default for PerformanceModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a specific target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetStatistics {
    pub target_name: String,
    pub total_measurements: usize,
    pub performance_profile: TargetPerformanceProfile,
    pub recent_accuracy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::qemu::{QemuMemoryStats, QemuPerformanceCounters};

    #[test]
    fn test_performance_model_creation() {
        let model = PerformanceModel::new();
        assert_eq!(model.target_profiles.len(), 4);
        assert!(model.target_profiles.contains_key("esp32"));
        assert!(model.target_profiles.contains_key("raspberry_pi"));
    }

    #[test]
    fn test_esp32_performance_analysis() {
        let model = PerformanceModel::new();

        let qemu_result = QemuExecutionResult {
            exit_code: 0,
            stdout: "esp32 execution completed".to_string(),
            stderr: String::new(),
            execution_time_ms: 200,
            memory_stats: QemuMemoryStats {
                peak_memory_usage: 300_000,
                average_memory_usage: 250_000,
                memory_accesses: 50_000,
                cache_hit_rate: 0.85,
            },
            performance_counters: QemuPerformanceCounters {
                instructions_executed: 100_000,
                cpu_cycles: 125_000,
                instructions_per_second: 500_000,
                branch_prediction_accuracy: 0.88,
                interrupt_count: 25,
            },
        };

        let metrics = model.analyze_execution(&qemu_result).unwrap();
        assert_eq!(metrics.target_name, "esp32");
        assert!(metrics.inference_time_us > 0);
        assert_eq!(metrics.memory_usage_bytes, 300_000);
        assert!(metrics.power_consumption_mw > 0.0);
        assert!(metrics.efficiency_score > 0.0 && metrics.efficiency_score <= 1.0);
    }

    #[test]
    fn test_target_statistics() {
        let mut model = PerformanceModel::new();

        // Add some test data
        let data_point = PerformanceDataPoint {
            target_name: "esp32".to_string(),
            timestamp: chrono::Utc::now(),
            measured_metrics: SimulationMetrics {
                target_name: "esp32".to_string(),
                inference_time_us: 150_000,
                memory_usage_bytes: 280_000,
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
            },
            qemu_data: None,
            measurement_source: MeasurementSource::QemuSimulation,
        };

        model.add_performance_data(data_point);

        let stats = model.get_target_statistics("esp32").unwrap();
        assert_eq!(stats.target_name, "esp32");
        assert_eq!(stats.total_measurements, 1);
        assert!(stats.recent_accuracy > 0.0);
    }
}
