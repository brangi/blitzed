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

//! Performance Calibration System
//!
//! This module provides calibration capabilities to improve simulation
//! accuracy by comparing predicted results with actual measurements.

use crate::simulation::performance::{PerformanceModel, SimulationMetrics};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance calibrator for improving simulation accuracy
#[derive(Debug, Clone)]
pub struct PerformanceCalibrator {
    /// Target accuracy threshold for calibration
    _accuracy_threshold: f32,
    /// Calibration data by target
    calibration_data: HashMap<String, Vec<CalibrationData>>,
    /// Current accuracy scores by target
    accuracy_scores: HashMap<String, f32>,
    /// Calibration statistics
    stats: CalibrationStatistics,
}

/// Calibration data point comparing predictions with measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Target hardware name
    pub target_name: String,
    /// Timestamp of calibration
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Predicted performance metrics
    pub predicted_metrics: SimulationMetrics,
    /// Actual measured metrics (from physical hardware)
    pub measured_metrics: SimulationMetrics,
    /// Model characteristics that generated these results
    pub model_info: ModelCalibrationInfo,
    /// Calibration quality score (0.0-1.0)
    pub quality_score: f32,
    /// Environmental conditions during measurement
    pub environment: EnvironmentalConditions,
}

/// Model information for calibration context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibrationInfo {
    /// Model size in parameters
    pub parameter_count: usize,
    /// Model complexity (operations count)
    pub operations_count: usize,
    /// Quantization level applied
    pub quantization_bits: u8,
    /// Whether pruning was applied
    pub is_pruned: bool,
    /// Model architecture type
    pub architecture_type: String,
}

/// Environmental conditions during measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalConditions {
    /// Ambient temperature in Celsius
    pub ambient_temperature_celsius: f32,
    /// CPU frequency at time of measurement
    pub cpu_frequency_mhz: u32,
    /// System load during measurement
    pub system_load: f32,
    /// Power supply voltage
    pub supply_voltage_v: f32,
}

/// Calibration statistics and accuracy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStatistics {
    /// Total calibration data points
    pub total_data_points: usize,
    /// Average accuracy across all targets
    pub average_accuracy: f32,
    /// Accuracy by target
    pub target_accuracies: HashMap<String, f32>,
    /// Last calibration update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Calibration confidence score
    pub confidence_score: f32,
}

/// Results from calibration process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Whether calibration improved accuracy
    pub improved: bool,
    /// New accuracy score
    pub new_accuracy: f32,
    /// Previous accuracy score
    pub previous_accuracy: f32,
    /// Number of data points used
    pub data_points_used: usize,
    /// Calibration adjustments applied
    pub adjustments: Vec<CalibrationAdjustment>,
}

/// Specific calibration adjustment made to performance model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAdjustment {
    /// Type of adjustment made
    pub adjustment_type: AdjustmentType,
    /// Parameter that was adjusted
    pub parameter_name: String,
    /// Previous value
    pub previous_value: f32,
    /// New adjusted value
    pub new_value: f32,
    /// Confidence in this adjustment
    pub confidence: f32,
}

/// Types of calibration adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentType {
    /// Timing model adjustment
    TimingModel,
    /// Memory model adjustment
    MemoryModel,
    /// Power model adjustment
    PowerModel,
    /// Thermal model adjustment
    ThermalModel,
    /// CPU performance adjustment
    CpuPerformance,
}

impl Default for EnvironmentalConditions {
    fn default() -> Self {
        Self {
            ambient_temperature_celsius: 25.0,
            cpu_frequency_mhz: 1000,
            system_load: 0.1,
            supply_voltage_v: 3.3,
        }
    }
}

impl Default for CalibrationStatistics {
    fn default() -> Self {
        Self {
            total_data_points: 0,
            average_accuracy: 0.5,
            target_accuracies: HashMap::new(),
            last_update: chrono::Utc::now(),
            confidence_score: 0.0,
        }
    }
}

impl PerformanceCalibrator {
    /// Create new performance calibrator
    pub fn new(accuracy_threshold: f32) -> Self {
        Self {
            _accuracy_threshold: accuracy_threshold,
            calibration_data: HashMap::new(),
            accuracy_scores: HashMap::new(),
            stats: CalibrationStatistics::default(),
        }
    }

    /// Add calibration data point
    pub fn add_calibration_data(&mut self, data: CalibrationData) {
        let target_name = data.target_name.clone();

        // Calculate quality score for this data point
        let quality = self.calculate_data_quality(&data);
        let mut data = data;
        data.quality_score = quality;

        // Store calibration data
        self.calibration_data
            .entry(target_name.clone())
            .or_insert_with(Vec::new)
            .push(data);

        // Keep calibration data manageable (last 500 points per target)
        if let Some(target_data) = self.calibration_data.get_mut(&target_name) {
            if target_data.len() > 500 {
                target_data.remove(0);
            }
        }

        // Update statistics
        self.update_statistics();
    }

    /// Calibrate performance model using available data
    pub fn calibrate(
        &mut self,
        performance_model: &mut PerformanceModel,
        new_data: Vec<CalibrationData>,
    ) -> Result<f32> {
        // Add new calibration data
        for data in new_data {
            self.add_calibration_data(data);
        }

        // Perform calibration for each target with sufficient data
        let mut _total_improvements = 0;
        let mut _total_targets = 0;

        for (target_name, target_data) in &self.calibration_data {
            if target_data.len() >= 5 {
                // Need at least 5 data points for calibration
                let previous_accuracy = self
                    .accuracy_scores
                    .get(target_name)
                    .copied()
                    .unwrap_or(0.5);

                if let Ok(result) =
                    self.calibrate_target(performance_model, target_name, target_data)
                {
                    self.accuracy_scores
                        .insert(target_name.clone(), result.new_accuracy);

                    if result.improved {
                        _total_improvements += 1;
                    }
                    _total_targets += 1;

                    log::info!(
                        "Calibration for {}: {:.2}% -> {:.2}% ({})",
                        target_name,
                        previous_accuracy * 100.0,
                        result.new_accuracy * 100.0,
                        if result.improved {
                            "improved"
                        } else {
                            "unchanged"
                        }
                    );
                }
            }
        }

        // Update overall statistics
        self.update_statistics();

        // Return overall accuracy improvement
        Ok(self.stats.average_accuracy)
    }

    /// Calibrate performance model for specific target
    fn calibrate_target(
        &self,
        _performance_model: &mut PerformanceModel,
        target_name: &str,
        calibration_data: &[CalibrationData],
    ) -> Result<CalibrationResult> {
        let previous_accuracy = self
            .accuracy_scores
            .get(target_name)
            .copied()
            .unwrap_or(0.5);

        // Analyze prediction vs measurement differences
        let timing_errors: Vec<f32> = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.inference_time_us as f32;
                let measured = data.measured_metrics.inference_time_us as f32;
                (predicted - measured).abs() / measured.max(1.0) // Relative error
            })
            .collect();

        let memory_errors: Vec<f32> = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.memory_usage_bytes as f32;
                let measured = data.measured_metrics.memory_usage_bytes as f32;
                (predicted - measured).abs() / measured.max(1.0)
            })
            .collect();

        let power_errors: Vec<f32> = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.power_consumption_mw;
                let measured = data.measured_metrics.power_consumption_mw;
                (predicted - measured).abs() / measured.max(1.0)
            })
            .collect();

        // Calculate new accuracy based on error reduction
        let avg_timing_error = timing_errors.iter().sum::<f32>() / timing_errors.len() as f32;
        let avg_memory_error = memory_errors.iter().sum::<f32>() / memory_errors.len() as f32;
        let avg_power_error = power_errors.iter().sum::<f32>() / power_errors.len() as f32;

        // Overall accuracy is inverse of average error
        let overall_error = (avg_timing_error + avg_memory_error + avg_power_error) / 3.0;
        let new_accuracy = (1.0 - overall_error.min(1.0)).max(0.0);

        // Generate calibration adjustments
        let adjustments = self.generate_adjustments(target_name, calibration_data);

        let improved = new_accuracy > previous_accuracy;

        Ok(CalibrationResult {
            improved,
            new_accuracy,
            previous_accuracy,
            data_points_used: calibration_data.len(),
            adjustments,
        })
    }

    /// Generate calibration adjustments based on error analysis
    fn generate_adjustments(
        &self,
        _target_name: &str,
        calibration_data: &[CalibrationData],
    ) -> Vec<CalibrationAdjustment> {
        let mut adjustments = Vec::new();

        // Analyze timing model adjustments needed
        let timing_bias = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.inference_time_us as f32;
                let measured = data.measured_metrics.inference_time_us as f32;
                predicted / measured.max(1.0) // Ratio of predicted to measured
            })
            .sum::<f32>()
            / calibration_data.len() as f32;

        if (timing_bias - 1.0).abs() > 0.1 {
            // 10% bias threshold
            adjustments.push(CalibrationAdjustment {
                adjustment_type: AdjustmentType::TimingModel,
                parameter_name: "inference_time_multiplier".to_string(),
                previous_value: 1.0,
                new_value: 1.0 / timing_bias, // Inverse adjustment
                confidence: self.calculate_adjustment_confidence(calibration_data.len()),
            });
        }

        // Analyze memory model adjustments
        let memory_bias = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.memory_usage_bytes as f32;
                let measured = data.measured_metrics.memory_usage_bytes as f32;
                predicted / measured.max(1.0)
            })
            .sum::<f32>()
            / calibration_data.len() as f32;

        if (memory_bias - 1.0).abs() > 0.15 {
            // 15% bias threshold for memory
            adjustments.push(CalibrationAdjustment {
                adjustment_type: AdjustmentType::MemoryModel,
                parameter_name: "memory_usage_multiplier".to_string(),
                previous_value: 1.0,
                new_value: 1.0 / memory_bias,
                confidence: self.calculate_adjustment_confidence(calibration_data.len()),
            });
        }

        // Analyze power model adjustments
        let power_bias = calibration_data
            .iter()
            .map(|data| {
                let predicted = data.predicted_metrics.power_consumption_mw;
                let measured = data.measured_metrics.power_consumption_mw;
                predicted / measured.max(1.0)
            })
            .sum::<f32>()
            / calibration_data.len() as f32;

        if (power_bias - 1.0).abs() > 0.2 {
            // 20% bias threshold for power
            adjustments.push(CalibrationAdjustment {
                adjustment_type: AdjustmentType::PowerModel,
                parameter_name: "power_consumption_multiplier".to_string(),
                previous_value: 1.0,
                new_value: 1.0 / power_bias,
                confidence: self.calculate_adjustment_confidence(calibration_data.len()),
            });
        }

        adjustments
    }

    /// Calculate confidence in calibration adjustment
    fn calculate_adjustment_confidence(&self, data_points: usize) -> f32 {
        match data_points {
            0..=2 => 0.1,    // Very low confidence
            3..=5 => 0.4,    // Low confidence
            6..=10 => 0.7,   // Moderate confidence
            11..=20 => 0.85, // High confidence
            _ => 0.95,       // Very high confidence
        }
    }

    /// Calculate quality score for calibration data point
    fn calculate_data_quality(&self, data: &CalibrationData) -> f32 {
        let mut quality: f32 = 1.0;

        // Reduce quality for extreme environmental conditions
        let temp_deviation = (data.environment.ambient_temperature_celsius - 25.0).abs();
        if temp_deviation > 10.0 {
            quality -= 0.2; // High temperature variation reduces quality
        }

        // Reduce quality for high system load during measurement
        if data.environment.system_load > 0.5 {
            quality -= 0.3; // High system load affects measurement accuracy
        }

        // Increase quality for larger models (more representative)
        if data.model_info.parameter_count > 1_000_000 {
            quality += 0.1;
        }

        // Reduce quality for very small models (less representative)
        if data.model_info.parameter_count < 10_000 {
            quality -= 0.2;
        }

        // Ensure quality stays within bounds
        quality.max(0.0).min(1.0)
    }

    /// Calculate confidence score for simulation predictions
    pub fn calculate_confidence(
        &self,
        metrics: &SimulationMetrics,
        target_name: &str,
    ) -> Result<f32> {
        let base_confidence = match self.accuracy_scores.get(target_name) {
            Some(&accuracy) => accuracy,
            None => 0.5, // Default confidence when no calibration data available
        };

        // Adjust confidence based on data availability
        let data_count = self
            .calibration_data
            .get(target_name)
            .map(|d| d.len())
            .unwrap_or(0);
        let data_confidence_multiplier = match data_count {
            0..=2 => 0.6,   // Low confidence with limited data
            3..=10 => 0.8,  // Moderate confidence
            11..=50 => 1.0, // Full confidence
            _ => 1.1,       // High confidence with lots of data
        };

        // Adjust confidence based on metric ranges
        let range_confidence = self.calculate_range_confidence(metrics, target_name);

        let final_confidence =
            (base_confidence * data_confidence_multiplier * range_confidence).min(1.0);
        Ok(final_confidence)
    }

    /// Calculate confidence based on whether metrics are within expected ranges
    fn calculate_range_confidence(&self, metrics: &SimulationMetrics, target_name: &str) -> f32 {
        // This would ideally use learned ranges from calibration data
        // For now, use basic range checking
        let mut confidence = 1.0;

        // Check if metrics are within reasonable ranges for target
        match target_name {
            "esp32" => {
                if metrics.inference_time_us > 2_000_000 {
                    // 2 seconds
                    confidence *= 0.7;
                }
                if metrics.memory_usage_bytes > 520_000 {
                    // ESP32 RAM limit
                    confidence *= 0.5;
                }
            }
            "raspberry_pi" => {
                if metrics.inference_time_us > 500_000 {
                    // 500ms
                    confidence *= 0.8;
                }
                if metrics.memory_usage_bytes > 100_000_000 {
                    // 100MB
                    confidence *= 0.7;
                }
            }
            _ => {} // Use default confidence for other targets
        }

        confidence
    }

    /// Update calibration statistics
    fn update_statistics(&mut self) {
        let mut total_points = 0;
        let mut accuracy_sum = 0.0;
        let mut target_count = 0;

        for (target_name, data) in &self.calibration_data {
            total_points += data.len();

            if let Some(&accuracy) = self.accuracy_scores.get(target_name) {
                accuracy_sum += accuracy;
                target_count += 1;
            }
        }

        self.stats = CalibrationStatistics {
            total_data_points: total_points,
            average_accuracy: if target_count > 0 {
                accuracy_sum / target_count as f32
            } else {
                0.5
            },
            target_accuracies: self.accuracy_scores.clone(),
            last_update: chrono::Utc::now(),
            confidence_score: self.calculate_overall_confidence(),
        };
    }

    /// Calculate overall confidence in calibration system
    fn calculate_overall_confidence(&self) -> f32 {
        if self.stats.total_data_points == 0 {
            return 0.0;
        }

        let data_confidence = (self.stats.total_data_points as f32 / 100.0).min(1.0); // Full confidence at 100+ data points
        let accuracy_confidence = self.stats.average_accuracy;

        (data_confidence + accuracy_confidence) / 2.0
    }

    /// Get current accuracy for a target
    pub fn current_accuracy(&self) -> f32 {
        self.stats.average_accuracy
    }

    /// Get calibration statistics
    pub fn get_statistics(&self) -> &CalibrationStatistics {
        &self.stats
    }

    /// Get target-specific calibration data
    pub fn get_target_data(&self, target_name: &str) -> Option<&Vec<CalibrationData>> {
        self.calibration_data.get(target_name)
    }

    /// Clear calibration data for a target
    pub fn clear_target_data(&mut self, target_name: &str) {
        self.calibration_data.remove(target_name);
        self.accuracy_scores.remove(target_name);
        self.update_statistics();
    }

    /// Export calibration data for external analysis
    pub fn export_calibration_data(&self) -> HashMap<String, Vec<CalibrationData>> {
        self.calibration_data.clone()
    }

    /// Import calibration data from external source
    pub fn import_calibration_data(&mut self, data: HashMap<String, Vec<CalibrationData>>) {
        for (_target_name, target_data) in data {
            for point in target_data {
                self.add_calibration_data(point);
            }
        }
        self.update_statistics();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::performance::{SimulationMetrics, ThermalMetrics};

    #[test]
    fn test_calibrator_creation() {
        let calibrator = PerformanceCalibrator::new(0.9);
        assert_eq!(calibrator._accuracy_threshold, 0.9);
        assert_eq!(calibrator.calibration_data.len(), 0);
        assert_eq!(calibrator.stats.total_data_points, 0);
    }

    #[test]
    fn test_add_calibration_data() {
        let mut calibrator = PerformanceCalibrator::new(0.9);

        let predicted_metrics = SimulationMetrics {
            target_name: "esp32".to_string(),
            inference_time_us: 100_000,
            memory_usage_bytes: 300_000,
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

        let measured_metrics = SimulationMetrics {
            target_name: "esp32".to_string(),
            inference_time_us: 110_000,  // 10% slower than predicted
            memory_usage_bytes: 280_000, // Slightly less memory
            power_consumption_mw: 130.0, // Slightly more power
            cpu_utilization_percent: 78.0,
            memory_bandwidth_utilization_percent: 32.0,
            cache_hit_rate: 0.83,
            efficiency_score: 0.80,
            thermal_metrics: ThermalMetrics {
                temperature_rise_celsius: 9.0,
                throttling_likelihood: 0.12,
                sustained_performance: 0.93,
            },
        };

        let calibration_data = CalibrationData {
            target_name: "esp32".to_string(),
            timestamp: chrono::Utc::now(),
            predicted_metrics,
            measured_metrics,
            model_info: ModelCalibrationInfo {
                parameter_count: 100_000,
                operations_count: 500_000,
                quantization_bits: 8,
                is_pruned: false,
                architecture_type: "CNN".to_string(),
            },
            quality_score: 0.0, // Will be calculated
            environment: EnvironmentalConditions::default(),
        };

        calibrator.add_calibration_data(calibration_data);

        assert_eq!(calibrator.calibration_data.len(), 1);
        assert_eq!(calibrator.stats.total_data_points, 1);

        let esp32_data = calibrator.calibration_data.get("esp32").unwrap();
        assert_eq!(esp32_data.len(), 1);
        assert!(esp32_data[0].quality_score > 0.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut calibrator = PerformanceCalibrator::new(0.9);

        // Add some accuracy data
        calibrator.accuracy_scores.insert("esp32".to_string(), 0.85);

        let metrics = SimulationMetrics {
            target_name: "esp32".to_string(),
            inference_time_us: 100_000,
            memory_usage_bytes: 300_000,
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

        let confidence = calibrator.calculate_confidence(&metrics, "esp32").unwrap();
        assert!(confidence > 0.0 && confidence <= 1.0);

        // Confidence should be lower for unknown target
        let unknown_confidence = calibrator
            .calculate_confidence(&metrics, "unknown_target")
            .unwrap();
        assert!(unknown_confidence <= confidence);
    }

    #[test]
    fn test_data_quality_calculation() {
        let calibrator = PerformanceCalibrator::new(0.9);

        let high_quality_data = CalibrationData {
            target_name: "esp32".to_string(),
            timestamp: chrono::Utc::now(),
            predicted_metrics: SimulationMetrics {
                target_name: "esp32".to_string(),
                inference_time_us: 100_000,
                memory_usage_bytes: 300_000,
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
            measured_metrics: SimulationMetrics {
                target_name: "esp32".to_string(),
                inference_time_us: 110_000,
                memory_usage_bytes: 280_000,
                power_consumption_mw: 130.0,
                cpu_utilization_percent: 78.0,
                memory_bandwidth_utilization_percent: 32.0,
                cache_hit_rate: 0.83,
                efficiency_score: 0.80,
                thermal_metrics: ThermalMetrics {
                    temperature_rise_celsius: 9.0,
                    throttling_likelihood: 0.12,
                    sustained_performance: 0.93,
                },
            },
            model_info: ModelCalibrationInfo {
                parameter_count: 1_500_000, // Large model
                operations_count: 500_000,
                quantization_bits: 8,
                is_pruned: false,
                architecture_type: "CNN".to_string(),
            },
            quality_score: 0.0,
            environment: EnvironmentalConditions {
                ambient_temperature_celsius: 25.0, // Ideal conditions
                cpu_frequency_mhz: 240,
                system_load: 0.1, // Low system load
                supply_voltage_v: 3.3,
            },
        };

        let quality = calibrator.calculate_data_quality(&high_quality_data);
        assert!(quality > 0.8); // Should be high quality

        // Test low quality data
        let low_quality_data = CalibrationData {
            environment: EnvironmentalConditions {
                ambient_temperature_celsius: 45.0, // High temperature
                system_load: 0.8,                  // High system load
                ..Default::default()
            },
            model_info: ModelCalibrationInfo {
                parameter_count: 5_000, // Very small model
                ..high_quality_data.model_info.clone()
            },
            ..high_quality_data.clone()
        };

        let low_quality = calibrator.calculate_data_quality(&low_quality_data);
        assert!(low_quality < quality); // Should be lower quality
    }
}
