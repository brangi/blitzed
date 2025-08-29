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

//! Performance benchmarking suite for comparing Blitzed against industry standards
//!
//! This module provides comprehensive benchmarking capabilities to compare
//! Blitzed's optimization pipeline against TensorFlow Lite, ONNX Runtime, and other
//! industry-standard edge AI frameworks.

pub mod metrics;
pub mod suite;

// Basic imports will be added as needed by submodules
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Supported competitive frameworks for benchmarking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompetitiveFramework {
    /// TensorFlow Lite
    TensorFlowLite,
    /// ONNX Runtime
    OnnxRuntime,
    /// PyTorch Mobile
    PyTorchMobile,
    /// Our Blitzed framework (baseline)
    Blitzed,
}

impl CompetitiveFramework {
    pub fn name(&self) -> &'static str {
        match self {
            CompetitiveFramework::TensorFlowLite => "TensorFlow Lite",
            CompetitiveFramework::OnnxRuntime => "ONNX Runtime",
            CompetitiveFramework::PyTorchMobile => "PyTorch Mobile",
            CompetitiveFramework::Blitzed => "Blitzed",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            CompetitiveFramework::TensorFlowLite => "TFLite",
            CompetitiveFramework::OnnxRuntime => "ONNX",
            CompetitiveFramework::PyTorchMobile => "PyTorch",
            CompetitiveFramework::Blitzed => "Blitzed",
        }
    }
}

/// Hardware platforms for benchmarking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwarePlatform {
    /// ESP32 dual-core ARM with WiFi/Bluetooth
    ESP32,
    /// Arduino Nano 33 BLE ultra-low-power
    ArduinoNano33BLE,
    /// STM32F4 ARM Cortex-M4 with FPU
    STM32F4,
    /// Raspberry Pi 4 ARM Cortex-A72
    RaspberryPi4,
    /// Mobile ARM processors (Android/iOS)
    MobileARM,
    /// x86 desktop/laptop processors
    X86Desktop,
    /// x86 server processors  
    X86Server,
}

impl HardwarePlatform {
    pub fn name(&self) -> &'static str {
        match self {
            HardwarePlatform::ESP32 => "ESP32",
            HardwarePlatform::ArduinoNano33BLE => "Arduino Nano 33 BLE",
            HardwarePlatform::STM32F4 => "STM32F4",
            HardwarePlatform::RaspberryPi4 => "Raspberry Pi 4",
            HardwarePlatform::MobileARM => "Mobile ARM",
            HardwarePlatform::X86Desktop => "x86 Desktop",
            HardwarePlatform::X86Server => "x86 Server",
        }
    }

    /// Get typical memory constraints for this platform (in bytes)
    pub fn memory_limit(&self) -> usize {
        match self {
            HardwarePlatform::ESP32 => 520 * 1024,            // 520KB
            HardwarePlatform::ArduinoNano33BLE => 256 * 1024, // 256KB
            HardwarePlatform::STM32F4 => 192 * 1024,          // 192KB
            HardwarePlatform::RaspberryPi4 => 4 * 1024 * 1024 * 1024, // 4GB
            HardwarePlatform::MobileARM => 2 * 1024 * 1024 * 1024, // 2GB
            HardwarePlatform::X86Desktop => 8 * 1024 * 1024 * 1024, // 8GB
            HardwarePlatform::X86Server => 32 * 1024 * 1024 * 1024, // 32GB
        }
    }

    /// Check if this is an embedded/edge platform
    pub fn is_embedded(&self) -> bool {
        matches!(
            self,
            HardwarePlatform::ESP32
                | HardwarePlatform::ArduinoNano33BLE
                | HardwarePlatform::STM32F4
        )
    }
}

/// Standard models for benchmarking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StandardModel {
    /// MobileNetV2 for image classification
    MobileNetV2,
    /// MobileNetV3 for image classification
    MobileNetV3Small,
    /// ResNet-18 for image classification
    ResNet18,
    /// EfficientNet-B0 for image classification
    EfficientNetB0,
    /// YOLOv5 nano for object detection
    YoloV5Nano,
    /// Custom model (user-provided)
    Custom(String),
}

impl StandardModel {
    pub fn name(&self) -> &str {
        match self {
            StandardModel::MobileNetV2 => "MobileNetV2",
            StandardModel::MobileNetV3Small => "MobileNetV3-Small",
            StandardModel::ResNet18 => "ResNet-18",
            StandardModel::EfficientNetB0 => "EfficientNet-B0",
            StandardModel::YoloV5Nano => "YOLOv5-Nano",
            StandardModel::Custom(name) => name,
        }
    }

    /// Get expected input shape for this model
    pub fn input_shape(&self) -> Vec<i64> {
        match self {
            StandardModel::MobileNetV2
            | StandardModel::MobileNetV3Small
            | StandardModel::ResNet18
            | StandardModel::EfficientNetB0 => vec![1, 3, 224, 224], // ImageNet standard
            StandardModel::YoloV5Nano => vec![1, 3, 640, 640], // YOLO standard
            StandardModel::Custom(_) => vec![1, 3, 224, 224],  // Default assumption
        }
    }

    /// Get expected model size range (in MB)
    pub fn expected_size_mb(&self) -> (f32, f32) {
        match self {
            StandardModel::MobileNetV2 => (9.0, 14.0),
            StandardModel::MobileNetV3Small => (5.0, 9.0),
            StandardModel::ResNet18 => (42.0, 50.0),
            StandardModel::EfficientNetB0 => (20.0, 30.0),
            StandardModel::YoloV5Nano => (3.0, 7.0),
            StandardModel::Custom(_) => (1.0, 100.0), // Wide range for custom
        }
    }
}

/// Configuration for benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Frameworks to benchmark against
    pub frameworks: Vec<CompetitiveFramework>,
    /// Hardware platforms to test on
    pub platforms: Vec<HardwarePlatform>,
    /// Models to benchmark
    pub models: Vec<StandardModel>,
    /// Number of warmup runs before timing
    pub warmup_runs: u32,
    /// Number of benchmark runs for averaging
    pub benchmark_runs: u32,
    /// Timeout for individual benchmark runs
    pub timeout: Duration,
    /// Whether to include accuracy validation
    pub validate_accuracy: bool,
    /// Whether to measure power consumption (if supported)
    pub measure_power: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            frameworks: vec![
                CompetitiveFramework::Blitzed,
                CompetitiveFramework::TensorFlowLite,
                CompetitiveFramework::OnnxRuntime,
            ],
            platforms: vec![
                HardwarePlatform::X86Desktop,
                HardwarePlatform::ESP32,
                HardwarePlatform::RaspberryPi4,
            ],
            models: vec![StandardModel::MobileNetV2, StandardModel::ResNet18],
            warmup_runs: 5,
            benchmark_runs: 10,
            timeout: Duration::from_secs(30),
            validate_accuracy: true,
            measure_power: false,
        }
    }
}

/// Complete benchmark results for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Framework that was benchmarked
    pub framework: CompetitiveFramework,
    /// Hardware platform used
    pub platform: HardwarePlatform,
    /// Model that was benchmarked
    pub model: StandardModel,
    /// Performance metrics collected
    pub metrics: metrics::PerformanceMetrics,
    /// Whether the benchmark completed successfully
    pub success: bool,
    /// Error message if benchmark failed
    pub error: Option<String>,
}

/// Summary of all benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Performance comparison matrix
    pub comparisons: HashMap<(CompetitiveFramework, CompetitiveFramework), ComparisonResult>,
    /// Overall statistics
    pub statistics: BenchmarkStatistics,
    /// Timestamp when benchmark was run
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Total benchmark execution time
    pub total_duration: Duration,
}

/// Comparison between two frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Framework being compared against (baseline)
    pub baseline: CompetitiveFramework,
    /// Framework being measured
    pub target: CompetitiveFramework,
    /// Speedup factor (>1.0 means target is faster)
    pub speedup: f64,
    /// Size reduction factor (>1.0 means target is smaller)
    pub size_reduction: f64,
    /// Memory efficiency improvement (>1.0 means target uses less memory)
    pub memory_efficiency: f64,
    /// Accuracy difference (positive means target is more accurate)
    pub accuracy_difference: f64,
}

/// Overall benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    /// Total number of benchmarks run
    pub total_benchmarks: usize,
    /// Number of successful benchmarks
    pub successful_benchmarks: usize,
    /// Average speedup of Blitzed vs competitors
    pub average_blitzed_speedup: f64,
    /// Average size reduction of Blitzed vs competitors
    pub average_blitzed_compression: f64,
    /// Average accuracy retention of Blitzed vs competitors
    pub average_accuracy_retention: f64,
    /// Best performing platform for Blitzed
    pub best_platform: Option<HardwarePlatform>,
    /// Most challenging model (lowest relative performance)
    pub most_challenging_model: Option<StandardModel>,
}

/// Trait for benchmark result analysis
pub trait BenchmarkAnalysis {
    /// Generate a textual summary of results
    fn summary(&self) -> String;

    /// Find the best performing framework for given criteria
    fn best_framework(&self, metric: &str) -> Option<CompetitiveFramework>;

    /// Calculate competitive advantage of Blitzed
    fn blitzed_advantage(&self) -> HashMap<String, f64>;
}

impl BenchmarkAnalysis for BenchmarkSummary {
    fn summary(&self) -> String {
        let success_rate = (self.statistics.successful_benchmarks as f64)
            / (self.statistics.total_benchmarks as f64)
            * 100.0;

        format!(
            "Benchmark Summary:\n\
             - Total tests: {}\n\
             - Success rate: {:.1}%\n\
             - Average Blitzed speedup: {:.2}x\n\
             - Average compression: {:.2}x\n\
             - Accuracy retention: {:.1}%\n\
             - Duration: {:.1}s",
            self.statistics.total_benchmarks,
            success_rate,
            self.statistics.average_blitzed_speedup,
            self.statistics.average_blitzed_compression,
            self.statistics.average_accuracy_retention * 100.0,
            self.total_duration.as_secs_f64()
        )
    }

    fn best_framework(&self, metric: &str) -> Option<CompetitiveFramework> {
        // Find best framework based on specified metric
        let mut best_framework = None;
        let mut best_value = match metric {
            "latency" => f64::INFINITY, // Lower is better
            "memory" => f64::INFINITY,  // Lower is better
            "size" => f64::INFINITY,    // Lower is better
            _ => f64::NEG_INFINITY,     // Higher is better (accuracy, throughput)
        };

        for result in &self.results {
            if !result.success {
                continue;
            }

            let value = match metric {
                "latency" => result.metrics.avg_inference_time_ms as f64,
                "memory" => result.metrics.peak_memory_usage as f64,
                "size" => result.metrics.model_size_bytes as f64,
                "accuracy" => result.metrics.accuracy_score,
                "throughput" => result.metrics.throughput_ops_per_sec,
                _ => continue,
            };

            let is_better = match metric {
                "latency" | "memory" | "size" => value < best_value,
                _ => value > best_value,
            };

            if is_better {
                best_value = value;
                best_framework = Some(result.framework.clone());
            }
        }

        best_framework
    }

    fn blitzed_advantage(&self) -> HashMap<String, f64> {
        let mut advantages = HashMap::new();

        // Calculate average advantages across all comparisons
        let blitzed_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.framework == CompetitiveFramework::Blitzed && r.success)
            .collect();

        let competitor_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.framework != CompetitiveFramework::Blitzed && r.success)
            .collect();

        if !blitzed_results.is_empty() && !competitor_results.is_empty() {
            // Average metrics for Blitzed
            let avg_blitzed_latency: f64 = blitzed_results
                .iter()
                .map(|r| r.metrics.avg_inference_time_ms as f64)
                .sum::<f64>()
                / blitzed_results.len() as f64;
            let avg_blitzed_memory: f64 = blitzed_results
                .iter()
                .map(|r| r.metrics.peak_memory_usage as f64)
                .sum::<f64>()
                / blitzed_results.len() as f64;
            let avg_blitzed_size: f64 = blitzed_results
                .iter()
                .map(|r| r.metrics.model_size_bytes as f64)
                .sum::<f64>()
                / blitzed_results.len() as f64;

            // Average metrics for competitors
            let avg_competitor_latency: f64 = competitor_results
                .iter()
                .map(|r| r.metrics.avg_inference_time_ms as f64)
                .sum::<f64>()
                / competitor_results.len() as f64;
            let avg_competitor_memory: f64 = competitor_results
                .iter()
                .map(|r| r.metrics.peak_memory_usage as f64)
                .sum::<f64>()
                / competitor_results.len() as f64;
            let avg_competitor_size: f64 = competitor_results
                .iter()
                .map(|r| r.metrics.model_size_bytes as f64)
                .sum::<f64>()
                / competitor_results.len() as f64;

            // Calculate advantages (higher values mean Blitzed is better)
            advantages.insert(
                "speed_advantage".to_string(),
                avg_competitor_latency / avg_blitzed_latency,
            );
            advantages.insert(
                "memory_advantage".to_string(),
                avg_competitor_memory / avg_blitzed_memory,
            );
            advantages.insert(
                "size_advantage".to_string(),
                avg_competitor_size / avg_blitzed_size,
            );
        }

        advantages
    }
}
