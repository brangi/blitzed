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

//! Performance metrics collection for benchmarking

use crate::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Comprehensive performance metrics for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Model size in bytes
    pub model_size_bytes: usize,

    /// Average inference time in milliseconds
    pub avg_inference_time_ms: u32,

    /// Standard deviation of inference times
    pub inference_time_stddev_ms: f32,

    /// Peak memory usage during inference (bytes)
    pub peak_memory_usage: usize,

    /// Average memory usage during inference (bytes)
    pub avg_memory_usage: usize,

    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,

    /// Model accuracy (0.0 to 1.0)
    pub accuracy_score: f64,

    /// Power consumption in milliwatts (if available)
    pub power_consumption_mw: Option<f32>,

    /// CPU utilization percentage (0.0 to 100.0)
    pub cpu_utilization: f32,

    /// Memory efficiency score (output quality per byte)
    pub memory_efficiency: f64,

    /// Additional custom metrics
    pub custom_metrics: std::collections::HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            model_size_bytes: 0,
            avg_inference_time_ms: 0,
            inference_time_stddev_ms: 0.0,
            peak_memory_usage: 0,
            avg_memory_usage: 0,
            throughput_ops_per_sec: 0.0,
            accuracy_score: 0.0,
            power_consumption_mw: None,
            cpu_utilization: 0.0,
            memory_efficiency: 0.0,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

/// Performance metrics collector
pub struct MetricsCollector {
    /// Inference timing measurements
    inference_times: Vec<Duration>,
    /// Memory usage samples
    memory_samples: Vec<usize>,
    /// Start time for overall measurement
    start_time: Option<Instant>,
    /// CPU utilization samples
    cpu_samples: Vec<f32>,
    /// Power consumption samples (if available)
    power_samples: Vec<f32>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            inference_times: Vec::new(),
            memory_samples: Vec::new(),
            start_time: None,
            cpu_samples: Vec::new(),
            power_samples: Vec::new(),
        }
    }

    /// Start measurement collection
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.inference_times.clear();
        self.memory_samples.clear();
        self.cpu_samples.clear();
        self.power_samples.clear();
    }

    /// Record a single inference timing
    pub fn record_inference_time(&mut self, time: Duration) {
        self.inference_times.push(time);
    }

    /// Record current memory usage
    pub fn record_memory_usage(&mut self, memory_bytes: usize) {
        self.memory_samples.push(memory_bytes);
    }

    /// Record CPU utilization sample
    pub fn record_cpu_utilization(&mut self, cpu_percent: f32) {
        self.cpu_samples.push(cpu_percent);
    }

    /// Record power consumption sample
    pub fn record_power_consumption(&mut self, power_mw: f32) {
        self.power_samples.push(power_mw);
    }

    /// Finalize metrics collection and compute results
    pub fn finalize(&self, model_size_bytes: usize, accuracy: f64) -> PerformanceMetrics {
        // Calculate timing statistics
        let (avg_time_ms, stddev_time_ms) = if !self.inference_times.is_empty() {
            let times_ms: Vec<f32> = self
                .inference_times
                .iter()
                .map(|d| d.as_secs_f32() * 1000.0)
                .collect();

            let avg = times_ms.iter().sum::<f32>() / times_ms.len() as f32;
            let variance =
                times_ms.iter().map(|&x| (x - avg).powi(2)).sum::<f32>() / times_ms.len() as f32;
            let stddev = variance.sqrt();

            (avg as u32, stddev)
        } else {
            (0, 0.0)
        };

        // Calculate memory statistics
        let (peak_memory, avg_memory) = if !self.memory_samples.is_empty() {
            let peak = *self.memory_samples.iter().max().unwrap();
            let avg = self.memory_samples.iter().sum::<usize>() / self.memory_samples.len();
            (peak, avg)
        } else {
            (0, 0)
        };

        // Calculate throughput
        let throughput = if avg_time_ms > 0 {
            1000.0 / (avg_time_ms as f64)
        } else {
            0.0
        };

        // Calculate CPU utilization
        let cpu_utilization = if !self.cpu_samples.is_empty() {
            self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32
        } else {
            0.0
        };

        // Calculate average power consumption
        let power_consumption = if !self.power_samples.is_empty() {
            Some(self.power_samples.iter().sum::<f32>() / self.power_samples.len() as f32)
        } else {
            None
        };

        // Calculate memory efficiency (accuracy per byte)
        let memory_efficiency = if peak_memory > 0 {
            accuracy / (peak_memory as f64)
        } else {
            0.0
        };

        PerformanceMetrics {
            model_size_bytes,
            avg_inference_time_ms: avg_time_ms,
            inference_time_stddev_ms: stddev_time_ms,
            peak_memory_usage: peak_memory,
            avg_memory_usage: avg_memory,
            throughput_ops_per_sec: throughput,
            accuracy_score: accuracy,
            power_consumption_mw: power_consumption,
            cpu_utilization,
            memory_efficiency,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated performance metrics for testing/development
pub fn simulate_performance_metrics(
    framework_name: &str,
    model_name: &str,
    platform_name: &str,
    base_model_size: usize,
) -> PerformanceMetrics {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create deterministic but realistic-looking metrics based on inputs
    let mut hasher = DefaultHasher::new();
    framework_name.hash(&mut hasher);
    model_name.hash(&mut hasher);
    platform_name.hash(&mut hasher);
    let seed = hasher.finish();

    // Use seed for pseudo-random but deterministic values
    let random_factor = (seed % 1000) as f32 / 1000.0;

    // Simulate framework-specific characteristics
    let (speed_multiplier, size_multiplier, accuracy_base) = match framework_name {
        "Blitzed" => (1.2, 0.7, 0.85), // Faster, smaller, good accuracy
        "TensorFlow Lite" => (1.0, 0.8, 0.88), // Baseline performance
        "ONNX Runtime" => (1.1, 0.85, 0.87), // Slightly faster, moderate size
        "PyTorch Mobile" => (0.9, 0.95, 0.89), // Slower but accurate
        _ => (1.0, 1.0, 0.8),          // Default values
    };

    // Platform-specific adjustments
    let platform_multiplier = match platform_name {
        "ESP32" => 0.3,               // Much slower on embedded
        "Arduino Nano 33 BLE" => 0.2, // Very slow on microcontroller
        "STM32F4" => 0.4,             // Slow but FPU helps
        "Raspberry Pi 4" => 0.8,      // Moderate performance
        "Mobile ARM" => 1.2,          // Good mobile performance
        "x86 Desktop" => 1.5,         // Fast desktop performance
        "x86 Server" => 2.0,          // Very fast server performance
        _ => 1.0,
    };

    // Calculate realistic metrics
    let inference_time_base = match model_name {
        "MobileNetV2" => 50.0,
        "MobileNetV3-Small" => 30.0,
        "ResNet-18" => 120.0,
        "EfficientNet-B0" => 80.0,
        "YOLOv5-Nano" => 200.0,
        _ => 100.0, // Default
    };

    let avg_inference_time_ms = ((inference_time_base / (speed_multiplier * platform_multiplier))
        * (0.8 + random_factor * 0.4)) as u32;

    let model_size_bytes =
        ((base_model_size as f32) * size_multiplier * (0.9 + random_factor * 0.2)) as usize;

    let accuracy = (accuracy_base + (random_factor - 0.5) * 0.1).clamp(0.0, 1.0);

    // Memory usage scales with model size and platform
    let base_memory = model_size_bytes * 2; // Typical 2x overhead
    let peak_memory = (base_memory as f32 * (1.0 + random_factor * 0.5)) as usize;
    let avg_memory = (peak_memory as f32 * 0.8) as usize;

    // Calculate derived metrics
    let throughput = if avg_inference_time_ms > 0 {
        1000.0 / (avg_inference_time_ms as f64)
    } else {
        0.0
    };

    let memory_efficiency = (accuracy as f64) / (peak_memory as f64);

    PerformanceMetrics {
        model_size_bytes,
        avg_inference_time_ms,
        inference_time_stddev_ms: (avg_inference_time_ms as f32) * 0.1, // 10% variance
        peak_memory_usage: peak_memory,
        avg_memory_usage: avg_memory,
        throughput_ops_per_sec: throughput,
        accuracy_score: accuracy as f64,
        power_consumption_mw: if platform_name.contains("ESP32")
            || platform_name.contains("Arduino")
        {
            Some(100.0 + random_factor * 50.0) // 100-150mW for embedded
        } else {
            None
        },
        cpu_utilization: 20.0 + random_factor * 60.0, // 20-80% CPU usage
        memory_efficiency,
        custom_metrics: std::collections::HashMap::new(),
    }
}

/// Benchmark runner for measuring performance
pub struct BenchmarkRunner {
    warmup_runs: u32,
    benchmark_runs: u32,
    timeout: Duration,
}

impl BenchmarkRunner {
    pub fn new(warmup_runs: u32, benchmark_runs: u32, timeout: Duration) -> Self {
        Self {
            warmup_runs,
            benchmark_runs,
            timeout,
        }
    }

    /// Run benchmark with metrics collection
    pub fn run_benchmark<F>(&self, mut inference_fn: F) -> Result<MetricsCollector>
    where
        F: FnMut() -> Result<()>,
    {
        let mut collector = MetricsCollector::new();
        collector.start();

        // Warmup runs (not measured)
        log::debug!("Running {} warmup iterations", self.warmup_runs);
        for _ in 0..self.warmup_runs {
            let start = Instant::now();
            inference_fn()?;
            let duration = start.elapsed();

            if duration > self.timeout {
                return Err(crate::BlitzedError::Internal(format!(
                    "Warmup run exceeded timeout: {:?}",
                    duration
                )));
            }
        }

        // Measured benchmark runs
        log::debug!("Running {} measured iterations", self.benchmark_runs);
        for i in 0..self.benchmark_runs {
            let start = Instant::now();

            // Record memory before inference (simulated)
            let memory_before = Self::get_current_memory_usage();
            collector.record_memory_usage(memory_before);

            // Run inference
            inference_fn()?;

            let duration = start.elapsed();

            if duration > self.timeout {
                return Err(crate::BlitzedError::Internal(format!(
                    "Benchmark run {} exceeded timeout: {:?}",
                    i, duration
                )));
            }

            // Record metrics
            collector.record_inference_time(duration);

            // Record memory after inference (simulated peak)
            let memory_after =
                Self::get_current_memory_usage() + (duration.as_millis() as usize * 1000);
            collector.record_memory_usage(memory_after);

            // Record CPU utilization (simulated)
            let cpu_usage = 30.0 + (i as f32 * 5.0) % 40.0; // Simulate varying CPU usage
            collector.record_cpu_utilization(cpu_usage);
        }

        Ok(collector)
    }

    /// Get current memory usage (simulated for now)
    fn get_current_memory_usage() -> usize {
        // TODO: Implement actual memory measurement using system APIs
        // For now, return a realistic baseline
        50 * 1024 * 1024 // 50MB baseline
    }
}

/// Performance comparison utilities
pub mod comparison {
    use super::PerformanceMetrics;

    /// Compare two performance metrics and return improvement factors
    pub fn compare_metrics(
        baseline: &PerformanceMetrics,
        target: &PerformanceMetrics,
    ) -> MetricsComparison {
        MetricsComparison {
            speedup: if target.avg_inference_time_ms > 0 {
                baseline.avg_inference_time_ms as f64 / target.avg_inference_time_ms as f64
            } else {
                1.0
            },
            size_reduction: if target.model_size_bytes > 0 {
                baseline.model_size_bytes as f64 / target.model_size_bytes as f64
            } else {
                1.0
            },
            memory_efficiency_gain: target.memory_efficiency / baseline.memory_efficiency.max(1e-9),
            accuracy_difference: target.accuracy_score - baseline.accuracy_score,
            throughput_improvement: target.throughput_ops_per_sec
                / baseline.throughput_ops_per_sec.max(1e-9),
        }
    }

    #[derive(Debug, Clone)]
    pub struct MetricsComparison {
        pub speedup: f64,
        pub size_reduction: f64,
        pub memory_efficiency_gain: f64,
        pub accuracy_difference: f64,
        pub throughput_improvement: f64,
    }

    impl MetricsComparison {
        pub fn summary(&self) -> String {
            format!(
                "Performance Comparison:\n\
                 - Speedup: {:.2}x\n\
                 - Size reduction: {:.2}x\n\
                 - Memory efficiency: {:.2}x\n\
                 - Accuracy change: {:.1}%\n\
                 - Throughput improvement: {:.2}x",
                self.speedup,
                self.size_reduction,
                self.memory_efficiency_gain,
                self.accuracy_difference * 100.0,
                self.throughput_improvement
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();

        assert_eq!(metrics.model_size_bytes, 0);
        assert_eq!(metrics.avg_inference_time_ms, 0);
        assert_eq!(metrics.inference_time_stddev_ms, 0.0);
        assert_eq!(metrics.peak_memory_usage, 0);
        assert_eq!(metrics.avg_memory_usage, 0);
        assert_eq!(metrics.throughput_ops_per_sec, 0.0);
        assert_eq!(metrics.accuracy_score, 0.0);
        assert!(metrics.power_consumption_mw.is_none());
        assert_eq!(metrics.cpu_utilization, 0.0);
        assert_eq!(metrics.memory_efficiency, 0.0);
        assert!(metrics.custom_metrics.is_empty());
    }

    #[test]
    fn test_metrics_collector_new_and_start() {
        let mut collector = MetricsCollector::new();

        // Should be able to start
        collector.start();

        // After start, should have cleared any previous data
        assert!(collector.inference_times.is_empty());
        assert!(collector.memory_samples.is_empty());
        assert!(collector.cpu_samples.is_empty());
        assert!(collector.power_samples.is_empty());
        assert!(collector.start_time.is_some());
    }

    #[test]
    fn test_metrics_collector_record_and_finalize() {
        let mut collector = MetricsCollector::new();
        collector.start();

        // Record several inference times
        collector.record_inference_time(Duration::from_millis(100));
        collector.record_inference_time(Duration::from_millis(110));
        collector.record_inference_time(Duration::from_millis(90));

        // Record memory samples
        collector.record_memory_usage(1_000_000);
        collector.record_memory_usage(1_200_000);
        collector.record_memory_usage(1_100_000);

        // Record CPU samples
        collector.record_cpu_utilization(45.0);
        collector.record_cpu_utilization(50.0);
        collector.record_cpu_utilization(55.0);

        // Record power samples
        collector.record_power_consumption(120.0);
        collector.record_power_consumption(125.0);

        let metrics = collector.finalize(500_000, 0.85);

        // Verify computed metrics
        assert_eq!(metrics.model_size_bytes, 500_000);
        assert!(metrics.avg_inference_time_ms > 0);
        assert_eq!(metrics.avg_inference_time_ms, 100); // Average of 100, 110, 90
        assert!(metrics.inference_time_stddev_ms > 0.0);
        assert_eq!(metrics.peak_memory_usage, 1_200_000);
        assert!(metrics.avg_memory_usage > 0);
        assert!(metrics.avg_memory_usage <= metrics.peak_memory_usage);
        assert_eq!(metrics.avg_memory_usage, 1_100_000); // Average of samples
        assert!(metrics.throughput_ops_per_sec > 0.0);
        assert_eq!(metrics.accuracy_score, 0.85);
        assert!(metrics.power_consumption_mw.is_some());
        assert_eq!(metrics.power_consumption_mw.unwrap(), 122.5); // Average of 120, 125
        assert!(metrics.cpu_utilization > 0.0);
        assert_eq!(metrics.cpu_utilization, 50.0); // Average of 45, 50, 55
        assert!(metrics.memory_efficiency > 0.0);
    }

    #[test]
    fn test_metrics_collector_finalize_empty() {
        let collector = MetricsCollector::new();
        let metrics = collector.finalize(1_000_000, 0.9);

        // With no data recorded, most metrics should be zero
        assert_eq!(metrics.model_size_bytes, 1_000_000);
        assert_eq!(metrics.avg_inference_time_ms, 0);
        assert_eq!(metrics.inference_time_stddev_ms, 0.0);
        assert_eq!(metrics.peak_memory_usage, 0);
        assert_eq!(metrics.avg_memory_usage, 0);
        assert_eq!(metrics.throughput_ops_per_sec, 0.0);
        assert_eq!(metrics.accuracy_score, 0.9);
        assert!(metrics.power_consumption_mw.is_none());
        assert_eq!(metrics.cpu_utilization, 0.0);
        assert_eq!(metrics.memory_efficiency, 0.0);
    }

    #[test]
    fn test_metrics_collector_finalize_with_power() {
        let mut collector = MetricsCollector::new();
        collector.start();

        // Record only power samples
        collector.record_power_consumption(100.0);
        collector.record_power_consumption(150.0);
        collector.record_power_consumption(125.0);

        let metrics = collector.finalize(100_000, 0.8);

        // Power consumption should be average
        assert!(metrics.power_consumption_mw.is_some());
        assert_eq!(metrics.power_consumption_mw.unwrap(), 125.0);
    }

    #[test]
    fn test_simulate_performance_metrics_deterministic() {
        let metrics1 = simulate_performance_metrics("Blitzed", "MobileNetV2", "ESP32", 1_000_000);

        let metrics2 = simulate_performance_metrics("Blitzed", "MobileNetV2", "ESP32", 1_000_000);

        // Same inputs should produce identical results
        assert_eq!(metrics1.model_size_bytes, metrics2.model_size_bytes);
        assert_eq!(
            metrics1.avg_inference_time_ms,
            metrics2.avg_inference_time_ms
        );
        assert_eq!(metrics1.peak_memory_usage, metrics2.peak_memory_usage);
        assert_eq!(metrics1.accuracy_score, metrics2.accuracy_score);
        assert_eq!(metrics1.cpu_utilization, metrics2.cpu_utilization);
    }

    #[test]
    fn test_simulate_performance_metrics_framework_differences() {
        let blitzed_metrics =
            simulate_performance_metrics("Blitzed", "MobileNetV2", "ESP32", 1_000_000);

        let tflite_metrics =
            simulate_performance_metrics("TensorFlow Lite", "MobileNetV2", "ESP32", 1_000_000);

        // Different frameworks should produce different metrics
        assert_ne!(
            blitzed_metrics.model_size_bytes,
            tflite_metrics.model_size_bytes
        );
        assert_ne!(
            blitzed_metrics.avg_inference_time_ms,
            tflite_metrics.avg_inference_time_ms
        );
    }

    #[test]
    fn test_simulate_performance_metrics_blitzed_characteristics() {
        let metrics = simulate_performance_metrics("Blitzed", "MobileNetV2", "ESP32", 1_000_000);

        // Blitzed should have reasonable non-zero values
        assert!(metrics.model_size_bytes > 0);
        assert!(metrics.avg_inference_time_ms > 0);
        assert!(metrics.peak_memory_usage > 0);
        assert!(metrics.avg_memory_usage > 0);
        assert!(metrics.throughput_ops_per_sec > 0.0);
        assert!(metrics.accuracy_score > 0.0);
        assert!(metrics.accuracy_score <= 1.0);

        // ESP32 should have power consumption
        assert!(metrics.power_consumption_mw.is_some());

        // CPU utilization should be within valid range
        assert!(metrics.cpu_utilization >= 0.0);
        assert!(metrics.cpu_utilization <= 100.0);
    }

    #[test]
    fn test_benchmark_runner_new() {
        let runner = BenchmarkRunner::new(2, 5, Duration::from_secs(5));

        assert_eq!(runner.warmup_runs, 2);
        assert_eq!(runner.benchmark_runs, 5);
        assert_eq!(runner.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_benchmark_runner_run() {
        let runner = BenchmarkRunner::new(2, 3, Duration::from_secs(5));

        let mut counter = 0;
        let result = runner.run_benchmark(|| {
            counter += 1;
            Ok(())
        });

        assert!(result.is_ok());
        let collector = result.unwrap();

        // Should have run warmup + benchmark iterations
        assert_eq!(counter, 5); // 2 warmup + 3 benchmark

        // Collector should have recorded data from benchmark runs only
        assert_eq!(collector.inference_times.len(), 3);
        assert!(!collector.memory_samples.is_empty());
        assert!(!collector.cpu_samples.is_empty());
    }

    #[test]
    fn test_compare_metrics() {
        let baseline = PerformanceMetrics {
            model_size_bytes: 1_000_000,
            avg_inference_time_ms: 100,
            inference_time_stddev_ms: 10.0,
            peak_memory_usage: 2_000_000,
            avg_memory_usage: 1_800_000,
            throughput_ops_per_sec: 10.0,
            accuracy_score: 0.85,
            power_consumption_mw: Some(150.0),
            cpu_utilization: 60.0,
            memory_efficiency: 0.85 / 2_000_000.0,
            custom_metrics: std::collections::HashMap::new(),
        };

        let target = PerformanceMetrics {
            model_size_bytes: 700_000,
            avg_inference_time_ms: 80,
            inference_time_stddev_ms: 8.0,
            peak_memory_usage: 1_400_000,
            avg_memory_usage: 1_200_000,
            throughput_ops_per_sec: 12.5,
            accuracy_score: 0.86,
            power_consumption_mw: Some(120.0),
            cpu_utilization: 50.0,
            memory_efficiency: 0.86 / 1_400_000.0,
            custom_metrics: std::collections::HashMap::new(),
        };

        let comparison = comparison::compare_metrics(&baseline, &target);

        // Speedup: 100/80 = 1.25x
        assert!((comparison.speedup - 1.25).abs() < 0.01);

        // Size reduction: 1_000_000/700_000 ≈ 1.43x
        assert!((comparison.size_reduction - 1.428571).abs() < 0.01);

        // Memory efficiency should have improved
        assert!(comparison.memory_efficiency_gain > 1.0);

        // Accuracy difference: 0.86 - 0.85 = 0.01
        assert!((comparison.accuracy_difference - 0.01).abs() < 0.001);

        // Throughput improvement: 12.5/10.0 = 1.25x
        assert!((comparison.throughput_improvement - 1.25).abs() < 0.01);
    }

    #[test]
    fn test_metrics_comparison_summary() {
        let comparison = comparison::MetricsComparison {
            speedup: 1.5,
            size_reduction: 2.0,
            memory_efficiency_gain: 1.8,
            accuracy_difference: 0.02,
            throughput_improvement: 1.6,
        };

        let summary = comparison.summary();

        // Verify summary contains expected keywords and values
        assert!(summary.contains("Performance Comparison"));
        assert!(summary.contains("Speedup"));
        assert!(summary.contains("1.50"));
        assert!(summary.contains("Size reduction"));
        assert!(summary.contains("2.00"));
        assert!(summary.contains("Memory efficiency"));
        assert!(summary.contains("1.80"));
        assert!(summary.contains("Accuracy change"));
        assert!(summary.contains("2.0%"));
        assert!(summary.contains("Throughput improvement"));
        assert!(summary.contains("1.60"));
    }
}
