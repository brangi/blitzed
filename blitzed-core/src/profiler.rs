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

//! Performance profiling and benchmarking utilities

use crate::{Model, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance metrics for model operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Model loading time
    pub load_time_ms: u64,
    /// Average inference time
    pub inference_time_ms: u64,
    /// Memory usage during inference
    pub memory_usage_bytes: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Throughput (inferences per second)
    pub throughput: f32,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of benchmark runs
    pub benchmark_runs: u32,
    /// Collect detailed memory statistics
    pub detailed_memory: bool,
    /// Measure CPU utilization
    pub measure_cpu: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 10,
            benchmark_runs: 100,
            detailed_memory: true,
            measure_cpu: false, // CPU measurement can be expensive
        }
    }
}

/// Model profiler for performance analysis
pub struct Profiler {
    config: ProfilingConfig,
}

impl Profiler {
    /// Create a new profiler with configuration
    pub fn new(config: ProfilingConfig) -> Self {
        Self { config }
    }

    /// Profile model loading performance
    pub fn profile_loading<F>(&self, load_fn: F) -> Result<Duration>
    where
        F: FnOnce() -> Result<Model>,
    {
        log::info!("Profiling model loading...");
        let start = Instant::now();
        let _model = load_fn()?;
        let duration = start.elapsed();
        log::info!("Model loading took {} ms", duration.as_millis());
        Ok(duration)
    }

    /// Profile model inference performance
    pub fn profile_inference(&self, model: &Model) -> Result<PerformanceMetrics> {
        log::info!(
            "Starting inference profiling with {} warmup runs and {} benchmark runs",
            self.config.warmup_runs,
            self.config.benchmark_runs
        );

        // Generate dummy input data
        let input_data = self.generate_dummy_input(model)?;

        // Warmup runs
        log::info!("Running warmup...");
        for _ in 0..self.config.warmup_runs {
            self.run_inference_dummy(model, &input_data)?;
        }

        // Benchmark runs
        log::info!("Running benchmarks...");
        let mut inference_times = Vec::new();
        let start_time = Instant::now();

        for _ in 0..self.config.benchmark_runs {
            let run_start = Instant::now();
            self.run_inference_dummy(model, &input_data)?;
            inference_times.push(run_start.elapsed());
        }

        let total_time = start_time.elapsed();

        // Calculate statistics
        let avg_inference_time =
            inference_times.iter().sum::<Duration>() / inference_times.len() as u32;
        let throughput = self.config.benchmark_runs as f32 / total_time.as_secs_f32();

        let metrics = PerformanceMetrics {
            load_time_ms: 0, // Not measured in this function
            inference_time_ms: avg_inference_time.as_millis() as u64,
            memory_usage_bytes: model.estimate_memory_usage(),
            cpu_utilization: 0.0, // TODO: Implement CPU measurement
            throughput,
        };

        log::info!("Inference profiling completed:");
        log::info!("  Average inference time: {} ms", metrics.inference_time_ms);
        log::info!("  Throughput: {:.2} inferences/sec", metrics.throughput);
        log::info!("  Memory usage: {} KB", metrics.memory_usage_bytes / 1024);

        Ok(metrics)
    }

    /// Generate dummy input data for profiling
    fn generate_dummy_input(&self, model: &Model) -> Result<Vec<Vec<f32>>> {
        let mut inputs = Vec::new();

        for shape in &model.info().input_shapes {
            let size = shape.iter().product::<i64>() as usize;
            let dummy_data: Vec<f32> = (0..size).map(|i| (i % 255) as f32 / 255.0).collect();
            inputs.push(dummy_data);
        }

        Ok(inputs)
    }

    /// Run a dummy inference for profiling
    fn run_inference_dummy(&self, _model: &Model, _input: &[Vec<f32>]) -> Result<()> {
        // TODO: Implement actual inference call
        // This is a placeholder that simulates inference work
        std::thread::sleep(Duration::from_micros(100)); // Simulate 0.1ms work
        Ok(())
    }

    /// Compare performance between two models
    pub fn compare_models(&self, original: &Model, optimized: &Model) -> Result<ComparisonResult> {
        log::info!("Comparing model performance...");

        let original_metrics = self.profile_inference(original)?;
        let optimized_metrics = self.profile_inference(optimized)?;

        let comparison = ComparisonResult {
            speedup_ratio: original_metrics.inference_time_ms as f32
                / optimized_metrics.inference_time_ms as f32,
            memory_reduction: 1.0
                - (optimized_metrics.memory_usage_bytes as f32
                    / original_metrics.memory_usage_bytes as f32),
            throughput_improvement: optimized_metrics.throughput / original_metrics.throughput,
            original_metrics,
            optimized_metrics,
        };

        log::info!("Performance comparison:");
        log::info!("  Speedup: {:.2}x", comparison.speedup_ratio);
        log::info!(
            "  Memory reduction: {:.1}%",
            comparison.memory_reduction * 100.0
        );
        log::info!(
            "  Throughput improvement: {:.2}x",
            comparison.throughput_improvement
        );

        Ok(comparison)
    }

    /// Benchmark against hardware constraints
    pub fn benchmark_constraints(
        &self,
        model: &Model,
        memory_limit: usize,
        latency_limit_ms: u64,
    ) -> Result<ConstraintReport> {
        let metrics = self.profile_inference(model)?;

        let memory_ok = metrics.memory_usage_bytes <= memory_limit;
        let latency_ok = metrics.inference_time_ms <= latency_limit_ms;

        let report = ConstraintReport {
            memory_constraint_met: memory_ok,
            latency_constraint_met: latency_ok,
            memory_usage: metrics.memory_usage_bytes,
            memory_limit,
            latency_ms: metrics.inference_time_ms,
            latency_limit_ms,
            overall_compatible: memory_ok && latency_ok,
        };

        log::info!("Constraint validation:");
        log::info!(
            "  Memory: {} / {} KB ({})",
            report.memory_usage / 1024,
            report.memory_limit / 1024,
            if report.memory_constraint_met {
                "OK"
            } else {
                "FAIL"
            }
        );
        log::info!(
            "  Latency: {} / {} ms ({})",
            report.latency_ms,
            report.latency_limit_ms,
            if report.latency_constraint_met {
                "OK"
            } else {
                "FAIL"
            }
        );

        Ok(report)
    }
}

/// Result of model performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub original_metrics: PerformanceMetrics,
    pub optimized_metrics: PerformanceMetrics,
    pub speedup_ratio: f32,
    pub memory_reduction: f32,
    pub throughput_improvement: f32,
}

/// Hardware constraint validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintReport {
    pub memory_constraint_met: bool,
    pub latency_constraint_met: bool,
    pub memory_usage: usize,
    pub memory_limit: usize,
    pub latency_ms: u64,
    pub latency_limit_ms: u64,
    pub overall_compatible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelData, ModelFormat, ModelInfo};

    fn create_test_model() -> Model {
        let info = ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 1000000,
            model_size_bytes: 4000000,
            operations_count: 500000,
            layers: vec![], // Empty for test
        };

        Model {
            info,
            data: ModelData::Raw(vec![0u8; 1000]),
        }
    }

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();
        assert_eq!(config.warmup_runs, 10);
        assert_eq!(config.benchmark_runs, 100);
        assert!(config.detailed_memory);
    }

    #[test]
    fn test_dummy_input_generation() {
        let profiler = Profiler::new(ProfilingConfig::default());
        let model = create_test_model();

        let inputs = profiler.generate_dummy_input(&model).unwrap();
        assert_eq!(inputs.len(), 1); // One input tensor
        assert_eq!(inputs[0].len(), 3 * 224 * 224); // Expected size
    }

    #[test]
    fn test_constraint_report() {
        let report = ConstraintReport {
            memory_constraint_met: true,
            latency_constraint_met: false,
            memory_usage: 1000000,
            memory_limit: 2000000,
            latency_ms: 150,
            latency_limit_ms: 100,
            overall_compatible: false,
        };

        assert!(report.memory_constraint_met);
        assert!(!report.latency_constraint_met);
        assert!(!report.overall_compatible);
    }
}
