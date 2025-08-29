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

//! Main benchmarking suite orchestrating performance comparisons

use super::{
    metrics::{simulate_performance_metrics, PerformanceMetrics},
    BenchmarkConfig, BenchmarkResult, BenchmarkStatistics, BenchmarkSummary, ComparisonResult,
    CompetitiveFramework, HardwarePlatform, StandardModel,
};
use crate::model::LayerInfo;
use crate::{Config, Model, Optimizer, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Main benchmarking suite for competitive analysis
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    optimizer: Optimizer,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        let optimizer = Optimizer::new(Config::default());
        Self { config, optimizer }
    }

    /// Create benchmark suite with default configuration
    pub fn with_default_config() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Run complete benchmark suite
    pub fn run_benchmarks(&self) -> Result<BenchmarkSummary> {
        let start_time = Instant::now();
        log::info!(
            "🚀 Starting benchmark suite with {} frameworks, {} platforms, {} models",
            self.config.frameworks.len(),
            self.config.platforms.len(),
            self.config.models.len()
        );

        let mut results = Vec::new();
        let mut total_benchmarks = 0;
        let mut successful_benchmarks = 0;

        // Run benchmarks for each combination
        for framework in &self.config.frameworks {
            for platform in &self.config.platforms {
                for model in &self.config.models {
                    total_benchmarks += 1;
                    log::debug!(
                        "Benchmarking {} on {} with {}",
                        framework.name(),
                        platform.name(),
                        model.name()
                    );

                    match self.run_single_benchmark(framework, platform, model) {
                        Ok(result) => {
                            if result.success {
                                successful_benchmarks += 1;
                            }
                            results.push(result);
                        }
                        Err(e) => {
                            log::warn!(
                                "Benchmark failed for {}/{}/{}: {}",
                                framework.name(),
                                platform.name(),
                                model.name(),
                                e
                            );
                            results.push(BenchmarkResult {
                                framework: framework.clone(),
                                platform: platform.clone(),
                                model: model.clone(),
                                metrics: PerformanceMetrics::default(),
                                success: false,
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }
            }
        }

        let total_duration = start_time.elapsed();

        // Calculate comparisons between frameworks
        let comparisons = self.calculate_comparisons(&results);

        // Generate statistics
        let statistics =
            self.calculate_statistics(&results, successful_benchmarks, total_benchmarks);

        log::info!(
            "✅ Benchmark suite completed: {}/{} successful in {:.1}s",
            successful_benchmarks,
            total_benchmarks,
            total_duration.as_secs_f64()
        );

        Ok(BenchmarkSummary {
            results,
            comparisons,
            statistics,
            timestamp: chrono::Utc::now(),
            total_duration,
        })
    }

    /// Run benchmark for a single framework/platform/model combination
    fn run_single_benchmark(
        &self,
        framework: &CompetitiveFramework,
        platform: &HardwarePlatform,
        model: &StandardModel,
    ) -> Result<BenchmarkResult> {
        // Simulate model creation and optimization
        let test_model = self.create_test_model(model)?;

        // Get base model size for simulation
        let base_size = test_model.info().model_size_bytes;

        // Simulate framework-specific optimization and measurement
        let metrics = match framework {
            CompetitiveFramework::Blitzed => {
                self.benchmark_blitzed(&test_model, platform, model, base_size)?
            }
            CompetitiveFramework::TensorFlowLite => {
                self.benchmark_tensorflow_lite(&test_model, platform, model, base_size)?
            }
            CompetitiveFramework::OnnxRuntime => {
                self.benchmark_onnx_runtime(&test_model, platform, model, base_size)?
            }
            CompetitiveFramework::PyTorchMobile => {
                self.benchmark_pytorch_mobile(&test_model, platform, model, base_size)?
            }
        };

        Ok(BenchmarkResult {
            framework: framework.clone(),
            platform: platform.clone(),
            model: model.clone(),
            metrics,
            success: true,
            error: None,
        })
    }

    /// Create a test model for benchmarking
    fn create_test_model(&self, model: &StandardModel) -> Result<Model> {
        // For now, create a synthetic model based on the standard model type
        use crate::model::{ModelData, ModelFormat, ModelInfo};

        let input_shape = model.input_shape();
        let (min_size_mb, max_size_mb) = model.expected_size_mb();
        let model_size_bytes = ((min_size_mb + max_size_mb) / 2.0 * 1024.0 * 1024.0) as usize;

        // Create synthetic layers based on model type
        let layers = match model {
            StandardModel::MobileNetV2 | StandardModel::MobileNetV3Small => {
                self.create_mobilenet_layers(&input_shape)
            }
            StandardModel::ResNet18 => self.create_resnet_layers(&input_shape),
            StandardModel::EfficientNetB0 => self.create_efficientnet_layers(&input_shape),
            StandardModel::YoloV5Nano => self.create_yolo_layers(&input_shape),
            StandardModel::Custom(_) => self.create_generic_layers(&input_shape),
        };

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch, // Default format
            input_shapes: vec![input_shape.clone()],
            output_shapes: match model {
                StandardModel::YoloV5Nano => vec![vec![1, 25200, 85]], // YOLO output format
                _ => vec![vec![1, 1000]],                              // ImageNet classification
            },
            parameter_count: total_params,
            model_size_bytes,
            operations_count: total_flops as usize,
            layers,
        };

        Ok(Model {
            info: model_info,
            data: ModelData::Raw(vec![0; model_size_bytes]), // Placeholder data
        })
    }

    /// Create MobileNet-style layers
    fn create_mobilenet_layers(&self, input_shape: &[i64]) -> Vec<LayerInfo> {
        vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: input_shape.to_vec(),
                output_shape: vec![1, 32, 112, 112],
                parameter_count: 864, // 3*32*3*3
                flops: 10_838_016,
            },
            LayerInfo {
                name: "depthwise_conv".to_string(),
                layer_type: "depthwise_conv2d".to_string(),
                input_shape: vec![1, 32, 112, 112],
                output_shape: vec![1, 32, 112, 112],
                parameter_count: 288, // 32*3*3
                flops: 1_204_224,
            },
            LayerInfo {
                name: "pointwise_conv".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 32, 112, 112],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 2_048, // 32*64*1*1
                flops: 25_690_112,
            },
            LayerInfo {
                name: "avgpool".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 64, 7, 7],
                output_shape: vec![1, 64, 1, 1],
                parameter_count: 0,
                flops: 3_136,
            },
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 64],
                output_shape: vec![1, 1000],
                parameter_count: 65_000, // 64*1000 + 1000
                flops: 64_000,
            },
        ]
    }

    /// Create ResNet-style layers
    fn create_resnet_layers(&self, input_shape: &[i64]) -> Vec<LayerInfo> {
        vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: input_shape.to_vec(),
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 9_408, // 3*64*7*7
                flops: 118_013_952,
            },
            LayerInfo {
                name: "layer1.0.conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 64, 56, 56],
                output_shape: vec![1, 64, 56, 56],
                parameter_count: 36_864, // 64*64*3*3
                flops: 115_605_504,
            },
            LayerInfo {
                name: "layer1.0.conv2".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 64, 56, 56],
                output_shape: vec![1, 64, 56, 56],
                parameter_count: 36_864,
                flops: 115_605_504,
            },
            LayerInfo {
                name: "avgpool".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 512, 7, 7],
                output_shape: vec![1, 512, 1, 1],
                parameter_count: 0,
                flops: 25_088,
            },
            LayerInfo {
                name: "fc".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 512],
                output_shape: vec![1, 1000],
                parameter_count: 513_000, // 512*1000 + 1000
                flops: 512_000,
            },
        ]
    }

    /// Create EfficientNet-style layers
    fn create_efficientnet_layers(&self, input_shape: &[i64]) -> Vec<LayerInfo> {
        vec![
            LayerInfo {
                name: "stem_conv".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: input_shape.to_vec(),
                output_shape: vec![1, 32, 112, 112],
                parameter_count: 864,
                flops: 10_838_016,
            },
            LayerInfo {
                name: "blocks.0".to_string(),
                layer_type: "mbconv".to_string(),
                input_shape: vec![1, 32, 112, 112],
                output_shape: vec![1, 16, 112, 112],
                parameter_count: 1_448,
                flops: 21_676_032,
            },
            LayerInfo {
                name: "head_conv".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 320, 7, 7],
                output_shape: vec![1, 1280, 7, 7],
                parameter_count: 409_600,
                flops: 20_070_400,
            },
            LayerInfo {
                name: "avgpool".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 1280, 7, 7],
                output_shape: vec![1, 1280, 1, 1],
                parameter_count: 0,
                flops: 62_720,
            },
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 1280],
                output_shape: vec![1, 1000],
                parameter_count: 1_281_000,
                flops: 1_280_000,
            },
        ]
    }

    /// Create YOLO-style layers
    fn create_yolo_layers(&self, input_shape: &[i64]) -> Vec<LayerInfo> {
        vec![
            LayerInfo {
                name: "backbone.0".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: input_shape.to_vec(),
                output_shape: vec![1, 32, 320, 320],
                parameter_count: 928, // 3*32*3*3 + 32
                flops: 94_371_840,
            },
            LayerInfo {
                name: "backbone.1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 32, 320, 320],
                output_shape: vec![1, 64, 160, 160],
                parameter_count: 18_496, // 32*64*3*3 + 64
                flops: 47_185_920,
            },
            LayerInfo {
                name: "neck.0".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 256, 20, 20],
                output_shape: vec![1, 256, 20, 20],
                parameter_count: 590_080, // 256*256*3*3 + 256
                flops: 236_032_000,
            },
            LayerInfo {
                name: "head".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 256, 20, 20],
                output_shape: vec![1, 255, 20, 20], // 85 * 3 anchors
                parameter_count: 65_535,            // 256*255 + 255
                flops: 26_214_400,
            },
        ]
    }

    /// Create generic layers for custom models
    fn create_generic_layers(&self, input_shape: &[i64]) -> Vec<LayerInfo> {
        vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: input_shape.to_vec(),
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 9_408,
                flops: 118_013_952,
            },
            LayerInfo {
                name: "conv2".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 64, 112, 112],
                output_shape: vec![1, 128, 56, 56],
                parameter_count: 73_856,
                flops: 231_211_008,
            },
            LayerInfo {
                name: "fc".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 2048],
                output_shape: vec![1, 1000],
                parameter_count: 2_049_000,
                flops: 2_048_000,
            },
        ]
    }

    /// Benchmark with Blitzed optimization
    fn benchmark_blitzed(
        &self,
        model: &Model,
        platform: &HardwarePlatform,
        standard_model: &StandardModel,
        base_size: usize,
    ) -> Result<PerformanceMetrics> {
        // Apply Blitzed optimization
        let _optimization_result = self.optimizer.optimize(model)?;

        // Simulate optimized performance
        let metrics = simulate_performance_metrics(
            "Blitzed",
            standard_model.name(),
            platform.name(),
            base_size,
        );

        log::debug!(
            "Blitzed benchmark: {}ms inference, {} bytes, {:.3} accuracy",
            metrics.avg_inference_time_ms,
            metrics.model_size_bytes,
            metrics.accuracy_score
        );

        Ok(metrics)
    }

    /// Benchmark with TensorFlow Lite
    fn benchmark_tensorflow_lite(
        &self,
        _model: &Model,
        platform: &HardwarePlatform,
        standard_model: &StandardModel,
        base_size: usize,
    ) -> Result<PerformanceMetrics> {
        // Simulate TensorFlow Lite conversion and benchmarking
        // TODO: Integrate with actual TensorFlow Lite when available

        let metrics = simulate_performance_metrics(
            "TensorFlow Lite",
            standard_model.name(),
            platform.name(),
            base_size,
        );

        log::debug!(
            "TensorFlow Lite benchmark: {}ms inference, {} bytes, {:.3} accuracy",
            metrics.avg_inference_time_ms,
            metrics.model_size_bytes,
            metrics.accuracy_score
        );

        Ok(metrics)
    }

    /// Benchmark with ONNX Runtime
    fn benchmark_onnx_runtime(
        &self,
        _model: &Model,
        platform: &HardwarePlatform,
        standard_model: &StandardModel,
        base_size: usize,
    ) -> Result<PerformanceMetrics> {
        // Simulate ONNX Runtime benchmarking
        // TODO: Integrate with actual ONNX Runtime when available

        let metrics = simulate_performance_metrics(
            "ONNX Runtime",
            standard_model.name(),
            platform.name(),
            base_size,
        );

        log::debug!(
            "ONNX Runtime benchmark: {}ms inference, {} bytes, {:.3} accuracy",
            metrics.avg_inference_time_ms,
            metrics.model_size_bytes,
            metrics.accuracy_score
        );

        Ok(metrics)
    }

    /// Benchmark with PyTorch Mobile
    fn benchmark_pytorch_mobile(
        &self,
        _model: &Model,
        platform: &HardwarePlatform,
        standard_model: &StandardModel,
        base_size: usize,
    ) -> Result<PerformanceMetrics> {
        // Simulate PyTorch Mobile benchmarking
        // TODO: Integrate with actual PyTorch Mobile when available

        let metrics = simulate_performance_metrics(
            "PyTorch Mobile",
            standard_model.name(),
            platform.name(),
            base_size,
        );

        log::debug!(
            "PyTorch Mobile benchmark: {}ms inference, {} bytes, {:.3} accuracy",
            metrics.avg_inference_time_ms,
            metrics.model_size_bytes,
            metrics.accuracy_score
        );

        Ok(metrics)
    }

    /// Calculate pairwise comparisons between frameworks
    fn calculate_comparisons(
        &self,
        results: &[BenchmarkResult],
    ) -> HashMap<(CompetitiveFramework, CompetitiveFramework), ComparisonResult> {
        let mut comparisons = HashMap::new();

        // Group results by platform and model for fair comparison
        let mut grouped_results: HashMap<(HardwarePlatform, StandardModel), Vec<&BenchmarkResult>> =
            HashMap::new();

        for result in results {
            if result.success {
                let key = (result.platform.clone(), result.model.clone());
                grouped_results.entry(key).or_default().push(result);
            }
        }

        // Compare frameworks within each group
        for group_results in grouped_results.values() {
            for baseline_result in group_results {
                for target_result in group_results {
                    if baseline_result.framework != target_result.framework {
                        let key = (
                            baseline_result.framework.clone(),
                            target_result.framework.clone(),
                        );

                        comparisons.entry(key).or_insert_with(|| ComparisonResult {
                            baseline: baseline_result.framework.clone(),
                            target: target_result.framework.clone(),
                            speedup: if target_result.metrics.avg_inference_time_ms > 0 {
                                baseline_result.metrics.avg_inference_time_ms as f64
                                    / target_result.metrics.avg_inference_time_ms as f64
                            } else {
                                1.0
                            },
                            size_reduction: if target_result.metrics.model_size_bytes > 0 {
                                baseline_result.metrics.model_size_bytes as f64
                                    / target_result.metrics.model_size_bytes as f64
                            } else {
                                1.0
                            },
                            memory_efficiency: if baseline_result.metrics.peak_memory_usage > 0 {
                                baseline_result.metrics.peak_memory_usage as f64
                                    / target_result.metrics.peak_memory_usage as f64
                            } else {
                                1.0
                            },
                            accuracy_difference: target_result.metrics.accuracy_score
                                - baseline_result.metrics.accuracy_score,
                        });
                    }
                }
            }
        }

        comparisons
    }

    /// Calculate overall benchmark statistics
    fn calculate_statistics(
        &self,
        results: &[BenchmarkResult],
        successful_benchmarks: usize,
        total_benchmarks: usize,
    ) -> BenchmarkStatistics {
        let blitzed_results: Vec<_> = results
            .iter()
            .filter(|r| r.framework == CompetitiveFramework::Blitzed && r.success)
            .collect();

        let competitor_results: Vec<_> = results
            .iter()
            .filter(|r| r.framework != CompetitiveFramework::Blitzed && r.success)
            .collect();

        let (average_blitzed_speedup, average_blitzed_compression, average_accuracy_retention) =
            if !blitzed_results.is_empty() && !competitor_results.is_empty() {
                let blitzed_avg_latency: f64 = blitzed_results
                    .iter()
                    .map(|r| r.metrics.avg_inference_time_ms as f64)
                    .sum::<f64>()
                    / blitzed_results.len() as f64;

                let competitor_avg_latency: f64 = competitor_results
                    .iter()
                    .map(|r| r.metrics.avg_inference_time_ms as f64)
                    .sum::<f64>()
                    / competitor_results.len() as f64;

                let blitzed_avg_size: f64 = blitzed_results
                    .iter()
                    .map(|r| r.metrics.model_size_bytes as f64)
                    .sum::<f64>()
                    / blitzed_results.len() as f64;

                let competitor_avg_size: f64 = competitor_results
                    .iter()
                    .map(|r| r.metrics.model_size_bytes as f64)
                    .sum::<f64>()
                    / competitor_results.len() as f64;

                let blitzed_avg_accuracy: f64 = blitzed_results
                    .iter()
                    .map(|r| r.metrics.accuracy_score)
                    .sum::<f64>()
                    / blitzed_results.len() as f64;

                let speedup = if blitzed_avg_latency > 0.0 {
                    competitor_avg_latency / blitzed_avg_latency
                } else {
                    1.0
                };

                let compression = if blitzed_avg_size > 0.0 {
                    competitor_avg_size / blitzed_avg_size
                } else {
                    1.0
                };

                (speedup, compression, blitzed_avg_accuracy)
            } else {
                (1.0, 1.0, 0.0)
            };

        // Find best platform for Blitzed
        let best_platform = blitzed_results
            .iter()
            .max_by(|a, b| {
                let a_score =
                    a.metrics.throughput_ops_per_sec / (a.metrics.model_size_bytes as f64);
                let b_score =
                    b.metrics.throughput_ops_per_sec / (b.metrics.model_size_bytes as f64);
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.platform.clone());

        // Find most challenging model (lowest relative performance vs competitors)
        let most_challenging_model = blitzed_results
            .iter()
            .min_by(|a, b| {
                a.metrics
                    .accuracy_score
                    .partial_cmp(&b.metrics.accuracy_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.model.clone());

        BenchmarkStatistics {
            total_benchmarks,
            successful_benchmarks,
            average_blitzed_speedup,
            average_blitzed_compression,
            average_accuracy_retention,
            best_platform,
            most_challenging_model,
        }
    }
}
