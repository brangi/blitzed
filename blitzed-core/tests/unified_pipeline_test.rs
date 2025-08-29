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

//! Unified optimization pipeline integration tests
//!
//! This test suite validates the complete Load → Quantize → Prune → Distill → Deploy
//! pipeline orchestration including:
//! - Chained optimization techniques with cumulative effects
//! - Accuracy budget management across techniques
//! - Hardware-specific optimization strategies
//! - Performance benchmarking and impact validation

#[cfg(test)]
mod tests {
    use blitzed_core::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::Optimizer;
    use blitzed_core::Config;

    fn create_edge_optimized_model() -> Model {
        // Create a small model suitable for edge deployment testing (~7KB)
        let layers = vec![
            // Small input convolution
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 32, 32], // Smaller input size
                output_shape: vec![1, 16, 16, 16],
                parameter_count: 432, // 3*16*3*3
                flops: 110_592,
            },
            // Depthwise convolution
            LayerInfo {
                name: "dw_conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 16, 16, 16],
                output_shape: vec![1, 16, 16, 16],
                parameter_count: 144, // 16*3*3
                flops: 36_864,
            },
            // Pointwise convolution
            LayerInfo {
                name: "pw_conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 16, 16, 16],
                output_shape: vec![1, 32, 16, 16],
                parameter_count: 512, // 16*32*1*1
                flops: 131_072,
            },
            // Another depthwise
            LayerInfo {
                name: "dw_conv2".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 32, 16, 16],
                output_shape: vec![1, 32, 8, 8],
                parameter_count: 288, // 32*3*3
                flops: 18_432,
            },
            // Global pooling
            LayerInfo {
                name: "global_pool".to_string(),
                layer_type: "avgpool".to_string(),
                input_shape: vec![1, 32, 8, 8],
                output_shape: vec![1, 32, 1, 1],
                parameter_count: 0,
                flops: 2_048,
            },
            // Small classifier for edge use case (10 classes instead of 1000)
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 32],
                output_shape: vec![1, 10],
                parameter_count: 330, // 32*10 + 10
                flops: 320,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum(); // ~1.7K parameters
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 32, 32]], // CIFAR-10 like input
            output_shapes: vec![vec![1, 10]],       // 10 classes
            parameter_count: total_params,          // ~1.7K parameters
            model_size_bytes: total_params * 4,     // FP32 = ~7KB model
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    /// Test unified pipeline with quantization only
    #[test]
    fn test_unified_pipeline_quantization_only() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();
        let original_size = model.info().model_size_bytes;

        // Configure for ESP32 with quantization only
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = false;
        config.optimization.enable_distillation = false;
        config.optimization.max_accuracy_loss = 10.0; // More lenient for testing

        let optimizer = Optimizer::new(config);
        let result = optimizer.optimize(&model).unwrap();

        println!("🧪 Quantization-Only Pipeline Results:");
        println!(
            "  Original size: {:.1} MB",
            original_size as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Optimized size: {:.1} MB",
            result.optimized_size as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Compression ratio: {:.1}%",
            result.compression_ratio * 100.0
        );
        println!(
            "  Estimated accuracy loss: {:.1}%",
            result.estimated_accuracy_loss
        );
        println!("  Estimated speedup: {:.1}x", result.estimated_speedup);
        println!("  Optimization time: {} ms", result.optimization_time_ms);
        println!("  Techniques: {:?}", result.techniques_applied);

        // Verify quantization was applied
        assert!(!result.techniques_applied.is_empty());
        assert!(result
            .techniques_applied
            .iter()
            .any(|t| t.contains("Quantization")));
        assert!(result.compression_ratio > 0.0);
        assert!(result.estimated_accuracy_loss > 0.0);
        assert!(result.estimated_speedup >= 1.0);
        assert!(result.optimization_time_ms < 1000); // Should be fast
    }

    /// Test unified pipeline with all optimization techniques
    #[test]
    fn test_unified_pipeline_all_techniques() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();
        let original_size = model.info().model_size_bytes;

        // Configure for ESP32 with all optimizations enabled
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.enable_distillation = true;
        config.optimization.max_accuracy_loss = 15.0; // Higher budget for multiple techniques

        let optimizer = Optimizer::new(config);
        let result = optimizer.optimize(&model).unwrap();

        println!("🚀 Full Pipeline Results (Quantize + Prune + Distill):");
        println!(
            "  Original size: {:.1} MB",
            original_size as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Optimized size: {:.1} MB",
            result.optimized_size as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Compression ratio: {:.1}%",
            result.compression_ratio * 100.0
        );
        println!(
            "  Estimated accuracy loss: {:.1}%",
            result.estimated_accuracy_loss
        );
        println!("  Estimated speedup: {:.1}x", result.estimated_speedup);
        println!("  Optimization time: {} ms", result.optimization_time_ms);
        println!("  Techniques applied:");
        for technique in &result.techniques_applied {
            println!("    - {}", technique);
        }

        // Verify all techniques were considered/applied
        assert!(!result.techniques_applied.is_empty());

        // Should achieve significant compression with multiple techniques
        assert!(result.compression_ratio > 0.1); // At least 10% reduction

        // Accuracy loss should be within budget
        assert!(result.estimated_accuracy_loss <= 15.0);

        // Should have significant speedup with all optimizations
        assert!(result.estimated_speedup >= 1.0);

        // Multiple techniques should give better compression than single technique
        if result.techniques_applied.len() > 1 {
            assert!(result.compression_ratio > 0.2); // Better compression with multiple techniques
        }
    }

    /// Test pipeline with accuracy budget constraints
    #[test]
    fn test_pipeline_accuracy_budget_management() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();

        // Test with very restrictive accuracy budget
        let mut config = Config::default();
        config.hardware.target = "raspberry_pi".to_string(); // Less aggressive than ESP32
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.enable_distillation = true;
        config.optimization.max_accuracy_loss = 2.0; // Very restrictive budget

        let optimizer = Optimizer::new(config);
        let result = optimizer.optimize(&model).unwrap();

        println!("🎯 Budget-Constrained Pipeline Results:");
        println!("  Accuracy budget: 2.0%");
        println!(
            "  Actual accuracy loss: {:.1}%",
            result.estimated_accuracy_loss
        );
        println!("  Techniques applied: {}", result.techniques_applied.len());
        println!(
            "  Compression achieved: {:.1}%",
            result.compression_ratio * 100.0
        );

        // Should respect accuracy budget
        assert!(result.estimated_accuracy_loss <= 2.0);

        // May have fewer techniques applied due to budget constraints
        // But should still achieve some optimization
        assert!(result.compression_ratio >= 0.0);

        // Should complete successfully without errors (time can be 0 for very fast operations)
    }

    /// Test hardware-specific optimization strategies
    #[test]
    fn test_hardware_specific_strategies() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();

        let targets = vec![
            ("esp32", "ESP32 should prefer aggressive quantization"),
            ("raspberry_pi", "RPi should balance speed and accuracy"),
            ("stm32", "STM32 should prioritize memory efficiency"),
        ];

        for (target, description) in targets {
            let mut config = Config::default();
            config.hardware.target = target.to_string();
            config.optimization.enable_quantization = true;
            config.optimization.enable_pruning = true;
            config.optimization.max_accuracy_loss = 10.0;

            let optimizer = Optimizer::new(config);
            let result = optimizer.optimize(&model).unwrap();

            println!("🔧 {} Strategy Results:", target.to_uppercase());
            println!("  {}", description);
            println!("  Compression: {:.1}%", result.compression_ratio * 100.0);
            println!("  Accuracy loss: {:.1}%", result.estimated_accuracy_loss);
            println!("  Speedup: {:.1}x", result.estimated_speedup);
            println!("  Techniques: {:?}", result.techniques_applied);
            println!();

            // All targets should achieve some optimization
            assert!(result.compression_ratio > 0.0);
            assert!(!result.techniques_applied.is_empty());
            assert!(result.estimated_accuracy_loss <= 10.0);
        }
    }

    /// Test optimization impact estimation vs actual results
    #[test]
    fn test_impact_estimation_accuracy() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();

        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.max_accuracy_loss = 12.0;

        let optimizer = Optimizer::new(config);

        // Get impact estimation
        let estimated_impact = optimizer.estimate_impact(&model).unwrap();

        // Get actual optimization results
        let actual_result = optimizer.optimize(&model).unwrap();

        println!("📊 Impact Estimation vs Actual Results:");
        println!(
            "  Estimated size reduction: {:.1}%",
            estimated_impact.size_reduction * 100.0
        );
        println!(
            "  Actual compression ratio: {:.1}%",
            actual_result.compression_ratio * 100.0
        );
        println!(
            "  Estimated accuracy loss: {:.1}%",
            estimated_impact.accuracy_loss
        );
        println!(
            "  Actual accuracy loss: {:.1}%",
            actual_result.estimated_accuracy_loss
        );
        println!(
            "  Estimated speedup: {:.1}x",
            estimated_impact.speed_improvement
        );
        println!("  Actual speedup: {:.1}x", actual_result.estimated_speedup);

        // Estimations should be in reasonable ballpark of actual results
        let size_diff = (estimated_impact.size_reduction - actual_result.compression_ratio).abs();
        assert!(size_diff < 0.5, "Size reduction estimation too far off");

        let accuracy_diff =
            (estimated_impact.accuracy_loss - actual_result.estimated_accuracy_loss).abs();
        assert!(accuracy_diff < 5.0, "Accuracy loss estimation too far off");

        let speedup_ratio = estimated_impact.speed_improvement / actual_result.estimated_speedup;
        assert!(
            speedup_ratio > 0.5 && speedup_ratio < 2.0,
            "Speedup estimation too far off"
        );
    }

    /// Test configuration validation and error handling
    #[test]
    fn test_pipeline_configuration_validation() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();

        // Test with impossible accuracy constraints
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.enable_distillation = true;
        config.optimization.max_accuracy_loss = 0.1; // Impossible to achieve with multiple techniques

        let optimizer = Optimizer::new(config);

        // Should either succeed with minimal optimization or fail gracefully
        match optimizer.optimize(&model) {
            Ok(result) => {
                println!("✅ Pipeline succeeded with minimal optimization:");
                println!("  Techniques applied: {}", result.techniques_applied.len());
                println!("  Accuracy loss: {:.1}%", result.estimated_accuracy_loss);
                assert!(result.estimated_accuracy_loss <= 0.1);
            }
            Err(e) => {
                println!(
                    "⚠️ Pipeline failed as expected with restrictive constraints: {}",
                    e
                );
                // This is acceptable - very restrictive constraints may not be achievable
            }
        }
    }

    /// Test optimization recommendations
    #[test]
    fn test_optimization_recommendations() {
        env_logger::try_init().ok();

        let model = create_edge_optimized_model();

        let targets = vec!["esp32", "raspberry_pi", "stm32"];

        for target in targets {
            let mut config = Config::default();
            config.hardware.target = target.to_string();

            let optimizer = Optimizer::new(config);
            let recommendations = optimizer.recommend(&model).unwrap();

            println!(
                "💡 Optimization Recommendations for {}:",
                target.to_uppercase()
            );
            for (i, recommendation) in recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, recommendation);
            }
            println!();

            // Should have meaningful recommendations
            assert!(!recommendations.is_empty());

            // Should mention the target hardware (check for both lowercase and formatted names)
            let target_mentioned = recommendations.iter().any(|r| {
                let r_lower = r.to_lowercase();
                r_lower.contains(target)
                    || r_lower.contains(&target.replace("_", " "))
                    || r.contains(&target.to_uppercase())
                    || r.contains("ESP32") && target == "esp32"
                    || r.contains("Raspberry Pi") && target == "raspberry_pi"
                    || r.contains("STM32") && target == "stm32"
            });
            assert!(
                target_mentioned,
                "Target {} not mentioned in recommendations: {:?}",
                target, recommendations
            );

            // Should have optimization-specific recommendations
            assert!(recommendations.iter().any(|r| r.contains("quantization")
                || r.contains("pruning")
                || r.contains("optimization")
                || r.contains("memory")
                || r.contains("accelerator")));
        }
    }
}
