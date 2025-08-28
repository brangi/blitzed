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

//! Comprehensive pruning integration tests
//!
//! This test suite validates the complete pruning implementation including:
//! - Magnitude-based pruning (unstructured and structured)
//! - Layer sensitivity handling
//! - Integration with the optimization pipeline
//! - Performance and accuracy trade-offs

#[cfg(test)]
mod tests {
    use blitzed_core::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::pruning::{Pruner, PruningConfig, PruningMethod};
    use blitzed_core::optimization::OptimizationTechnique;

    fn create_realistic_model() -> Model {
        // Create a realistic model similar to MobileNetV2 or similar efficient architecture
        let layers = vec![
            // Initial convolution
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 32, 112, 112],
                parameter_count: 864, // 3*32*3*3
                flops: 21_676_032,    // 864 * 112 * 112
            },
            // Depthwise separable blocks
            LayerInfo {
                name: "dw_conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 32, 112, 112],
                output_shape: vec![1, 32, 112, 112],
                parameter_count: 288, // 32*3*3
                flops: 3_612_672,     // 288 * 112 * 112
            },
            LayerInfo {
                name: "pw_conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 32, 112, 112],
                output_shape: vec![1, 16, 112, 112],
                parameter_count: 512, // 32*16*1*1
                flops: 6_422_528,     // 512 * 112 * 112
            },
            // Bottleneck blocks (simplified)
            LayerInfo {
                name: "bottleneck1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 16, 112, 112],
                output_shape: vec![1, 96, 56, 56],
                parameter_count: 13_824, // Multiple convs combined
                flops: 43_319_296,
            },
            LayerInfo {
                name: "bottleneck2".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 96, 56, 56],
                output_shape: vec![1, 320, 7, 7],
                parameter_count: 434_176, // Multiple layers
                flops: 95_420_416,
            },
            // Global average pooling (no parameters)
            LayerInfo {
                name: "global_pool".to_string(),
                layer_type: "avgpool".to_string(),
                input_shape: vec![1, 320, 7, 7],
                output_shape: vec![1, 320, 1, 1],
                parameter_count: 0,
                flops: 15_680,
            },
            // Final classifier
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 320],
                output_shape: vec![1, 1000],
                parameter_count: 321_000, // 320*1000 + 1000
                flops: 320_000,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: total_params,      // ~771K parameters
            model_size_bytes: total_params * 4, // FP32
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    /// Test magnitude-based unstructured pruning on realistic model
    #[test]
    fn test_pruning_realistic_model_unstructured() {
        env_logger::try_init().ok();

        let model = create_realistic_model();

        let config = PruningConfig {
            target_sparsity: 0.5,
            structured: false,
            method: PruningMethod::Magnitude,
            fine_tune_epochs: 0,
        };

        let pruner = Pruner::new(config);
        let result = pruner.prune_magnitude(&model).unwrap();

        println!("🔬 Unstructured Pruning Results for Realistic Model:");
        println!(
            "  Original parameters: {}",
            result.original_model_info.parameter_count
        );
        println!("  Pruned parameters: {}", result.pruned_parameters);
        println!("  Remaining parameters: {}", result.remaining_parameters);
        println!("  Sparsity achieved: {:.1}%", result.sparsity_ratio * 100.0);
        println!("  Compression ratio: {:.1}x", result.compression_ratio());
        println!("  Estimated accuracy loss: {:.1}%", result.accuracy_loss);

        // Verify reasonable pruning results
        assert!(result.sparsity_ratio >= 0.35); // Should achieve reasonable sparsity
        assert!(result.sparsity_ratio <= 0.6); // But not too extreme
        assert!(result.compression_ratio() >= 1.4); // Good compression
        assert!(result.accuracy_loss < 30.0); // Reasonable accuracy loss

        // Check layer-wise results
        for layer in &result.layers {
            if layer.original_parameters > 0 {
                println!(
                    "    {}: {:.1}% sparsity ({}/{} params)",
                    layer.name,
                    layer.sparsity_achieved * 100.0,
                    layer.pruned_parameters,
                    layer.original_parameters
                );

                // Verify layer sparsity is reasonable
                assert!(layer.sparsity_achieved >= 0.0);
                assert!(layer.sparsity_achieved <= 0.8); // Max 80% for any single layer
            }
        }
    }

    /// Test magnitude-based structured pruning on realistic model  
    #[test]
    fn test_pruning_realistic_model_structured() {
        env_logger::try_init().ok();

        let model = create_realistic_model();

        let config = PruningConfig {
            target_sparsity: 0.3, // More conservative for structured
            structured: true,
            method: PruningMethod::Magnitude,
            fine_tune_epochs: 5,
        };

        let pruner = Pruner::new(config);
        let result = pruner.prune_magnitude(&model).unwrap();

        println!("🏗️ Structured Pruning Results for Realistic Model:");
        println!(
            "  Original parameters: {}",
            result.original_model_info.parameter_count
        );
        println!("  Pruned parameters: {}", result.pruned_parameters);
        println!("  Remaining parameters: {}", result.remaining_parameters);
        println!("  Sparsity achieved: {:.1}%", result.sparsity_ratio * 100.0);
        println!("  Compression ratio: {:.1}x", result.compression_ratio());
        println!("  Estimated accuracy loss: {:.1}%", result.accuracy_loss);

        // Structured pruning should have different characteristics
        assert!(result.structured);
        assert!(result.sparsity_ratio >= 0.15); // More conservative
        assert!(result.sparsity_ratio <= 0.4);
        assert!(result.accuracy_loss < 15.0); // Should have lower accuracy loss

        // All layers should be marked as structured
        for layer in &result.layers {
            assert!(layer.structured_pruning_applied);
        }
    }

    /// Test pruning impact estimation accuracy
    #[test]
    fn test_pruning_impact_vs_actual() {
        let model = create_realistic_model();

        let configs = vec![
            PruningConfig {
                target_sparsity: 0.2,
                structured: false,
                method: PruningMethod::Magnitude,
                fine_tune_epochs: 0,
            },
            PruningConfig {
                target_sparsity: 0.5,
                structured: false,
                method: PruningMethod::Magnitude,
                fine_tune_epochs: 0,
            },
            PruningConfig {
                target_sparsity: 0.3,
                structured: true,
                method: PruningMethod::Magnitude,
                fine_tune_epochs: 0,
            },
        ];

        for config in configs {
            let pruner = Pruner::new(config.clone());

            // Get impact estimation
            let impact = pruner.estimate_impact(&model, &config).unwrap();

            // Get actual results
            let result = pruner.optimize(&model, &config).unwrap();

            println!(
                "📊 Impact vs Actual for sparsity {:.0}%, structured: {}:",
                config.target_sparsity * 100.0,
                config.structured
            );
            println!("  Estimated accuracy loss: {:.1}%", impact.accuracy_loss);
            println!("  Actual accuracy loss: {:.1}%", result.accuracy_loss);
            println!(
                "  Estimated size reduction: {:.1}%",
                impact.size_reduction * 100.0
            );
            println!("  Actual sparsity: {:.1}%", result.sparsity_ratio * 100.0);

            // Verify estimations are in reasonable ballpark
            let accuracy_diff = (impact.accuracy_loss - result.accuracy_loss).abs();
            assert!(accuracy_diff < 10.0, "Accuracy loss estimation too far off");

            // Size reduction should be correlated with sparsity
            assert!(impact.size_reduction > 0.0);
            assert!(impact.speed_improvement >= 1.0);
        }
    }

    /// Test pruning with different methods (future compatibility)
    #[test]
    fn test_pruning_method_fallbacks() {
        let model = create_realistic_model();

        let methods = vec![
            PruningMethod::Magnitude,
            PruningMethod::Gradient, // Should fall back to magnitude
            PruningMethod::Random,   // Should fall back to magnitude
        ];

        for method in methods {
            let config = PruningConfig {
                target_sparsity: 0.4,
                structured: false,
                method,
                fine_tune_epochs: 0,
            };

            let pruner = Pruner::new(config.clone());
            let result = pruner.optimize(&model, &config).unwrap();

            // All methods should currently use magnitude-based pruning
            assert_eq!(result.method_used, PruningMethod::Magnitude);
            assert!(result.sparsity_ratio > 0.0);

            println!(
                "✅ Method {:?} completed (using magnitude fallback)",
                method
            );
        }
    }

    /// Test layer sensitivity is working correctly
    #[test]
    fn test_layer_sensitivity_behavior() {
        let model = create_realistic_model();

        let config = PruningConfig {
            target_sparsity: 0.6, // High sparsity to see differences
            structured: false,
            method: PruningMethod::Magnitude,
            fine_tune_epochs: 0,
        };

        let pruner = Pruner::new(config);
        let result = pruner.prune_magnitude(&model).unwrap();

        println!("🎯 Layer Sensitivity Analysis:");

        // Find conv and linear layers to compare
        let conv_layer = result
            .layers
            .iter()
            .find(|l| l.layer_type.contains("conv") && l.original_parameters > 0)
            .expect("Should have conv layer");

        let linear_layer = result
            .layers
            .iter()
            .find(|l| l.layer_type == "linear")
            .expect("Should have linear layer");

        println!(
            "  Conv layer '{}': {:.1}% sparsity",
            conv_layer.name,
            conv_layer.sparsity_achieved * 100.0
        );
        println!(
            "  Linear layer '{}': {:.1}% sparsity",
            linear_layer.name,
            linear_layer.sparsity_achieved * 100.0
        );

        // Linear layers should have lower sparsity due to sensitivity adjustment
        assert!(
            linear_layer.sparsity_achieved < conv_layer.sparsity_achieved,
            "Linear layers should be pruned more conservatively"
        );

        // Conv layers should get closer to target sparsity
        assert!(
            conv_layer.sparsity_achieved >= 0.5,
            "Conv layers should achieve higher sparsity"
        );
    }

    /// Test model size and parameter count calculations
    #[test]
    fn test_pruning_metrics_accuracy() {
        let model = create_realistic_model();
        let original_params = model.info().parameter_count;

        let config = PruningConfig {
            target_sparsity: 0.4,
            structured: false,
            method: PruningMethod::Magnitude,
            fine_tune_epochs: 0,
        };

        let pruner = Pruner::new(config);
        let result = pruner.prune_magnitude(&model).unwrap();

        // Verify parameter counting is consistent
        let layer_total_original: usize = result.layers.iter().map(|l| l.original_parameters).sum();
        let layer_total_pruned: usize = result.layers.iter().map(|l| l.pruned_parameters).sum();

        assert_eq!(layer_total_original, original_params);
        assert_eq!(layer_total_pruned, result.pruned_parameters);
        assert_eq!(
            layer_total_original - layer_total_pruned,
            result.remaining_parameters
        );

        // Verify sparsity calculation
        let calculated_sparsity = result.pruned_parameters as f32 / original_params as f32;
        assert!((calculated_sparsity - result.sparsity_ratio).abs() < 0.001);

        // Verify compression ratio
        let calculated_compression = original_params as f32 / result.remaining_parameters as f32;
        assert!((calculated_compression - result.compression_ratio()).abs() < 0.01);

        println!("📏 Metrics Verification:");
        println!("  Parameter accounting: ✅");
        println!("  Sparsity calculation: ✅");
        println!("  Compression ratio: ✅");
    }
}
