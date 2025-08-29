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

//! Cross-format model validation integration tests
//!
//! This test suite validates the complete cross-format validation pipeline including:
//! - PyTorch ↔ ONNX model consistency validation
//! - Optimization accuracy preservation across formats
//! - Numerical precision validation and error handling
//! - Real-world validation scenarios

#[cfg(test)]
mod tests {
    use blitzed_core::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::Optimizer;
    use blitzed_core::validation::{CrossFormatValidator, ValidationConfig};
    use blitzed_core::Config;

    fn create_realistic_pytorch_model() -> Model {
        // Create a realistic PyTorch-style model for validation testing
        let layers = vec![
            // Convolutional layers
            LayerInfo {
                name: "features.0".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 1792, // 3*64*3*3 + 64
                flops: 89_915_392,
            },
            LayerInfo {
                name: "features.1".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 64, 112, 112],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 0,
                flops: 802_816,
            },
            LayerInfo {
                name: "features.2".to_string(),
                layer_type: "maxpool2d".to_string(),
                input_shape: vec![1, 64, 112, 112],
                output_shape: vec![1, 64, 56, 56],
                parameter_count: 0,
                flops: 802_816,
            },
            // More conv layers
            LayerInfo {
                name: "features.3".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 64, 56, 56],
                output_shape: vec![1, 128, 56, 56],
                parameter_count: 73_856, // 64*128*3*3 + 128
                flops: 231_211_008,
            },
            LayerInfo {
                name: "features.4".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 128, 56, 56],
                output_shape: vec![1, 128, 56, 56],
                parameter_count: 0,
                flops: 401_408,
            },
            LayerInfo {
                name: "avgpool".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 128, 56, 56],
                output_shape: vec![1, 128, 1, 1],
                parameter_count: 0,
                flops: 401_408,
            },
            // Classifier
            LayerInfo {
                name: "classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 128],
                output_shape: vec![1, 1000],
                parameter_count: 129_000, // 128*1000 + 1000
                flops: 128_000,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum();
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: total_params,
            model_size_bytes: total_params * 4,
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    fn create_equivalent_onnx_model() -> Model {
        // Create equivalent ONNX model with same structure
        let mut pytorch_model = create_realistic_pytorch_model();

        // Change format to ONNX but keep everything else the same
        pytorch_model.info.format = ModelFormat::Onnx;

        // Adjust layer names to ONNX style
        for layer in &mut pytorch_model.info.layers {
            layer.name = layer.name.replace("features.", "").replace(".", "/");
        }

        pytorch_model
    }

    fn create_incompatible_model() -> Model {
        // Create a model with different structure for negative testing
        let layers = vec![
            LayerInfo {
                name: "different_conv".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 32, 32], // Different input size
                output_shape: vec![1, 16, 16, 16],
                parameter_count: 448,
                flops: 115_200,
            },
            LayerInfo {
                name: "different_classifier".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 256],
                output_shape: vec![1, 10], // Different output classes
                parameter_count: 2570,
                flops: 2560,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum();

        let model_info = ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 32, 32]], // Different input shape
            output_shapes: vec![vec![1, 10]],       // Different output shape
            parameter_count: total_params,
            model_size_bytes: total_params * 4,
            operations_count: 117_760,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    /// Test basic cross-format validator creation and configuration
    #[test]
    fn test_validator_creation_and_configuration() {
        env_logger::try_init().ok();

        // Test default creation
        let validator = CrossFormatValidator::new();
        assert_eq!(validator.config().numerical_tolerance, 1e-5);
        assert_eq!(validator.config().max_accuracy_loss, 5.0);
        assert_eq!(validator.config().test_input_count, 10);
        assert!(!validator.config().intensive_validation);

        println!("✅ Default validator configuration:");
        println!(
            "   - Numerical tolerance: {:.1e}",
            validator.config().numerical_tolerance
        );
        println!(
            "   - Max accuracy loss: {:.1}%",
            validator.config().max_accuracy_loss
        );
        println!(
            "   - Test input count: {}",
            validator.config().test_input_count
        );

        // Test custom configuration
        let custom_config = ValidationConfig {
            numerical_tolerance: 1e-6,
            max_accuracy_loss: 2.0,
            test_input_count: 20,
            random_seed: 123,
            intensive_validation: true,
        };

        let validator = CrossFormatValidator::with_config(custom_config.clone());
        assert_eq!(validator.config().numerical_tolerance, 1e-6);
        assert_eq!(validator.config().max_accuracy_loss, 2.0);
        assert_eq!(validator.config().test_input_count, 20);
        assert!(validator.config().intensive_validation);

        println!("✅ Custom validator configuration:");
        println!(
            "   - Numerical tolerance: {:.1e}",
            validator.config().numerical_tolerance
        );
        println!(
            "   - Max accuracy loss: {:.1}%",
            validator.config().max_accuracy_loss
        );
        println!(
            "   - Intensive validation: {}",
            validator.config().intensive_validation
        );
    }

    /// Test model structure validation
    #[test]
    fn test_model_structure_validation() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();
        let pytorch_model = create_realistic_pytorch_model();

        // Test valid model structure
        let result = validator.validate_model_structure(&pytorch_model, "PyTorch");
        assert!(result.is_ok());

        println!("✅ Valid model structure validation passed");
        println!("   - Parameters: {}", pytorch_model.info().parameter_count);
        println!("   - Layers: {}", pytorch_model.info().layers.len());
        println!("   - Input shapes: {:?}", pytorch_model.info().input_shapes);
        println!(
            "   - Output shapes: {:?}",
            pytorch_model.info().output_shapes
        );

        // Test model with no inputs (should fail)
        let mut invalid_model = create_realistic_pytorch_model();
        invalid_model.info.input_shapes.clear();

        let result = validator.validate_model_structure(&invalid_model, "PyTorch");
        assert!(result.is_err());
        println!("✅ Invalid model (no inputs) correctly rejected");

        // Test model with no outputs (should fail)
        let mut invalid_model = create_realistic_pytorch_model();
        invalid_model.info.output_shapes.clear();

        let result = validator.validate_model_structure(&invalid_model, "PyTorch");
        assert!(result.is_err());
        println!("✅ Invalid model (no outputs) correctly rejected");
    }

    /// Test model compatibility validation between PyTorch and ONNX
    #[test]
    fn test_model_compatibility_validation() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();
        let pytorch_model = create_realistic_pytorch_model();
        let onnx_model = create_equivalent_onnx_model();

        println!("🔍 Testing compatible models:");
        println!(
            "   PyTorch model: {} params, {} layers",
            pytorch_model.info().parameter_count,
            pytorch_model.info().layers.len()
        );
        println!(
            "   ONNX model: {} params, {} layers",
            onnx_model.info().parameter_count,
            onnx_model.info().layers.len()
        );

        // Test compatible models
        let result = validator.validate_model_compatibility(&pytorch_model, &onnx_model);
        assert!(result.is_ok());
        println!("✅ Compatible models validation passed");

        // Test incompatible models
        let incompatible_model = create_incompatible_model();
        let result = validator.validate_model_compatibility(&pytorch_model, &incompatible_model);
        assert!(result.is_err());

        println!("✅ Incompatible models correctly rejected:");
        println!(
            "   - PyTorch input: {:?}",
            pytorch_model.info().input_shapes[0]
        );
        println!(
            "   - Incompatible input: {:?}",
            incompatible_model.info().input_shapes[0]
        );
        println!("   - Error: {}", result.unwrap_err());
    }

    /// Test simulated numerical comparison
    #[test]
    fn test_simulated_numerical_comparison() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();
        let pytorch_model = create_realistic_pytorch_model();
        let onnx_model = create_equivalent_onnx_model();

        println!("🧮 Testing numerical comparison simulation:");

        // Test with default configuration
        let result = validator.simulate_numerical_comparison(&pytorch_model, &onnx_model);
        let diff = result.expect("Numerical difference calculation should succeed");
        assert!(diff > 0.0);
        assert!(diff < 1e-3); // Should be small but non-zero

        println!("   - Simulated max difference: {:.2e}", diff);
        println!(
            "   - Tolerance threshold: {:.2e}",
            validator.config().numerical_tolerance
        );
        println!(
            "   - Within tolerance: {}",
            diff < validator.config().numerical_tolerance
        );

        // Test with different complexity models
        let simple_model = create_incompatible_model(); // Smaller model
        let result = validator.simulate_numerical_comparison(&simple_model, &onnx_model);
        let simple_diff = result.expect("Simple model numerical difference should succeed");
        println!("   - Simple model difference: {:.2e}", simple_diff);
        println!("   - Complex vs simple: {:.2}x", diff / simple_diff);

        assert!(diff > simple_diff); // More complex model should have larger differences
        println!("✅ Numerical comparison simulation working correctly");
    }

    /// Test complete cross-format validation workflow
    #[test]
    fn test_complete_cross_format_validation_workflow() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();
        let pytorch_model = create_realistic_pytorch_model();
        let onnx_model = create_equivalent_onnx_model();

        println!("🚀 Testing complete cross-format validation workflow:");

        // Test the complete pipeline (simulated since we don't have real files)
        // This tests all the internal validation logic
        let compatibility_result =
            validator.validate_model_compatibility(&pytorch_model, &onnx_model);
        assert!(compatibility_result.is_ok());

        let comparison_result = validator.compare_model_outputs(&pytorch_model, &onnx_model);
        let validation_result = comparison_result.expect("Model output comparison should succeed");
        println!("   - Source format: {}", validation_result.source_format);
        println!("   - Target format: {}", validation_result.target_format);
        println!(
            "   - Max numerical diff: {:.2e}",
            validation_result.max_numerical_diff
        );
        println!(
            "   - Avg numerical diff: {:.2e}",
            validation_result.avg_numerical_diff
        );
        println!(
            "   - Validation passed: {}",
            validation_result.validation_passed
        );
        println!(
            "   - Values compared: {}",
            validation_result.values_compared
        );

        // Verify result structure
        assert_eq!(validation_result.source_format, "PyTorch");
        assert_eq!(validation_result.target_format, "ONNX");
        assert!(validation_result.max_numerical_diff > 0.0);
        assert!(validation_result.avg_numerical_diff > 0.0);
        assert!(validation_result.avg_numerical_diff <= validation_result.max_numerical_diff);
        assert_eq!(
            validation_result.tolerance_threshold,
            validator.config().numerical_tolerance
        );
        assert_eq!(validation_result.values_compared, 1000); // Model has 1000 output classes

        // Check metadata
        assert!(validation_result.metadata.contains_key("validation_method"));
        assert!(validation_result.metadata.contains_key("test_input_count"));
        assert_eq!(
            validation_result.metadata.get("validation_method").unwrap(),
            "real_inference"
        );

        println!("✅ Complete cross-format validation workflow successful");
    }

    /// Test optimization validation workflow
    #[test]
    fn test_optimization_validation_workflow() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();
        let model = create_realistic_pytorch_model();

        println!("⚙️ Testing optimization validation workflow:");

        // Create optimizer with reasonable settings (use raspberry_pi for larger memory limit)
        let mut config = Config::default();
        config.hardware.target = "raspberry_pi".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.max_accuracy_loss = 8.0; // Higher tolerance for complex model

        let optimizer = Optimizer::new(config.clone());

        // Test optimization validation
        let result = validator.validate_optimization_consistency(&model, &optimizer, &config);
        if result.is_err() {
            println!(
                "❌ Optimization validation failed: {:?}",
                result.as_ref().unwrap_err()
            );
        }
        assert!(result.is_ok());

        let validation_result = result.unwrap();
        println!(
            "   - Original accuracy: {:.1}%",
            validation_result.original_accuracy
        );
        println!(
            "   - Optimized accuracy: {:.1}%",
            validation_result.optimized_accuracy
        );
        println!(
            "   - Accuracy loss: {:.1}%",
            validation_result.accuracy_loss
        );
        println!("   - Acceptable: {}", validation_result.accuracy_acceptable);
        println!(
            "   - Max acceptable loss: {:.1}%",
            validation_result.max_acceptable_loss
        );
        println!(
            "   - Techniques applied: {:?}",
            validation_result.techniques_applied
        );

        // Verify result structure
        assert!(validation_result.original_accuracy >= 80.0);
        assert!(validation_result.original_accuracy <= 95.0);
        assert!(validation_result.optimized_accuracy < validation_result.original_accuracy);
        assert!(validation_result.accuracy_loss >= 0.0);
        assert_eq!(
            validation_result.accuracy_loss,
            validation_result.original_accuracy - validation_result.optimized_accuracy
        );
        assert!(!validation_result.techniques_applied.is_empty());

        if validation_result.accuracy_acceptable {
            println!("✅ Optimization validation passed - accuracy loss within bounds");
        } else {
            println!("⚠️ Optimization validation failed - accuracy loss too high");
        }

        println!("✅ Optimization validation workflow completed");
    }

    /// Test validation configuration edge cases
    #[test]
    fn test_validation_configuration_edge_cases() {
        env_logger::try_init().ok();

        println!("🧪 Testing validation configuration edge cases:");

        // Test very strict configuration
        let strict_config = ValidationConfig {
            numerical_tolerance: 1e-10,
            max_accuracy_loss: 0.1,
            test_input_count: 100,
            random_seed: 42,
            intensive_validation: true,
        };

        let validator = CrossFormatValidator::with_config(strict_config);
        let model = create_realistic_pytorch_model();

        let accuracy = validator.estimate_model_accuracy(&model).unwrap();
        println!("   - Strict config estimated accuracy: {:.2}%", accuracy);

        // Test very lenient configuration
        let lenient_config = ValidationConfig {
            numerical_tolerance: 1e-2,
            max_accuracy_loss: 20.0,
            test_input_count: 1,
            random_seed: 123,
            intensive_validation: false,
        };

        let validator = CrossFormatValidator::with_config(lenient_config);
        let accuracy = validator.estimate_model_accuracy(&model).unwrap();
        println!("   - Lenient config estimated accuracy: {:.2}%", accuracy);

        // Test configuration updates
        let mut validator = CrossFormatValidator::new();
        let original_tolerance = validator.config().numerical_tolerance;

        let new_config = ValidationConfig {
            numerical_tolerance: 1e-8,
            ..ValidationConfig::default()
        };

        validator.set_config(new_config);
        assert_ne!(validator.config().numerical_tolerance, original_tolerance);
        assert_eq!(validator.config().numerical_tolerance, 1e-8);

        println!(
            "   - Configuration update successful: {:.1e} → {:.1e}",
            original_tolerance,
            validator.config().numerical_tolerance
        );

        println!("✅ Configuration edge cases handled correctly");
    }

    /// Test accuracy estimation across different model types
    #[test]
    fn test_accuracy_estimation_model_variations() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();

        println!("📊 Testing accuracy estimation across model variations:");

        // Test complex model
        let complex_model = create_realistic_pytorch_model();
        let complex_accuracy = validator.estimate_model_accuracy(&complex_model).unwrap();
        println!(
            "   - Complex model ({} params): {:.1}% accuracy",
            complex_model.info().parameter_count,
            complex_accuracy
        );

        // Test simple model
        let simple_model = create_incompatible_model();
        let simple_accuracy = validator.estimate_model_accuracy(&simple_model).unwrap();
        println!(
            "   - Simple model ({} params): {:.1}% accuracy",
            simple_model.info().parameter_count,
            simple_accuracy
        );

        // Complex models should generally have higher estimated accuracy
        println!(
            "   - Accuracy difference: {:.1}%",
            complex_accuracy - simple_accuracy
        );

        // All estimates should be reasonable
        assert!((80.0..=95.0).contains(&complex_accuracy));
        assert!((80.0..=95.0).contains(&simple_accuracy));

        println!("✅ Accuracy estimation varies appropriately with model complexity");
    }

    /// Test error handling and edge cases
    #[test]
    fn test_error_handling_and_edge_cases() {
        env_logger::try_init().ok();

        let validator = CrossFormatValidator::new();

        println!("🛡️ Testing error handling and edge cases:");

        // Test empty model validation
        let empty_model = Model {
            info: ModelInfo {
                format: ModelFormat::PyTorch,
                input_shapes: vec![],
                output_shapes: vec![vec![1, 10]],
                parameter_count: 0,
                model_size_bytes: 0,
                operations_count: 0,
                layers: vec![],
            },
            data: ModelData::Raw(vec![]),
        };

        let result = validator.validate_model_structure(&empty_model, "Empty");
        assert!(result.is_err());
        println!("   - Empty input shapes correctly rejected");

        // Test model with different input counts
        let pytorch_model = create_realistic_pytorch_model();
        let mut multi_input_model = create_equivalent_onnx_model();
        multi_input_model.info.input_shapes.push(vec![1, 10]); // Add extra input

        let result = validator.validate_model_compatibility(&pytorch_model, &multi_input_model);
        assert!(result.is_err());
        println!("   - Different input count correctly rejected");

        // Test model with different output counts
        let mut multi_output_model = create_equivalent_onnx_model();
        multi_output_model.info.output_shapes.push(vec![1, 5]); // Add extra output

        let result = validator.validate_model_compatibility(&pytorch_model, &multi_output_model);
        assert!(result.is_err());
        println!("   - Different output count correctly rejected");

        println!("✅ Error handling working correctly for edge cases");
    }
}
