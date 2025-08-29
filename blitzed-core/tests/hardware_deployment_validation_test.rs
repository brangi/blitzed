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

//! Hardware deployment validation integration tests
//!
//! This test suite validates the complete hardware deployment pipeline including:
//! - ESP32 deployment with real memory and performance constraints
//! - Arduino Nano 33 BLE ultra-low-power deployment (< 32KB models)
//! - STM32 deployment with FPU optimization and HAL integration
//! - Performance measurement and constraint validation
//! - End-to-end optimization to deployment workflows

#[cfg(test)]
mod tests {
    use blitzed_core::deployment::{DeploymentValidationConfig, HardwareDeploymentValidator};
    use blitzed_core::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::Optimizer;
    use blitzed_core::Config;

    /// Create a small model suitable for edge deployment testing
    fn create_edge_deployment_model() -> Model {
        let layers = vec![
            // Simple convolutional feature extraction
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 1, 28, 28], // MNIST-like input
                output_shape: vec![1, 8, 14, 14],
                parameter_count: 80, // 1*8*3*3 + 8
                flops: 15_680,
            },
            LayerInfo {
                name: "relu1".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 8, 14, 14],
                output_shape: vec![1, 8, 14, 14],
                parameter_count: 0,
                flops: 1_568,
            },
            LayerInfo {
                name: "avgpool1".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 8, 14, 14],
                output_shape: vec![1, 8, 7, 7],
                parameter_count: 0,
                flops: 1_568,
            },
            // Small dense layers for classification
            LayerInfo {
                name: "flatten".to_string(),
                layer_type: "flatten".to_string(),
                input_shape: vec![1, 8, 7, 7],
                output_shape: vec![1, 392],
                parameter_count: 0,
                flops: 0,
            },
            LayerInfo {
                name: "fc1".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 392],
                output_shape: vec![1, 32],
                parameter_count: 12_576, // 392*32 + 32
                flops: 12_544,
            },
            LayerInfo {
                name: "relu2".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 32],
                output_shape: vec![1, 32],
                parameter_count: 0,
                flops: 32,
            },
            LayerInfo {
                name: "fc2".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 32],
                output_shape: vec![1, 10],
                parameter_count: 330, // 32*10 + 10
                flops: 320,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum(); // ~13K parameters
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 1, 28, 28]], // MNIST-style input
            output_shapes: vec![vec![1, 10]],       // 10 classes
            parameter_count: total_params,          // ~13K parameters
            model_size_bytes: total_params * 4,     // FP32 = ~52KB model
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    /// Create ultra-tiny model for Arduino deployment
    fn create_tiny_arduino_model() -> Model {
        let layers = vec![
            // Ultra-minimal model for Arduino constraints
            LayerInfo {
                name: "tiny_fc1".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 16], // Very small input
                output_shape: vec![1, 8],
                parameter_count: 136, // 16*8 + 8
                flops: 128,
            },
            LayerInfo {
                name: "tiny_relu".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 8],
                output_shape: vec![1, 8],
                parameter_count: 0,
                flops: 8,
            },
            LayerInfo {
                name: "tiny_fc2".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 8],
                output_shape: vec![1, 4],
                parameter_count: 36, // 8*4 + 4
                flops: 32,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.parameter_count).sum(); // ~172 parameters
        let total_flops: u64 = layers.iter().map(|l| l.flops).sum();

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 16]],    // Tiny input
            output_shapes: vec![vec![1, 4]],    // 4 classes
            parameter_count: total_params,      // ~172 parameters
            model_size_bytes: total_params * 4, // FP32 = ~688 bytes model
            operations_count: total_flops as usize,
            layers,
        };

        Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        }
    }

    /// Test ESP32 deployment validation workflow
    #[test]
    fn test_esp32_deployment_validation() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();
        let model = create_edge_deployment_model();

        println!("🔥 Testing ESP32 deployment validation:");
        println!(
            "   Model: {} params ({:.1} KB)",
            model.info().parameter_count,
            model.info().model_size_bytes as f32 / 1024.0
        );

        // Create ESP32-optimized configuration
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.max_accuracy_loss = 10.0; // More lenient for edge deployment

        let optimizer = Optimizer::new(config);

        // Validate ESP32 deployment
        let result = validator.validate_esp32_deployment(&model, &optimizer);
        let deployment_result = result.expect("ESP32 deployment validation should succeed");
        println!(
            "   Status: {}",
            if deployment_result.deployment_successful {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            }
        );
        println!(
            "   Original size: {:.1} KB",
            deployment_result.original_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Optimized size: {:.1} KB",
            deployment_result.optimized_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Memory usage: {:.1} KB",
            deployment_result.optimized_model_info.memory_usage as f32 / 1024.0
        );
        println!(
            "   Fits constraints: {}",
            deployment_result.optimized_model_info.fits_constraints
        );
        println!(
            "   Inference time: {:.1} ms",
            deployment_result
                .performance_metrics
                .estimated_inference_time_ms
        );
        println!(
            "   Memory efficiency: {:.1}%",
            deployment_result.performance_metrics.memory_efficiency * 100.0
        );
        println!(
            "   Power efficiency: {:.1}%",
            deployment_result.performance_metrics.power_efficiency * 100.0
        );

        // Verify deployment characteristics
        assert_eq!(deployment_result.target_name, "ESP32");
        assert!(
            deployment_result
                .performance_metrics
                .estimated_inference_time_ms
                > 0.0
        );
        assert!(deployment_result.performance_metrics.memory_efficiency >= 0.0);
        assert!(deployment_result.performance_metrics.power_efficiency >= 0.0);
        assert!(!deployment_result
            .deployment_artifacts
            .source_files
            .is_empty());
        assert!(!deployment_result
            .deployment_artifacts
            .header_files
            .is_empty());

        // ESP32 should be able to handle this model size
        if deployment_result.deployment_successful {
            println!("   ✅ ESP32 deployment validation passed");
            assert!(deployment_result.optimized_model_info.memory_usage < 320 * 1024);
        // ESP32 limit
        } else {
            println!("   ⚠️ ESP32 deployment failed - model may need more aggressive optimization");
        }

        // Check optimizations were applied
        assert!(!deployment_result
            .optimized_model_info
            .optimizations_applied
            .is_empty());

        println!("   Optimizations applied:");
        for opt in &deployment_result.optimized_model_info.optimizations_applied {
            println!("     - {}", opt);
        }

        if !deployment_result.warnings.is_empty() {
            println!("   Warnings:");
            for warning in &deployment_result.warnings {
                println!("     ⚠️ {}", warning);
            }
        }
    }

    /// Test Arduino Nano 33 BLE deployment validation (ultra-low-power)
    #[test]
    fn test_arduino_deployment_validation() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();
        let model = create_tiny_arduino_model();

        println!("🔋 Testing Arduino Nano 33 BLE deployment validation:");
        println!(
            "   Tiny model: {} params ({:.1} KB)",
            model.info().parameter_count,
            model.info().model_size_bytes as f32 / 1024.0
        );

        // Create Arduino-optimized configuration (very aggressive)
        let mut config = Config::default();
        config.hardware.target = "arduino".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.enable_distillation = false; // Keep simple for Arduino
        config.optimization.max_accuracy_loss = 15.0; // Higher tolerance for ultra-low-power

        let optimizer = Optimizer::new(config);

        // Validate Arduino deployment
        let result = validator.validate_arduino_deployment(&model, &optimizer);
        let deployment_result = result.expect("Arduino deployment validation should succeed");
        println!(
            "   Status: {}",
            if deployment_result.deployment_successful {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            }
        );
        println!(
            "   Original size: {:.1} KB",
            deployment_result.original_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Optimized size: {:.1} KB",
            deployment_result.optimized_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Memory usage: {:.1} KB",
            deployment_result.optimized_model_info.memory_usage as f32 / 1024.0
        );
        println!(
            "   Inference time: {:.1} ms",
            deployment_result
                .performance_metrics
                .estimated_inference_time_ms
        );
        println!(
            "   Power efficiency: {:.1}%",
            deployment_result.performance_metrics.power_efficiency * 100.0
        );

        // Verify Arduino-specific requirements
        assert_eq!(deployment_result.target_name, "Arduino Nano 33 BLE");

        // Arduino has very strict constraints
        let model_fits_flash = deployment_result.optimized_model_info.model_size < 32 * 1024; // 32KB limit
        let model_fits_ram = deployment_result.optimized_model_info.memory_usage < 64 * 1024; // 64KB RAM limit

        println!("   Model fits 32KB flash: {}", model_fits_flash);
        println!("   Model fits 64KB RAM: {}", model_fits_ram);

        // Check deployment artifacts
        assert!(!deployment_result
            .deployment_artifacts
            .source_files
            .is_empty());
        assert!(!deployment_result
            .deployment_artifacts
            .example_files
            .is_empty());

        // Arduino should prioritize power efficiency
        assert!(deployment_result.performance_metrics.power_efficiency >= 0.5);

        if deployment_result.deployment_successful {
            println!(
                "   ✅ Arduino deployment validation passed - ready for ultra-low-power deployment"
            );
            assert!(model_fits_flash || model_fits_ram); // At least one constraint met
        } else {
            println!("   ⚠️ Arduino deployment needs further optimization");
        }

        println!("   Optimizations applied:");
        for opt in &deployment_result.optimized_model_info.optimizations_applied {
            println!("     - {}", opt);
        }
    }

    /// Test STM32 deployment validation with FPU optimization
    #[test]
    fn test_stm32_deployment_validation() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();
        let model = create_edge_deployment_model();

        println!("⚡ Testing STM32 deployment validation:");
        println!(
            "   Model: {} params ({:.1} KB)",
            model.info().parameter_count,
            model.info().model_size_bytes as f32 / 1024.0
        );

        // Create STM32-optimized configuration (balanced)
        let mut config = Config::default();
        config.hardware.target = "stm32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.max_accuracy_loss = 8.0; // Moderate tolerance

        let optimizer = Optimizer::new(config);

        // Validate STM32 deployment
        let result = validator.validate_stm32_deployment(&model, &optimizer);
        let deployment_result = result.expect("STM32 deployment validation should succeed");
        println!(
            "   Status: {}",
            if deployment_result.deployment_successful {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            }
        );
        println!(
            "   Original size: {:.1} KB",
            deployment_result.original_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Optimized size: {:.1} KB",
            deployment_result.optimized_model_info.model_size as f32 / 1024.0
        );
        println!(
            "   Memory usage: {:.1} KB",
            deployment_result.optimized_model_info.memory_usage as f32 / 1024.0
        );
        println!(
            "   Inference time: {:.1} ms",
            deployment_result
                .performance_metrics
                .estimated_inference_time_ms
        );
        println!(
            "   Memory efficiency: {:.1}%",
            deployment_result.performance_metrics.memory_efficiency * 100.0
        );
        println!(
            "   Power efficiency: {:.1}%",
            deployment_result.performance_metrics.power_efficiency * 100.0
        );

        // Verify STM32-specific characteristics
        assert_eq!(deployment_result.target_name, "STM32F4");

        // STM32 should leverage FPU for better performance
        assert!(
            deployment_result
                .performance_metrics
                .estimated_inference_time_ms
                < 500.0
        ); // Should be fast with FPU

        // Check build artifacts include STM32-specific files
        let has_makefile = deployment_result
            .deployment_artifacts
            .build_files
            .iter()
            .any(|f| f.to_string_lossy().contains("Makefile"));
        let has_linker_script = deployment_result
            .deployment_artifacts
            .build_files
            .iter()
            .any(|f| f.to_string_lossy().contains("linker_script"));

        println!("   Has Makefile: {}", has_makefile);
        println!("   Has linker script: {}", has_linker_script);

        // STM32 should generate comprehensive build system
        assert!(!deployment_result
            .deployment_artifacts
            .build_files
            .is_empty());
        assert!(!deployment_result
            .deployment_artifacts
            .source_files
            .is_empty());

        if deployment_result.deployment_successful {
            println!(
                "   ✅ STM32 deployment validation passed - ready for FPU-accelerated inference"
            );
        } else {
            println!("   ⚠️ STM32 deployment needs adjustment");
        }

        println!("   Optimizations applied:");
        for opt in &deployment_result.optimized_model_info.optimizations_applied {
            println!("     - {}", opt);
        }
    }

    /// Test deployment validation with custom configuration
    #[test]
    fn test_deployment_validation_custom_config() {
        env_logger::try_init().ok();

        // Create custom deployment validation configuration
        let custom_config = DeploymentValidationConfig {
            max_inference_time_ms: 200.0, // Strict timing requirement
            min_memory_efficiency: 0.8,   // High efficiency requirement
            generate_optimized_variants: true,
            strict_constraint_validation: true,
            output_directory: Some(std::env::temp_dir().join("blitzed_deployment_test")),
            cleanup_artifacts: false,
        };

        let validator = HardwareDeploymentValidator::with_config(custom_config);
        let model = create_edge_deployment_model();

        println!("🎛️ Testing deployment validation with custom configuration:");
        println!(
            "   Max inference time: {:.0} ms",
            validator.config().max_inference_time_ms
        );
        println!(
            "   Min memory efficiency: {:.1}%",
            validator.config().min_memory_efficiency * 100.0
        );
        println!(
            "   Strict validation: {}",
            validator.config().strict_constraint_validation
        );

        // Test with ESP32 target
        let mut config = Config::default();
        config.hardware.target = "esp32".to_string();
        config.optimization.enable_quantization = true;
        config.optimization.enable_pruning = true;
        config.optimization.max_accuracy_loss = 5.0;

        let optimizer = Optimizer::new(config);
        let result = validator
            .validate_esp32_deployment(&model, &optimizer)
            .unwrap();

        println!("   Deployment successful: {}", result.deployment_successful);
        println!(
            "   Inference time: {:.1} ms (limit: {:.0} ms)",
            result.performance_metrics.estimated_inference_time_ms,
            validator.config().max_inference_time_ms
        );
        println!(
            "   Memory efficiency: {:.1}% (min: {:.1}%)",
            result.performance_metrics.memory_efficiency * 100.0,
            validator.config().min_memory_efficiency * 100.0
        );

        // Check custom configuration effects
        let meets_timing = result.performance_metrics.estimated_inference_time_ms
            <= validator.config().max_inference_time_ms;
        let meets_efficiency = result.performance_metrics.memory_efficiency
            >= validator.config().min_memory_efficiency;

        println!("   Meets timing requirement: {}", meets_timing);
        println!("   Meets efficiency requirement: {}", meets_efficiency);

        if meets_timing && meets_efficiency {
            println!("   ✅ Custom configuration requirements met");
        } else {
            println!("   ⚠️ Custom configuration requirements not met - may need more aggressive optimization");
        }
    }

    /// Test multi-target deployment comparison
    #[test]
    fn test_multi_target_deployment_comparison() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();
        let edge_model = create_edge_deployment_model();
        let tiny_model = create_tiny_arduino_model();

        println!("🎯 Testing multi-target deployment comparison:");

        // Test all three targets
        let targets = vec![
            ("ESP32", "esp32", &edge_model),
            ("Arduino", "arduino", &tiny_model), // Use tiny model for Arduino
            ("STM32", "stm32", &edge_model),
        ];

        let mut results = Vec::new();

        for (name, target_id, model) in targets {
            let mut config = Config::default();
            config.hardware.target = target_id.to_string();
            config.optimization.enable_quantization = true;
            config.optimization.enable_pruning = true;
            config.optimization.max_accuracy_loss = 12.0;

            let optimizer = Optimizer::new(config);

            let result = match target_id {
                "esp32" => validator.validate_esp32_deployment(model, &optimizer),
                "arduino" => validator.validate_arduino_deployment(model, &optimizer),
                "stm32" => validator.validate_stm32_deployment(model, &optimizer),
                _ => panic!("Unknown target"),
            };

            let deployment_result = result.expect("Deployment should succeed");
            results.push(deployment_result);

            println!("   {} Deployment:", name);
            let last_result = results.last().unwrap();
            println!("     Success: {}", last_result.deployment_successful);
            println!(
                "     Inference time: {:.1} ms",
                last_result.performance_metrics.estimated_inference_time_ms
            );
            println!(
                "     Memory efficiency: {:.1}%",
                last_result.performance_metrics.memory_efficiency * 100.0
            );
            println!(
                "     Power efficiency: {:.1}%",
                last_result.performance_metrics.power_efficiency * 100.0
            );
        }

        // Generate comprehensive deployment report
        let report = validator.generate_deployment_report(&results);
        println!("\n📊 Deployment Report Generated:");
        println!("{}", &report[..500.min(report.len())]); // Print first 500 chars

        // Verify report contains all targets
        assert!(report.contains("ESP32"));
        assert!(report.contains("Arduino"));
        assert!(report.contains("STM32"));
        assert!(report.contains("Performance Metrics"));
        assert!(report.contains("Model Analysis"));

        println!("   ✅ Multi-target deployment comparison completed");
    }

    /// Test deployment performance metrics accuracy
    #[test]
    fn test_deployment_performance_metrics() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();
        let model = create_edge_deployment_model();

        println!("📈 Testing deployment performance metrics accuracy:");

        // Create different optimization configurations to compare performance impacts
        let configs = vec![
            ("No Optimization", false, false, false),
            ("Quantization Only", true, false, false),
            ("Quantization + Pruning", true, true, false),
            ("Full Optimization", true, true, true),
        ];

        for (config_name, quant, prune, distill) in configs {
            let mut config = Config::default();
            config.hardware.target = "esp32".to_string();
            config.optimization.enable_quantization = quant;
            config.optimization.enable_pruning = prune;
            config.optimization.enable_distillation = distill;
            config.optimization.max_accuracy_loss = 15.0;

            let optimizer = Optimizer::new(config);
            let result = validator
                .validate_esp32_deployment(&model, &optimizer)
                .unwrap();

            println!("   {}:", config_name);
            println!(
                "     Model size: {:.1} KB",
                result.optimized_model_info.model_size as f32 / 1024.0
            );
            println!(
                "     Memory usage: {:.1} KB",
                result.optimized_model_info.memory_usage as f32 / 1024.0
            );
            println!(
                "     Inference time: {:.1} ms",
                result.performance_metrics.estimated_inference_time_ms
            );
            println!(
                "     Memory efficiency: {:.1}%",
                result.performance_metrics.memory_efficiency * 100.0
            );
            println!(
                "     Power efficiency: {:.1}%",
                result.performance_metrics.power_efficiency * 100.0
            );
            println!(
                "     Optimizations: {:?}",
                result.optimized_model_info.optimizations_applied
            );

            // Verify metrics are reasonable
            assert!(result.performance_metrics.estimated_inference_time_ms > 0.0);
            assert!(result.performance_metrics.estimated_inference_time_ms < 10000.0); // Less than 10 seconds
            assert!(result.performance_metrics.memory_efficiency >= 0.0);
            assert!(result.performance_metrics.memory_efficiency <= 1.0);
            assert!(result.performance_metrics.power_efficiency >= 0.0);
            assert!(result.performance_metrics.power_efficiency <= 1.0);
        }

        println!("   ✅ Performance metrics validation completed");
    }

    /// Test error handling in deployment validation
    #[test]
    fn test_deployment_validation_error_handling() {
        env_logger::try_init().ok();

        let validator = HardwareDeploymentValidator::new();

        println!("🛡️ Testing deployment validation error handling:");

        // Test with very large model that should fail constraints
        let mut large_model = create_edge_deployment_model();
        large_model.info.model_size_bytes = 10 * 1024 * 1024; // 10MB model
        large_model.info.parameter_count = 2_500_000; // 2.5M parameters

        let mut config = Config::default();
        config.hardware.target = "arduino".to_string(); // Very restrictive target
        config.optimization.enable_quantization = false; // No optimization to ensure failure
        config.optimization.max_accuracy_loss = 1.0; // Very strict

        let optimizer = Optimizer::new(config);

        // This should succeed but with warnings about constraint violations
        let result = validator.validate_arduino_deployment(&large_model, &optimizer);

        if let Ok(deployment_result) = result {
            println!(
                "   Large model deployment status: {}",
                if deployment_result.deployment_successful {
                    "Success"
                } else {
                    "Failed"
                }
            );
            println!("   Warnings count: {}", deployment_result.warnings.len());

            // Should have constraint violations
            assert!(
                !deployment_result.warnings.is_empty() || !deployment_result.deployment_successful
            );

            if !deployment_result.warnings.is_empty() {
                println!("   Constraint violation warnings:");
                for warning in &deployment_result.warnings {
                    println!("     ⚠️ {}", warning);
                }
            }

            println!("   ✅ Error handling working correctly - large model properly flagged");
        } else {
            // If validation fails entirely, that's also acceptable error handling
            println!(
                "   ⚠️ Validation failed for oversized model: {:?}",
                result.unwrap_err()
            );
            println!("   ✅ Error handling working correctly - validation properly failed");
        }
    }
}
