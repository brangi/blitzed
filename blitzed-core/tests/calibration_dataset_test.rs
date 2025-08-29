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

//! Comprehensive tests for calibration dataset functionality
//!
//! This test suite validates:
//! - Calibration dataset creation and validation
//! - Statistical analysis of calibration data
//! - Different calibration methods (MinMax, Percentile, Entropy, MSE)
//! - Quantization with calibration data vs. without
//! - Quality scoring and warning systems
//! - Integration with existing quantization pipeline

#[cfg(test)]
mod tests {
    use blitzed_core::model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::quantization::{
        CalibrationMethod, QuantizationConfig, QuantizationType, Quantizer,
    };

    fn create_test_model() -> Model {
        let layers = vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 9408,
                flops: 118_013_952,
            },
            LayerInfo {
                name: "fc".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 2048],
                output_shape: vec![1, 1000],
                parameter_count: 2_048_000,
                flops: 2_048_000,
            },
        ];

        Model {
            info: ModelInfo {
                format: ModelFormat::PyTorch,
                input_shapes: vec![vec![1, 3, 224, 224]],
                output_shapes: vec![vec![1, 1000]],
                parameter_count: 2_057_408,
                model_size_bytes: 8_229_632, // ~8MB
                operations_count: 120_061_952,
                layers,
            },
            data: ModelData::Raw(vec![0u8; 100]),
        }
    }

    /// Create realistic calibration data with different distributions
    fn create_calibration_data_realistic() -> (Vec<Vec<Vec<f32>>>, Vec<Vec<usize>>) {
        let sample_count = 200;
        let input_shapes = vec![vec![1, 3, 224, 224]];

        // Create calibration data that simulates ImageNet-like distribution
        let mut inputs = vec![Vec::new()];

        for sample_idx in 0..sample_count {
            let mut sample_data = Vec::new();

            // Generate image-like data with realistic statistics
            for pixel_idx in 0..(3 * 224 * 224) {
                // Simulate normalized ImageNet values (mean ~0, std ~1)
                let base_val = (sample_idx as f32 * 0.01 + pixel_idx as f32 * 0.0001) % 6.28; // 0 to 2π
                let normalized_val = (base_val.sin() * 0.5 + base_val.cos() * 0.3).tanh(); // -1 to 1 range
                sample_data.push(normalized_val);
            }

            inputs[0].push(sample_data);
        }

        (inputs, input_shapes)
    }

    /// Create calibration data with outliers to test robustness
    fn create_calibration_data_with_outliers() -> (Vec<Vec<Vec<f32>>>, Vec<Vec<usize>>) {
        let sample_count = 100;
        let input_shapes = vec![vec![1, 512]]; // Smaller for testing

        let mut inputs = vec![Vec::new()];

        for sample_idx in 0..sample_count {
            let mut sample_data = Vec::new();

            for feature_idx in 0..512 {
                let base_val = (sample_idx as f32 * 0.1 + feature_idx as f32 * 0.01) % 3.14;

                // Add outliers to 5% of the data
                let val = if (sample_idx + feature_idx) % 20 == 0 {
                    // Outlier values
                    base_val * 10.0
                } else {
                    // Normal values
                    base_val
                };

                sample_data.push(val);
            }

            inputs[0].push(sample_data);
        }

        (inputs, input_shapes)
    }

    #[test]
    fn test_calibration_dataset_creation() {
        let (inputs, input_shapes) = create_calibration_data_realistic();

        let dataset = Quantizer::create_calibration_dataset(
            inputs,
            input_shapes.clone(),
            "test_dataset".to_string(),
        )
        .expect("Failed to create calibration dataset");

        assert_eq!(dataset.metadata.sample_count, 200);
        assert_eq!(dataset.metadata.input_shapes, input_shapes);
        assert_eq!(dataset.metadata.data_source, "test_dataset");
        assert_eq!(dataset.inputs.len(), 1);
        assert_eq!(dataset.inputs[0].len(), 200);
        assert_eq!(dataset.inputs[0][0].len(), 3 * 224 * 224);
    }

    #[test]
    fn test_calibration_dataset_validation() {
        // Test mismatched sample counts
        let inputs = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]], // 2 samples
            vec![vec![5.0, 6.0]],                 // 1 sample - mismatch!
        ];
        let input_shapes = vec![vec![2], vec![2]];

        let result =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string());

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("has 1 samples, expected 2"));
    }

    #[test]
    fn test_calibration_analysis_minmax() {
        let config = QuantizationConfig {
            calibration_method: CalibrationMethod::MinMax,
            ..Default::default()
        };

        let (inputs, input_shapes) = create_calibration_data_realistic();
        let dataset =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string())
                .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Verify statistics were computed
        assert_eq!(result.stats.sample_count, 200);
        assert_eq!(result.stats.min_values.len(), 1);
        assert_eq!(result.stats.max_values.len(), 1);
        assert_eq!(result.stats.mean_values.len(), 1);
        assert_eq!(result.stats.std_values.len(), 1);

        // Verify quantization parameters were generated
        assert_eq!(result.recommended_params.len(), 1);
        assert!(result.recommended_params[0].scale > 0.0);

        // Verify quality score is reasonable
        assert!(result.quality_score > 0.0);
        assert!(result.quality_score <= 1.0);

        println!("MinMax calibration quality: {:.3}", result.quality_score);
        println!("MinMax warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_calibration_analysis_percentile() {
        let config = QuantizationConfig {
            calibration_method: CalibrationMethod::Percentile,
            percentile_threshold: 0.02, // 2% and 98% percentiles
            ..Default::default()
        };

        let (inputs, input_shapes) = create_calibration_data_with_outliers();
        let dataset = Quantizer::create_calibration_dataset(
            inputs,
            input_shapes,
            "test_with_outliers".to_string(),
        )
        .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Percentile method should handle outliers better
        assert!(result.quality_score > 0.0);

        // Check percentile values
        assert_eq!(result.stats.percentile_values.len(), 1);
        let (low_percentile, high_percentile) = result.stats.percentile_values[0];
        assert!(low_percentile < high_percentile);

        println!(
            "Percentile calibration quality: {:.3}",
            result.quality_score
        );
        println!(
            "Percentiles: {:.3} to {:.3}",
            low_percentile, high_percentile
        );
        println!("Percentile warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_calibration_analysis_entropy() {
        let config = QuantizationConfig {
            calibration_method: CalibrationMethod::Entropy,
            ..Default::default()
        };

        let (inputs, input_shapes) = create_calibration_data_realistic();
        let dataset =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string())
                .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Entropy method should produce reasonable results
        assert!(result.quality_score > 0.0);
        assert_eq!(result.recommended_params.len(), 1);

        println!("Entropy calibration quality: {:.3}", result.quality_score);
        println!("Entropy warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_calibration_analysis_mse() {
        let config = QuantizationConfig {
            calibration_method: CalibrationMethod::MSE,
            ..Default::default()
        };

        let (inputs, input_shapes) = create_calibration_data_realistic();
        let dataset =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string())
                .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // MSE method should optimize for reconstruction error
        assert!(result.quality_score > 0.0);
        assert_eq!(result.recommended_params.len(), 1);

        println!("MSE calibration quality: {:.3}", result.quality_score);
        println!("MSE warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_quantization_with_calibration_vs_without() {
        env_logger::try_init().ok();

        let model = create_test_model();

        // Test without calibration
        let config_without = QuantizationConfig::default();
        let quantizer_without = Quantizer::new(config_without);
        let result_without = quantizer_without
            .quantize_post_training(&model)
            .expect("Quantization without calibration failed");

        // Test with calibration
        let config_with = QuantizationConfig::default();
        let (inputs, input_shapes) = create_calibration_data_realistic();
        let dataset =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string())
                .expect("Failed to create calibration dataset");

        let quantizer_with = Quantizer::with_calibration(config_with, dataset);
        let result_with = quantizer_with
            .quantize_post_training(&model)
            .expect("Quantization with calibration failed");

        // Both should succeed
        assert!(result_without.quantized_size > 0);
        assert!(result_with.quantized_size > 0);

        // Results might be different due to calibration
        println!(
            "Without calibration: {:.3}% accuracy loss",
            result_without.accuracy_loss
        );
        println!(
            "With calibration: {:.3}% accuracy loss",
            result_with.accuracy_loss
        );
        println!(
            "Compression ratio without: {:.2}x",
            result_without.compression_ratio()
        );
        println!(
            "Compression ratio with: {:.2}x",
            result_with.compression_ratio()
        );
    }

    #[test]
    fn test_calibration_quality_warnings() {
        let config = QuantizationConfig::default();

        // Create calibration data with problematic characteristics
        let problematic_inputs = vec![vec![
            vec![1.0; 1000],    // Constant values - should trigger warning
            vec![1.0001; 1000], // Nearly constant - should trigger warning
        ]];
        let input_shapes = vec![vec![1000]];

        let dataset = Quantizer::create_calibration_dataset(
            problematic_inputs,
            input_shapes,
            "problematic_dataset".to_string(),
        )
        .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Should have low quality score and warnings
        assert!(result.quality_score < 0.5);
        assert!(!result.warnings.is_empty());

        println!("Problematic data quality: {:.3}", result.quality_score);
        println!("Warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_histogram_generation() {
        let config = QuantizationConfig::default();
        let (inputs, input_shapes) = create_calibration_data_realistic();
        let dataset =
            Quantizer::create_calibration_dataset(inputs, input_shapes, "test_dataset".to_string())
                .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Verify histogram was generated
        assert_eq!(result.stats.histogram_bins.len(), 1);
        assert_eq!(result.stats.histogram_bins[0].len(), 256);

        // Histogram should sum to total number of values
        let total_bins: usize = result.stats.histogram_bins[0].iter().sum();
        let expected_total = 200 * 3 * 224 * 224; // samples * channels * height * width
        assert_eq!(total_bins, expected_total);
    }

    #[test]
    fn test_multiple_quantization_methods_with_calibration() {
        env_logger::try_init().ok();

        let model = create_test_model();
        let (inputs, input_shapes) = create_calibration_data_realistic();

        let quantization_types = vec![
            QuantizationType::Int8,
            QuantizationType::Int4,
            QuantizationType::Mixed,
        ];

        for quant_type in quantization_types {
            println!("Testing {:?} quantization with calibration", quant_type);

            let config = QuantizationConfig {
                quantization_type: quant_type,
                ..Default::default()
            };

            let dataset = Quantizer::create_calibration_dataset(
                inputs.clone(),
                input_shapes.clone(),
                format!("test_dataset_{:?}", quant_type),
            )
            .expect("Failed to create calibration dataset");

            let quantizer = Quantizer::with_calibration(config, dataset);
            let result = quantizer
                .quantize_post_training(&model)
                .expect("Quantization failed");

            // Verify quantization succeeded
            assert!(result.quantized_size > 0);
            assert!(result.accuracy_loss >= 0.0);
            assert!(result.compression_ratio() > 1.0);

            println!(
                "  {:?}: {:.2}x compression, {:.3}% accuracy loss",
                quant_type,
                result.compression_ratio(),
                result.accuracy_loss
            );
        }
    }

    #[test]
    fn test_calibration_with_insufficient_data() {
        let config = QuantizationConfig {
            calibration_dataset_size: 1000, // Request more than we provide
            ..Default::default()
        };

        // Provide only 10 samples
        let small_inputs = vec![vec![vec![1.0, 2.0, 3.0]; 10]];
        let input_shapes = vec![vec![3]];

        let dataset = Quantizer::create_calibration_dataset(
            small_inputs,
            input_shapes,
            "small_dataset".to_string(),
        )
        .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Should have warnings about insufficient data
        assert!(result.quality_score < 0.8); // Penalized for small dataset
        assert!(!result.warnings.is_empty());

        println!("Small dataset quality: {:.3}", result.quality_score);
        println!("Small dataset warnings: {:?}", result.warnings);
    }

    #[test]
    fn test_calibration_statistics_accuracy() {
        let config = QuantizationConfig::default();

        // Create known distribution for testing
        let known_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let inputs = vec![vec![known_values.clone(); 100]]; // 100 samples of the same values
        let input_shapes = vec![vec![5]];

        let dataset = Quantizer::create_calibration_dataset(
            inputs,
            input_shapes,
            "known_distribution".to_string(),
        )
        .expect("Failed to create calibration dataset");

        let mut quantizer = Quantizer::new(config);
        quantizer.set_calibration_data(dataset);

        let result = quantizer
            .analyze_calibration_data()
            .expect("Failed to analyze calibration data");

        // Verify statistics are correct
        assert_eq!(result.stats.min_values[0], 1.0);
        assert_eq!(result.stats.max_values[0], 5.0);
        assert!((result.stats.mean_values[0] - 3.0).abs() < 0.1); // Mean should be ~3.0

        println!("Known distribution stats:");
        println!("  Min: {}", result.stats.min_values[0]);
        println!("  Max: {}", result.stats.max_values[0]);
        println!("  Mean: {}", result.stats.mean_values[0]);
        println!("  Std: {}", result.stats.std_values[0]);
    }
}
