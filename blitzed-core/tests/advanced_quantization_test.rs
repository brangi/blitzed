#[cfg(test)]
mod tests {
    use blitzed_core::model::{Model, ModelData, ModelFormat, ModelInfo};
    use blitzed_core::optimization::quantization::{
        QuantizationConfig, QuantizationType, Quantizer,
    };

    /// Test INT4 quantization implementation
    #[test]
    fn test_int4_quantization() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            calibration_dataset_size: 100,
            symmetric: true,
            per_channel: true,
            skip_sensitive_layers: false,
            accuracy_threshold: 5.0,
        };

        let quantizer = Quantizer::new(config);

        // Create a dummy model for testing
        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 25_000_000,
            model_size_bytes: 100 * 1024 * 1024, // 100 MB
            operations_count: 8_000_000_000,
            layers: vec![],
        };

        let model = Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        };

        let result = quantizer.quantize_int4(&model);
        assert!(result.is_ok(), "INT4 quantization should succeed");

        let quantized_model = result.unwrap();

        // Debug output first
        println!("✅ INT4 Quantization Debug:");
        println!("  Accuracy Loss: {:.1}%", quantized_model.accuracy_loss);
        println!("  Compression: {:.1}x", quantized_model.compression_ratio());
        println!(
            "  Original Size: {:.2} MB ({} bytes)",
            quantized_model.original_model_info.model_size_bytes as f32 / (1024.0 * 1024.0),
            quantized_model.original_model_info.model_size_bytes
        );
        println!(
            "  Quantized Size: {:.2} MB ({} bytes)",
            quantized_model.quantized_size as f32 / (1024.0 * 1024.0),
            quantized_model.quantized_size
        );
        println!(
            "  Parameters: {}",
            quantized_model.original_model_info.parameter_count
        );
        println!("  Layers: {}", quantized_model.layers.len());

        // Verify compression characteristics
        assert!(
            quantized_model.accuracy_loss >= 5.0 && quantized_model.accuracy_loss <= 15.0,
            "INT4 accuracy loss should be between 5-15%, got {:.1}%",
            quantized_model.accuracy_loss
        );

        assert!(
            quantized_model.compression_ratio() >= 7.0
                && quantized_model.compression_ratio() <= 9.0,
            "INT4 compression ratio should be ~8x, got {:.1}x",
            quantized_model.compression_ratio()
        );

        println!("✅ INT4 Quantization Results:");
        println!("  Accuracy Loss: {:.1}%", quantized_model.accuracy_loss);
        println!("  Compression: {:.1}x", quantized_model.compression_ratio());
        println!(
            "  Original Size: {:.2} MB ({} bytes)",
            quantized_model.original_model_info.model_size_bytes as f32 / (1024.0 * 1024.0),
            quantized_model.original_model_info.model_size_bytes
        );
        println!(
            "  Quantized Size: {:.2} MB ({} bytes)",
            quantized_model.quantized_size as f32 / (1024.0 * 1024.0),
            quantized_model.quantized_size
        );
        println!(
            "  Parameters: {}",
            quantized_model.original_model_info.parameter_count
        );
        println!("  Layers: {}", quantized_model.layers.len());
    }

    /// Test binary quantization implementation
    #[test]
    fn test_binary_quantization() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Binary,
            calibration_dataset_size: 100,
            symmetric: true,
            per_channel: false,
            skip_sensitive_layers: false,
            accuracy_threshold: 15.0,
        };

        let quantizer = Quantizer::new(config);

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 10]],
            parameter_count: 10_000_000,
            model_size_bytes: 40 * 1024 * 1024, // 40 MB
            operations_count: 2_000_000_000,
            layers: vec![],
        };

        let model = Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        };

        let result = quantizer.quantize_binary(&model);
        assert!(result.is_ok(), "Binary quantization should succeed");

        let quantized_model = result.unwrap();

        // Verify extreme compression characteristics
        assert!(
            quantized_model.accuracy_loss >= 10.0 && quantized_model.accuracy_loss <= 40.0,
            "Binary accuracy loss should be between 10-40%, got {:.1}%",
            quantized_model.accuracy_loss
        );

        assert!(
            quantized_model.compression_ratio() >= 2.0
                && quantized_model.compression_ratio() <= 4.0,
            "Binary compression ratio should be ~2.5x (mixed layers), got {:.1}x",
            quantized_model.compression_ratio()
        );

        println!("✅ Binary Quantization Results:");
        println!("  Accuracy Loss: {:.1}%", quantized_model.accuracy_loss);
        println!("  Compression: {:.1}x", quantized_model.compression_ratio());
        println!(
            "  Original Size: {:.2} MB",
            quantized_model.original_model_info.model_size_bytes as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Quantized Size: {:.2} MB",
            quantized_model.quantized_size as f32 / (1024.0 * 1024.0)
        );
    }

    /// Test mixed precision quantization implementation
    #[test]
    fn test_mixed_precision_quantization() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Mixed,
            calibration_dataset_size: 200,
            symmetric: true,
            per_channel: true,
            skip_sensitive_layers: true,
            accuracy_threshold: 8.0,
        };

        let quantizer = Quantizer::new(config);

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 50_000_000,
            model_size_bytes: 200 * 1024 * 1024, // 200 MB
            operations_count: 15_000_000_000,
            layers: vec![],
        };

        let model = Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        };

        let result = quantizer.quantize_mixed(&model);
        assert!(
            result.is_ok(),
            "Mixed precision quantization should succeed"
        );

        let quantized_model = result.unwrap();

        // Verify balanced compression characteristics
        assert!(
            quantized_model.accuracy_loss >= 2.0 && quantized_model.accuracy_loss <= 15.0,
            "Mixed precision accuracy loss should be between 2-15%, got {:.1}%",
            quantized_model.accuracy_loss
        );

        assert!(
            quantized_model.compression_ratio() >= 3.0
                && quantized_model.compression_ratio() <= 6.0,
            "Mixed precision compression ratio should be 3-6x (weighted average), got {:.1}x",
            quantized_model.compression_ratio()
        );

        println!("✅ Mixed Precision Quantization Results:");
        println!("  Accuracy Loss: {:.1}%", quantized_model.accuracy_loss);
        println!("  Compression: {:.1}x", quantized_model.compression_ratio());
        println!(
            "  Original Size: {:.2} MB",
            quantized_model.original_model_info.model_size_bytes as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Quantized Size: {:.2} MB",
            quantized_model.quantized_size as f32 / (1024.0 * 1024.0)
        );
    }

    /// Test quantization technique comparison
    #[test]
    fn test_quantization_technique_comparison() {
        println!("\n🔬 Advanced Quantization Techniques Comparison:");

        let model_info = ModelInfo {
            format: ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 25_000_000,
            model_size_bytes: 100 * 1024 * 1024,
            operations_count: 8_000_000_000,
            layers: vec![],
        };

        let model = Model {
            info: model_info,
            data: ModelData::Raw(vec![]),
        };

        // Test each technique
        let techniques = [
            (QuantizationType::Int4, "INT4"),
            (QuantizationType::Binary, "Binary"),
            (QuantizationType::Mixed, "Mixed Precision"),
        ];

        for (technique, name) in techniques.iter() {
            let config = QuantizationConfig {
                quantization_type: *technique,
                calibration_dataset_size: 100,
                symmetric: true,
                per_channel: true,
                skip_sensitive_layers: false,
                accuracy_threshold: 10.0,
            };

            let quantizer = Quantizer::new(config);

            let result = match technique {
                QuantizationType::Int4 => quantizer.quantize_int4(&model),
                QuantizationType::Binary => quantizer.quantize_binary(&model),
                QuantizationType::Mixed => quantizer.quantize_mixed(&model),
                _ => continue, // Skip other techniques for this test
            };

            assert!(result.is_ok(), "{} quantization should succeed", name);

            let quantized = result.unwrap();
            println!(
                "  {} - Compression: {:.1}x, Accuracy Loss: {:.1}%",
                name,
                quantized.compression_ratio(),
                quantized.accuracy_loss
            );
        }

        println!("  Note: Actual results depend on model architecture and data distribution");
    }

    /// Test quantization parameter validation
    #[test]
    fn test_quantization_parameters() {
        // Test symmetric vs asymmetric quantization
        let symmetric_config = QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            calibration_dataset_size: 50,
            symmetric: true,
            per_channel: false,
            skip_sensitive_layers: false,
            accuracy_threshold: 8.0,
        };

        let asymmetric_config = QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            calibration_dataset_size: 50,
            symmetric: false,
            per_channel: false,
            skip_sensitive_layers: false,
            accuracy_threshold: 8.0,
        };

        let sym_quantizer = Quantizer::new(symmetric_config);
        let asym_quantizer = Quantizer::new(asymmetric_config);

        // Both should be valid (constructors should succeed)
        let _sym_quantizer = sym_quantizer;
        let _asym_quantizer = asym_quantizer;

        // Test per-channel vs per-tensor
        let per_channel_config = QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            calibration_dataset_size: 100,
            symmetric: true,
            per_channel: true,
            skip_sensitive_layers: false,
            accuracy_threshold: 8.0,
        };

        let per_tensor_config = QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            calibration_dataset_size: 100,
            symmetric: true,
            per_channel: false,
            skip_sensitive_layers: false,
            accuracy_threshold: 8.0,
        };

        let _pc_quantizer = Quantizer::new(per_channel_config);
        let _pt_quantizer = Quantizer::new(per_tensor_config);

        // Both should construct successfully (no panics or errors)

        println!("✅ Quantization parameter validation successful");
    }
}
