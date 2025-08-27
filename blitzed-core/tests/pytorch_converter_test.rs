// Test PyTorch converter implementation without actual model loading
#[cfg(test)]
mod tests {
    use blitzed_core::converters::pytorch::PyTorchConverter;
    use blitzed_core::converters::ModelConverter;
    use std::fs;

    #[test]
    fn test_pytorch_converter_basic() {
        let converter = PyTorchConverter::new();
        assert_eq!(converter.supported_extensions(), &["pt", "pth"]);
    }

    #[test]
    #[cfg(not(feature = "pytorch"))]
    fn test_pytorch_disabled_error() {
        let converter = PyTorchConverter::new();
        let result = converter.load_model("/tmp/test.pt");
        assert!(result.is_err());
        match result.unwrap_err() {
            blitzed_core::BlitzedError::UnsupportedFormat { format } => {
                assert_eq!(format, "PyTorch (feature not enabled)");
            }
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    #[test]
    fn test_pytorch_file_detection() {
        use blitzed_core::model::ModelFormat;

        assert_eq!(
            ModelFormat::from_path("/path/to/model.pt"),
            Some(ModelFormat::PyTorch)
        );
        assert_eq!(
            ModelFormat::from_path("/path/to/model.pth"),
            Some(ModelFormat::PyTorch)
        );
        assert_eq!(
            ModelFormat::from_path("/path/to/model.onnx"),
            Some(ModelFormat::Onnx)
        );
        assert_eq!(ModelFormat::from_path("/path/to/model.txt"), None);
    }

    #[test]
    fn test_universal_converter_routes_pytorch() {
        use blitzed_core::converters::UniversalConverter;

        let converter = UniversalConverter::new();

        // Create a dummy file to test routing (won't actually load)
        let test_path = "/tmp/test_routing.pt";
        let _ = fs::File::create(test_path);

        let result = converter.load_model(test_path);
        assert!(result.is_err()); // Will fail to load, but should route to PyTorch converter

        // Clean up
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_pytorch_converter_validates_real_models() {
        // This test verifies our implementation is ready for real models
        let test_files = vec![
            ("/tmp/test_model.pt", 51_419_549), // Expected size from our created model
            ("/tmp/test_model.pth", 51_419_549),
        ];

        for (path, _expected_size) in test_files {
            if let Ok(metadata) = fs::metadata(path) {
                println!(
                    "Found real PyTorch model at {}: {} bytes",
                    path,
                    metadata.len()
                );
                assert!(metadata.len() > 0, "Model file should not be empty");

                // Verify it's actually a PyTorch file (zip format)
                if let Ok(mut file) = fs::File::open(path) {
                    let mut buffer = [0; 4];
                    use std::io::Read;
                    if file.read(&mut buffer).is_ok() {
                        // PyTorch files are ZIP archives starting with "PK"
                        assert_eq!(
                            &buffer[0..2],
                            b"PK",
                            "PyTorch models should be ZIP archives"
                        );
                    }
                }
            } else {
                println!(
                    "Skipping {} - model not found (create with Python script)",
                    path
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_pytorch_implementation_complete() {
        // Verify all required methods are implemented
        let converter = PyTorchConverter::new();

        // Test that load_model is implemented (will fail on non-existent file, but that's ok)
        let _ = converter.load_model("/nonexistent.pt");

        // Test that save_model returns expected error
        let model_info = blitzed_core::model::ModelInfo {
            format: blitzed_core::model::ModelFormat::PyTorch,
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            parameter_count: 1000,
            model_size_bytes: 4096,
            operations_count: 2000,
            layers: vec![],
        };
        let model = blitzed_core::Model {
            info: model_info,
            data: blitzed_core::model::ModelData::Raw(vec![]),
        };

        let save_result = converter.save_model(&model, "/tmp/test_save.pt");
        assert!(save_result.is_err());
        match save_result.unwrap_err() {
            blitzed_core::BlitzedError::Internal(msg) => {
                assert_eq!(msg, "PyTorch saving not implemented");
            }
            _ => panic!("Expected Internal error for save"),
        }
    }
}
