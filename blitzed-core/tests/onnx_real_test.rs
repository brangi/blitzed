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

#[cfg(feature = "onnx")]
#[test]
fn test_load_real_onnx_models() {
    use blitzed_core::converters::onnx::OnnxConverter;
    use blitzed_core::converters::ModelConverter;
    use std::path::Path;

    // Only run this test if the test models exist
    let test_model_onnx = Path::new("/tmp/test_model.onnx");

    if !test_model_onnx.exists() {
        println!("Skipping real ONNX model test - test model not found at /tmp/");
        println!("To run this test, create a model with:");
        println!("  python3 -c \"");
        println!("import torch");
        println!("import torch.onnx");
        println!("import torch.nn as nn");
        println!("model = nn.Sequential(");
        println!("    nn.Conv2d(3, 32, 3, padding=1),");
        println!("    nn.ReLU(),");
        println!("    nn.AdaptiveAvgPool2d((1, 1)),");
        println!("    nn.Flatten(),");
        println!("    nn.Linear(32, 10)");
        println!(");");
        println!("dummy_input = torch.randn(1, 3, 32, 32)");
        println!("torch.onnx.export(model, dummy_input, '/tmp/test_model.onnx', ");
        println!("                  input_names=['input'], output_names=['output'])");
        println!("\"");
        return;
    }

    // Test loading .onnx file
    let converter = OnnxConverter::new();

    let result = converter.load_model("/tmp/test_model.onnx");
    assert!(
        result.is_ok(),
        "Failed to load .onnx file: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(model.info.format, blitzed_core::model::ModelFormat::Onnx);
    assert!(
        !model.info.input_shapes.is_empty(),
        "Input shapes should not be empty"
    );
    assert!(
        !model.info.output_shapes.is_empty(),
        "Output shapes should not be empty"
    );
    assert!(
        model.info.model_size_bytes > 0,
        "Model size should be greater than 0"
    );
    assert!(
        model.info.parameter_count > 0,
        "Parameter count should be greater than 0"
    );
    assert!(
        model.info.operations_count > 0,
        "Operations count should be greater than 0"
    );

    println!("Successfully loaded .onnx file:");
    println!("  Input shapes: {:?}", model.info.input_shapes);
    println!("  Output shapes: {:?}", model.info.output_shapes);
    println!("  Parameters: {}", model.info.parameter_count);
    println!(
        "  Model size: {:.2} MB",
        model.info.model_size_bytes as f32 / (1024.0 * 1024.0)
    );
}

#[cfg(feature = "onnx")]
#[test]
fn test_onnx_model_inference() {
    use blitzed_core::converters::onnx::OnnxConverter;
    use blitzed_core::converters::ModelConverter;
    use std::path::Path;

    let test_model = Path::new("/tmp/test_model.onnx");
    if !test_model.exists() {
        println!("Skipping ONNX inference test - model not found");
        return;
    }

    let converter = OnnxConverter::new();
    let result = converter.load_model("/tmp/test_model.onnx");
    let model = result.expect("Failed to load model for inference test");

    // Verify we got an ONNX model with session data
    #[cfg(feature = "onnx")]
    {
        use blitzed_core::model::ModelData;
        match &model.data {
            ModelData::Onnx(session) => {
                println!("✅ Model loaded as ONNX Runtime session");

                // For now, just verify the session exists and has metadata
                println!("✅ Model session created successfully");
                println!("  Session inputs: {}", session.inputs.len());
                println!("  Session outputs: {}", session.outputs.len());

                // TODO: Implement proper inference testing once ort API is stabilized
                // The current ort version requires more complex setup for inference
            }
            _ => panic!("Model data is not ONNX Runtime session"),
        }
    }
}
