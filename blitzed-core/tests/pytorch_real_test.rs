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

#[cfg(feature = "pytorch")]
#[test]
fn test_load_real_pytorch_models() {
    use blitzed_core::converters::pytorch::PyTorchConverter;
    use blitzed_core::converters::ModelConverter;
    use std::path::Path;

    // Only run this test if the test models exist
    let test_model_pt = Path::new("/tmp/test_model.pt");
    let test_model_pth = Path::new("/tmp/test_model.pth");

    if !test_model_pt.exists() || !test_model_pth.exists() {
        println!("Skipping real PyTorch model test - test models not found at /tmp/");
        println!("To run this test, create models with:");
        println!("  python3 -c \"import torch; import torch.nn as nn; model = nn.Linear(10, 5); torch.jit.script(model).save('/tmp/test_model.pt')\"");
        return;
    }

    // Test loading .pt file
    let converter = PyTorchConverter::new();

    let result = converter.load_model("/tmp/test_model.pt");
    assert!(
        result.is_ok(),
        "Failed to load .pt file: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(model.info.format, blitzed_core::model::ModelFormat::PyTorch);
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

    println!("Successfully loaded .pt file:");
    println!("  Input shapes: {:?}", model.info.input_shapes);
    println!("  Output shapes: {:?}", model.info.output_shapes);
    println!("  Parameters: {}", model.info.parameter_count);
    println!(
        "  Model size: {:.2} MB",
        model.info.model_size_bytes as f32 / (1024.0 * 1024.0)
    );

    // Test loading .pth file
    let result = converter.load_model("/tmp/test_model.pth");
    assert!(
        result.is_ok(),
        "Failed to load .pth file: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(model.info.format, blitzed_core::model::ModelFormat::PyTorch);
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

    println!("Successfully loaded .pth file:");
    println!("  Input shapes: {:?}", model.info.input_shapes);
    println!("  Output shapes: {:?}", model.info.output_shapes);
    println!("  Parameters: {}", model.info.parameter_count);
    println!(
        "  Model size: {:.2} MB",
        model.info.model_size_bytes as f32 / (1024.0 * 1024.0)
    );
}

#[cfg(feature = "pytorch")]
#[test]
fn test_pytorch_model_forward_pass() {
    use blitzed_core::converters::pytorch::PyTorchConverter;
    use blitzed_core::converters::ModelConverter;
    use std::path::Path;

    let test_model = Path::new("/tmp/test_model.pt");
    if !test_model.exists() {
        println!("Skipping forward pass test - model not found");
        return;
    }

    let converter = PyTorchConverter::new();
    let result = converter.load_model("/tmp/test_model.pt");
    assert!(result.is_ok(), "Failed to load model for forward pass test");

    let model = result.unwrap();

    // Verify we got a PyTorch model with CModule data
    #[cfg(feature = "pytorch")]
    {
        use blitzed_core::model::ModelData;
        match &model.data {
            ModelData::PyTorch(module) => {
                println!("✅ Model loaded as PyTorch CModule");

                // Try a forward pass with dummy data
                use tch::{Device, Tensor};
                let dummy_input = Tensor::randn(&[1, 3, 224, 224], (tch::Kind::Float, Device::Cpu));

                match module.forward_ts(&[dummy_input]) {
                    Ok(output) => {
                        let output_shape: Vec<i64> = output.size();
                        println!("✅ Forward pass successful!");
                        println!("  Output shape: {:?}", output_shape);
                        assert_eq!(output_shape, vec![1, 10], "Expected output shape [1, 10]");
                    }
                    Err(e) => {
                        panic!("Forward pass failed: {}", e);
                    }
                }
            }
            _ => panic!("Model data is not PyTorch CModule"),
        }
    }
}
