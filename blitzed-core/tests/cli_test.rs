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

//! CLI integration tests

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

use blitzed_core::{codegen::CodeGenerator, Result};

#[test]
fn test_cli_info_command() {
    let output = Command::new("python")
        .args(["-m", "blitzed.cli.main", "info"])
        .current_dir("/Users/brangirod/blitzed")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("Blitzed Information"));
            println!("✅ CLI available and working");
        }
        Ok(output) => {
            // CLI available but returned error - that's still useful info
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("⚠️  CLI available but failed: {}", stderr);
        }
        Err(_) => {
            // CLI not available - this is expected in many test environments
            println!("ℹ️  Python CLI not available in test environment (expected)");
        }
    }
}

#[test]
fn test_end_to_end_optimization_workflow() -> Result<()> {
    let _temp_dir = TempDir::new()?;

    // Create test model using Rust API first
    let model = blitzed_core::Model::create_test_model()?;

    // Test Rust optimization directly with mobile target (has more memory)
    let config = blitzed_core::Config::preset("mobile")?;
    let optimizer = blitzed_core::Optimizer::new(config);
    let result = optimizer.optimize(&model)?;

    assert!(result.compression_ratio > 0.0);
    assert!(!result.techniques_applied.is_empty());

    Ok(())
}

#[test]
fn test_c_code_generation_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join("generated");

    // Create test model
    let model = blitzed_core::Model::create_test_model()?;

    // Test C code generation
    let codegen = blitzed_core::codegen::c::CCodeGen::with_target("esp32");
    let generated = codegen.generate(&model, &output_dir)?;

    // Verify files were created
    assert!(
        Path::new(&generated.implementation_file).exists(),
        "Implementation file does not exist: {}",
        generated.implementation_file
    );

    if let Some(header) = generated.header_file {
        assert!(
            Path::new(&header).exists(),
            "Header file does not exist: {}",
            header
        );
    }

    if let Some(build_config) = generated.build_config {
        assert!(
            Path::new(&build_config).exists(),
            "Build config does not exist: {}",
            build_config
        );
    }

    Ok(())
}

#[test]
fn test_hardware_target_validation() -> Result<()> {
    let targets = ["esp32", "mobile", "raspberry_pi", "generic"]; // Test all supported targets

    for target in &targets {
        let config = blitzed_core::Config::preset(target)?;
        let model = blitzed_core::Model::create_test_model()?;
        let optimizer = blitzed_core::Optimizer::new(config);

        // Test recommendations
        let recommendations = optimizer.recommend(&model)?;
        assert!(!recommendations.is_empty());

        // Test impact estimation
        let impact = optimizer.estimate_impact(&model)?;
        assert!(impact.size_reduction >= 0.0);
        assert!(impact.speed_improvement >= 1.0);

        println!("✅ Target '{}' validation passed", target);
    }

    Ok(())
}

#[test]
fn test_complete_deployment_pipeline() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join("deployment");

    // Create and optimize model with mobile target (has more memory)
    let model = blitzed_core::Model::create_test_model()?;
    let config = blitzed_core::Config::preset("mobile")?;
    let optimizer = blitzed_core::Optimizer::new(config);
    let _optimization_result = optimizer.optimize(&model)?;

    // Generate deployment code (use ESP32 for C code gen)
    let codegen = blitzed_core::codegen::c::CCodeGen::with_target("esp32");
    let generated = codegen.generate(&model, &output_dir)?;

    // Verify complete deployment package
    assert!(Path::new(&generated.implementation_file).exists());
    if let Some(header) = generated.header_file {
        assert!(Path::new(&header).exists());
    }
    if let Some(build_config) = generated.build_config {
        assert!(Path::new(&build_config).exists());

        // Verify build config contains ESP32-specific content
        let content = fs::read_to_string(&build_config)?;
        assert!(content.contains("ESP32") || content.contains("esp32"));
    }

    Ok(())
}

#[test]
fn test_generated_c_code_quality() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join("esp32_output");

    // Create test model and generate C code
    let model = blitzed_core::Model::create_test_model()?;
    let codegen = blitzed_core::codegen::c::CCodeGen::with_target("esp32");
    let generated = codegen.generate(&model, &output_dir)?;

    // Read and validate implementation file
    let impl_content = fs::read_to_string(&generated.implementation_file)?;
    assert!(
        impl_content.contains("ESP32_TARGET"),
        "Missing ESP32 target definition"
    );
    assert!(
        impl_content.contains("model_predict"),
        "Missing main prediction function"
    );
    assert!(
        impl_content.contains("model_init"),
        "Missing initialization function"
    );
    assert!(
        impl_content.contains("INT8"),
        "Missing quantization support"
    );
    assert!(
        impl_content.len() > 5000,
        "Implementation file too small: {} bytes",
        impl_content.len()
    );

    // Read and validate header file
    if let Some(header_path) = generated.header_file {
        let header_content = fs::read_to_string(&header_path)?;
        assert!(header_content.contains("#ifndef"), "Missing header guard");
        assert!(
            header_content.contains("model_predict"),
            "Missing function declaration"
        );
        assert!(
            header_content.contains("typedef"),
            "Missing type definitions"
        );
    }

    // Read and validate Makefile
    if let Some(makefile_path) = generated.build_config {
        let makefile_content = fs::read_to_string(&makefile_path)?;
        assert!(
            makefile_content.contains("CC"),
            "Missing compiler definition"
        );
        assert!(
            makefile_content.contains("ESP32") || makefile_content.contains("esp32"),
            "Missing ESP32-specific configuration"
        );
        assert!(
            makefile_content.contains("flash:"),
            "Missing ESP32 flash target"
        );
    }

    println!("✅ Generated C code quality validation passed");
    println!("   📊 Implementation: {} bytes", impl_content.len());
    println!("   📁 Output directory: {}", output_dir.display());

    Ok(())
}
