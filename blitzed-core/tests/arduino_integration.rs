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

//! Arduino integration test

use blitzed_core::{codegen::CodeGenerator, Model};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_complete_arduino_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("arduino_test");
    
    // Create a tiny model suitable for Arduino
    let model = Model::create_test_model().unwrap();
    
    // Generate Arduino code
    let codegen = blitzed_core::codegen::arduino::ArduinoCodeGen::new();
    let result = codegen.generate(&model, &output_dir).unwrap();
    
    // Verify all files were created
    assert!(Path::new(&result.implementation_file).exists());
    assert!(result.header_file.is_some());
    assert!(Path::new(result.header_file.as_ref().unwrap()).exists());
    assert!(result.example_file.is_some());
    assert!(Path::new(result.example_file.as_ref().unwrap()).exists());
    
    // Verify file contents
    let header_content = std::fs::read_to_string(result.header_file.unwrap()).unwrap();
    assert!(header_content.contains("ARDUINO_TARGET"));
    assert!(header_content.contains("model_predict"));
    assert!(header_content.contains("model_init"));
    
    let ino_content = std::fs::read_to_string(result.example_file.unwrap()).unwrap();
    assert!(ino_content.contains("void setup()"));
    assert!(ino_content.contains("void loop()"));
    assert!(ino_content.contains("Serial.begin"));
    assert!(ino_content.contains("model_predict"));
    
    println!("✅ Arduino integration test passed!");
    println!("   Generated files in: {}", output_dir.display());
}