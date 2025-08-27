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

//! STM32 integration test

use blitzed_core::{codegen::CodeGenerator, Model};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_complete_stm32_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("stm32_test");
    
    // Create a moderate-sized model suitable for STM32
    let model = Model::create_test_model().unwrap();
    
    // Generate STM32 code
    let codegen = blitzed_core::codegen::stm32::Stm32CodeGen::new();
    let result = codegen.generate(&model, &output_dir).unwrap();
    
    // Verify all files were created
    assert!(Path::new(&result.implementation_file).exists());
    assert!(result.header_file.is_some());
    assert!(Path::new(result.header_file.as_ref().unwrap()).exists());
    assert!(result.example_file.is_some());
    assert!(Path::new(result.example_file.as_ref().unwrap()).exists());
    assert!(result.build_config.is_some());
    assert!(Path::new(result.build_config.as_ref().unwrap()).exists());
    
    // Verify header content
    let header_content = std::fs::read_to_string(result.header_file.unwrap()).unwrap();
    assert!(header_content.contains("STM32_TARGET"));
    assert!(header_content.contains("USE_HAL_DRIVER"));
    assert!(header_content.contains("USE_FPU"));
    assert!(header_content.contains("model_predict"));
    assert!(header_content.contains("model_init"));
    
    // Verify main.c content
    let main_content = std::fs::read_to_string(result.example_file.unwrap()).unwrap();
    assert!(main_content.contains("int main(void)"));
    assert!(main_content.contains("HAL_Init()"));
    assert!(main_content.contains("SystemClock_Config()"));
    assert!(main_content.contains("model_predict"));
    assert!(main_content.contains("printf"));
    assert!(main_content.contains("stm32f4xx_hal.h"));
    
    // Verify Makefile content
    let makefile_content = std::fs::read_to_string(result.build_config.unwrap()).unwrap();
    assert!(makefile_content.contains("arm-none-eabi-gcc"));
    assert!(makefile_content.contains("-mcpu=cortex-m4"));
    assert!(makefile_content.contains("-mfpu=fpv4-sp-d16"));
    assert!(makefile_content.contains("flash:"));
    assert!(makefile_content.contains("st-flash"));
    
    println!("✅ STM32 integration test passed!");
    println!("   Generated files in: {}", output_dir.display());
}