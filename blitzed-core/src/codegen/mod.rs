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

//! Code generation for deployment targets

pub mod arduino;
pub mod esp32;
pub mod c;
pub mod raspberry_pi;

use crate::{Model, Result, BlitzedError};
use std::path::Path;

/// Trait for code generators
pub trait CodeGenerator {
    /// Generate deployment code for the optimized model
    fn generate(&self, model: &Model, output_dir: &Path) -> Result<GeneratedCode>;
    
    /// Get the target name this generator supports
    fn target_name(&self) -> &str;
    
    /// Get required dependencies/libraries
    fn dependencies(&self) -> Vec<String>;
}

/// Generated code result
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// Main implementation file
    pub implementation_file: String,
    /// Header file (if applicable)
    pub header_file: Option<String>,
    /// Example usage file
    pub example_file: Option<String>,
    /// Build configuration (Makefile, CMakeLists.txt, etc.)
    pub build_config: Option<String>,
    /// Required dependencies
    pub dependencies: Vec<String>,
}

/// Universal code generator that delegates to target-specific generators
pub struct UniversalCodeGenerator {
    generators: std::collections::HashMap<String, Box<dyn CodeGenerator>>,
}

impl UniversalCodeGenerator {
    pub fn new() -> Self {
        let mut generator = Self {
            generators: std::collections::HashMap::new(),
        };
        
        // Register built-in generators
        generator.register("arduino", Box::new(arduino::ArduinoCodeGen::new()));
        generator.register("esp32", Box::new(esp32::Esp32CodeGen::new()));
        generator.register("c", Box::new(c::CCodeGen::new()));
        generator.register("raspberry_pi", Box::new(raspberry_pi::RaspberryPiCodeGen::new()));
        
        generator
    }
    
    pub fn register(&mut self, target: &str, generator: Box<dyn CodeGenerator>) {
        self.generators.insert(target.to_string(), generator);
    }
    
    pub fn generate(&self, target: &str, model: &Model, output_dir: &Path) -> Result<GeneratedCode> {
        let generator = self.generators.get(target)
            .ok_or_else(|| BlitzedError::UnsupportedTarget {
                target: target.to_string(),
            })?;
        
        generator.generate(model, output_dir)
    }
    
    pub fn list_targets(&self) -> Vec<&str> {
        self.generators.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for UniversalCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}