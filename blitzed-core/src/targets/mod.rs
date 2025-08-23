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

//! Target hardware abstraction and optimization

pub mod arduino;
pub mod esp32;
pub mod mobile;
pub mod raspberry_pi;
pub mod stm32;

use crate::{BlitzedError, Result};
use serde::{Deserialize, Serialize};

/// Hardware constraints for target devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Available RAM in bytes
    pub memory_limit: usize,
    /// Available flash storage in bytes
    pub storage_limit: usize,
    /// CPU frequency in MHz
    pub cpu_frequency: u32,
    /// Architecture (ARM, x86, etc.)
    pub architecture: String,
    /// Word size in bits (8, 16, 32, 64)
    pub word_size: u8,
    /// Has floating point unit
    pub has_fpu: bool,
    /// Hardware accelerators available
    pub accelerators: Vec<String>,
}

/// Trait for target hardware platforms
pub trait HardwareTarget {
    /// Get hardware constraints for this target
    fn constraints(&self) -> &HardwareConstraints;

    /// Get target name
    fn name(&self) -> &str;

    /// Check if model fits on this target
    fn check_compatibility(&self, model_size: usize, memory_usage: usize) -> Result<()> {
        let constraints = self.constraints();

        if model_size > constraints.storage_limit {
            return Err(BlitzedError::HardwareConstraint {
                constraint: format!(
                    "Model size {}KB exceeds storage limit {}KB",
                    model_size / 1024,
                    constraints.storage_limit / 1024
                ),
            });
        }

        if memory_usage > constraints.memory_limit {
            return Err(BlitzedError::HardwareConstraint {
                constraint: format!(
                    "Memory usage {}KB exceeds limit {}KB",
                    memory_usage / 1024,
                    constraints.memory_limit / 1024
                ),
            });
        }

        Ok(())
    }

    /// Get recommended optimization strategy
    fn optimization_strategy(&self) -> OptimizationStrategy;
}

/// Optimization strategy recommendations for hardware targets
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub aggressive_quantization: bool,
    pub enable_pruning: bool,
    pub target_precision: String,
    pub memory_optimization: bool,
    pub speed_optimization: bool,
}

/// Target registry for managing hardware targets
pub struct TargetRegistry {
    targets: std::collections::HashMap<String, Box<dyn HardwareTarget>>,
}

impl TargetRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            targets: std::collections::HashMap::new(),
        };

        // Register built-in targets
        registry.register_target("esp32", Box::new(esp32::Esp32Target::new()));
        registry.register_target("arduino", Box::new(arduino::ArduinoTarget::new()));
        registry.register_target("stm32", Box::new(stm32::Stm32Target::new()));
        registry.register_target("mobile", Box::new(mobile::MobileTarget::new()));
        registry.register_target(
            "raspberry_pi",
            Box::new(raspberry_pi::RaspberryPiTarget::new()),
        );

        registry
    }

    pub fn register_target(&mut self, name: &str, target: Box<dyn HardwareTarget>) {
        self.targets.insert(name.to_string(), target);
    }

    pub fn get_target(&self, name: &str) -> Result<&dyn HardwareTarget> {
        self.targets
            .get(name)
            .map(|t| t.as_ref())
            .ok_or_else(|| BlitzedError::UnsupportedTarget {
                target: name.to_string(),
            })
    }

    pub fn list_targets(&self) -> Vec<&str> {
        self.targets.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for TargetRegistry {
    fn default() -> Self {
        Self::new()
    }
}
