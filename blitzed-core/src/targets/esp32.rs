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

//! ESP32 hardware target implementation

use super::{HardwareTarget, HardwareConstraints, OptimizationStrategy};

/// ESP32 hardware target
pub struct Esp32Target {
    constraints: HardwareConstraints,
}

impl Esp32Target {
    pub fn new() -> Self {
        Self {
            constraints: HardwareConstraints {
                memory_limit: 320 * 1024, // 320KB RAM
                storage_limit: 4 * 1024 * 1024, // 4MB Flash
                cpu_frequency: 240, // MHz
                architecture: "Xtensa".to_string(),
                word_size: 32,
                has_fpu: true,
                accelerators: vec![], // No ML accelerators
            },
        }
    }
}

impl HardwareTarget for Esp32Target {
    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn name(&self) -> &str {
        "ESP32"
    }

    fn optimization_strategy(&self) -> OptimizationStrategy {
        OptimizationStrategy {
            aggressive_quantization: true, // Memory constrained
            enable_pruning: true,
            target_precision: "int8".to_string(),
            memory_optimization: true,
            speed_optimization: false, // Memory more important than speed
        }
    }
}

impl Default for Esp32Target {
    fn default() -> Self {
        Self::new()
    }
}