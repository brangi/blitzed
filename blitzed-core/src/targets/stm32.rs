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

//! STM32 hardware target implementation

use super::{HardwareTarget, HardwareConstraints, OptimizationStrategy};

/// STM32 hardware target
pub struct Stm32Target {
    constraints: HardwareConstraints,
}

impl Stm32Target {
    pub fn new() -> Self {
        Self {
            constraints: HardwareConstraints {
                memory_limit: 128 * 1024, // 128KB RAM
                storage_limit: 1024 * 1024, // 1MB Flash
                cpu_frequency: 72, // MHz
                architecture: "ARM Cortex-M".to_string(),
                word_size: 32,
                has_fpu: true, // Many STM32 have FPU
                accelerators: vec![],
            },
        }
    }
}

impl HardwareTarget for Stm32Target {
    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn name(&self) -> &str {
        "STM32"
    }

    fn optimization_strategy(&self) -> OptimizationStrategy {
        OptimizationStrategy {
            aggressive_quantization: true,
            enable_pruning: true,
            target_precision: "int8".to_string(),
            memory_optimization: true,
            speed_optimization: true, // Can leverage FPU when needed
        }
    }
}

impl Default for Stm32Target {
    fn default() -> Self {
        Self::new()
    }
}