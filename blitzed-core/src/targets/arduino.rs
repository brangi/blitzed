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

//! Arduino hardware target implementation

use super::{HardwareConstraints, HardwareTarget, OptimizationStrategy};

/// Arduino Uno/Nano hardware target
pub struct ArduinoTarget {
    constraints: HardwareConstraints,
}

impl ArduinoTarget {
    pub fn new() -> Self {
        Self {
            constraints: HardwareConstraints {
                memory_limit: 2 * 1024,   // 2KB RAM
                storage_limit: 32 * 1024, // 32KB Flash
                cpu_frequency: 16,        // MHz
                architecture: "AVR".to_string(),
                word_size: 8,
                has_fpu: false,
                accelerators: vec![],
            },
        }
    }
}

impl HardwareTarget for ArduinoTarget {
    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn name(&self) -> &str {
        "Arduino"
    }

    fn optimization_strategy(&self) -> OptimizationStrategy {
        OptimizationStrategy {
            aggressive_quantization: true, // Extremely memory constrained
            enable_pruning: true,
            target_precision: "int4".to_string(), // Even more aggressive
            memory_optimization: true,
            speed_optimization: false,
        }
    }
}

impl Default for ArduinoTarget {
    fn default() -> Self {
        Self::new()
    }
}
