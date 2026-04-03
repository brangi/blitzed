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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arduino_constraints() {
        let target = ArduinoTarget::new();
        let constraints = target.constraints();
        assert_eq!(constraints.memory_limit, 2048);
        assert_eq!(constraints.storage_limit, 32768);
        assert_eq!(constraints.cpu_frequency, 16);
        assert_eq!(constraints.architecture, "AVR");
        assert_eq!(constraints.word_size, 8);
        assert!(!constraints.has_fpu);
        assert!(constraints.accelerators.is_empty());
    }

    #[test]
    fn test_arduino_name() {
        let target = ArduinoTarget::new();
        assert_eq!(target.name(), "Arduino");
    }

    #[test]
    fn test_arduino_optimization_strategy() {
        let target = ArduinoTarget::new();
        let strategy = target.optimization_strategy();
        assert!(strategy.aggressive_quantization);
        assert!(strategy.enable_pruning);
        assert_eq!(strategy.target_precision, "int4");
        assert!(strategy.memory_optimization);
        assert!(!strategy.speed_optimization);
    }

    #[test]
    fn test_arduino_default() {
        let _target = ArduinoTarget::default();
    }
}
