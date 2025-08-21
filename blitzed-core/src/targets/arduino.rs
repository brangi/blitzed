//! Arduino hardware target implementation

use super::{HardwareTarget, HardwareConstraints, OptimizationStrategy};

/// Arduino Uno/Nano hardware target
pub struct ArduinoTarget {
    constraints: HardwareConstraints,
}

impl ArduinoTarget {
    pub fn new() -> Self {
        Self {
            constraints: HardwareConstraints {
                memory_limit: 2 * 1024, // 2KB RAM
                storage_limit: 32 * 1024, // 32KB Flash
                cpu_frequency: 16, // MHz
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