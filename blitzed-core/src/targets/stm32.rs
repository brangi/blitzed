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