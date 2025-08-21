//! Mobile device hardware target implementation

use super::{HardwareTarget, HardwareConstraints, OptimizationStrategy};

/// Mobile device hardware target (iOS/Android)
pub struct MobileTarget {
    constraints: HardwareConstraints,
}

impl MobileTarget {
    pub fn new() -> Self {
        Self {
            constraints: HardwareConstraints {
                memory_limit: 100 * 1024 * 1024, // 100MB available for ML
                storage_limit: 500 * 1024 * 1024, // 500MB app storage
                cpu_frequency: 2000, // MHz (modern mobile CPU)
                architecture: "ARM64".to_string(),
                word_size: 64,
                has_fpu: true,
                accelerators: vec![
                    "Neural Engine".to_string(),
                    "GPU".to_string(),
                    "DSP".to_string(),
                ],
            },
        }
    }
}

impl HardwareTarget for MobileTarget {
    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn name(&self) -> &str {
        "Mobile"
    }

    fn optimization_strategy(&self) -> OptimizationStrategy {
        OptimizationStrategy {
            aggressive_quantization: false, // More flexible with memory
            enable_pruning: false, // Focus on accuracy
            target_precision: "fp16".to_string(), // Good balance
            memory_optimization: false,
            speed_optimization: true, // Optimize for user experience
        }
    }
}

impl Default for MobileTarget {
    fn default() -> Self {
        Self::new()
    }
}