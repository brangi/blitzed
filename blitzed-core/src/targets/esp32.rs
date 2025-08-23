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
use crate::{Model, Result};
use serde::{Deserialize, Serialize};

/// ESP32 hardware specifications and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Esp32Specs {
    /// Variant of ESP32 (ESP32, ESP32-S2, ESP32-S3, ESP32-C3)
    pub variant: Esp32Variant,
    /// Available SRAM in bytes (faster access)
    pub sram_size: usize,
    /// Available DRAM in bytes (general purpose)
    pub dram_size: usize,
    /// Flash memory size in bytes
    pub flash_size: usize,
    /// CPU cores and frequency
    pub cpu_cores: u8,
    pub cpu_frequency: u32,
    /// Hardware capabilities
    pub has_fpu: bool,
    pub has_dsp: bool,
    pub wifi_enabled: bool,
    pub bluetooth_enabled: bool,
}

/// ESP32 variants with different capabilities
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Esp32Variant {
    /// Original ESP32 - dual-core, WiFi+BT
    Esp32,
    /// ESP32-S2 - single-core, WiFi only
    Esp32S2,
    /// ESP32-S3 - dual-core, WiFi+BT, AI acceleration
    Esp32S3,
    /// ESP32-C3 - single-core RISC-V, WiFi+BT
    Esp32C3,
}

/// ESP32 performance characteristics
#[derive(Debug, Clone)]
pub struct Esp32Performance {
    /// Instructions per second (approximate)
    pub ips: u64,
    /// Memory bandwidth in bytes/second
    pub memory_bandwidth: u64,
    /// Typical power consumption in milliwatts
    pub power_consumption: u32,
    /// Operating temperature range
    pub temp_range: (i8, i8),
}

/// ESP32 hardware target
pub struct Esp32Target {
    specs: Esp32Specs,
    performance: Esp32Performance,
    constraints: HardwareConstraints,
}

impl Esp32Target {
    /// Create standard ESP32 target (original variant)
    pub fn new() -> Self {
        Self::with_variant(Esp32Variant::Esp32)
    }
    
    /// Create ESP32 target with specific variant
    pub fn with_variant(variant: Esp32Variant) -> Self {
        let specs = match variant {
            Esp32Variant::Esp32 => Esp32Specs {
                variant,
                sram_size: 320 * 1024,      // 320KB SRAM
                dram_size: 320 * 1024,      // 320KB DRAM (overlaps with SRAM)
                flash_size: 4 * 1024 * 1024, // 4MB Flash
                cpu_cores: 2,
                cpu_frequency: 240,          // MHz
                has_fpu: true,
                has_dsp: false,
                wifi_enabled: true,
                bluetooth_enabled: true,
            },
            Esp32Variant::Esp32S2 => Esp32Specs {
                variant,
                sram_size: 320 * 1024,
                dram_size: 320 * 1024,
                flash_size: 4 * 1024 * 1024,
                cpu_cores: 1,
                cpu_frequency: 240,
                has_fpu: true,
                has_dsp: false,
                wifi_enabled: true,
                bluetooth_enabled: false,   // No Bluetooth
            },
            Esp32Variant::Esp32S3 => Esp32Specs {
                variant,
                sram_size: 512 * 1024,      // 512KB SRAM
                dram_size: 512 * 1024,
                flash_size: 8 * 1024 * 1024, // 8MB Flash
                cpu_cores: 2,
                cpu_frequency: 240,
                has_fpu: true,
                has_dsp: true,              // AI acceleration
                wifi_enabled: true,
                bluetooth_enabled: true,
            },
            Esp32Variant::Esp32C3 => Esp32Specs {
                variant,
                sram_size: 400 * 1024,      // 400KB SRAM
                dram_size: 400 * 1024,
                flash_size: 4 * 1024 * 1024,
                cpu_cores: 1,
                cpu_frequency: 160,          // RISC-V core
                has_fpu: false,              // No FPU
                has_dsp: false,
                wifi_enabled: true,
                bluetooth_enabled: true,
            },
        };
        
        let performance = Self::calculate_performance(&specs);
        let constraints = Self::build_constraints(&specs);
        
        Self {
            specs,
            performance,
            constraints,
        }
    }
    
    /// Calculate performance characteristics based on specs
    fn calculate_performance(specs: &Esp32Specs) -> Esp32Performance {
        // Approximate performance calculations based on real ESP32 benchmarks
        let base_ips = match specs.variant {
            Esp32Variant::Esp32 => 600_000_000,      // ~600 MIPS total
            Esp32Variant::Esp32S2 => 300_000_000,    // Single core
            Esp32Variant::Esp32S3 => 700_000_000,    // Improved architecture
            Esp32Variant::Esp32C3 => 400_000_000,    // RISC-V efficiency
        };
        
        // Memory bandwidth varies by variant
        let memory_bandwidth = match specs.variant {
            Esp32Variant::Esp32 => 40_000_000,       // ~40 MB/s
            Esp32Variant::Esp32S2 => 35_000_000,
            Esp32Variant::Esp32S3 => 50_000_000,     // Improved memory controller
            Esp32Variant::Esp32C3 => 30_000_000,
        };
        
        // Power consumption estimates (active mode)
        let power_consumption = match specs.variant {
            Esp32Variant::Esp32 => 240,              // ~240mW
            Esp32Variant::Esp32S2 => 180,            // More efficient
            Esp32Variant::Esp32S3 => 250,            // More features
            Esp32Variant::Esp32C3 => 120,            // Very efficient
        };
        
        Esp32Performance {
            ips: base_ips,
            memory_bandwidth,
            power_consumption,
            temp_range: (-40, 85), // Industrial temperature range
        }
    }
    
    /// Build hardware constraints from specs
    fn build_constraints(specs: &Esp32Specs) -> HardwareConstraints {
        let mut accelerators = Vec::new();
        if specs.has_dsp {
            accelerators.push("ESP-DSP".to_string());
        }
        if specs.wifi_enabled {
            accelerators.push("WiFi".to_string());
        }
        if specs.bluetooth_enabled {
            accelerators.push("Bluetooth".to_string());
        }
        
        let architecture = match specs.variant {
            Esp32Variant::Esp32C3 => "RISC-V".to_string(),
            _ => "Xtensa".to_string(),
        };
        
        HardwareConstraints {
            memory_limit: specs.sram_size,
            storage_limit: specs.flash_size,
            cpu_frequency: specs.cpu_frequency,
            architecture,
            word_size: 32,
            has_fpu: specs.has_fpu,
            accelerators,
        }
    }
    
    /// Get detailed ESP32 specifications
    pub fn specs(&self) -> &Esp32Specs {
        &self.specs
    }
    
    /// Get performance characteristics
    pub fn performance(&self) -> &Esp32Performance {
        &self.performance
    }
    
    /// Estimate inference latency for a model
    pub fn estimate_inference_latency(&self, model: &Model) -> Result<InferenceEstimate> {
        let model_info = model.info();
        
        // Calculate computational requirements
        let total_ops = model_info.operations_count as u64;
        
        // Account for quantized operations being faster
        let ops_per_second = self.performance.ips;
        
        // ESP32-specific optimizations
        let optimization_factor = match self.specs.variant {
            Esp32Variant::Esp32S3 => 1.2,  // DSP acceleration
            Esp32Variant::Esp32C3 => 0.9,  // No FPU
            _ => 1.0,
        };
        
        let effective_ops_per_second = (ops_per_second as f32 * optimization_factor) as u64;
        
        // Estimate latency
        let inference_time_us = (total_ops * 1_000_000) / effective_ops_per_second.max(1);
        
        // Memory requirements
        let model_memory = model_info.model_size_bytes;
        let activation_memory = self.estimate_activation_memory(model_info);
        let total_memory = model_memory + activation_memory;
        
        // Power estimation
        let inference_energy_mj = (self.performance.power_consumption as u64 * inference_time_us) / 1_000;
        
        Ok(InferenceEstimate {
            latency_us: inference_time_us,
            memory_usage: total_memory,
            energy_consumption_mj: inference_energy_mj,
            fits_in_memory: total_memory <= self.specs.sram_size,
            estimated_fps: if inference_time_us > 0 { 1_000_000 / inference_time_us } else { 0 },
        })
    }
    
    /// Estimate activation memory requirements
    fn estimate_activation_memory(&self, model_info: &crate::model::ModelInfo) -> usize {
        // Estimate based on input/output tensors
        let input_memory: usize = model_info.input_shapes.iter()
            .map(|shape| shape.iter().product::<i64>() as usize * 1) // INT8 = 1 byte
            .sum();
            
        let output_memory: usize = model_info.output_shapes.iter()
            .map(|shape| shape.iter().product::<i64>() as usize * 1)
            .sum();
            
        // Add intermediate activations (rough estimate)
        let intermediate_memory = (input_memory + output_memory) * 2;
        
        input_memory + output_memory + intermediate_memory
    }
}

impl HardwareTarget for Esp32Target {
    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn name(&self) -> &str {
        match self.specs.variant {
            Esp32Variant::Esp32 => "ESP32",
            Esp32Variant::Esp32S2 => "ESP32-S2",
            Esp32Variant::Esp32S3 => "ESP32-S3",
            Esp32Variant::Esp32C3 => "ESP32-C3",
        }
    }

    fn optimization_strategy(&self) -> OptimizationStrategy {
        // Tailor optimization strategy based on ESP32 variant capabilities
        let (aggressive_quant, enable_pruning, speed_opt) = match self.specs.variant {
            Esp32Variant::Esp32 => (true, true, false),     // Balanced
            Esp32Variant::Esp32S2 => (true, true, false),   // Memory focused
            Esp32Variant::Esp32S3 => (false, true, true),   // Can handle more complexity
            Esp32Variant::Esp32C3 => (true, true, false),   // Very constrained
        };
        
        let target_precision = if self.specs.has_fpu {
            "int8".to_string()
        } else {
            "int8".to_string() // INT8 even without FPU for efficiency
        };
        
        OptimizationStrategy {
            aggressive_quantization: aggressive_quant,
            enable_pruning,
            target_precision,
            memory_optimization: true, // Always optimize for memory on ESP32
            speed_optimization: speed_opt,
        }
    }
}

/// Inference performance estimate for ESP32
#[derive(Debug, Clone)]
pub struct InferenceEstimate {
    /// Estimated inference latency in microseconds
    pub latency_us: u64,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Energy consumption in millijoules
    pub energy_consumption_mj: u64,
    /// Whether the model fits in available memory
    pub fits_in_memory: bool,
    /// Estimated frames per second
    pub estimated_fps: u64,
}

impl InferenceEstimate {
    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f32 {
        self.latency_us as f32 / 1000.0
    }
    
    /// Get memory usage in KB
    pub fn memory_usage_kb(&self) -> f32 {
        self.memory_usage as f32 / 1024.0
    }
    
    /// Get energy consumption in millijoules
    pub fn energy_mj(&self) -> u64 {
        self.energy_consumption_mj
    }
    
    /// Check if performance meets real-time requirements
    pub fn meets_realtime_requirements(&self, target_fps: u64) -> bool {
        self.fits_in_memory && self.estimated_fps >= target_fps
    }
}

impl Default for Esp32Target {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Model, ModelInfo, ModelFormat, ModelData, LayerInfo};
    
    fn create_test_model() -> Model {
        let info = ModelInfo {
            format: ModelFormat::Onnx,
            input_shapes: vec![vec![1, 3, 32, 32]], // Smaller input for ESP32
            output_shapes: vec![vec![1, 10]], // Smaller output
            parameter_count: 50000, // Smaller parameter count
            model_size_bytes: 200000, // 200KB - fits in ESP32 memory
            operations_count: 100000, // Fewer operations
            layers: vec![
                LayerInfo {
                    name: "conv1".to_string(),
                    layer_type: "Conv2d".to_string(),
                    input_shape: vec![1, 3, 32, 32],
                    output_shape: vec![1, 64, 16, 16],
                    parameter_count: 1728, // 3*3*3*64
                    flops: 1769472, // Smaller FLOPS
                }
            ],
        };
        
        Model {
            info,
            data: ModelData::Raw(vec![0u8; 1000]),
        }
    }
    
    #[test]
    fn test_esp32_variants() {
        let esp32 = Esp32Target::with_variant(Esp32Variant::Esp32);
        let esp32s3 = Esp32Target::with_variant(Esp32Variant::Esp32S3);
        let esp32c3 = Esp32Target::with_variant(Esp32Variant::Esp32C3);
        
        assert_eq!(esp32.specs().cpu_cores, 2);
        assert_eq!(esp32s3.specs().sram_size, 512 * 1024);
        assert_eq!(esp32c3.specs().has_fpu, false);
        
        assert!(esp32s3.specs().has_dsp);
        assert!(!esp32.specs().has_dsp);
    }
    
    #[test]
    fn test_performance_estimation() {
        let esp32 = Esp32Target::new();
        let model = create_test_model();
        
        let estimate = esp32.estimate_inference_latency(&model).unwrap();
        
        assert!(estimate.latency_us > 0);
        assert!(estimate.memory_usage > 0);
        assert!(estimate.energy_consumption_mj > 0);
        
        // Should fit in ESP32 memory for this small model
        assert!(estimate.fits_in_memory);
    }
    
    #[test]
    fn test_optimization_strategies() {
        let esp32 = Esp32Target::with_variant(Esp32Variant::Esp32);
        let esp32s3 = Esp32Target::with_variant(Esp32Variant::Esp32S3);
        
        let strategy_esp32 = esp32.optimization_strategy();
        let strategy_s3 = esp32s3.optimization_strategy();
        
        assert!(strategy_esp32.aggressive_quantization);
        assert!(strategy_esp32.memory_optimization);
        
        // S3 can handle more complexity
        assert!(strategy_s3.speed_optimization);
        assert!(!strategy_s3.aggressive_quantization);
    }
    
    #[test]
    fn test_memory_compatibility() {
        let esp32 = Esp32Target::new();
        let model = create_test_model();
        
        let model_size = model.info().model_size_bytes;
        let memory_usage = esp32.estimate_activation_memory(model.info());
        
        let result = esp32.check_compatibility(model_size, memory_usage);
        assert!(result.is_ok());
        
        // Test with oversized model
        let oversized_result = esp32.check_compatibility(10_000_000, memory_usage);
        assert!(oversized_result.is_err());
    }
}