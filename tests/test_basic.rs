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

//! Basic integration tests for Blitzed

use blitzed_core::{
    init, Config, 
    optimization::{Optimizer, QuantizationConfig, QuantizationType},
    targets::TargetRegistry,
    profiler::{Profiler, ProfilingConfig},
};

#[test]
fn test_library_initialization() {
    assert!(init().is_ok());
}

#[test]
fn test_config_creation() {
    let config = Config::default();
    assert!(config.validate().is_ok());
    
    // Test preset configs
    let esp32_config = Config::preset("esp32").unwrap();
    assert_eq!(esp32_config.hardware.target, "esp32");
    assert!(esp32_config.validate().is_ok());
}

#[test]
fn test_quantization_config() {
    let config = QuantizationConfig::default();
    assert_eq!(config.quantization_type, QuantizationType::Int8);
    assert!(config.symmetric);
}

#[test]
fn test_target_registry() {
    let registry = TargetRegistry::new();
    let targets = registry.list_targets();
    
    assert!(targets.contains(&"esp32"));
    assert!(targets.contains(&"arduino"));
    assert!(targets.contains(&"mobile"));
    
    // Test getting a target
    let esp32_target = registry.get_target("esp32").unwrap();
    assert_eq!(esp32_target.name(), "ESP32");
    
    let constraints = esp32_target.constraints();
    assert_eq!(constraints.memory_limit, 320 * 1024);
}

#[test]
fn test_optimizer_creation() {
    let config = Config::default();
    let optimizer = Optimizer::new(config);
    
    // Should create without errors
    assert_eq!(optimizer.config.optimization.max_accuracy_loss, 5.0);
}

#[test]
fn test_profiler_creation() {
    let config = ProfilingConfig::default();
    let profiler = Profiler::new(config);
    
    // Basic creation test
    assert_eq!(profiler.config.warmup_runs, 10);
    assert_eq!(profiler.config.benchmark_runs, 100);
}

#[test]
fn test_error_handling() {
    // Test unsupported target
    let result = Config::preset("invalid_target");
    assert!(result.is_err());
    
    // Test invalid config validation
    let mut config = Config::default();
    config.optimization.max_accuracy_loss = -1.0; // Invalid
    assert!(config.validate().is_err());
}