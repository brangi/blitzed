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

//! Configuration management for Blitzed

use crate::{BlitzedError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Global configuration for Blitzed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub optimization: OptimizationConfig,
    pub hardware: HardwareConfig,
    pub deployment: DeploymentConfig,
}

/// Optimization-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Maximum allowed accuracy loss (percentage)
    pub max_accuracy_loss: f32,
    /// Enable quantization
    pub enable_quantization: bool,
    /// Enable pruning
    pub enable_pruning: bool,
    /// Enable knowledge distillation
    pub enable_distillation: bool,
    /// Target compression ratio
    pub target_compression: f32,
}

/// Hardware-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Target hardware platform
    pub target: String,
    /// Available RAM in bytes
    pub memory_limit: usize,
    /// Available flash storage in bytes
    pub storage_limit: usize,
    /// CPU frequency in MHz
    pub cpu_frequency: u32,
    /// Use hardware accelerators if available
    pub use_accelerators: bool,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Output directory for generated code
    pub output_dir: String,
    /// Target framework for deployment
    pub target_framework: String,
    /// Include preprocessing code
    pub include_preprocessing: bool,
    /// Include postprocessing code
    pub include_postprocessing: bool,
    /// Generate example usage code
    pub generate_examples: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            optimization: OptimizationConfig {
                max_accuracy_loss: 5.0,
                enable_quantization: true,
                enable_pruning: false,
                enable_distillation: false,
                target_compression: 0.25, // 75% size reduction
            },
            hardware: HardwareConfig {
                target: "generic".to_string(),
                memory_limit: 1024 * 1024,      // 1MB
                storage_limit: 4 * 1024 * 1024, // 4MB
                cpu_frequency: 240,             // MHz (ESP32-like)
                use_accelerators: false,
            },
            deployment: DeploymentConfig {
                output_dir: "./output".to_string(),
                target_framework: "arduino".to_string(),
                include_preprocessing: true,
                include_postprocessing: true,
                generate_examples: true,
            },
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config =
            toml::from_str(&content).map_err(|e| BlitzedError::Configuration(e.to_string()))?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content =
            toml::to_string_pretty(self).map_err(|e| BlitzedError::Configuration(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> Result<()> {
        if self.optimization.max_accuracy_loss < 0.0 || self.optimization.max_accuracy_loss > 100.0
        {
            return Err(BlitzedError::Configuration(
                "max_accuracy_loss must be between 0 and 100".to_string(),
            ));
        }

        if self.optimization.target_compression <= 0.0
            || self.optimization.target_compression >= 1.0
        {
            return Err(BlitzedError::Configuration(
                "target_compression must be between 0 and 1".to_string(),
            ));
        }

        if self.hardware.memory_limit == 0 {
            return Err(BlitzedError::Configuration(
                "memory_limit must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get hardware-specific preset configurations
    pub fn preset(target: &str) -> Result<Self> {
        let mut config = Self::default();

        match target.to_lowercase().as_str() {
            "esp32" => {
                config.hardware.target = "esp32".to_string();
                config.hardware.memory_limit = 320 * 1024; // 320KB RAM
                config.hardware.storage_limit = 4 * 1024 * 1024; // 4MB Flash
                config.hardware.cpu_frequency = 240;
                config.deployment.target_framework = "arduino".to_string();
            }
            "arduino" => {
                config.hardware.target = "arduino".to_string();
                config.hardware.memory_limit = 32 * 1024; // 32KB RAM
                config.hardware.storage_limit = 256 * 1024; // 256KB Flash
                config.hardware.cpu_frequency = 16;
                config.deployment.target_framework = "arduino".to_string();
            }
            "stm32" => {
                config.hardware.target = "stm32".to_string();
                config.hardware.memory_limit = 128 * 1024; // 128KB RAM
                config.hardware.storage_limit = 1024 * 1024; // 1MB Flash
                config.hardware.cpu_frequency = 72;
                config.deployment.target_framework = "bare_metal".to_string();
            }
            "mobile" => {
                config.hardware.target = "mobile".to_string();
                config.hardware.memory_limit = 100 * 1024 * 1024; // 100MB
                config.hardware.storage_limit = 500 * 1024 * 1024; // 500MB
                config.hardware.cpu_frequency = 2000;
                config.hardware.use_accelerators = true;
                config.deployment.target_framework = "tflite".to_string();
            }
            _ => {
                return Err(BlitzedError::UnsupportedTarget {
                    target: target.to_string(),
                });
            }
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(
            config.optimization.max_accuracy_loss,
            deserialized.optimization.max_accuracy_loss
        );
    }

    #[test]
    fn test_config_file_operations() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();

        config.save(temp_file.path()).unwrap();
        let loaded = Config::load(temp_file.path()).unwrap();

        assert_eq!(
            config.optimization.max_accuracy_loss,
            loaded.optimization.max_accuracy_loss
        );
    }

    #[test]
    fn test_preset_configs() {
        let esp32_config = Config::preset("esp32").unwrap();
        assert_eq!(esp32_config.hardware.target, "esp32");
        assert_eq!(esp32_config.hardware.memory_limit, 320 * 1024);

        let mobile_config = Config::preset("mobile").unwrap();
        assert_eq!(mobile_config.hardware.target, "mobile");
        assert!(mobile_config.hardware.use_accelerators);
    }

    #[test]
    fn test_invalid_target() {
        let result = Config::preset("invalid_target");
        assert!(result.is_err());
    }
}
