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

//! Optimization algorithms and techniques for edge AI deployment

pub mod quantization;
pub mod pruning;
pub mod distillation;
pub mod optimizer;

pub use optimizer::{Optimizer, OptimizationConfig, OptimizationResult};
pub use quantization::{QuantizationConfig, QuantizationType, Quantizer};

/// Trait for all optimization techniques
pub trait OptimizationTechnique {
    type Config;
    type Output;
    
    /// Apply the optimization to a model
    fn optimize(&self, model: &crate::Model, config: &Self::Config) -> crate::Result<Self::Output>;
    
    /// Estimate the impact of optimization
    fn estimate_impact(&self, model: &crate::Model, config: &Self::Config) -> crate::Result<OptimizationImpact>;
}

/// Impact estimation for optimization techniques
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Estimated size reduction ratio (0.0 - 1.0)
    pub size_reduction: f32,
    /// Estimated speed improvement ratio (1.0 = no change, 2.0 = 2x faster)
    pub speed_improvement: f32,
    /// Estimated accuracy loss (0.0 - 100.0 percentage points)
    pub accuracy_loss: f32,
    /// Estimated memory reduction ratio (0.0 - 1.0)
    pub memory_reduction: f32,
}

impl OptimizationImpact {
    /// Combine multiple optimization impacts
    pub fn combine(impacts: &[Self]) -> Self {
        let size_reduction = impacts.iter().map(|i| i.size_reduction).product();
        let speed_improvement = impacts.iter().map(|i| i.speed_improvement).product();
        let accuracy_loss = impacts.iter().map(|i| i.accuracy_loss).sum();
        let memory_reduction = impacts.iter().map(|i| i.memory_reduction).product();
        
        Self {
            size_reduction,
            speed_improvement,
            accuracy_loss,
            memory_reduction,
        }
    }
}