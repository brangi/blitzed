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

pub mod distillation;
pub mod optimizer;
pub mod pruning;
pub mod quantization;

pub use optimizer::{OptimizationConfig, OptimizationResult, Optimizer};
pub use quantization::{QuantizationConfig, QuantizationType, Quantizer};

/// Trait for all optimization techniques
pub trait OptimizationTechnique {
    type Config;
    type Output;

    /// Apply the optimization to a model
    fn optimize(&self, model: &crate::Model, config: &Self::Config) -> crate::Result<Self::Output>;

    /// Estimate the impact of optimization
    fn estimate_impact(
        &self,
        model: &crate::Model,
        config: &Self::Config,
    ) -> crate::Result<OptimizationImpact>;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_impact_combine_empty() {
        let impacts: Vec<OptimizationImpact> = vec![];
        let result = OptimizationImpact::combine(&impacts);
        assert_eq!(result.size_reduction, 1.0);
        assert_eq!(result.speed_improvement, 1.0);
        assert_eq!(result.accuracy_loss, 0.0);
        assert_eq!(result.memory_reduction, 1.0);
    }

    #[test]
    fn test_optimization_impact_combine_single() {
        let impact = OptimizationImpact {
            size_reduction: 0.5,
            speed_improvement: 2.0,
            accuracy_loss: 1.5,
            memory_reduction: 0.75,
        };
        let impacts = vec![impact.clone()];
        let result = OptimizationImpact::combine(&impacts);
        assert_eq!(result.size_reduction, 0.5);
        assert_eq!(result.speed_improvement, 2.0);
        assert_eq!(result.accuracy_loss, 1.5);
        assert_eq!(result.memory_reduction, 0.75);
    }

    #[test]
    fn test_optimization_impact_combine_multiple() {
        let impact1 = OptimizationImpact {
            size_reduction: 0.5,
            speed_improvement: 2.0,
            accuracy_loss: 1.0,
            memory_reduction: 0.8,
        };
        let impact2 = OptimizationImpact {
            size_reduction: 0.4,
            speed_improvement: 1.5,
            accuracy_loss: 0.5,
            memory_reduction: 0.9,
        };
        let impacts = vec![impact1, impact2];
        let result = OptimizationImpact::combine(&impacts);
        assert_eq!(result.size_reduction, 0.5 * 0.4);
        assert_eq!(result.speed_improvement, 2.0 * 1.5);
        assert_eq!(result.accuracy_loss, 1.0 + 0.5);
        assert_eq!(result.memory_reduction, 0.8 * 0.9);
    }
}
