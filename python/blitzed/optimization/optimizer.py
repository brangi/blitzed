# Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main optimizer orchestrating multiple optimization techniques.

This module provides the high-level Optimizer class that coordinates
various optimization techniques to achieve target compression and
performance goals.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .quantization import QuantizationConfig


@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""
    quantization: Optional[QuantizationConfig] = None
    pruning_enabled: bool = False
    distillation_enabled: bool = False
    target_compression_ratio: float = 0.75
    max_accuracy_loss: float = 5.0
    optimization_passes: int = 1

    def __post_init__(self):
        """Set default quantization config if optimization enabled but config not provided."""
        if self.quantization is None:
            # Enable quantization by default
            self.quantization = QuantizationConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "quantization": self.quantization.to_dict() if self.quantization else None,
            "pruning_enabled": self.pruning_enabled,
            "distillation_enabled": self.distillation_enabled,
            "target_compression_ratio": self.target_compression_ratio,
            "max_accuracy_loss": self.max_accuracy_loss,
            "optimization_passes": self.optimization_passes,
        }


@dataclass
class OptimizationResult:
    """Result of the optimization process."""
    original_size: int
    optimized_size: int
    compression_ratio: float
    estimated_accuracy_loss: float
    estimated_speedup: float
    optimization_time_ms: int
    techniques_applied: List[str]

    @property
    def size_reduction_mb(self) -> float:
        """Size reduction in megabytes."""
        return (self.original_size - self.optimized_size) / (1024 * 1024)

    @property
    def meets_target(self) -> bool:
        """Check if result meets basic targets (placeholder)."""
        return self.compression_ratio >= 0.5 and self.estimated_accuracy_loss <= 10.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Optimization Summary:\n"
            f"  Size: {self.original_size / (1024*1024):.1f} MB → "
            f"{self.optimized_size / (1024*1024):.1f} MB "
            f"({self.compression_ratio*100:.1f}% reduction)\n"
            f"  Estimated accuracy loss: {self.estimated_accuracy_loss:.2f}%\n"
            f"  Estimated speedup: {self.estimated_speedup:.1f}x\n"
            f"  Optimization time: {self.optimization_time_ms} ms\n"
            f"  Techniques: {', '.join(self.techniques_applied)}"
        )


class Optimizer:
    """
    Main optimizer for applying multiple optimization techniques.
    
    The Optimizer coordinates various optimization techniques like quantization,
    pruning, and knowledge distillation to achieve target performance goals
    while respecting accuracy constraints.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: Optimization configuration. If None, uses default config.
        """
        self.config = config or OptimizationConfig()

    def optimize(self, model_path: str, output_path: Optional[str] = None, 
                 target: Optional[str] = None) -> OptimizationResult:
        """
        Optimize a model using configured techniques.
        
        Args:
            model_path: Path to input model file
            output_path: Path for optimized model output
            target: Target hardware (e.g., 'esp32', 'arduino', 'mobile')
            
        Returns:
            OptimizationResult with details about the optimization
            
        Raises:
            ValueError: If model format is not supported
            RuntimeError: If optimization fails
        """
        try:
            # Import core functions
            from .._core import optimize_model
            
            if output_path is None:
                # Generate output path based on input
                import os
                name, ext = os.path.splitext(model_path)
                output_path = f"{name}_optimized{ext}"
            
            # Prepare configuration
            config_dict = self.config.to_dict()
            if target:
                config_dict["target"] = target
            
            # Call Rust implementation
            result = optimize_model(model_path, output_path, config_dict)
            
            return OptimizationResult(
                original_size=result["original_size"],
                optimized_size=result["optimized_size"], 
                compression_ratio=result["compression_ratio"],
                estimated_accuracy_loss=result["estimated_accuracy_loss"],
                estimated_speedup=result["estimated_speedup"],
                optimization_time_ms=result["optimization_time_ms"],
                techniques_applied=result["techniques_applied"],
            )
            
        except ImportError:
            raise RuntimeError("Blitzed core extension not available")
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

    def estimate_impact(self, model_path: str, target: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate optimization impact without applying changes.
        
        Args:
            model_path: Path to model file
            target: Target hardware platform
            
        Returns:
            Dictionary with estimated impact metrics
        """
        try:
            from .._core import estimate_optimization_impact
            
            config_dict = self.config.to_dict()
            if target:
                config_dict["target"] = target
                
            return estimate_optimization_impact(model_path, config_dict)
        except ImportError:
            # Fallback estimates
            return {
                "size_reduction": 0.7,
                "speed_improvement": 2.0,
                "accuracy_loss": 3.0,
                "memory_reduction": 0.6,
            }

    def recommend(self, model_path: str, target: Optional[str] = None) -> List[str]:
        """
        Get optimization recommendations for a model and target.
        
        Args:
            model_path: Path to model file
            target: Target hardware platform
            
        Returns:
            List of recommendation strings
        """
        try:
            from .._core import get_optimization_recommendations
            
            config_dict = self.config.to_dict()
            if target:
                config_dict["target"] = target
                
            return get_optimization_recommendations(model_path, config_dict)
        except ImportError:
            # Fallback recommendations
            recommendations = ["Consider INT8 quantization for size reduction"]
            
            if target == "arduino":
                recommendations.append("Arduino target: Use aggressive quantization")
            elif target == "esp32":
                recommendations.append("ESP32 target: INT8 quantization recommended")
            elif target == "mobile":
                recommendations.append("Mobile target: Consider mixed precision")
                
            return recommendations

    def profile(self, model_path: str, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Profile model performance characteristics.
        
        Args:
            model_path: Path to model file
            target: Target hardware platform
            
        Returns:
            Dictionary with profiling results
        """
        try:
            from .._core import profile_model
            
            config_dict = {"target": target} if target else {}
            return profile_model(model_path, config_dict)
        except ImportError:
            # Mock profiling results
            import os
            model_size = os.path.getsize(model_path)
            return {
                "model_size_bytes": model_size,
                "estimated_memory_usage": model_size * 2,
                "estimated_inference_time_ms": 100,
                "estimated_throughput": 10.0,
            }