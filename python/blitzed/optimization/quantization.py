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
Quantization optimization for model compression.

This module provides Python interfaces for quantizing neural networks
to reduce model size and improve inference speed on edge devices.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class QuantizationType(Enum):
    """Supported quantization types."""
    INT8 = "int8"
    INT4 = "int4" 
    BINARY = "binary"
    MIXED = "mixed"


@dataclass
class QuantizationConfig:
    """Configuration for quantization optimization."""
    quantization_type: QuantizationType = QuantizationType.INT8
    calibration_dataset_size: int = 100
    symmetric: bool = True
    per_channel: bool = True
    skip_sensitive_layers: bool = True
    accuracy_threshold: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "quantization_type": self.quantization_type.value,
            "calibration_dataset_size": self.calibration_dataset_size,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "skip_sensitive_layers": self.skip_sensitive_layers,
            "accuracy_threshold": self.accuracy_threshold,
        }


class Quantizer:
    """High-level quantization interface."""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer with configuration.
        
        Args:
            config: Quantization configuration. If None, uses default config.
        """
        self.config = config or QuantizationConfig()
    
    def quantize(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Quantize a model file.
        
        Args:
            model_path: Path to input model file
            output_path: Path for quantized model output
            
        Returns:
            Path to quantized model
            
        Raises:
            ValueError: If model format is not supported
            RuntimeError: If quantization fails
        """
        try:
            # Import core functions
            from .._core import quantize_model
            
            if output_path is None:
                # Generate output path based on input
                import os
                name, ext = os.path.splitext(model_path)
                output_path = f"{name}_quantized{ext}"
            
            # Call Rust implementation
            result_path = quantize_model(
                model_path, 
                output_path, 
                self.config.to_dict()
            )
            
            return result_path
            
        except ImportError:
            raise RuntimeError("Blitzed core extension not available")
        except Exception as e:
            raise RuntimeError(f"Quantization failed: {e}")
    
    def estimate_impact(self, model_path: str) -> Dict[str, float]:
        """
        Estimate the impact of quantization without applying it.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with estimated impact metrics
        """
        try:
            from .._core import estimate_quantization_impact
            return estimate_quantization_impact(model_path, self.config.to_dict())
        except ImportError:
            # Fallback estimates
            impact_map = {
                QuantizationType.INT8: {"size_reduction": 0.75, "accuracy_loss": 2.0},
                QuantizationType.INT4: {"size_reduction": 0.875, "accuracy_loss": 5.0},
                QuantizationType.BINARY: {"size_reduction": 0.96875, "accuracy_loss": 15.0},
                QuantizationType.MIXED: {"size_reduction": 0.6, "accuracy_loss": 3.0},
            }
            return impact_map.get(
                self.config.quantization_type,
                {"size_reduction": 0.5, "accuracy_loss": 10.0}
            )