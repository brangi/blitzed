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
Base classes for model format converters.

This module defines the abstract base class and configuration structures
that all format-specific converters inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    optimize: bool = True
    dynamic_range_quantization: bool = False
    integer_quantization: bool = False
    target_platform: Optional[str] = None
    representative_dataset: Optional[Any] = None
    inference_input_type: str = "float32"
    inference_output_type: str = "float32"


class BaseConverter(ABC):
    """
    Abstract base class for model format converters.
    
    All format-specific converters should inherit from this class and
    implement the convert method.
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """
        Initialize converter with configuration.
        
        Args:
            config: Conversion configuration. If None, uses default config.
        """
        self.config = config or ConversionConfig()
    
    @abstractmethod
    def convert(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a model to the target format.
        
        Args:
            model_path: Path to input model file
            output_path: Path for converted model output
            
        Returns:
            Path to converted model
            
        Raises:
            ValueError: If model format is not supported
            RuntimeError: If conversion fails
        """
        pass
    
    def validate_model(self, model_path: str) -> bool:
        """
        Validate that the model can be converted.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model is valid for conversion
        """
        # Basic file existence check
        import os
        return os.path.exists(model_path) and os.path.isfile(model_path)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model information
        """
        import os
        if not self.validate_model(model_path):
            return {}
            
        return {
            "path": model_path,
            "size_bytes": os.path.getsize(model_path),
            "filename": os.path.basename(model_path),
            "extension": os.path.splitext(model_path)[1]
        }