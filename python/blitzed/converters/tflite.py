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
TensorFlow Lite converter for Blitzed.

This module provides functionality to convert TensorFlow, Keras, and SavedModel
formats to TensorFlow Lite (.tflite) for edge device deployment.
"""

from typing import Optional
from .base import BaseConverter, ConversionConfig


class TFLiteConverter(BaseConverter):
    """
    Converter for TensorFlow Lite format.
    
    This converter transforms TensorFlow/Keras models and SavedModels
    into TensorFlow Lite format with optional quantization for edge deployment.
    """
    
    def convert(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a model to TensorFlow Lite format.
        
        Args:
            model_path: Path to input model file (.h5, .pb, or SavedModel directory)
            output_path: Path for .tflite output. If None, creates based on input name
            
        Returns:
            Path to converted .tflite model
            
        Raises:
            ValueError: If model format is not supported for TFLite conversion
            RuntimeError: If conversion fails
        """
        if not self.validate_model(model_path):
            raise ValueError(f"Invalid model path: {model_path}")
            
        try:
            # Import core functions
            from ..._core import convert_to_tflite
            
            if output_path is None:
                import os
                name, _ = os.path.splitext(model_path)
                output_path = f"{name}.tflite"
            
            # Call Rust implementation
            result_path = convert_to_tflite(
                model_path,
                output_path,
                self.config.__dict__
            )
            
            return result_path
            
        except ImportError:
            raise RuntimeError("Blitzed core extension not available for TFLite conversion")
        except Exception as e:
            raise RuntimeError(f"TFLite conversion failed: {e}")


# Convenience functions
def convert_to_tflite(model_path: str, 
                     output_path: Optional[str] = None,
                     optimize: bool = True,
                     quantize: bool = False) -> str:
    """
    Convenience function to convert a model to TensorFlow Lite.
    
    Args:
        model_path: Path to input model file
        output_path: Path for .tflite output
        optimize: Whether to apply default optimizations
        quantize: Whether to apply integer quantization
        
    Returns:
        Path to converted .tflite model
    """
    config = ConversionConfig(
        optimize=optimize,
        integer_quantization=quantize
    )
    converter = TFLiteConverter(config)
    return converter.convert(model_path, output_path)