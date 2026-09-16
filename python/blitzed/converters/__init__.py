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

"""Model format converters for Blitzed.

Optional format-specific converters are imported when their implementation is
present. This keeps the base Python package usable in lightweight installs.
"""

from .base import BaseConverter, ConversionConfig
from .tflite import TFLiteConverter

__all__ = ["BaseConverter", "ConversionConfig", "TFLiteConverter"]

try:
    from .onnx import ONNXConverter
except ModuleNotFoundError:
    ONNXConverter = None
else:
    __all__.append("ONNXConverter")

try:
    from .pytorch import PyTorchConverter
except ModuleNotFoundError:
    PyTorchConverter = None
else:
    __all__.append("PyTorchConverter")

try:
    from .tensorflow import TensorFlowConverter
except ModuleNotFoundError:
    TensorFlowConverter = None
else:
    __all__.append("TensorFlowConverter")
