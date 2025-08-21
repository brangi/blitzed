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
Blitzed - High-performance edge AI optimization framework

This package provides tools for optimizing and deploying machine learning models
on edge devices including microcontrollers, mobile devices, and embedded systems.

Key features:
- Model compression (quantization, pruning, knowledge distillation)
- Hardware-aware optimization
- Cross-platform deployment code generation
- Performance profiling and validation
"""

__version__ = "0.1.0"
__author__ = "Gibran Rodriguez <brangi000@gmail.com>"

# Import core functionality
from . import optimization
from . import converters
from . import targets
from . import cli

# Import main classes for convenience
try:
    from ._core import init as _init_core
    _init_core()
except ImportError:
    # Fallback if Rust extension is not available
    import warnings
    warnings.warn(
        "Blitzed core extension not available. Some features may be limited.",
        ImportWarning
    )

# Public API
__all__ = [
    "optimization",
    "converters", 
    "targets",
    "cli",
    "__version__",
]