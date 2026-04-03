import pytest
import numpy as np
from blitzed.optimization.quantization import estimate_impact
from unittest.mock import patch


def test_estimate_impact_happy_path():
    # Test with supported PyTorch model
    class MockPyTorchModel:
        def __init__(self):
            self.size = 1000000  # 1MB

    model = MockPyTorchModel()
    impact = estimate_impact(model)
    assert isinstance(impact, dict), "Impact should be a dictionary"
    assert 'size_reduction' in impact
    assert 0 < impact['size_reduction'] < 1
    assert impact['accuracy_impact'] == 0
    assert impact['estimated_quantized_size'] < model.size


def test_estimate_impact_onnx_model():
    # Test with supported ONNX model
    class MockOnnxDomain:
        def __init__(self):
            self.model_size = 500000  # 500KB

    class MockONNXModel:
        domain = MockOnnxDomain()

    model = MockONNXModel()
    impact = estimate_impact(model)
    assert isinstance(impact, dict)
    assert 'accuracy_impact' in impact


def test_unsupported_model():
    # Test ValueError for unsupported model type
    class MockUnsupportedModel:
        pass

    model = MockUnsupportedModel()
    with pytest.raises(ValueError, match="Unsupported model format: MockUnsupportedModel"):
        estimate_impact(model)

@patch('blitzed.optimization.quantization._core', None)
def test_missing_core_extension():
    # Test RuntimeError when core is missing
    with pytest.raises(RuntimeError, match="Blitzed core extension not available"):
        estimate_impact(None)


def test_edge_case_empty_model():
    # Test error handling for invalid model input
    class MockBrokenModel:
        size = None

    model = MockBrokenModel()
    with pytest.raises(ValueError, match="Invalid model size: NoneType"):
        estimate_impact(model)


def test_accuracy_impact_range():
    # Test the accuracy impact calculation boundaries
    class MockPyTorchModel:
        def __init__(self):
            self.size = 1000000

    model = MockPyTorchModel()
    impact = estimate_impact(model)
    # Assuming accuracy impact is calculated as 0 for PyTorch fallback
    assert impact['accuracy_impact'] == 0

    # Add another test with higher impact scenario if quantization actually happens
