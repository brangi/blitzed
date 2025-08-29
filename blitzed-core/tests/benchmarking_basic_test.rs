// Basic benchmarking functionality test
use blitzed_core::benchmarking::{CompetitiveFramework, HardwarePlatform, StandardModel};

#[test]
fn test_basic_enums() {
    // Test that enums work properly
    let model = StandardModel::MobileNetV2;
    assert_eq!(model.name(), "MobileNetV2");

    let framework = CompetitiveFramework::Blitzed;
    assert_eq!(framework.name(), "Blitzed");

    let platform = HardwarePlatform::ESP32;
    assert_eq!(platform.name(), "ESP32");
}

#[test]
fn test_model_properties() {
    let model = StandardModel::MobileNetV2;
    assert_eq!(model.name(), "MobileNetV2");
    assert_eq!(model.input_shape(), vec![1, 3, 224, 224]);
    assert!(model.expected_size_mb().0 > 0.0);
}
