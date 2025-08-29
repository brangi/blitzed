// Integration test for the benchmark suite
use blitzed_core::benchmarking::{
    BenchmarkConfig, CompetitiveFramework, HardwarePlatform, StandardModel,
};
use blitzed_core::BenchmarkSuite;
use std::time::Duration;

#[test]
fn test_benchmark_suite_creation() {
    let config = BenchmarkConfig {
        frameworks: vec![
            CompetitiveFramework::Blitzed,
            CompetitiveFramework::TensorFlowLite,
        ],
        platforms: vec![HardwarePlatform::X86Desktop, HardwarePlatform::ESP32],
        models: vec![StandardModel::MobileNetV2],
        warmup_runs: 2,
        benchmark_runs: 5,
        timeout: Duration::from_secs(10),
        validate_accuracy: false,
        measure_power: false,
    };

    let _suite = BenchmarkSuite::new(config);

    // Verify suite was created successfully - we can't access private fields
    // but we can verify it was created without error
    // Suite creation should not panic or error
}

#[test]
fn test_benchmark_config_default() {
    let config = BenchmarkConfig::default();

    // Verify default configuration is sensible
    assert!(config.frameworks.contains(&CompetitiveFramework::Blitzed));
    assert!(config
        .frameworks
        .contains(&CompetitiveFramework::TensorFlowLite));
    assert!(!config.models.is_empty());
    assert!(!config.platforms.is_empty());
    assert!(config.warmup_runs > 0);
    assert!(config.benchmark_runs > 0);
}

#[test]
fn test_simple_benchmark_run() {
    // Use minimal configuration for quick test
    let config = BenchmarkConfig {
        frameworks: vec![CompetitiveFramework::Blitzed],
        platforms: vec![HardwarePlatform::X86Desktop],
        models: vec![StandardModel::MobileNetV2],
        warmup_runs: 1,
        benchmark_runs: 2,
        timeout: Duration::from_secs(5),
        validate_accuracy: false,
        measure_power: false,
    };

    let suite = BenchmarkSuite::new(config);

    // Run the benchmark suite
    let result = suite.run_benchmarks();

    // Verify the benchmark ran successfully
    assert!(
        result.is_ok(),
        "Benchmark should run successfully: {:?}",
        result.err()
    );

    let summary = result.unwrap();
    assert!(
        !summary.results.is_empty(),
        "Should have at least one result"
    );
    assert!(
        summary.total_duration.as_secs() < 10,
        "Should complete quickly"
    );

    // Check that we have results for the expected configuration
    let blitzed_results: Vec<_> = summary
        .results
        .iter()
        .filter(|r| r.framework == CompetitiveFramework::Blitzed)
        .collect();
    assert!(!blitzed_results.is_empty(), "Should have Blitzed results");
}
