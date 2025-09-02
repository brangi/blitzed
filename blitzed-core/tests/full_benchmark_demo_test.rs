// Full demonstration test of the benchmarking suite
use blitzed_core::benchmarking::{
    BenchmarkAnalysis, BenchmarkConfig, CompetitiveFramework, HardwarePlatform, StandardModel,
};
use blitzed_core::BenchmarkSuite;
use std::time::Duration;

#[test]
#[ignore = "Expensive test - comprehensive benchmark, run with --ignored or slow-tests feature"]
fn test_comprehensive_benchmark_demo() {
    // Create a comprehensive benchmark configuration
    let config = BenchmarkConfig {
        frameworks: vec![
            CompetitiveFramework::Blitzed,
            CompetitiveFramework::TensorFlowLite,
            CompetitiveFramework::OnnxRuntime,
        ],
        platforms: vec![HardwarePlatform::X86Desktop, HardwarePlatform::ESP32],
        models: vec![StandardModel::MobileNetV2, StandardModel::ResNet18],
        warmup_runs: 2,
        benchmark_runs: 3,
        timeout: Duration::from_secs(5),
        validate_accuracy: true,
        measure_power: false,
    };

    println!("📊 Running comprehensive benchmark test...");
    println!(
        "   Frameworks: {:?}",
        config
            .frameworks
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );
    println!(
        "   Platforms: {:?}",
        config
            .platforms
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
    );
    println!(
        "   Models: {:?}",
        config.models.iter().map(|m| m.name()).collect::<Vec<_>>()
    );

    // Create and run benchmark suite
    let suite = BenchmarkSuite::new(config);
    let summary = suite
        .run_benchmarks()
        .expect("Benchmark should run successfully");

    // Verify comprehensive results
    assert!(!summary.results.is_empty(), "Should have benchmark results");
    println!("✅ Generated {} benchmark results", summary.results.len());

    // Should have results for each framework
    let blitzed_results = summary
        .results
        .iter()
        .filter(|r| r.framework == CompetitiveFramework::Blitzed)
        .count();
    let tflite_results = summary
        .results
        .iter()
        .filter(|r| r.framework == CompetitiveFramework::TensorFlowLite)
        .count();
    let onnx_results = summary
        .results
        .iter()
        .filter(|r| r.framework == CompetitiveFramework::OnnxRuntime)
        .count();

    assert!(blitzed_results > 0, "Should have Blitzed results");
    assert!(tflite_results > 0, "Should have TensorFlow Lite results");
    assert!(onnx_results > 0, "Should have ONNX Runtime results");

    println!("   Blitzed results: {}", blitzed_results);
    println!("   TensorFlow Lite results: {}", tflite_results);
    println!("   ONNX Runtime results: {}", onnx_results);

    // Test summary functionality
    let summary_text = summary.summary();
    assert!(
        summary_text.contains("Benchmark Summary"),
        "Summary should contain header"
    );
    assert!(
        summary_text.contains("Total tests"),
        "Summary should contain test count"
    );
    println!("📋 Summary generated successfully");

    // Test competitive advantage analysis
    let advantages = summary.blitzed_advantage();
    assert!(
        !advantages.is_empty(),
        "Should have competitive advantages calculated"
    );
    println!("🏆 Calculated {} competitive advantages", advantages.len());

    for (metric, advantage) in &advantages {
        println!("   {}: {:.2}x", metric, advantage);
        assert!(*advantage > 0.0, "Advantage should be positive");
    }

    // Test best framework analysis
    let metrics = ["latency", "memory", "accuracy"];
    for metric in &metrics {
        if let Some(framework) = summary.best_framework(metric) {
            println!("   Best for {}: {}", metric, framework.name());
        }
    }

    // Verify statistics
    assert!(
        summary.statistics.total_benchmarks > 0,
        "Should have total benchmark count"
    );
    assert!(
        summary.statistics.successful_benchmarks > 0,
        "Should have successful benchmarks"
    );
    assert!(
        summary.statistics.average_blitzed_speedup > 0.0,
        "Should have speedup calculated"
    );
    assert!(
        summary.statistics.average_blitzed_compression > 0.0,
        "Should have compression calculated"
    );
    assert!(
        summary.statistics.average_accuracy_retention >= 0.0,
        "Should have accuracy retention"
    );

    println!("📈 Statistics validation passed");
    println!(
        "   Total benchmarks: {}",
        summary.statistics.total_benchmarks
    );
    println!(
        "   Success rate: {:.1}%",
        (summary.statistics.successful_benchmarks as f64
            / summary.statistics.total_benchmarks as f64)
            * 100.0
    );
    println!(
        "   Average Blitzed speedup: {:.2}x",
        summary.statistics.average_blitzed_speedup
    );
    println!(
        "   Average compression: {:.2}x",
        summary.statistics.average_blitzed_compression
    );
    println!(
        "   Accuracy retention: {:.1}%",
        summary.statistics.average_accuracy_retention * 100.0
    );

    // Verify timing
    assert!(
        summary.total_duration.as_secs() < 60,
        "Benchmark should complete in reasonable time"
    );
    println!(
        "⏱️  Completed in {:.2}s",
        summary.total_duration.as_secs_f64()
    );

    println!("🎉 Comprehensive benchmark test completed successfully!");
}
