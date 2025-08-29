// Demonstration of the Blitzed benchmarking suite
use blitzed_core::benchmarking::{
    BenchmarkConfig, CompetitiveFramework, HardwarePlatform, StandardModel,
    BenchmarkAnalysis,
};
use blitzed_core::{BenchmarkSuite, init};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Blitzed
    init()?;

    println!("🚀 Blitzed Performance Benchmarking Suite Demo");
    println!("===============================================");

    // Create a comprehensive benchmark configuration
    let config = BenchmarkConfig {
        frameworks: vec![
            CompetitiveFramework::Blitzed,
            CompetitiveFramework::TensorFlowLite,
            CompetitiveFramework::OnnxRuntime,
            CompetitiveFramework::PyTorchMobile,
        ],
        platforms: vec![
            HardwarePlatform::X86Desktop,
            HardwarePlatform::RaspberryPi4,
            HardwarePlatform::ESP32,
        ],
        models: vec![
            StandardModel::MobileNetV2,
            StandardModel::MobileNetV3Small,
            StandardModel::ResNet18,
            StandardModel::EfficientNetB0,
        ],
        warmup_runs: 3,
        benchmark_runs: 10,
        timeout: Duration::from_secs(30),
        validate_accuracy: true,
        measure_power: true,
    };

    println!("📊 Benchmark Configuration:");
    println!("   Frameworks: {:?}", config.frameworks.iter().map(|f| f.name()).collect::<Vec<_>>());
    println!("   Platforms: {:?}", config.platforms.iter().map(|p| p.name()).collect::<Vec<_>>());
    println!("   Models: {:?}", config.models.iter().map(|m| m.name()).collect::<Vec<_>>());
    println!("   Benchmark runs: {}", config.benchmark_runs);
    println!();

    // Create and run benchmark suite
    println!("🏃 Running benchmarks...");
    let suite = BenchmarkSuite::new(config);
    let summary = suite.run_benchmarks()?;

    // Display results
    println!("✅ Benchmarking completed!");
    println!();
    println!("{}", summary.summary());
    println!();

    // Show competitive advantages
    println!("🏆 Blitzed Competitive Analysis:");
    let advantages = summary.blitzed_advantage();
    for (metric, advantage) in advantages {
        println!("   {}: {:.2}x advantage", metric, advantage);
    }
    println!();

    // Find best performing framework for different metrics
    let metrics = ["latency", "memory", "size", "accuracy"];
    println!("🎯 Best Performing Framework by Metric:");
    for metric in &metrics {
        if let Some(framework) = summary.best_framework(metric) {
            println!("   Best for {}: {}", metric, framework.name());
        }
    }
    println!();

    // Show detailed statistics
    println!("📈 Detailed Statistics:");
    println!("   Total benchmarks: {}", summary.statistics.total_benchmarks);
    println!("   Successful benchmarks: {}", summary.statistics.successful_benchmarks);
    println!("   Success rate: {:.1}%", 
        (summary.statistics.successful_benchmarks as f64 / summary.statistics.total_benchmarks as f64) * 100.0
    );
    println!("   Average Blitzed speedup: {:.2}x", summary.statistics.average_blitzed_speedup);
    println!("   Average compression: {:.2}x", summary.statistics.average_blitzed_compression);
    println!("   Accuracy retention: {:.1}%", summary.statistics.average_accuracy_retention * 100.0);
    println!("   Total duration: {:.2}s", summary.total_duration.as_secs_f64());

    if let Some(best_platform) = &summary.statistics.best_platform {
        println!("   Best platform for Blitzed: {}", best_platform.name());
    }

    if let Some(challenging_model) = &summary.statistics.most_challenging_model {
        println!("   Most challenging model: {}", challenging_model.name());
    }

    println!();
    println!("🎉 Benchmark demo completed successfully!");

    Ok(())
}