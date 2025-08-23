// Copyright 2025 Gibran Rodriguez <brangi000@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Performance benchmarks for Blitzed optimization algorithms

use blitzed_core::{
    model::{LayerInfo, Model, ModelData, ModelFormat, ModelInfo},
    optimization::{OptimizationTechnique, QuantizationConfig, Quantizer},
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn create_benchmark_model(size_mb: usize) -> Model {
    let model_size = size_mb * 1024 * 1024; // Convert MB to bytes
    let param_count = model_size / 4; // Assume 4 bytes per parameter (FP32)

    let info = ModelInfo {
        format: ModelFormat::Onnx,
        input_shapes: vec![vec![1, 3, 224, 224]],
        output_shapes: vec![vec![1, 1000]],
        parameter_count: param_count,
        model_size_bytes: model_size,
        operations_count: param_count / 2, // Rough FLOPS estimate
        layers: vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "Conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 64, 112, 112],
                parameter_count: 9408,
                flops: 118013952,
            },
            LayerInfo {
                name: "fc".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: vec![1, 2048],
                output_shape: vec![1, 1000],
                parameter_count: 2048000,
                flops: 2048000,
            },
        ],
    };

    Model {
        info,
        data: ModelData::Raw(vec![0u8; 1000]),
    }
}

fn bench_int8_quantization(c: &mut Criterion) {
    let model = create_benchmark_model(10); // 10MB model
    let config = QuantizationConfig::default();
    let quantizer = Quantizer::new(config.clone());

    c.bench_function("int8_quantization_10mb", |b| {
        b.iter(|| {
            let impact = quantizer.estimate_impact(black_box(&model), black_box(&config));
            black_box(impact)
        })
    });
}

fn bench_quantization_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_scaling");

    for size_mb in [1, 5, 10, 25, 50].iter() {
        let model = create_benchmark_model(*size_mb);
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config.clone());

        group.bench_with_input(format!("{}mb_model", size_mb), size_mb, |b, _| {
            b.iter(|| {
                let impact = quantizer.estimate_impact(black_box(&model), black_box(&config));
                black_box(impact)
            })
        });
    }
    group.finish();
}

fn bench_model_analysis(c: &mut Criterion) {
    let model = create_benchmark_model(25); // ResNet-50 sized model

    c.bench_function("model_analysis_resnet50", |b| {
        b.iter(|| {
            let info = black_box(&model).info();
            let memory_usage = black_box(&model).estimate_memory_usage();
            black_box((info, memory_usage))
        })
    });
}

fn bench_hardware_targeting(c: &mut Criterion) {
    use blitzed_core::targets::{esp32::Esp32Target, raspberry_pi::RaspberryPiTarget};

    let model = create_benchmark_model(10);
    let esp32_target = Esp32Target::new();
    let rpi_target = RaspberryPiTarget::new();

    let mut group = c.benchmark_group("hardware_targeting");

    group.bench_function("esp32_inference_estimation", |b| {
        b.iter(|| {
            let estimate = esp32_target.estimate_inference_latency(black_box(&model));
            black_box(estimate)
        })
    });

    group.bench_function("raspberry_pi_inference_estimation", |b| {
        b.iter(|| {
            let estimate = rpi_target.estimate_inference_latency(black_box(&model));
            black_box(estimate)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_int8_quantization,
    bench_quantization_scaling,
    bench_model_analysis,
    bench_hardware_targeting
);
criterion_main!(benches);
