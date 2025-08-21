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

//! Python bindings for Blitzed Core
//! 
//! This crate provides PyO3-based Python bindings for the Blitzed edge AI
//! optimization framework, exposing core Rust functionality to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

use blitzed_core::{
    self as core,
    Config, Model, Optimizer, OptimizationConfig as CoreOptConfig,
    optimization::{QuantizationConfig, QuantizationType, Quantizer},
    targets::TargetRegistry,
    codegen::UniversalCodeGenerator,
    profiler::{Profiler, ProfilingConfig},
};

/// Initialize the Blitzed core library
#[pyfunction]
fn init() -> PyResult<()> {
    core::init().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

/// Get Blitzed core version
#[pyfunction]
fn version() -> &'static str {
    core::VERSION
}

/// Load and analyze a model file
#[pyfunction]
fn load_model(path: String) -> PyResult<PyDict> {
    let model = Model::load(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        let info = model.info();
        
        dict.set_item("format", format!("{:?}", info.format))?;
        dict.set_item("model_size_bytes", info.model_size_bytes)?;
        dict.set_item("parameter_count", info.parameter_count)?;
        dict.set_item("operations_count", info.operations_count)?;
        
        // Convert shapes to Python lists
        let input_shapes = PyList::new(py, &info.input_shapes);
        let output_shapes = PyList::new(py, &info.output_shapes);
        dict.set_item("input_shapes", input_shapes)?;
        dict.set_item("output_shapes", output_shapes)?;
        
        // Memory estimation
        dict.set_item("estimated_memory_usage", model.estimate_memory_usage())?;
        
        Ok(dict.into())
    })
}

/// Quantize a model with specified configuration
#[pyfunction]
fn quantize_model(input_path: String, output_path: String, config: &PyDict) -> PyResult<String> {
    let model = Model::load(&input_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    // Parse quantization configuration from Python dict
    let quant_type = match config.get_item("quantization_type")? {
        Some(qt) => match qt.extract::<String>()? {
            qt if qt == "int8" => QuantizationType::Int8,
            qt if qt == "int4" => QuantizationType::Int4,
            qt if qt == "binary" => QuantizationType::Binary,
            qt if qt == "mixed" => QuantizationType::Mixed,
            _ => QuantizationType::Int8,
        },
        None => QuantizationType::Int8,
    };
    
    let quant_config = QuantizationConfig {
        quantization_type: quant_type,
        calibration_dataset_size: config.get_item("calibration_dataset_size")?
            .map(|v| v.extract().unwrap_or(100))
            .unwrap_or(100),
        symmetric: config.get_item("symmetric")?
            .map(|v| v.extract().unwrap_or(true))
            .unwrap_or(true),
        per_channel: config.get_item("per_channel")?
            .map(|v| v.extract().unwrap_or(true))
            .unwrap_or(true),
        skip_sensitive_layers: config.get_item("skip_sensitive_layers")?
            .map(|v| v.extract().unwrap_or(true))
            .unwrap_or(true),
        accuracy_threshold: config.get_item("accuracy_threshold")?
            .map(|v| v.extract().unwrap_or(5.0))
            .unwrap_or(5.0),
    };
    
    let quantizer = Quantizer::new(quant_config.clone());
    let _quantized = quantizer.quantize_post_training(&model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // TODO: Save quantized model to output_path
    // For now, just return the output path
    Ok(output_path)
}

/// Estimate quantization impact without applying it
#[pyfunction]
fn estimate_quantization_impact(model_path: String, config: &PyDict) -> PyResult<PyDict> {
    let model = Model::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    // Parse config (simplified)
    let quant_type = match config.get_item("quantization_type")? {
        Some(qt) => match qt.extract::<String>()? {
            qt if qt == "int8" => QuantizationType::Int8,
            qt if qt == "int4" => QuantizationType::Int4,
            qt if qt == "binary" => QuantizationType::Binary,
            qt if qt == "mixed" => QuantizationType::Mixed,
            _ => QuantizationType::Int8,
        },
        None => QuantizationType::Int8,
    };
    
    let quant_config = QuantizationConfig {
        quantization_type: quant_type,
        ..Default::default()
    };
    
    let quantizer = Quantizer::new(quant_config);
    let impact = quantizer.estimate_impact(&model, &quant_config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("size_reduction", impact.size_reduction)?;
        dict.set_item("speed_improvement", impact.speed_improvement)?;
        dict.set_item("accuracy_loss", impact.accuracy_loss)?;
        dict.set_item("memory_reduction", impact.memory_reduction)?;
        Ok(dict.into())
    })
}

/// Optimize a model with full optimization pipeline
#[pyfunction]
fn optimize_model(input_path: String, output_path: String, config: &PyDict) -> PyResult<PyDict> {
    // Load target configuration if specified
    let target = config.get_item("target")?
        .map(|t| t.extract::<String>())
        .transpose()?;
    
    let blitzed_config = if let Some(target_name) = target {
        Config::preset(&target_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        Config::default()
    };
    
    let model = Model::load(&input_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let optimizer = Optimizer::new(blitzed_config);
    let result = optimizer.optimize(&model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // TODO: Save optimized model to output_path
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("original_size", result.original_size)?;
        dict.set_item("optimized_size", result.optimized_size)?;
        dict.set_item("compression_ratio", result.compression_ratio)?;
        dict.set_item("estimated_accuracy_loss", result.estimated_accuracy_loss)?;
        dict.set_item("estimated_speedup", result.estimated_speedup)?;
        dict.set_item("optimization_time_ms", result.optimization_time_ms)?;
        
        let techniques = PyList::new(py, &result.techniques_applied);
        dict.set_item("techniques_applied", techniques)?;
        
        Ok(dict.into())
    })
}

/// Estimate optimization impact for full pipeline
#[pyfunction]
fn estimate_optimization_impact(model_path: String, config: &PyDict) -> PyResult<PyDict> {
    let target = config.get_item("target")?
        .map(|t| t.extract::<String>())
        .transpose()?;
    
    let blitzed_config = if let Some(target_name) = target {
        Config::preset(&target_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        Config::default()
    };
    
    let model = Model::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let optimizer = Optimizer::new(blitzed_config);
    let impact = optimizer.estimate_impact(&model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("size_reduction", impact.size_reduction)?;
        dict.set_item("speed_improvement", impact.speed_improvement)?;
        dict.set_item("accuracy_loss", impact.accuracy_loss)?;
        dict.set_item("memory_reduction", impact.memory_reduction)?;
        Ok(dict.into())
    })
}

/// Get optimization recommendations
#[pyfunction]
fn get_optimization_recommendations(model_path: String, config: &PyDict) -> PyResult<Vec<String>> {
    let target = config.get_item("target")?
        .map(|t| t.extract::<String>())
        .transpose()?;
    
    let blitzed_config = if let Some(target_name) = target {
        Config::preset(&target_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        Config::default()
    };
    
    let model = Model::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let optimizer = Optimizer::new(blitzed_config);
    let recommendations = optimizer.recommend(&model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(recommendations)
}

/// Profile model performance
#[pyfunction]
fn profile_model(model_path: String, config: &PyDict) -> PyResult<PyDict> {
    let model = Model::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let profiler = Profiler::new(ProfilingConfig::default());
    let metrics = profiler.profile_inference(&model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("model_size_bytes", model.info().model_size_bytes)?;
        dict.set_item("estimated_memory_usage", metrics.memory_usage_bytes)?;
        dict.set_item("estimated_inference_time_ms", metrics.inference_time_ms)?;
        dict.set_item("estimated_throughput", metrics.throughput)?;
        Ok(dict.into())
    })
}

/// Generate deployment code for target platform
#[pyfunction]
fn generate_deployment_code(model_path: String, output_dir: String, target: String) -> PyResult<PyDict> {
    let model = Model::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let codegen = UniversalCodeGenerator::new();
    let output_path = PathBuf::from(output_dir);
    
    let generated = codegen.generate(&target, &model, &output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("implementation_file", generated.implementation_file)?;
        
        if let Some(header) = generated.header_file {
            dict.set_item("header_file", header)?;
        }
        
        if let Some(example) = generated.example_file {
            dict.set_item("example_file", example)?;
        }
        
        if let Some(build_config) = generated.build_config {
            dict.set_item("build_config", build_config)?;
        }
        
        let deps = PyList::new(py, &generated.dependencies);
        dict.set_item("dependencies", deps)?;
        
        Ok(dict.into())
    })
}

/// List supported target platforms
#[pyfunction]
fn list_targets() -> Vec<String> {
    let registry = TargetRegistry::new();
    registry.list_targets().into_iter().map(|s| s.to_string()).collect()
}

/// Python module definition
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_model, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_quantization_impact, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_model, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_optimization_impact, m)?)?;
    m.add_function(wrap_pyfunction!(get_optimization_recommendations, m)?)?;
    m.add_function(wrap_pyfunction!(profile_model, m)?)?;
    m.add_function(wrap_pyfunction!(generate_deployment_code, m)?)?;
    m.add_function(wrap_pyfunction!(list_targets, m)?)?;
    
    // Add version constant
    m.add("VERSION", core::VERSION)?;
    
    Ok(())
}