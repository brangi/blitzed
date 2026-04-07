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

//! Native inference engine for neural networks
//!
//! This module provides a complete inference engine that can execute
//! neural network models using the core tensor operations. It supports:
//!
//! - Graph execution with automatic memory management
//! - Multiple optimization levels (speed vs memory)
//! - Support for quantized and pruned models
//! - Batch processing and parallel execution
//! - Memory pooling for edge device deployment

use crate::error::{BlitzedError, Result};
use crate::model::{LayerInfo, Model};
use crate::tensor_ops::{Tensor, TensorData, TensorOps};
use std::collections::HashMap;

/// Configuration for the inference engine
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Batch size for processing multiple inputs
    pub batch_size: usize,
    /// Memory optimization level (0=speed, 1=balanced, 2=memory)
    pub memory_optimization: u8,
    /// Enable parallel processing for operations
    pub enable_parallel: bool,
    /// Maximum memory pool size in MB
    pub max_memory_pool_mb: f32,
    /// Enable layer fusion optimizations
    pub enable_fusion: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            memory_optimization: 1,
            enable_parallel: true,
            max_memory_pool_mb: 32.0,
            enable_fusion: true,
        }
    }
}

/// Execution graph node representing a neural network layer
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Layer information from the model
    pub layer_info: LayerInfo,
    /// Node type for execution
    pub node_type: NodeType,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor name
    pub output: String,
    /// Layer parameters (weights, biases)
    pub parameters: HashMap<String, Tensor>,
    /// Operation-specific configuration
    pub config: NodeConfig,
}

/// Types of nodes in the execution graph
#[derive(Debug, Clone)]
pub enum NodeType {
    /// Convolution layer
    Conv2D {
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    },
    /// Linear/Dense layer
    Linear,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Max pooling
    MaxPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Average pooling
    AvgPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Batch normalization
    BatchNorm,
    /// Element-wise addition
    Add,
    /// Flatten operation
    Flatten,
    /// Reshape operation
    Reshape { target_shape: Vec<usize> },
}

/// Node-specific configuration
#[derive(Debug, Clone, Default)]
pub struct NodeConfig {
    /// Whether this node should use quantized execution
    pub quantized: bool,
    /// Custom memory allocation hint
    pub memory_hint: Option<usize>,
    /// Whether to fuse with next operation
    pub fuse_next: bool,
}

/// Execution graph for neural network inference
#[derive(Debug)]
pub struct ExecutionGraph {
    /// Ordered list of nodes for execution
    pub nodes: Vec<GraphNode>,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications  
    pub outputs: Vec<TensorSpec>,
    /// Intermediate tensor cache
    pub tensor_cache: HashMap<String, Tensor>,
}

/// Specification for input/output tensors
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Expected shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
}

/// Supported data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Int8,
    Int4,
    Binary,
}

/// Native inference engine
pub struct InferenceEngine {
    /// Engine configuration
    config: InferenceConfig,
    /// Execution graph
    graph: Option<ExecutionGraph>,
    /// Memory pool for tensor allocation (reserved for future optimization)
    #[allow(dead_code)]
    memory_pool: MemoryPool,
    /// Performance statistics
    stats: InferenceStats,
}

/// Memory pool for efficient tensor allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Pre-allocated memory blocks
    blocks: Vec<Vec<u8>>,
    /// Available block indices
    available: Vec<usize>,
    /// Block size in bytes
    block_size: usize,
    /// Maximum number of blocks
    max_blocks: usize,
}

/// Performance statistics for inference
#[derive(Debug, Default)]
pub struct InferenceStats {
    /// Total inference time in milliseconds
    pub total_time_ms: f32,
    /// Number of inferences performed
    pub inference_count: u64,
    /// Memory usage statistics
    pub peak_memory_mb: f32,
    /// Per-layer timing breakdown
    pub layer_times_ms: HashMap<String, f32>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_size_mb: f32, block_size: usize) -> Self {
        let max_blocks = (max_size_mb * 1024.0 * 1024.0) as usize / block_size;

        Self {
            blocks: Vec::new(),
            available: Vec::new(),
            block_size,
            max_blocks,
        }
    }

    /// Allocate a memory block
    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(block_id) = self.available.pop() {
            Some(block_id)
        } else if self.blocks.len() < self.max_blocks {
            let block_id = self.blocks.len();
            self.blocks.push(vec![0u8; self.block_size]);
            Some(block_id)
        } else {
            None
        }
    }

    /// Deallocate a memory block
    pub fn deallocate(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.available.push(block_id);
        }
    }
}

impl ExecutionGraph {
    /// Create a new execution graph from a model
    pub fn from_model(model: &Model) -> Result<Self> {
        let mut nodes = Vec::new();
        let tensor_cache = HashMap::new();

        // Create input specifications
        let inputs = model
            .info
            .input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| TensorSpec {
                name: format!("input_{}", i),
                shape: shape.iter().map(|&x| x as usize).collect(),
                dtype: DataType::Float32,
            })
            .collect();

        // Create output specifications
        let outputs = model
            .info
            .output_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| TensorSpec {
                name: format!("output_{}", i),
                shape: shape.iter().map(|&x| x as usize).collect(),
                dtype: DataType::Float32,
            })
            .collect();

        // Convert model layers to graph nodes
        for (i, layer) in model.info.layers.iter().enumerate() {
            let input_name = if i == 0 {
                "input_0".to_string()
            } else {
                format!("layer_{}_out", i - 1)
            };

            let output_name = if i == model.info.layers.len() - 1 {
                "output_0".to_string()
            } else {
                format!("layer_{}_out", i)
            };

            let node = Self::create_node_from_layer(layer, input_name, output_name)?;
            nodes.push(node);
        }

        Ok(Self {
            nodes,
            inputs,
            outputs,
            tensor_cache,
        })
    }

    /// Create a graph node from a layer
    fn create_node_from_layer(
        layer: &LayerInfo,
        input: String,
        output: String,
    ) -> Result<GraphNode> {
        let node_type = match layer.layer_type.as_str() {
            "conv2d" => NodeType::Conv2D {
                stride: (1, 1),
                padding: (1, 1),
                groups: 1,
            },
            "linear" => NodeType::Linear,
            "relu" => NodeType::ReLU,
            "sigmoid" => NodeType::Sigmoid,
            "tanh" => NodeType::Tanh,
            "maxpool2d" => NodeType::MaxPool2D {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            },
            "avgpool2d" => NodeType::AvgPool2D {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            },
            "batchnorm" => NodeType::BatchNorm,
            "flatten" => NodeType::Flatten,
            _ => {
                return Err(BlitzedError::TensorError {
                    message: format!("Unsupported layer type: {}", layer.layer_type),
                });
            }
        };

        // PLACEHOLDER: Creates zero-initialized parameters for structural graph execution.
        // These are NOT trained weights. For real inference with trained weights,
        // use the quantized inference path via Esp32CodeGen::generate_from_weights().
        let mut parameters = HashMap::new();

        match &node_type {
            NodeType::Conv2D { .. } => {
                let weight_shape = vec![64, 3, 3, 3];
                let weight_data = vec![0.0f32; 64 * 3 * 3 * 3];
                parameters.insert(
                    "weight".to_string(),
                    Tensor::new(weight_shape, TensorData::Float32(weight_data)),
                );

                let bias_data = vec![0.0f32; 64];
                parameters.insert(
                    "bias".to_string(),
                    Tensor::new(vec![64], TensorData::Float32(bias_data)),
                );
            }
            NodeType::Linear => {
                let input_size = *layer.input_shape.last().unwrap_or(&512) as usize;
                let output_size = *layer.output_shape.last().unwrap_or(&256) as usize;

                let weight_data = vec![0.0f32; input_size * output_size];
                parameters.insert(
                    "weight".to_string(),
                    Tensor::new(
                        vec![output_size, input_size],
                        TensorData::Float32(weight_data),
                    ),
                );

                let bias_data = vec![0.0f32; output_size];
                parameters.insert(
                    "bias".to_string(),
                    Tensor::new(vec![output_size], TensorData::Float32(bias_data)),
                );
            }
            NodeType::BatchNorm => {
                let num_features = *layer.output_shape.get(1).unwrap_or(&64) as usize;
                let ones = vec![1.0f32; num_features];
                let zeros = vec![0.0f32; num_features];

                parameters.insert(
                    "weight".to_string(),
                    Tensor::new(vec![num_features], TensorData::Float32(ones.clone())),
                );
                parameters.insert(
                    "bias".to_string(),
                    Tensor::new(vec![num_features], TensorData::Float32(zeros.clone())),
                );
                parameters.insert(
                    "running_mean".to_string(),
                    Tensor::new(vec![num_features], TensorData::Float32(zeros.clone())),
                );
                parameters.insert(
                    "running_var".to_string(),
                    Tensor::new(vec![num_features], TensorData::Float32(ones)),
                );
            }
            _ => {
                // No parameters needed for activation functions, pooling, etc.
            }
        }

        Ok(GraphNode {
            layer_info: layer.clone(),
            node_type,
            inputs: vec![input],
            output,
            parameters,
            config: NodeConfig::default(),
        })
    }

    /// Execute a single node in the graph
    pub fn execute_node(&mut self, node: &GraphNode) -> Result<Tensor> {
        // Get input tensor
        let input_tensor = if let Some(tensor) = self.tensor_cache.get(&node.inputs[0]) {
            tensor.clone()
        } else {
            return Err(BlitzedError::TensorError {
                message: format!("Input tensor not found: {}", node.inputs[0]),
            });
        };

        // Execute based on node type
        let output = match &node.node_type {
            NodeType::Conv2D {
                stride, padding, ..
            } => {
                let weight = node.parameters.get("weight").unwrap();
                let bias = node.parameters.get("bias");
                TensorOps::conv2d(&input_tensor, weight, bias, *stride, *padding)?
            }

            NodeType::Linear => {
                // Flatten input if needed
                let flattened = if input_tensor.ndim() > 2 {
                    let batch_size = input_tensor.shape[0];
                    let features: usize = input_tensor.shape[1..].iter().product();
                    let mut flat_tensor = input_tensor.clone();
                    flat_tensor.reshape(vec![batch_size, features])?;
                    flat_tensor
                } else {
                    input_tensor
                };

                let weight = node.parameters.get("weight").unwrap();
                let bias = node.parameters.get("bias");

                let output = TensorOps::matmul(&flattened, &{
                    // Transpose weight matrix for correct matmul
                    let mut transposed = weight.clone();
                    if let TensorData::Float32(ref data) = weight.data {
                        let rows = weight.shape[0];
                        let cols = weight.shape[1];
                        let mut new_data = vec![0.0; data.len()];
                        for i in 0..rows {
                            for j in 0..cols {
                                new_data[j * rows + i] = data[i * cols + j];
                            }
                        }
                        transposed.data = TensorData::Float32(new_data);
                        transposed.shape = vec![cols, rows];
                    }
                    transposed
                })?;

                // Add bias if available
                if let Some(bias) = bias {
                    TensorOps::add(&output, bias)?
                } else {
                    output
                }
            }

            NodeType::ReLU => TensorOps::relu(&input_tensor)?,
            NodeType::Sigmoid => TensorOps::sigmoid(&input_tensor)?,
            NodeType::Tanh => TensorOps::tanh(&input_tensor)?,

            NodeType::MaxPool2D {
                kernel_size,
                stride,
                padding,
            } => TensorOps::max_pool2d(&input_tensor, *kernel_size, *stride, *padding)?,

            NodeType::AvgPool2D {
                kernel_size,
                stride,
                padding,
            } => TensorOps::avg_pool2d(&input_tensor, *kernel_size, *stride, *padding)?,

            NodeType::BatchNorm => {
                let weight = node.parameters.get("weight").unwrap();
                let bias = node.parameters.get("bias").unwrap();
                let mean = node.parameters.get("running_mean").unwrap();
                let var = node.parameters.get("running_var").unwrap();
                TensorOps::batch_norm(&input_tensor, mean, var, Some(weight), Some(bias), 1e-5)?
            }

            NodeType::Flatten => {
                let batch_size = input_tensor.shape[0];
                let features: usize = input_tensor.shape[1..].iter().product();
                let mut flattened = input_tensor;
                flattened.reshape(vec![batch_size, features])?;
                flattened
            }

            NodeType::Reshape { target_shape } => {
                let mut reshaped = input_tensor;
                reshaped.reshape(target_shape.clone())?;
                reshaped
            }

            NodeType::Add => {
                // For add, we would need a second input tensor
                // For now, just return the input as-is
                input_tensor
            }
        };

        // Cache the output tensor
        self.tensor_cache
            .insert(node.output.clone(), output.clone());

        Ok(output)
    }
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        let memory_pool = MemoryPool::new(config.max_memory_pool_mb, 1024 * 1024); // 1MB blocks

        Self {
            config,
            graph: None,
            memory_pool,
            stats: InferenceStats::default(),
        }
    }

    /// Load a model into the inference engine
    pub fn load_model(&mut self, model: &Model) -> Result<()> {
        log::info!(
            "Loading model into inference engine: {} parameters",
            model.info.parameter_count
        );

        let graph = ExecutionGraph::from_model(model)?;
        log::info!("Created execution graph with {} nodes", graph.nodes.len());

        self.graph = Some(graph);
        Ok(())
    }

    /// Run inference on input data
    pub fn inference(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let start_time = std::time::Instant::now();

        let graph = self
            .graph
            .as_mut()
            .ok_or_else(|| BlitzedError::TensorError {
                message: "No model loaded in inference engine".to_string(),
            })?;

        // Validate inputs
        if inputs.len() != graph.inputs.len() {
            return Err(BlitzedError::TensorError {
                message: format!(
                    "Expected {} inputs, got {}",
                    graph.inputs.len(),
                    inputs.len()
                ),
            });
        }

        // Validate batch size configuration
        for (i, input_tensor) in inputs.iter().enumerate() {
            if input_tensor.shape[0] > self.config.batch_size {
                return Err(BlitzedError::TensorError {
                    message: format!(
                        "Input {} batch size {} exceeds configured maximum {}",
                        i, input_tensor.shape[0], self.config.batch_size
                    ),
                });
            }
        }

        // Load input tensors into cache
        for (i, input_tensor) in inputs.into_iter().enumerate() {
            let input_name = &graph.inputs[i].name;
            graph.tensor_cache.insert(input_name.clone(), input_tensor);
        }

        // Execute graph nodes in order
        for node in &graph.nodes.clone() {
            let node_start = std::time::Instant::now();

            graph.execute_node(node)?;

            let node_time = node_start.elapsed().as_millis() as f32;
            self.stats
                .layer_times_ms
                .insert(node.layer_info.name.clone(), node_time);
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for output_spec in &graph.outputs {
            if let Some(output_tensor) = graph.tensor_cache.get(&output_spec.name) {
                outputs.push(output_tensor.clone());
            } else {
                return Err(BlitzedError::TensorError {
                    message: format!("Output tensor not found: {}", output_spec.name),
                });
            }
        }

        // Update statistics
        let total_time = start_time.elapsed().as_millis() as f32;
        self.stats.total_time_ms += total_time;
        self.stats.inference_count += 1;

        log::info!("Inference completed in {:.2}ms", total_time);

        Ok(outputs)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &InferenceStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = InferenceStats::default();
    }

    /// Get average inference time
    pub fn average_inference_time_ms(&self) -> f32 {
        if self.stats.inference_count > 0 {
            self.stats.total_time_ms / self.stats.inference_count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelData, ModelFormat, ModelInfo};

    fn create_test_model() -> Model {
        let layers = vec![
            LayerInfo {
                name: "conv1".to_string(),
                layer_type: "conv2d".to_string(),
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 64, 224, 224],
                parameter_count: 1728, // 64*3*3*3
                flops: 1_000_000,
            },
            LayerInfo {
                name: "relu1".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 64, 224, 224],
                output_shape: vec![1, 64, 224, 224],
                parameter_count: 0,
                flops: 50_176, // 64*224*224
            },
            LayerInfo {
                name: "fc".to_string(),
                layer_type: "linear".to_string(),
                input_shape: vec![1, 3211264], // Flattened
                output_shape: vec![1, 10],
                parameter_count: 32_112_650,
                flops: 32_112_640,
            },
        ];

        Model {
            info: ModelInfo {
                format: ModelFormat::PyTorch,
                input_shapes: vec![vec![1, 3, 224, 224]],
                output_shapes: vec![vec![1, 10]],
                parameter_count: 32_114_378,
                model_size_bytes: 128_457_512,
                operations_count: 33_162_816,
                layers,
            },
            data: ModelData::Raw(vec![0u8; 1024]),
        }
    }

    #[test]
    fn test_inference_engine_creation() {
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config);

        assert_eq!(engine.stats.inference_count, 0);
        assert_eq!(engine.average_inference_time_ms(), 0.0);
    }

    #[test]
    fn test_execution_graph_creation() {
        let model = create_test_model();
        let graph = ExecutionGraph::from_model(&model);

        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_model_loading() {
        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);
        let model = create_test_model();

        let result = engine.load_model(&model);
        assert!(result.is_ok());
        assert!(engine.graph.is_some());
    }

    #[test]
    fn test_inference_execution() {
        env_logger::try_init().ok();

        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);
        let model = create_test_model();

        engine
            .load_model(&model)
            .expect("Model loading should succeed");

        // Create dummy input tensor
        let input_data = vec![0.5f32; 3 * 224 * 224];
        let input_tensor = Tensor::new(vec![1, 3, 224, 224], TensorData::Float32(input_data));

        let result = engine.inference(vec![input_tensor]);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![1, 10]);

        // Check stats
        assert_eq!(engine.get_stats().inference_count, 1);
        assert!(engine.average_inference_time_ms() > 0.0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(1.0, 1024); // 1MB, 1KB blocks

        let block1 = pool.allocate();
        assert!(block1.is_some());

        let block2 = pool.allocate();
        assert!(block2.is_some());

        // Deallocate and reallocate
        pool.deallocate(block1.unwrap());
        let block3 = pool.allocate();
        assert!(block3.is_some());
    }

    #[test]
    fn test_tensor_spec() {
        let spec = TensorSpec {
            name: "input_0".to_string(),
            shape: vec![1, 3, 224, 224],
            dtype: DataType::Float32,
        };

        assert_eq!(spec.name, "input_0");
        assert_eq!(spec.shape, vec![1, 3, 224, 224]);
        assert_eq!(spec.dtype, DataType::Float32);
    }

    #[test]
    fn test_execute_node_relu() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        // Create input with negative values
        let input_data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
        let input_tensor = Tensor::new(vec![1, 6], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        // Create ReLU node
        let relu_node = GraphNode {
            layer_info: LayerInfo {
                name: "relu_test".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 6],
                output_shape: vec![1, 6],
                parameter_count: 0,
                flops: 6,
            },
            node_type: NodeType::ReLU,
            inputs: vec!["input_0".to_string()],
            output: "relu_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&relu_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        if let TensorData::Float32(data) = &output.data {
            assert_eq!(data[0], 0.0); // -1.0 -> 0.0
            assert_eq!(data[1], 2.0); // 2.0 -> 2.0
            assert_eq!(data[2], 0.0); // -3.0 -> 0.0
            assert_eq!(data[3], 4.0); // 4.0 -> 4.0
        } else {
            panic!("Expected Float32 data");
        }
    }

    #[test]
    fn test_execute_node_sigmoid() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        let input_data = vec![0.0, 1.0, -1.0, 5.0, -5.0];
        let input_tensor = Tensor::new(vec![1, 5], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let sigmoid_node = GraphNode {
            layer_info: LayerInfo {
                name: "sigmoid_test".to_string(),
                layer_type: "sigmoid".to_string(),
                input_shape: vec![1, 5],
                output_shape: vec![1, 5],
                parameter_count: 0,
                flops: 5,
            },
            node_type: NodeType::Sigmoid,
            inputs: vec!["input_0".to_string()],
            output: "sigmoid_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&sigmoid_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        if let TensorData::Float32(data) = &output.data {
            // All values should be in (0, 1)
            for &val in data.iter() {
                assert!(val > 0.0 && val < 1.0);
            }
        } else {
            panic!("Expected Float32 data");
        }
    }

    #[test]
    fn test_execute_node_tanh() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        let input_data = vec![0.0, 1.0, -1.0, 3.0, -3.0];
        let input_tensor = Tensor::new(vec![1, 5], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let tanh_node = GraphNode {
            layer_info: LayerInfo {
                name: "tanh_test".to_string(),
                layer_type: "tanh".to_string(),
                input_shape: vec![1, 5],
                output_shape: vec![1, 5],
                parameter_count: 0,
                flops: 5,
            },
            node_type: NodeType::Tanh,
            inputs: vec!["input_0".to_string()],
            output: "tanh_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&tanh_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        if let TensorData::Float32(data) = &output.data {
            // All values should be in (-1, 1)
            for &val in data.iter() {
                assert!(val > -1.0 && val < 1.0);
            }
        } else {
            panic!("Expected Float32 data");
        }
    }

    #[test]
    fn test_execute_node_flatten() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        // Create [1, 3, 4, 4] tensor
        let input_data = vec![0.5f32; 48]; // 3*4*4 = 48
        let input_tensor = Tensor::new(vec![1, 3, 4, 4], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let flatten_node = GraphNode {
            layer_info: LayerInfo {
                name: "flatten_test".to_string(),
                layer_type: "flatten".to_string(),
                input_shape: vec![1, 3, 4, 4],
                output_shape: vec![1, 48],
                parameter_count: 0,
                flops: 0,
            },
            node_type: NodeType::Flatten,
            inputs: vec!["input_0".to_string()],
            output: "flatten_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&flatten_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, 48]);
    }

    #[test]
    fn test_execute_node_reshape() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        // Create [1, 48] tensor
        let input_data = vec![0.5f32; 48];
        let input_tensor = Tensor::new(vec![1, 48], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let reshape_node = GraphNode {
            layer_info: LayerInfo {
                name: "reshape_test".to_string(),
                layer_type: "reshape".to_string(),
                input_shape: vec![1, 48],
                output_shape: vec![1, 6, 8],
                parameter_count: 0,
                flops: 0,
            },
            node_type: NodeType::Reshape {
                target_shape: vec![1, 6, 8],
            },
            inputs: vec!["input_0".to_string()],
            output: "reshape_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&reshape_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, 6, 8]);
    }

    #[test]
    fn test_execute_node_add() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_tensor = Tensor::new(vec![1, 4], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let add_node = GraphNode {
            layer_info: LayerInfo {
                name: "add_test".to_string(),
                layer_type: "add".to_string(),
                input_shape: vec![1, 4],
                output_shape: vec![1, 4],
                parameter_count: 0,
                flops: 4,
            },
            node_type: NodeType::Add,
            inputs: vec!["input_0".to_string()],
            output: "add_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&add_node);
        assert!(result.is_ok());

        // Currently Add just returns input as-is (needs second input)
        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, 4]);
    }

    #[test]
    fn test_execute_node_batchnorm() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        let num_features = 4;
        // BatchNorm expects 4D input: [batch, channels, height, width]
        let input_data = vec![1.0f32; num_features * 2 * 2]; // [1, 4, 2, 2]
        let input_tensor =
            Tensor::new(vec![1, num_features, 2, 2], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let mut parameters = HashMap::new();
        parameters.insert(
            "weight".to_string(),
            Tensor::new(
                vec![num_features],
                TensorData::Float32(vec![1.0; num_features]),
            ),
        );
        parameters.insert(
            "bias".to_string(),
            Tensor::new(
                vec![num_features],
                TensorData::Float32(vec![0.0; num_features]),
            ),
        );
        parameters.insert(
            "running_mean".to_string(),
            Tensor::new(
                vec![num_features],
                TensorData::Float32(vec![0.0; num_features]),
            ),
        );
        parameters.insert(
            "running_var".to_string(),
            Tensor::new(
                vec![num_features],
                TensorData::Float32(vec![1.0; num_features]),
            ),
        );

        let batchnorm_node = GraphNode {
            layer_info: LayerInfo {
                name: "batchnorm_test".to_string(),
                layer_type: "batchnorm".to_string(),
                input_shape: vec![1, num_features as i64, 2, 2],
                output_shape: vec![1, num_features as i64, 2, 2],
                parameter_count: num_features * 4,
                flops: num_features as u64,
            },
            node_type: NodeType::BatchNorm,
            inputs: vec!["input_0".to_string()],
            output: "batchnorm_out".to_string(),
            parameters,
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&batchnorm_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, num_features, 2, 2]);
    }

    #[test]
    fn test_execute_node_avgpool2d() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        // Create [1, 2, 4, 4] tensor with known values
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input_tensor = Tensor::new(vec![1, 2, 4, 4], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let avgpool_node = GraphNode {
            layer_info: LayerInfo {
                name: "avgpool_test".to_string(),
                layer_type: "avgpool2d".to_string(),
                input_shape: vec![1, 2, 4, 4],
                output_shape: vec![1, 2, 2, 2],
                parameter_count: 0,
                flops: 32,
            },
            node_type: NodeType::AvgPool2D {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            },
            inputs: vec!["input_0".to_string()],
            output: "avgpool_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&avgpool_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, 2, 2, 2]);
    }

    #[test]
    fn test_execute_node_maxpool2d() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        // Create [1, 2, 4, 4] tensor
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input_tensor = Tensor::new(vec![1, 2, 4, 4], TensorData::Float32(input_data));
        graph
            .tensor_cache
            .insert("input_0".to_string(), input_tensor);

        let maxpool_node = GraphNode {
            layer_info: LayerInfo {
                name: "maxpool_test".to_string(),
                layer_type: "maxpool2d".to_string(),
                input_shape: vec![1, 2, 4, 4],
                output_shape: vec![1, 2, 2, 2],
                parameter_count: 0,
                flops: 32,
            },
            node_type: NodeType::MaxPool2D {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
            },
            inputs: vec!["input_0".to_string()],
            output: "maxpool_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&maxpool_node);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape, vec![1, 2, 2, 2]);
    }

    #[test]
    fn test_execute_node_input_not_found() {
        let mut graph = ExecutionGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_cache: HashMap::new(),
        };

        let relu_node = GraphNode {
            layer_info: LayerInfo {
                name: "relu_test".to_string(),
                layer_type: "relu".to_string(),
                input_shape: vec![1, 6],
                output_shape: vec![1, 6],
                parameter_count: 0,
                flops: 6,
            },
            node_type: NodeType::ReLU,
            inputs: vec!["missing_input".to_string()],
            output: "relu_out".to_string(),
            parameters: HashMap::new(),
            config: NodeConfig::default(),
        };

        let result = graph.execute_node(&relu_node);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_no_model_loaded() {
        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);

        let input_data = vec![0.5f32; 3 * 224 * 224];
        let input_tensor = Tensor::new(vec![1, 3, 224, 224], TensorData::Float32(input_data));

        let result = engine.inference(vec![input_tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_wrong_input_count() {
        env_logger::try_init().ok();

        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);
        let model = create_test_model();

        engine.load_model(&model).unwrap();

        // Model expects 1 input, provide 2
        let input1 = Tensor::new(
            vec![1, 3, 224, 224],
            TensorData::Float32(vec![0.5; 3 * 224 * 224]),
        );
        let input2 = Tensor::new(
            vec![1, 3, 224, 224],
            TensorData::Float32(vec![0.5; 3 * 224 * 224]),
        );

        let result = engine.inference(vec![input1, input2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_node_from_layer_conv2d() {
        let layer = LayerInfo {
            name: "conv_test".to_string(),
            layer_type: "conv2d".to_string(),
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 64, 224, 224],
            parameter_count: 1728,
            flops: 1_000_000,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::Conv2D { .. }));
        assert!(node.parameters.contains_key("weight"));
        assert!(node.parameters.contains_key("bias"));
    }

    #[test]
    fn test_create_node_from_layer_linear() {
        let layer = LayerInfo {
            name: "linear_test".to_string(),
            layer_type: "linear".to_string(),
            input_shape: vec![1, 512],
            output_shape: vec![1, 256],
            parameter_count: 131328,
            flops: 131072,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::Linear));
        assert!(node.parameters.contains_key("weight"));
        assert!(node.parameters.contains_key("bias"));
    }

    #[test]
    fn test_create_node_from_layer_relu() {
        let layer = LayerInfo {
            name: "relu_test".to_string(),
            layer_type: "relu".to_string(),
            input_shape: vec![1, 64],
            output_shape: vec![1, 64],
            parameter_count: 0,
            flops: 64,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::ReLU));
    }

    #[test]
    fn test_create_node_from_layer_sigmoid() {
        let layer = LayerInfo {
            name: "sigmoid_test".to_string(),
            layer_type: "sigmoid".to_string(),
            input_shape: vec![1, 64],
            output_shape: vec![1, 64],
            parameter_count: 0,
            flops: 64,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::Sigmoid));
    }

    #[test]
    fn test_create_node_from_layer_tanh() {
        let layer = LayerInfo {
            name: "tanh_test".to_string(),
            layer_type: "tanh".to_string(),
            input_shape: vec![1, 64],
            output_shape: vec![1, 64],
            parameter_count: 0,
            flops: 64,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::Tanh));
    }

    #[test]
    fn test_create_node_from_layer_maxpool2d() {
        let layer = LayerInfo {
            name: "maxpool_test".to_string(),
            layer_type: "maxpool2d".to_string(),
            input_shape: vec![1, 64, 224, 224],
            output_shape: vec![1, 64, 112, 112],
            parameter_count: 0,
            flops: 100_000,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::MaxPool2D { .. }));
    }

    #[test]
    fn test_create_node_from_layer_avgpool2d() {
        let layer = LayerInfo {
            name: "avgpool_test".to_string(),
            layer_type: "avgpool2d".to_string(),
            input_shape: vec![1, 64, 224, 224],
            output_shape: vec![1, 64, 112, 112],
            parameter_count: 0,
            flops: 100_000,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::AvgPool2D { .. }));
    }

    #[test]
    fn test_create_node_from_layer_batchnorm() {
        let layer = LayerInfo {
            name: "batchnorm_test".to_string(),
            layer_type: "batchnorm".to_string(),
            input_shape: vec![1, 64, 224, 224],
            output_shape: vec![1, 64, 224, 224],
            parameter_count: 256,
            flops: 200_000,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::BatchNorm));
        assert!(node.parameters.contains_key("weight"));
        assert!(node.parameters.contains_key("bias"));
        assert!(node.parameters.contains_key("running_mean"));
        assert!(node.parameters.contains_key("running_var"));
    }

    #[test]
    fn test_create_node_from_layer_flatten() {
        let layer = LayerInfo {
            name: "flatten_test".to_string(),
            layer_type: "flatten".to_string(),
            input_shape: vec![1, 64, 7, 7],
            output_shape: vec![1, 3136],
            parameter_count: 0,
            flops: 0,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_ok());

        let node = result.unwrap();
        assert!(matches!(node.node_type, NodeType::Flatten));
    }

    #[test]
    fn test_create_node_from_layer_unsupported() {
        let layer = LayerInfo {
            name: "unsupported_test".to_string(),
            layer_type: "custom_layer".to_string(),
            input_shape: vec![1, 64],
            output_shape: vec![1, 64],
            parameter_count: 0,
            flops: 0,
        };

        let result =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_pool_exhaustion() {
        let mut pool = MemoryPool::new(0.002, 1024); // 0.002MB = ~2KB, 1KB blocks = max 2 blocks

        let block1 = pool.allocate();
        assert!(block1.is_some());

        let block2 = pool.allocate();
        assert!(block2.is_some());

        // Should fail - pool exhausted
        let block3 = pool.allocate();
        assert!(block3.is_none());
    }

    #[test]
    fn test_memory_pool_deallocate_reuse() {
        let mut pool = MemoryPool::new(0.002, 1024); // ~2 blocks max

        let block1 = pool.allocate();
        assert!(block1.is_some());

        let block2 = pool.allocate();
        assert!(block2.is_some());

        // Pool exhausted
        assert!(pool.allocate().is_none());

        // Deallocate block1
        pool.deallocate(block1.unwrap());

        // Should now succeed
        let block3 = pool.allocate();
        assert!(block3.is_some());
    }

    #[test]
    fn test_inference_stats_reset() {
        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);

        // Manually set some stats
        engine.stats.inference_count = 5;
        engine.stats.total_time_ms = 123.45;
        engine.stats.peak_memory_mb = 10.0;

        engine.reset_stats();

        assert_eq!(engine.stats.inference_count, 0);
        assert_eq!(engine.stats.total_time_ms, 0.0);
        assert_eq!(engine.stats.peak_memory_mb, 0.0);
    }

    #[test]
    fn test_inference_config_custom() {
        let config = InferenceConfig {
            batch_size: 8,
            memory_optimization: 2,
            enable_parallel: false,
            max_memory_pool_mb: 64.0,
            enable_fusion: false,
        };

        assert_eq!(config.batch_size, 8);
        assert_eq!(config.memory_optimization, 2);
        assert!(!config.enable_parallel);
        assert_eq!(config.max_memory_pool_mb, 64.0);
        assert!(!config.enable_fusion);
    }

    #[test]
    fn test_placeholder_weights_are_zero() {
        // Build a model with a single Linear layer and create its graph node.
        let layer = LayerInfo {
            name: "fc".to_string(),
            layer_type: "linear".to_string(),
            input_shape: vec![1, 8],
            output_shape: vec![1, 4],
            parameter_count: 36,
            flops: 32,
        };

        let node =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string())
                .unwrap();

        // Both weight and bias tensors must be all zeros (changed from 0.1).
        for param_name in &["weight", "bias"] {
            let tensor = node
                .parameters
                .get(*param_name)
                .unwrap_or_else(|| panic!("Linear node missing '{}' parameter", param_name));

            if let TensorData::Float32(data) = &tensor.data {
                let all_zero = data.iter().all(|&v| v == 0.0f32);
                assert!(
                    all_zero,
                    "Linear placeholder '{}' contains non-zero values (old 0.1 dummy weights \
                     were not replaced); first value = {}",
                    param_name,
                    data.first().copied().unwrap_or(f32::NAN)
                );
            } else {
                panic!("Expected Float32 data for parameter '{}'", param_name);
            }
        }
    }

    #[test]
    fn test_placeholder_conv_weights_are_zero() {
        // Build a model with a Conv2D layer and verify its placeholder weights are zero.
        let layer = LayerInfo {
            name: "conv1".to_string(),
            layer_type: "conv2d".to_string(),
            input_shape: vec![1, 3, 16, 16],
            output_shape: vec![1, 64, 16, 16],
            parameter_count: 1728,
            flops: 442_368,
        };

        let node =
            ExecutionGraph::create_node_from_layer(&layer, "in".to_string(), "out".to_string())
                .unwrap();

        for param_name in &["weight", "bias"] {
            let tensor = node
                .parameters
                .get(*param_name)
                .unwrap_or_else(|| panic!("Conv2D node missing '{}' parameter", param_name));

            if let TensorData::Float32(data) = &tensor.data {
                let all_zero = data.iter().all(|&v| v == 0.0f32);
                assert!(
                    all_zero,
                    "Conv2D placeholder '{}' contains non-zero values (old 0.1 dummy weights \
                     were not replaced); first value = {}",
                    param_name,
                    data.first().copied().unwrap_or(f32::NAN)
                );
            } else {
                panic!("Expected Float32 data for parameter '{}'", param_name);
            }
        }
    }
}
