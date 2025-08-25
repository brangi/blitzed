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

//! Enhanced ONNX model analysis with proper operator support

use crate::model::LayerInfo;
use crate::Result;

/// ONNX operator types
#[derive(Debug, Clone, PartialEq)]
pub enum OnnxOperator {
    Conv,
    Linear,
    BatchNorm,
    ReLU,
    MaxPool,
    AveragePool,
    GlobalAveragePool,
    Flatten,
    Dropout,
    Softmax,
    Add,
    Mul,
    Concat,
    Reshape,
    Transpose,
}

impl OnnxOperator {
    /// Parse operator from string
    pub fn from_op_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "conv" | "conv2d" | "convolution" => Some(Self::Conv),
            "gemm" | "linear" | "matmul" => Some(Self::Linear),
            "batchnorm" | "batchnormalization" => Some(Self::BatchNorm),
            "relu" => Some(Self::ReLU),
            "maxpool" | "maxpool2d" => Some(Self::MaxPool),
            "averagepool" | "avgpool" => Some(Self::AveragePool),
            "globalaveragepool" => Some(Self::GlobalAveragePool),
            "flatten" => Some(Self::Flatten),
            "dropout" => Some(Self::Dropout),
            "softmax" => Some(Self::Softmax),
            "add" | "sum" => Some(Self::Add),
            "mul" | "multiply" => Some(Self::Mul),
            "concat" | "concatenate" => Some(Self::Concat),
            "reshape" => Some(Self::Reshape),
            "transpose" => Some(Self::Transpose),
            _ => None,
        }
    }

    /// Calculate FLOPs for this operator
    pub fn calculate_flops(
        &self,
        input_shape: &[i64],
        output_shape: &[i64],
        params: &OperatorParams,
    ) -> u64 {
        match self {
            Self::Conv => {
                // FLOPs = 2 * H_out * W_out * C_out * C_in * K_h * K_w
                if output_shape.len() >= 4 && input_shape.len() >= 4 {
                    let batch = output_shape[0] as u64;
                    let out_channels = output_shape[1] as u64;
                    let out_h = output_shape[2] as u64;
                    let out_w = output_shape[3] as u64;
                    let in_channels = input_shape[1] as u64;
                    let kernel_h = params.kernel_size.0 as u64;
                    let kernel_w = params.kernel_size.1 as u64;

                    batch * 2 * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w
                } else {
                    0
                }
            }
            Self::Linear => {
                // FLOPs = 2 * batch * in_features * out_features
                if output_shape.len() >= 2 && input_shape.len() >= 2 {
                    let batch = output_shape[0] as u64;
                    let in_features = input_shape[input_shape.len() - 1] as u64;
                    let out_features = output_shape[output_shape.len() - 1] as u64;

                    batch * 2 * in_features * out_features
                } else {
                    0
                }
            }
            Self::BatchNorm => {
                // FLOPs = 4 * num_elements (sub_mean, div_std, mul_gamma, add_beta)
                output_shape.iter().product::<i64>() as u64 * 4
            }
            Self::ReLU => {
                // FLOPs = num_elements (comparison)
                output_shape.iter().product::<i64>() as u64
            }
            Self::MaxPool | Self::AveragePool => {
                // FLOPs = num_output_elements * kernel_size (comparisons or additions)
                let output_elements = output_shape.iter().product::<i64>() as u64;
                let kernel_elements = (params.kernel_size.0 * params.kernel_size.1) as u64;
                output_elements * kernel_elements
            }
            Self::GlobalAveragePool => {
                // FLOPs = num_input_elements (sum) + num_output_elements (divide)
                let input_elements = input_shape.iter().product::<i64>() as u64;
                let output_elements = output_shape.iter().product::<i64>() as u64;
                input_elements + output_elements
            }
            Self::Softmax => {
                // FLOPs = 3 * num_elements (exp, sum, divide)
                output_shape.iter().product::<i64>() as u64 * 3
            }
            Self::Add | Self::Mul => {
                // FLOPs = num_elements
                output_shape.iter().product::<i64>() as u64
            }
            _ => 0, // No FLOPs for reshape, transpose, flatten, etc.
        }
    }

    /// Calculate parameter count for this operator
    pub fn calculate_params(
        &self,
        input_shape: &[i64],
        output_shape: &[i64],
        params: &OperatorParams,
    ) -> usize {
        match self {
            Self::Conv => {
                if output_shape.len() >= 4 && input_shape.len() >= 4 {
                    let out_channels = output_shape[1] as usize;
                    let in_channels = input_shape[1] as usize;
                    let kernel_h = params.kernel_size.0 as usize;
                    let kernel_w = params.kernel_size.1 as usize;

                    // Weights + bias
                    (out_channels * in_channels * kernel_h * kernel_w) + out_channels
                } else {
                    0
                }
            }
            Self::Linear => {
                if output_shape.len() >= 2 && input_shape.len() >= 2 {
                    let in_features = input_shape[input_shape.len() - 1] as usize;
                    let out_features = output_shape[output_shape.len() - 1] as usize;

                    // Weights + bias
                    (in_features * out_features) + out_features
                } else {
                    0
                }
            }
            Self::BatchNorm => {
                if input_shape.len() >= 2 {
                    // gamma, beta, running_mean, running_var
                    let channels = input_shape[1] as usize;
                    channels * 4
                } else {
                    0
                }
            }
            _ => 0, // Most other operators don't have parameters
        }
    }
}

/// Parameters for ONNX operators
#[derive(Debug, Clone)]
pub struct OperatorParams {
    pub kernel_size: (i32, i32),
    pub stride: (i32, i32),
    pub padding: (i32, i32),
    pub dilation: (i32, i32),
    pub groups: i32,
}

impl Default for OperatorParams {
    fn default() -> Self {
        Self {
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

/// Enhanced ONNX graph analyzer
pub struct OnnxGraphAnalyzer {
    layers: Vec<LayerInfo>,
    total_params: usize,
    total_flops: u64,
}

impl Default for OnnxGraphAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxGraphAnalyzer {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            total_params: 0,
            total_flops: 0,
        }
    }

    /// Analyze a layer and add it to the graph
    pub fn add_layer(
        &mut self,
        name: String,
        operator: OnnxOperator,
        input_shape: Vec<i64>,
        output_shape: Vec<i64>,
        params: OperatorParams,
    ) {
        let flops = operator.calculate_flops(&input_shape, &output_shape, &params);
        let param_count = operator.calculate_params(&input_shape, &output_shape, &params);

        self.total_flops += flops;
        self.total_params += param_count;

        self.layers.push(LayerInfo {
            name,
            layer_type: format!("{:?}", operator),
            input_shape,
            output_shape,
            parameter_count: param_count,
            flops,
        });
    }

    /// Get analysis results
    pub fn get_results(self) -> (Vec<LayerInfo>, usize, u64) {
        (self.layers, self.total_params, self.total_flops)
    }

    /// Create a sample CNN architecture for testing
    pub fn analyze_sample_cnn(&mut self, input_shape: Vec<i64>) -> Result<()> {
        let batch = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        // Conv1: 3->64, 7x7, stride 2
        self.add_layer(
            "conv1".to_string(),
            OnnxOperator::Conv,
            vec![batch, channels, height, width],
            vec![batch, 64, height / 2, width / 2],
            OperatorParams {
                kernel_size: (7, 7),
                stride: (2, 2),
                padding: (3, 3),
                ..Default::default()
            },
        );

        // BatchNorm1
        self.add_layer(
            "bn1".to_string(),
            OnnxOperator::BatchNorm,
            vec![batch, 64, height / 2, width / 2],
            vec![batch, 64, height / 2, width / 2],
            OperatorParams::default(),
        );

        // ReLU1
        self.add_layer(
            "relu1".to_string(),
            OnnxOperator::ReLU,
            vec![batch, 64, height / 2, width / 2],
            vec![batch, 64, height / 2, width / 2],
            OperatorParams::default(),
        );

        // MaxPool1
        self.add_layer(
            "maxpool1".to_string(),
            OnnxOperator::MaxPool,
            vec![batch, 64, height / 2, width / 2],
            vec![batch, 64, height / 4, width / 4],
            OperatorParams {
                kernel_size: (3, 3),
                stride: (2, 2),
                padding: (1, 1),
                ..Default::default()
            },
        );

        // Conv2: 64->128, 3x3
        self.add_layer(
            "conv2".to_string(),
            OnnxOperator::Conv,
            vec![batch, 64, height / 4, width / 4],
            vec![batch, 128, height / 4, width / 4],
            OperatorParams::default(),
        );

        // Global Average Pool
        self.add_layer(
            "global_avg_pool".to_string(),
            OnnxOperator::GlobalAveragePool,
            vec![batch, 128, height / 4, width / 4],
            vec![batch, 128, 1, 1],
            OperatorParams::default(),
        );

        // Flatten
        self.add_layer(
            "flatten".to_string(),
            OnnxOperator::Flatten,
            vec![batch, 128, 1, 1],
            vec![batch, 128],
            OperatorParams::default(),
        );

        // Linear: 128->1000
        self.add_layer(
            "fc".to_string(),
            OnnxOperator::Linear,
            vec![batch, 128],
            vec![batch, 1000],
            OperatorParams::default(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_parsing() {
        assert_eq!(
            OnnxOperator::from_op_string("Conv"),
            Some(OnnxOperator::Conv)
        );
        assert_eq!(
            OnnxOperator::from_op_string("conv2d"),
            Some(OnnxOperator::Conv)
        );
        assert_eq!(
            OnnxOperator::from_op_string("ReLU"),
            Some(OnnxOperator::ReLU)
        );
        assert_eq!(
            OnnxOperator::from_op_string("gemm"),
            Some(OnnxOperator::Linear)
        );
        assert_eq!(OnnxOperator::from_op_string("unknown"), None);
    }

    #[test]
    fn test_conv_flops_calculation() {
        let op = OnnxOperator::Conv;
        let input_shape = vec![1, 3, 224, 224];
        let output_shape = vec![1, 64, 112, 112];
        let params = OperatorParams {
            kernel_size: (7, 7),
            ..Default::default()
        };

        let flops = op.calculate_flops(&input_shape, &output_shape, &params);
        // Expected: 1 * 2 * 112 * 112 * 64 * 3 * 7 * 7 = 236,027,904
        assert_eq!(flops, 236_027_904);
    }

    #[test]
    fn test_conv_params_calculation() {
        let op = OnnxOperator::Conv;
        let input_shape = vec![1, 3, 224, 224];
        let output_shape = vec![1, 64, 112, 112];
        let params = OperatorParams {
            kernel_size: (7, 7),
            ..Default::default()
        };

        let param_count = op.calculate_params(&input_shape, &output_shape, &params);
        // Expected: (64 * 3 * 7 * 7) + 64 = 9408 + 64 = 9472
        assert_eq!(param_count, 9472);
    }

    #[test]
    fn test_linear_flops_calculation() {
        let op = OnnxOperator::Linear;
        let input_shape = vec![1, 128];
        let output_shape = vec![1, 1000];
        let params = OperatorParams::default();

        let flops = op.calculate_flops(&input_shape, &output_shape, &params);
        // Expected: 1 * 2 * 128 * 1000 = 256,000
        assert_eq!(flops, 256_000);
    }

    #[test]
    fn test_linear_params_calculation() {
        let op = OnnxOperator::Linear;
        let input_shape = vec![1, 128];
        let output_shape = vec![1, 1000];
        let params = OperatorParams::default();

        let param_count = op.calculate_params(&input_shape, &output_shape, &params);
        // Expected: (128 * 1000) + 1000 = 129,000
        assert_eq!(param_count, 129_000);
    }

    #[test]
    fn test_graph_analyzer_sample_cnn() {
        let mut analyzer = OnnxGraphAnalyzer::new();
        let input_shape = vec![1, 3, 224, 224];

        analyzer.analyze_sample_cnn(input_shape).unwrap();

        let (layers, total_params, total_flops) = analyzer.get_results();

        // Should have 8 layers
        assert_eq!(layers.len(), 8);

        // Check first layer is Conv1
        assert_eq!(layers[0].name, "conv1");
        assert_eq!(layers[0].layer_type, "Conv");

        // Check last layer is FC
        assert_eq!(layers[7].name, "fc");
        assert_eq!(layers[7].layer_type, "Linear");

        // Should have reasonable parameter count
        assert!(total_params > 100_000); // At least 100K params
        assert!(total_params < 10_000_000); // Less than 10M params

        // Should have reasonable FLOPs
        assert!(total_flops > 1_000_000); // At least 1M FLOPs
    }

    #[test]
    fn test_batchnorm_calculation() {
        let op = OnnxOperator::BatchNorm;
        let input_shape = vec![1, 64, 112, 112];
        let output_shape = vec![1, 64, 112, 112];
        let params = OperatorParams::default();

        let param_count = op.calculate_params(&input_shape, &output_shape, &params);
        // Expected: 64 * 4 = 256 (gamma, beta, running_mean, running_var)
        assert_eq!(param_count, 256);

        let flops = op.calculate_flops(&input_shape, &output_shape, &params);
        // Expected: 1 * 64 * 112 * 112 * 4 = 3,211,264
        assert_eq!(flops, 3_211_264);
    }

    #[test]
    fn test_pooling_calculation() {
        let op = OnnxOperator::MaxPool;
        let input_shape = vec![1, 64, 112, 112];
        let output_shape = vec![1, 64, 56, 56];
        let params = OperatorParams {
            kernel_size: (3, 3),
            stride: (2, 2),
            ..Default::default()
        };

        let param_count = op.calculate_params(&input_shape, &output_shape, &params);
        // Pooling has no parameters
        assert_eq!(param_count, 0);

        let flops = op.calculate_flops(&input_shape, &output_shape, &params);
        // Expected: 1 * 64 * 56 * 56 * 9 = 1,806,336
        assert_eq!(flops, 1_806_336);
    }
}
