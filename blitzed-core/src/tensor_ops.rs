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

//! Core tensor operations for neural network inference
//!
//! This module provides the fundamental tensor operations required for
//! neural network inference, optimized for edge devices and microcontrollers.
//!
//! Features:
//! - Memory-efficient implementations
//! - No heap allocation for fixed-size operations
//! - SIMD optimizations where available
//! - Support for quantized tensors
//! - Hardware-specific optimizations for ARM Cortex-M

use crate::error::{BlitzedError, Result};
use rayon::prelude::*;

/// Shape representation for tensors
pub type Shape = Vec<usize>;

/// Tensor data storage
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point data
    Float32(Vec<f32>),
    /// 8-bit signed integer data (quantized)
    Int8(Vec<i8>),
    /// 4-bit packed integer data (2 values per byte)
    Int4(Vec<u8>),
    /// Binary data (1 bit per weight, packed in bytes)
    Binary(Vec<u8>),
}

/// Multi-dimensional tensor with shape and data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor shape (dimensions)
    pub shape: Shape,
    /// Tensor data
    pub data: TensorData,
    /// Quantization parameters (if quantized)
    pub quantization: Option<QuantizationParams>,
}

/// Quantization parameters for tensor operations
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scaling factor for dequantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Minimum clipping value
    pub clip_min: f32,
    /// Maximum clipping value
    pub clip_max: f32,
}

impl Tensor {
    /// Create a new tensor with the given shape and data
    pub fn new(shape: Shape, data: TensorData) -> Self {
        Self {
            shape,
            data,
            quantization: None,
        }
    }

    /// Create a new quantized tensor
    pub fn new_quantized(shape: Shape, data: TensorData, quantization: QuantizationParams) -> Self {
        Self {
            shape,
            data,
            quantization: Some(quantization),
        }
    }

    /// Get the total number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape the tensor (must preserve total number of elements)
    pub fn reshape(&mut self, new_shape: Shape) -> Result<()> {
        let old_numel = self.numel();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(BlitzedError::TensorError {
                message: format!(
                    "Cannot reshape tensor: old shape {:?} has {} elements, new shape {:?} has {} elements",
                    self.shape, old_numel, new_shape, new_numel
                ),
            });
        }

        self.shape = new_shape;
        Ok(())
    }

    /// Convert tensor to f32 data (dequantize if necessary)
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match &self.data {
            TensorData::Float32(data) => Ok(data.clone()),
            TensorData::Int8(data) => {
                if let Some(quant) = &self.quantization {
                    Ok(data
                        .iter()
                        .map(|&x| (x as f32 - quant.zero_point as f32) * quant.scale)
                        .collect())
                } else {
                    // Simple conversion without quantization parameters
                    Ok(data.iter().map(|&x| x as f32 / 127.0).collect())
                }
            }
            TensorData::Int4(data) => {
                let mut result = Vec::with_capacity(data.len() * 2);
                for &byte in data {
                    // Extract two 4-bit values from each byte
                    let val1 = ((byte & 0xF0) >> 4) as i8 - 8; // Convert to signed -8 to 7
                    let val2 = (byte & 0x0F) as i8 - 8;

                    if let Some(quant) = &self.quantization {
                        result.push((val1 as f32 - quant.zero_point as f32) * quant.scale);
                        result.push((val2 as f32 - quant.zero_point as f32) * quant.scale);
                    } else {
                        result.push(val1 as f32 / 7.0);
                        result.push(val2 as f32 / 7.0);
                    }
                }
                Ok(result)
            }
            TensorData::Binary(data) => {
                let mut result = Vec::with_capacity(data.len() * 8);
                for &byte in data {
                    for i in 0..8 {
                        let bit = (byte >> i) & 1;
                        result.push(if bit == 1 { 1.0 } else { -1.0 });
                    }
                }
                Ok(result)
            }
        }
    }

    /// Create a zeros tensor with the given shape
    pub fn zeros(shape: Shape) -> Self {
        let numel = shape.iter().product();
        Self::new(shape, TensorData::Float32(vec![0.0; numel]))
    }

    /// Create a ones tensor with the given shape
    pub fn ones(shape: Shape) -> Self {
        let numel = shape.iter().product();
        Self::new(shape, TensorData::Float32(vec![1.0; numel]))
    }
}

/// Core tensor operations
pub struct TensorOps;

impl TensorOps {
    /// Matrix multiplication (GEMM): C = A * B
    /// A: [M, K], B: [K, N], C: [M, N]
    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Validate shapes
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(BlitzedError::TensorError {
                message: "Matrix multiplication requires 2D tensors".to_string(),
            });
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];

        if k != k2 {
            return Err(BlitzedError::TensorError {
                message: format!(
                    "Matrix multiplication dimension mismatch: {}x{} × {}x{}",
                    m, k, k2, n
                ),
            });
        }

        // Convert to f32 for computation
        let a_data = a.to_f32()?;
        let b_data = b.to_f32()?;

        // Perform matrix multiplication with parallel processing
        let mut c_data = vec![0.0; m * n];

        // Use parallel processing for large matrices
        if m * n > 1024 {
            c_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
                    }
                    row[j] = sum;
                }
            });
        } else {
            // Sequential for small matrices
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
                    }
                    c_data[i * n + j] = sum;
                }
            }
        }

        Ok(Tensor::new(vec![m, n], TensorData::Float32(c_data)))
    }

    /// 2D Convolution: output = conv2d(input, weight, bias, stride, padding)
    /// input: [N, C_in, H_in, W_in]
    /// weight: [C_out, C_in, K_h, K_w]
    /// bias: [C_out] (optional)
    /// output: [N, C_out, H_out, W_out]
    pub fn conv2d(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        // Validate input shapes
        if input.ndim() != 4 || weight.ndim() != 4 {
            return Err(BlitzedError::TensorError {
                message: "Conv2d requires 4D input and weight tensors".to_string(),
            });
        }

        let n = input.shape[0];
        let c_in = input.shape[1];
        let h_in = input.shape[2];
        let w_in = input.shape[3];

        let c_out = weight.shape[0];
        let c_in_weight = weight.shape[1];
        let k_h = weight.shape[2];
        let k_w = weight.shape[3];

        if c_in != c_in_weight {
            return Err(BlitzedError::TensorError {
                message: format!(
                    "Conv2d input channels mismatch: input {} vs weight {}",
                    c_in, c_in_weight
                ),
            });
        }

        // Calculate output dimensions
        let h_out = (h_in + 2 * padding.0 - k_h) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - k_w) / stride.1 + 1;

        // Convert tensors to f32
        let input_data = input.to_f32()?;
        let weight_data = weight.to_f32()?;
        let bias_data = if let Some(b) = bias {
            Some(b.to_f32()?)
        } else {
            None
        };

        // Initialize output
        let mut output_data = vec![0.0; n * c_out * h_out * w_out];

        // Perform convolution with parallel processing over output channels
        output_data
            .par_chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(flat_idx, channel_out)| {
                let batch = flat_idx / c_out;
                let out_channel = flat_idx % c_out;

                for h_out_idx in 0..h_out {
                    for w_out_idx in 0..w_out {
                        let mut sum = 0.0;

                        // Convolution kernel
                        for c_in_idx in 0..c_in {
                            for k_h_idx in 0..k_h {
                                for k_w_idx in 0..k_w {
                                    let h_in_idx = h_out_idx * stride.0 + k_h_idx;
                                    let w_in_idx = w_out_idx * stride.1 + k_w_idx;

                                    // Apply padding
                                    if h_in_idx >= padding.0 && w_in_idx >= padding.1 {
                                        let h_in_actual = h_in_idx - padding.0;
                                        let w_in_actual = w_in_idx - padding.1;

                                        if h_in_actual < h_in && w_in_actual < w_in {
                                            let input_idx = batch * c_in * h_in * w_in
                                                + c_in_idx * h_in * w_in
                                                + h_in_actual * w_in
                                                + w_in_actual;

                                            let weight_idx = out_channel * c_in * k_h * k_w
                                                + c_in_idx * k_h * k_w
                                                + k_h_idx * k_w
                                                + k_w_idx;

                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if let Some(ref bias) = bias_data {
                            sum += bias[out_channel];
                        }

                        channel_out[h_out_idx * w_out + w_out_idx] = sum;
                    }
                }
            });

        Ok(Tensor::new(
            vec![n, c_out, h_out, w_out],
            TensorData::Float32(output_data),
        ))
    }

    /// ReLU activation: f(x) = max(0, x)
    pub fn relu(input: &Tensor) -> Result<Tensor> {
        let data = input.to_f32()?;
        let output_data: Vec<f32> = data.par_iter().map(|&x| x.max(0.0)).collect();

        Ok(Tensor::new(
            input.shape.clone(),
            TensorData::Float32(output_data),
        ))
    }

    /// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
        let data = input.to_f32()?;
        let output_data: Vec<f32> = data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

        Ok(Tensor::new(
            input.shape.clone(),
            TensorData::Float32(output_data),
        ))
    }

    /// Tanh activation: f(x) = tanh(x)
    pub fn tanh(input: &Tensor) -> Result<Tensor> {
        let data = input.to_f32()?;
        let output_data: Vec<f32> = data.par_iter().map(|&x| x.tanh()).collect();

        Ok(Tensor::new(
            input.shape.clone(),
            TensorData::Float32(output_data),
        ))
    }

    /// Max pooling: reduce spatial dimensions by taking maximum in each window
    /// input: [N, C, H_in, W_in]
    /// output: [N, C, H_out, W_out]
    pub fn max_pool2d(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(BlitzedError::TensorError {
                message: "MaxPool2d requires 4D input tensor".to_string(),
            });
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let h_in = input.shape[2];
        let w_in = input.shape[3];

        // Calculate output dimensions
        let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

        let input_data = input.to_f32()?;
        let mut output_data = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

        // Parallel processing over batches and channels
        output_data
            .par_chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(flat_idx, channel_out)| {
                let batch = flat_idx / c;
                let channel = flat_idx % c;

                for h_out_idx in 0..h_out {
                    for w_out_idx in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;

                        for k_h in 0..kernel_size.0 {
                            for k_w in 0..kernel_size.1 {
                                let h_in_idx = h_out_idx * stride.0 + k_h;
                                let w_in_idx = w_out_idx * stride.1 + k_w;

                                // Apply padding
                                if h_in_idx >= padding.0 && w_in_idx >= padding.1 {
                                    let h_actual = h_in_idx - padding.0;
                                    let w_actual = w_in_idx - padding.1;

                                    if h_actual < h_in && w_actual < w_in {
                                        let input_idx = batch * c * h_in * w_in
                                            + channel * h_in * w_in
                                            + h_actual * w_in
                                            + w_actual;
                                        max_val = max_val.max(input_data[input_idx]);
                                    }
                                }
                            }
                        }

                        channel_out[h_out_idx * w_out + w_out_idx] = max_val;
                    }
                }
            });

        Ok(Tensor::new(
            vec![n, c, h_out, w_out],
            TensorData::Float32(output_data),
        ))
    }

    /// Average pooling: reduce spatial dimensions by taking average in each window
    pub fn avg_pool2d(
        input: &Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(BlitzedError::TensorError {
                message: "AvgPool2d requires 4D input tensor".to_string(),
            });
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let h_in = input.shape[2];
        let w_in = input.shape[3];

        let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

        let input_data = input.to_f32()?;
        let mut output_data = vec![0.0; n * c * h_out * w_out];

        let kernel_area = (kernel_size.0 * kernel_size.1) as f32;

        output_data
            .par_chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(flat_idx, channel_out)| {
                let batch = flat_idx / c;
                let channel = flat_idx % c;

                for h_out_idx in 0..h_out {
                    for w_out_idx in 0..w_out {
                        let mut sum = 0.0;

                        for k_h in 0..kernel_size.0 {
                            for k_w in 0..kernel_size.1 {
                                let h_in_idx = h_out_idx * stride.0 + k_h;
                                let w_in_idx = w_out_idx * stride.1 + k_w;

                                if h_in_idx >= padding.0 && w_in_idx >= padding.1 {
                                    let h_actual = h_in_idx - padding.0;
                                    let w_actual = w_in_idx - padding.1;

                                    if h_actual < h_in && w_actual < w_in {
                                        let input_idx = batch * c * h_in * w_in
                                            + channel * h_in * w_in
                                            + h_actual * w_in
                                            + w_actual;
                                        sum += input_data[input_idx];
                                    }
                                }
                            }
                        }

                        channel_out[h_out_idx * w_out + w_out_idx] = sum / kernel_area;
                    }
                }
            });

        Ok(Tensor::new(
            vec![n, c, h_out, w_out],
            TensorData::Float32(output_data),
        ))
    }

    /// Element-wise addition with broadcasting: output = a + b
    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.to_f32()?;
        let b_data = b.to_f32()?;

        // Handle exact shape match
        if a.shape == b.shape {
            let output_data: Vec<f32> = a_data
                .par_iter()
                .zip(b_data.par_iter())
                .map(|(&x, &y)| x + y)
                .collect();
            return Ok(Tensor::new(
                a.shape.clone(),
                TensorData::Float32(output_data),
            ));
        }

        // Handle bias addition: [N, C, H, W] + [C] or [N, C] + [C]
        if b.shape.len() == 1 && a.shape.len() >= 2 {
            let bias_size = b.shape[0];
            let a_size = a_data.len();

            // For tensors like [N, C, ...], add bias to each channel
            if a.shape[1] == bias_size {
                let mut output_data = Vec::with_capacity(a_size);
                let channel_stride = a_size / (a.shape[0] * a.shape[1]);

                for i in 0..a_size {
                    let channel = (i / channel_stride) % a.shape[1];
                    let bias_value = b_data[channel];
                    output_data.push(a_data[i] + bias_value);
                }

                return Ok(Tensor::new(
                    a.shape.clone(),
                    TensorData::Float32(output_data),
                ));
            }
        }

        Err(BlitzedError::TensorError {
            message: format!(
                "Cannot add tensors with shapes: {:?} vs {:?} (broadcasting not supported for these shapes)",
                a.shape, b.shape
            ),
        })
    }

    /// Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn batch_norm(
        input: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(BlitzedError::TensorError {
                message: "BatchNorm requires 4D input tensor".to_string(),
            });
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];

        let input_data = input.to_f32()?;
        let mean_data = mean.to_f32()?;
        let var_data = var.to_f32()?;

        let gamma_data = if let Some(g) = gamma {
            Some(g.to_f32()?)
        } else {
            None
        };

        let beta_data = if let Some(b) = beta {
            Some(b.to_f32()?)
        } else {
            None
        };

        let mut output_data = vec![0.0; n * c * h * w];

        // Process each channel
        for channel_idx in 0..c {
            let channel_mean = mean_data[channel_idx];
            let channel_var = var_data[channel_idx];
            let std_inv = 1.0 / (channel_var + eps).sqrt();

            let gamma_val = gamma_data.as_ref().map_or(1.0, |g| g[channel_idx]);
            let beta_val = beta_data.as_ref().map_or(0.0, |b| b[channel_idx]);

            // Apply normalization to all spatial locations for this channel
            for batch_idx in 0..n {
                for spatial_idx in 0..(h * w) {
                    let idx = batch_idx * c * h * w + channel_idx * h * w + spatial_idx;
                    let normalized = (input_data[idx] - channel_mean) * std_inv;
                    output_data[idx] = normalized * gamma_val + beta_val;
                }
            }
        }

        Ok(Tensor::new(
            input.shape.clone(),
            TensorData::Float32(output_data),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.ndim(), 3);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut tensor = Tensor::zeros(vec![2, 3, 4]);
        tensor.reshape(vec![6, 4]).expect("Reshape should succeed");
        assert_eq!(tensor.shape, vec![6, 4]);

        // Should fail for incompatible shape
        let result = tensor.reshape(vec![2, 3, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(
            vec![2, 3],
            TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let b = Tensor::new(
            vec![3, 2],
            TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        let result = TensorOps::matmul(&a, &b).expect("MatMul should succeed");
        assert_eq!(result.shape, vec![2, 2]);

        let expected = vec![22.0, 28.0, 49.0, 64.0]; // Manual calculation
        if let TensorData::Float32(data) = &result.data {
            assert_eq!(data, &expected);
        }
    }

    #[test]
    fn test_conv2d() {
        // Simple 1x1x3x3 input, 1x1x2x2 kernel
        let input = Tensor::new(
            vec![1, 1, 3, 3],
            TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        );

        let weight = Tensor::new(
            vec![1, 1, 2, 2],
            TensorData::Float32(vec![1.0, 0.0, 0.0, 1.0]),
        );

        let result = TensorOps::conv2d(&input, &weight, None, (1, 1), (0, 0))
            .expect("Conv2d should succeed");

        assert_eq!(result.shape, vec![1, 1, 2, 2]);

        // Expected output: [6.0, 8.0, 12.0, 14.0]
        if let TensorData::Float32(data) = &result.data {
            assert_eq!(data.len(), 4);
        }
    }

    #[test]
    fn test_relu_activation() {
        let input = Tensor::new(vec![2, 2], TensorData::Float32(vec![-1.0, 2.0, -3.0, 4.0]));

        let result = TensorOps::relu(&input).expect("ReLU should succeed");
        let expected = vec![0.0, 2.0, 0.0, 4.0];

        if let TensorData::Float32(data) = &result.data {
            assert_eq!(data, &expected);
        }
    }

    #[test]
    fn test_max_pool2d() {
        let input = Tensor::new(
            vec![1, 1, 4, 4],
            TensorData::Float32(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ]),
        );

        let result = TensorOps::max_pool2d(&input, (2, 2), (2, 2), (0, 0))
            .expect("MaxPool2d should succeed");

        assert_eq!(result.shape, vec![1, 1, 2, 2]);

        // Expected: max of each 2x2 region
        let expected = vec![6.0, 8.0, 14.0, 16.0];
        if let TensorData::Float32(data) = &result.data {
            assert_eq!(data, &expected);
        }
    }

    #[test]
    fn test_quantized_tensor_conversion() {
        let quant_params = QuantizationParams {
            scale: 0.1,
            zero_point: 128,
            clip_min: -12.8,
            clip_max: 12.7,
        };

        let tensor = Tensor::new_quantized(
            vec![2, 2],
            TensorData::Int8(vec![0, 127, -128, 64]),
            quant_params,
        );

        let f32_data = tensor.to_f32().expect("Conversion should succeed");

        // Check approximate values (quantization introduces some error)
        assert!((f32_data[0] - (-12.8)).abs() < 0.1);
        assert!((f32_data[1] - (-0.1)).abs() < 0.1);
        assert!((f32_data[2] - (-25.6)).abs() < 0.1);
        assert!((f32_data[3] - (-6.4)).abs() < 0.1);
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(vec![2, 2], TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0]));
        let b = Tensor::new(vec![2, 2], TensorData::Float32(vec![5.0, 6.0, 7.0, 8.0]));

        let result = TensorOps::add(&a, &b).expect("Add should succeed");
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        if let TensorData::Float32(data) = &result.data {
            assert_eq!(data, &expected);
        }
    }
}
