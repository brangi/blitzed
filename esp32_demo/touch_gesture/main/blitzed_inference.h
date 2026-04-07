// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#ifndef BLITZED_INFERENCE_H
#define BLITZED_INFERENCE_H

#include <stdint.h>

/**
 * Initialize the Blitzed inference engine.
 * Returns 0 on success, negative on error.
 */
int blitzed_model_init(void);

/**
 * Run INT8 quantized inference.
 *
 * Executes a 2-layer dense network: Dense(20, 32) + ReLU + Dense(32, 5).
 * All arithmetic is performed in INT32 accumulators to prevent overflow,
 * with requantization between layers.
 *
 * @param input      Quantized INT8 input tensor (20 elements: touch features)
 * @param input_len  Length of input tensor (must be 20)
 * @param output     INT8 output logits tensor (will be populated, 5 elements)
 * @param output_len Length of output tensor (must be >= 5)
 * @return 0 on success, negative on error
 */
int blitzed_model_predict(const int8_t* input, int input_len,
                           int8_t* output, int output_len);

/**
 * Quantize a single normalized touch feature to INT8.
 *
 * Features produced by extract_touch_features() are already normalized to
 * [0, 1], so INPUT_NORM_FACTOR = 1.0 and no division is applied before
 * quantization. This keeps the quantize path consistent with the vibration
 * classifier while correctly handling features that are already in range.
 *
 * @param value  Normalized feature value in [0, 1]
 * @return Quantized INT8 value
 */
int8_t blitzed_quantize_input(float value);

/**
 * Dequantize an INT8 output logit to float.
 *
 * @param value  Quantized output value
 * @return Dequantized float value
 */
float blitzed_dequantize_output(int8_t value);

/**
 * Find the index of the maximum value in the output tensor (argmax).
 *
 * @param output  INT8 output tensor
 * @param len     Length of tensor
 * @return Index of maximum value (predicted class), or -1 on error
 */
int blitzed_argmax(const int8_t* output, int len);

/**
 * Get the latency of the last inference call in microseconds.
 *
 * @return Inference latency in microseconds
 */
uint32_t blitzed_get_last_inference_us(void);

#endif // BLITZED_INFERENCE_H
