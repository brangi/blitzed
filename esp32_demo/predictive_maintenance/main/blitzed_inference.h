#ifndef BLITZED_INFERENCE_H
#define BLITZED_INFERENCE_H

#include <stdint.h>

/**
 * Initialize the Blitzed inference engine.
 *
 * Weights are stored as static const arrays in flash; this call simply
 * validates that the model constants are self-consistent.
 *
 * @return 0 on success, negative on error
 */
int blitzed_model_init(void);

/**
 * Run INT8 quantized inference.
 *
 * Executes a 2-layer dense network: Dense(4,32)+ReLU -> Dense(32,4).
 * All arithmetic uses INT32 accumulators with requantization between layers.
 *
 * @param input      Quantized INT8 input tensor (must have length >= 4)
 * @param input_len  Length of input tensor (expected: 4)
 * @param output     INT8 output tensor to populate (must have length >= 4)
 * @param output_len Length of output buffer (expected: 4)
 * @return           0 on success, negative on error
 */
int blitzed_model_predict(const int8_t *input, int input_len,
                           int8_t *output, int output_len);

/**
 * Quantize a normalized sensor feature to INT8.
 *
 * The caller is responsible for normalizing the raw value before passing it
 * here (temperature / 120.0, acceleration / 12.0).
 *
 * @param normalized_value  Feature value already normalized to approximately [-1, 1]
 * @return                  Quantized INT8 value
 */
int8_t blitzed_quantize_input(float normalized_value);

/**
 * Dequantize an INT8 output logit to float.
 *
 * @param value  Quantized output value
 * @return       Dequantized float value
 */
float blitzed_dequantize_output(int8_t value);

/**
 * Return the index of the maximum value in an INT8 tensor (argmax).
 *
 * @param output  INT8 tensor
 * @param len     Number of elements
 * @return        Index of the maximum element, or -1 on invalid input
 */
int blitzed_argmax(const int8_t *output, int len);

/**
 * Return the latency of the most recent blitzed_model_predict() call.
 *
 * @return Inference latency in microseconds
 */
uint32_t blitzed_get_last_inference_us(void);

#endif // BLITZED_INFERENCE_H
