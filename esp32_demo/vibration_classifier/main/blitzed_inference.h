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
 * Executes a 2-layer dense network: Dense(3, 32) + ReLU + Dense(32, 4).
 * All arithmetic is performed in INT32 accumulators to prevent overflow,
 * with requantization between layers.
 *
 * @param input      Quantized INT8 input tensor (3 elements: rms_x, rms_y, rms_z)
 * @param input_len  Length of input tensor (must be 3)
 * @param output     INT8 output logits tensor (will be populated, 4 elements)
 * @param output_len Length of output tensor (must be >= 4)
 * @return 0 on success, negative on error
 */
int blitzed_model_predict(const int8_t* input, int input_len,
                           int8_t* output, int output_len);

/**
 * Quantize a single raw RMS accelerometer value to INT8.
 *
 * Normalizes by INPUT_NORM_FACTOR (12.0g) to map typical vibration RMS
 * range to [0, 1], then applies INT8 quantization.
 *
 * @param rms_g  Raw RMS acceleration in g units
 * @return Quantized INT8 value
 */
int8_t blitzed_quantize_input(float rms_g);

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
