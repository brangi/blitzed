#ifndef BLITZED_INFERENCE_H
#define BLITZED_INFERENCE_H

#include <stdint.h>

/**
 * Initialize the Blitzed inference engine
 * Returns 0 on success, negative on error
 */
int blitzed_model_init(void);

/**
 * Run INT8 quantized inference
 *
 * @param input Quantized INT8 input tensor
 * @param input_len Length of input tensor
 * @param output Quantized INT8 output tensor (will be populated)
 * @param output_len Length of output tensor
 * @return 0 on success, negative on error
 */
int blitzed_model_predict(const int8_t* input, int input_len, int8_t* output, int output_len);

/**
 * Quantize a raw hall sensor reading to INT8
 *
 * @param raw_hall_value Raw hall sensor value (typically -500 to +500)
 * @return Quantized INT8 value
 */
int8_t blitzed_quantize_input(float raw_hall_value);

/**
 * Dequantize INT8 output logit to float
 *
 * @param value Quantized output value
 * @return Dequantized float value
 */
float blitzed_dequantize_output(int8_t value);

/**
 * Find index of maximum value in output tensor (argmax for classification)
 *
 * @param output INT8 output tensor
 * @param len Length of tensor
 * @return Index of maximum value
 */
int blitzed_argmax(const int8_t* output, int len);

/**
 * Get last inference time in microseconds
 *
 * @return Inference latency in microseconds
 */
uint32_t blitzed_get_last_inference_us(void);

#endif // BLITZED_INFERENCE_H
