#ifndef BLITZED_INFERENCE_H
#define BLITZED_INFERENCE_H

#include <stdint.h>

/**
 * Initialize the Blitzed inference engine for temperature anomaly detection.
 * Returns 0 on success, negative on error.
 */
int blitzed_model_init(void);

/**
 * Run INT8 quantized inference for temperature classification.
 *
 * @param input     Quantized INT8 input tensor (single temperature value)
 * @param input_len Length of input tensor (must be 1)
 * @param output    Quantized INT8 output tensor (4 class logits, caller-allocated)
 * @param output_len Length of output tensor (must be >= 4)
 * @return 0 on success, negative on error
 */
int blitzed_model_predict(const int8_t* input, int input_len, int8_t* output, int output_len);

/**
 * Quantize a raw temperature reading (in degrees Celsius) to INT8.
 *
 * The temperature is first normalized by dividing by INPUT_MAX (120.0), then
 * quantized using the model's input scale. Valid range: -10°C to 120°C.
 *
 * @param temp_celsius Raw temperature reading in degrees Celsius
 * @return Quantized INT8 value
 */
int8_t blitzed_quantize_input(float temp_celsius);

/**
 * Dequantize INT8 output logit to float.
 *
 * @param value Quantized output value
 * @return Dequantized float logit value
 */
float blitzed_dequantize_output(int8_t value);

/**
 * Find index of maximum value in output tensor (argmax for classification).
 *
 * @param output INT8 output tensor
 * @param len    Length of tensor
 * @return Index of maximum value, or -1 on invalid input
 */
int blitzed_argmax(const int8_t* output, int len);

/**
 * Get last inference time in microseconds.
 *
 * @return Inference latency in microseconds
 */
uint32_t blitzed_get_last_inference_us(void);

#endif // BLITZED_INFERENCE_H
