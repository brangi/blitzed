#include "blitzed_inference.h"
#include "blitzed_model_weights.h"
#include <string.h>
#include "esp_timer.h"

static uint32_t last_inference_us = 0;

int blitzed_model_init(void)
{
    // Weights are const arrays in flash; no dynamic initialization needed.
    // Validate compile-time dimensions for a basic sanity check.
#if LAYER1_INPUT_SIZE != 4
#error "Expected LAYER1_INPUT_SIZE == 4"
#endif
#if LAYER1_OUTPUT_SIZE != 32
#error "Expected LAYER1_OUTPUT_SIZE == 32"
#endif
#if LAYER2_INPUT_SIZE != 32
#error "Expected LAYER2_INPUT_SIZE == 32"
#endif
#if LAYER2_OUTPUT_SIZE != 4
#error "Expected LAYER2_OUTPUT_SIZE == 4"
#endif
    return 0;
}

int blitzed_model_predict(const int8_t *input, int input_len,
                           int8_t *output, int output_len)
{
    if (!input || !output) {
        return -1;
    }
    if (input_len < LAYER1_INPUT_SIZE || output_len < LAYER2_OUTPUT_SIZE) {
        return -2;
    }

    uint64_t start = esp_timer_get_time();

    // ----------------------------------------------------------------
    // Layer 1: Dense(4, 32) + ReLU
    //
    // Weight layout: layer1_weights[i * LAYER1_OUTPUT_SIZE + j]
    //   where i = input neuron index (0..3)
    //         j = output neuron index (0..31)
    //
    // INT32 accumulator prevents overflow: each INT8*INT8 product fits in
    // INT16, and we accumulate LAYER1_INPUT_SIZE=4 such products, so the
    // maximum accumulation is 4 * 127 * 127 = 64516 — well within INT32.
    // The bias is pre-scaled to the same quantization space as the matmul
    // result, so we can add it directly.
    // ----------------------------------------------------------------
    int32_t hidden[LAYER1_OUTPUT_SIZE];

    for (int j = 0; j < LAYER1_OUTPUT_SIZE; j++) {
        int32_t acc = layer1_bias[j];

        for (int i = 0; i < LAYER1_INPUT_SIZE; i++) {
            acc += (int32_t)input[i] * (int32_t)layer1_weights[i * LAYER1_OUTPUT_SIZE + j];
        }

        // Requantize: bring accumulated result from the matmul scale
        // (input_scale * weight_scale) down to the inter-layer scale
        // (layer1_output_scale), which is what layer 2 expects.
        //
        //   float_val        = acc * INPUT_SCALE * layer1_weight_scale
        //   requantized_int  = round(float_val / layer1_output_scale)
        //                    + layer1_output_zero_point
        float float_val = (float)acc * INPUT_SCALE * layer1_weight_scale;
        int32_t requantized = (int32_t)(float_val / layer1_output_scale)
                              + layer1_output_zero_point;

        // ReLU in quantized space: values below zero_point represent
        // negative floats, so clamp to zero_point.
        if (requantized < layer1_output_zero_point) {
            requantized = layer1_output_zero_point;
        }

        // Clamp to INT8 range [-128, 127]
        if (requantized > 127)  requantized = 127;
        if (requantized < -128) requantized = -128;

        hidden[j] = requantized;
    }

    // ----------------------------------------------------------------
    // Layer 2: Dense(32, 4) — output layer, no activation
    //
    // Weight layout: layer2_weights[i * LAYER2_OUTPUT_SIZE + j]
    //   where i = hidden neuron index (0..31)
    //         j = output class index (0..3)
    //
    // Maximum accumulation: 32 * 127 * 127 = 516128 — within INT32.
    // ----------------------------------------------------------------
    int out_len = LAYER2_OUTPUT_SIZE < output_len ? LAYER2_OUTPUT_SIZE : output_len;

    for (int j = 0; j < out_len; j++) {
        int32_t acc = layer2_bias[j];

        for (int i = 0; i < LAYER2_INPUT_SIZE; i++) {
            acc += (int32_t)hidden[i] * (int32_t)layer2_weights[i * LAYER2_OUTPUT_SIZE + j];
        }

        // Requantize from inter-layer scale to final output scale.
        //
        //   float_val   = acc * layer1_output_scale * layer2_weight_scale
        //   requantized = round(float_val / layer2_output_scale)
        //               + layer2_output_zero_point
        float float_val = (float)acc * layer1_output_scale * layer2_weight_scale;
        int32_t requantized = (int32_t)(float_val / layer2_output_scale)
                              + layer2_output_zero_point;

        // Clamp to INT8 range
        if (requantized > 127)  requantized = 127;
        if (requantized < -128) requantized = -128;

        output[j] = (int8_t)requantized;
    }

    uint64_t end = esp_timer_get_time();
    last_inference_us = (uint32_t)(end - start);

    return 0;
}

int8_t blitzed_quantize_input(float normalized_value)
{
    // normalized_value is already in approximately [-1, 1] (caller normalizes).
    // Clamp to valid range before quantizing.
    if (normalized_value >  1.0f) normalized_value =  1.0f;
    if (normalized_value < -1.0f) normalized_value = -1.0f;

    // q = round(v / scale) + zero_point
    int32_t quantized = (int32_t)(normalized_value / INPUT_SCALE) + INPUT_ZERO_POINT;

    if (quantized >  127) quantized =  127;
    if (quantized < -128) quantized = -128;

    return (int8_t)quantized;
}

float blitzed_dequantize_output(int8_t value)
{
    // real_value = (q - zero_point) * scale
    return ((float)value - (float)layer2_output_zero_point) * layer2_output_scale;
}

int blitzed_argmax(const int8_t *output, int len)
{
    if (!output || len <= 0) {
        return -1;
    }

    int max_idx = 0;
    int8_t max_val = output[0];

    for (int i = 1; i < len; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }

    return max_idx;
}

uint32_t blitzed_get_last_inference_us(void)
{
    return last_inference_us;
}
