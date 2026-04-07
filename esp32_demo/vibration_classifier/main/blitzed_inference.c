#include "blitzed_inference.h"
#include "blitzed_model_weights.h"
#include <string.h>
#include <esp_timer.h>

static uint32_t last_inference_us = 0;

int blitzed_model_init(void) {
    // Weights are const arrays in flash, no dynamic initialization needed
    return 0;
}

int blitzed_model_predict(const int8_t* input, int input_len,
                           int8_t* output, int output_len) {
    if (!input || !output) {
        return -1;
    }

    uint64_t start = esp_timer_get_time();

    // Layer 1: Dense(3, 32) + ReLU
    // Weights layout: layer1_weights[input_idx * LAYER1_OUTPUT_SIZE + output_idx]
    // Accumulate in INT32 to prevent overflow: acc = bias + sum_i(input[i] * weight[i][j])
    int32_t hidden[LAYER1_OUTPUT_SIZE];

    for (int j = 0; j < LAYER1_OUTPUT_SIZE; j++) {
        int32_t acc = layer1_bias[j];

        for (int i = 0; i < LAYER1_INPUT_SIZE && i < input_len; i++) {
            // INT8 * INT8 promotes to INT16 in C, but we accumulate in INT32
            // to safely handle up to 3 * 127 * 127 = 48387 without overflow
            acc += (int32_t)input[i] * (int32_t)layer1_weights[i * LAYER1_OUTPUT_SIZE + j];
        }

        // Requantize from accumulated scale to layer1_output_scale
        //   acc_float = acc * input_scale * weight_scale
        //   hidden_int8 = round(acc_float / output_scale) + output_zero_point
        float float_val = (float)acc * INPUT_SCALE * layer1_weight_scale;
        int32_t requantized = (int32_t)(float_val / layer1_output_scale)
                              + layer1_output_zero_point;

        // ReLU activation: max(zero_point, x) in quantized space
        if (requantized < layer1_output_zero_point) {
            requantized = layer1_output_zero_point;
        }

        // Clamp to INT8 range [-128, 127]
        if (requantized > 127) requantized = 127;
        if (requantized < -128) requantized = -128;

        hidden[j] = requantized;
    }

    // Layer 2: Dense(32, 4) — output layer
    // Weights layout: layer2_weights[input_idx * LAYER2_OUTPUT_SIZE + output_idx]
    int out_len = LAYER2_OUTPUT_SIZE < output_len ? LAYER2_OUTPUT_SIZE : output_len;

    for (int j = 0; j < out_len; j++) {
        int32_t acc = layer2_bias[j];

        for (int i = 0; i < LAYER2_INPUT_SIZE; i++) {
            // hidden[i] is INT32 holding a clamped INT8 value
            acc += hidden[i] * (int32_t)layer2_weights[i * LAYER2_OUTPUT_SIZE + j];
        }

        // Requantize to output scale
        //   acc_float = acc * layer1_output_scale * layer2_weight_scale
        //   output_int8 = round(acc_float / output_scale) + output_zero_point
        float float_val = (float)acc * layer1_output_scale * layer2_weight_scale;
        int32_t requantized = (int32_t)(float_val / layer2_output_scale)
                              + layer2_output_zero_point;

        // Clamp to INT8 range
        if (requantized > 127) requantized = 127;
        if (requantized < -128) requantized = -128;

        output[j] = (int8_t)requantized;
    }

    uint64_t end = esp_timer_get_time();
    last_inference_us = (uint32_t)(end - start);

    return 0;
}

int8_t blitzed_quantize_input(float rms_g) {
    // Normalize to [0, 1] range: divide by INPUT_NORM_FACTOR (12.0g max RMS range)
    float normalized = rms_g / INPUT_NORM_FACTOR;

    // Clamp to valid range
    if (normalized > 1.0f) normalized = 1.0f;
    if (normalized < 0.0f) normalized = 0.0f;

    // Quantize: q = round(normalized / scale) + zero_point
    int32_t quantized = (int32_t)(normalized / INPUT_SCALE) + INPUT_ZERO_POINT;

    // Clamp to INT8 range
    if (quantized > 127) quantized = 127;
    if (quantized < -128) quantized = -128;

    return (int8_t)quantized;
}

float blitzed_dequantize_output(int8_t value) {
    // Dequantize: real_value = (q - zero_point) * scale
    return ((float)value - (float)layer2_output_zero_point) * layer2_output_scale;
}

int blitzed_argmax(const int8_t* output, int len) {
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

uint32_t blitzed_get_last_inference_us(void) {
    return last_inference_us;
}
