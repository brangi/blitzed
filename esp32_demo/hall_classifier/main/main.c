#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "driver/adc.h"
#include "esp_timer.h"
#include "blitzed_inference.h"
#include "blitzed_model_weights.h"

static const char *TAG = "blitzed";

/**
 * Read the built-in hall effect sensor via ADC1.
 * The ESP32 hall sensor is wired to ADC1 channels 0 and 3.
 * We read channel 0 as a proxy — values shift when a magnet is nearby.
 * Returns raw ADC value (0-4095 at 12-bit, ~1800-2200 baseline, shifts with magnetic field).
 */
static int read_hall_sensor(void) {
    return adc1_get_raw(ADC1_CHANNEL_0);
}

/**
 * Run inference benchmark: measure min/mean/max latency over N iterations
 */
static void run_benchmark(int num_iterations) {
    ESP_LOGI(TAG, "Starting benchmark: %d iterations", num_iterations);

    int8_t input[1] = {0};  // Dummy input
    int8_t output[3];

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;
    uint64_t total_us = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Vary input slightly to prevent cache effects
        input[0] = (int8_t)(i % 127);

        uint64_t start = esp_timer_get_time();
        blitzed_model_predict(input, 1, output, 3);
        uint64_t end = esp_timer_get_time();

        uint32_t latency_us = (uint32_t)(end - start);

        if (latency_us < min_us) min_us = latency_us;
        if (latency_us > max_us) max_us = latency_us;
        total_us += latency_us;
    }

    uint32_t mean_us = (uint32_t)(total_us / num_iterations);

    ESP_LOGI(TAG, "Benchmark complete:");
    ESP_LOGI(TAG, "  Min latency:  %lu us", min_us);
    ESP_LOGI(TAG, "  Mean latency: %lu us", mean_us);
    ESP_LOGI(TAG, "  Max latency:  %lu us", max_us);
    ESP_LOGI(TAG, "  Throughput:   %lu inferences/sec", 1000000UL / mean_us);
}

void app_main(void) {
    ESP_LOGI(TAG, "Blitzed Hall Sensor Classifier");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());

    // Initialize ADC1 for hall sensor (channel 0, 12-bit, no attenuation)
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_0);

    // Initialize inference engine
    if (blitzed_model_init() != 0) {
        ESP_LOGE(TAG, "Failed to initialize model");
        return;
    }
    ESP_LOGI(TAG, "Model initialized successfully");

    // Run benchmark on startup
    run_benchmark(10000);

    ESP_LOGI(TAG, "Starting real-time classification loop");
    ESP_LOGI(TAG, "Bring a magnet near the ESP32 chip to see predictions change");
    ESP_LOGI(TAG, "");

    // Main classification loop
    int8_t input[1];
    int8_t output[3];
    int iteration = 0;

    while (1) {
        // Read hall sensor
        int hall_value = read_hall_sensor();

        // Quantize to INT8
        input[0] = blitzed_quantize_input((float)hall_value);

        // Run inference
        uint64_t start = esp_timer_get_time();
        int ret = blitzed_model_predict(input, 1, output, 3);
        uint64_t end = esp_timer_get_time();

        if (ret != 0) {
            ESP_LOGE(TAG, "Inference failed with error %d", ret);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        uint32_t inference_us = (uint32_t)(end - start);

        // Get predicted class
        int predicted_class = blitzed_argmax(output, 3);

        // Dequantize output logits for display
        float logit_0 = blitzed_dequantize_output(output[0]);
        float logit_1 = blitzed_dequantize_output(output[1]);
        float logit_2 = blitzed_dequantize_output(output[2]);

        // Log results every iteration
        ESP_LOGI(TAG, "[%05d] Hall: %4d | Predicted: %s | Logits: [%.2f, %.2f, %.2f] | "
                      "Latency: %lu us | Heap: %lu bytes",
                 iteration++,
                 hall_value,
                 class_labels[predicted_class],
                 logit_0, logit_1, logit_2,
                 inference_us,
                 esp_get_free_heap_size());

        // Wait 500ms before next reading
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
