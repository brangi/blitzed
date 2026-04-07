#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/i2c.h"
#include "mpu6050.h"
#include "blitzed_inference.h"
#include "blitzed_model_weights.h"

static const char *TAG = "blitzed";

// MPU6050 configuration
#define I2C_PORT     I2C_NUM_0
#define MPU6050_ADDR MPU6050_ADDR_DEFAULT

// Number of accelerometer samples to collect per inference window
// At ~125Hz sample rate, 32 samples = ~256ms of vibration data
#define ACCEL_RMS_SAMPLES 32

/**
 * Run inference benchmark: measure min/mean/max latency over N iterations.
 * Uses dummy inputs to stress the INT8 matmul kernel without blocking on I2C.
 */
static void run_benchmark(int num_iterations) {
    ESP_LOGI(TAG, "Starting benchmark: %d iterations", num_iterations);

    // Dummy inputs with non-zero values to exercise all multiply-accumulate paths
    int8_t input[3];
    int8_t output[4];

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;
    uint64_t total_us = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Vary inputs across INT8 range to prevent branch prediction shortcuts
        input[0] = (int8_t)((i * 37) % 127);
        input[1] = (int8_t)((i * 53) % 127);
        input[2] = (int8_t)((i * 71) % 127);

        uint64_t start = esp_timer_get_time();
        blitzed_model_predict(input, 3, output, 4);
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
    ESP_LOGI(TAG, "Blitzed Vibration Pattern Classifier");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "Model: Dense(3,32)+ReLU+Dense(32,4) | 164 params | INT8");

    // Initialize MPU6050 accelerometer on I2C0, GPIO21=SDA, GPIO22=SCL
    esp_err_t ret = mpu6050_init(I2C_PORT, MPU6050_ADDR);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize MPU6050: %s", esp_err_to_name(ret));
        ESP_LOGE(TAG, "Check wiring: SDA=GPIO21, SCL=GPIO22, VCC=3.3V, AD0=GND");
        return;
    }

    // Initialize inference engine (verifies flash-resident weight arrays are accessible)
    if (blitzed_model_init() != 0) {
        ESP_LOGE(TAG, "Failed to initialize inference engine");
        return;
    }
    ESP_LOGI(TAG, "Inference engine initialized successfully");

    // Run benchmark to characterize INT8 kernel performance before entering main loop
    run_benchmark(10000);

    ESP_LOGI(TAG, "Starting real-time vibration classification loop");
    ESP_LOGI(TAG, "Collecting %d samples per window at ~125Hz", ACCEL_RMS_SAMPLES);
    ESP_LOGI(TAG, "");

    // Main classification loop
    int8_t input[3];
    int8_t output[4];
    int iteration = 0;

    while (1) {
        float rms_x, rms_y, rms_z;

        // Collect ACCEL_RMS_SAMPLES accelerometer readings and compute per-axis RMS
        ret = mpu6050_read_accel_rms(&rms_x, &rms_y, &rms_z, ACCEL_RMS_SAMPLES);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read accelerometer: %s", esp_err_to_name(ret));
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // Quantize 3 RMS features to INT8
        // Each value is normalized by INPUT_NORM_FACTOR (12.0g) before quantizing
        input[0] = blitzed_quantize_input(rms_x);
        input[1] = blitzed_quantize_input(rms_y);
        input[2] = blitzed_quantize_input(rms_z);

        // Run INT8 inference
        uint64_t start = esp_timer_get_time();
        ret = blitzed_model_predict(input, 3, output, 4);
        uint64_t end = esp_timer_get_time();

        if (ret != 0) {
            ESP_LOGE(TAG, "Inference failed with error %d", ret);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        uint32_t inference_us = (uint32_t)(end - start);

        // Argmax over 4 class logits
        int predicted_class = blitzed_argmax(output, 4);

        // Log in the format requested: RMS values, class name, latency
        ESP_LOGI(TAG, "[%05d] Accel RMS: [%.2f, %.2f, %.2f] g | Predicted: %s | Latency: %lu us",
                 iteration++,
                 rms_x, rms_y, rms_z,
                 class_labels[predicted_class],
                 inference_us);

        // Wait 500ms between classification windows
        // The RMS collection above takes ~256ms, so total cycle is ~756ms
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
