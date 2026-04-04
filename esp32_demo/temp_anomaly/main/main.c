#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "blitzed_inference.h"
#include "blitzed_model_weights.h"

static const char *TAG = "blitzed";

// ESP-IDF v5.x temperature sensor API
// Falls back to legacy driver for older SDK versions
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#include "driver/temperature_sensor.h"

static temperature_sensor_handle_t temp_sensor = NULL;

static esp_err_t temp_sensor_init(void) {
    temperature_sensor_config_t temp_sensor_config = TEMPERATURE_SENSOR_CONFIG_DEFAULT(-10, 80);
    esp_err_t err = temperature_sensor_install(&temp_sensor_config, &temp_sensor);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install temperature sensor: %s", esp_err_to_name(err));
        return err;
    }
    err = temperature_sensor_enable(temp_sensor);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable temperature sensor: %s", esp_err_to_name(err));
        return err;
    }
    return ESP_OK;
}

static esp_err_t temp_sensor_read_celsius(float *out_celsius) {
    return temperature_sensor_get_celsius(temp_sensor, out_celsius);
}

#else
// Legacy ESP-IDF v4.x API
#include "driver/temp_sensor.h"

static esp_err_t temp_sensor_init(void) {
    temp_sensor_config_t tsens_config = TSENS_CONFIG_DEFAULT();
    esp_err_t err = temp_sensor_set_config(tsens_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure temperature sensor: %s", esp_err_to_name(err));
        return err;
    }
    return temp_sensor_start();
}

static esp_err_t temp_sensor_read_celsius(float *out_celsius) {
    return temp_sensor_read_celsius(out_celsius);
}
#endif // ESP_IDF_VERSION

/**
 * Run inference benchmark: measure min/mean/max latency over N iterations.
 */
static void run_benchmark(int num_iterations) {
    ESP_LOGI(TAG, "Starting benchmark: %d iterations", num_iterations);

    int8_t input[1] = {0};  // Dummy input
    int8_t output[NUM_CLASSES];

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;
    uint64_t total_us = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Vary input slightly to prevent cache effects
        input[0] = (int8_t)(i % 127);

        uint64_t start = esp_timer_get_time();
        blitzed_model_predict(input, 1, output, NUM_CLASSES);
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
    ESP_LOGI(TAG, "Blitzed Temperature Anomaly Detector");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "Classes: normal | cold_warning | hot_warning | critical_overheat");

    // Initialize internal temperature sensor
    if (temp_sensor_init() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize temperature sensor");
        return;
    }
    ESP_LOGI(TAG, "Temperature sensor initialized");

    // Initialize inference engine
    if (blitzed_model_init() != 0) {
        ESP_LOGE(TAG, "Failed to initialize model");
        return;
    }
    ESP_LOGI(TAG, "Model initialized successfully");
    ESP_LOGI(TAG, "Architecture: Dense(1,16)+ReLU+Dense(16,4) | INT8 quantized | %d params", 84);

    // Run benchmark on startup
    run_benchmark(10000);

    ESP_LOGI(TAG, "Starting real-time temperature classification loop");
    ESP_LOGI(TAG, "");

    // Main classification loop
    int8_t input[1];
    int8_t output[NUM_CLASSES];
    int iteration = 0;

    while (1) {
        // Read internal temperature sensor
        float temp_celsius = 0.0f;
        esp_err_t err = temp_sensor_read_celsius(&temp_celsius);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to read temperature sensor: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // Quantize to INT8 (normalizes by INPUT_MAX=120.0 then scales to INT8)
        input[0] = blitzed_quantize_input(temp_celsius);

        // Run inference and measure latency
        uint64_t start = esp_timer_get_time();
        int ret = blitzed_model_predict(input, 1, output, NUM_CLASSES);
        uint64_t end = esp_timer_get_time();

        if (ret != 0) {
            ESP_LOGE(TAG, "Inference failed with error %d", ret);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        uint32_t inference_us = (uint32_t)(end - start);

        // Get predicted class
        int predicted_class = blitzed_argmax(output, NUM_CLASSES);

        // Dequantize output logits for display
        float logit_0 = blitzed_dequantize_output(output[0]);
        float logit_1 = blitzed_dequantize_output(output[1]);
        float logit_2 = blitzed_dequantize_output(output[2]);
        float logit_3 = blitzed_dequantize_output(output[3]);

        // Log results every iteration at 1-second intervals
        ESP_LOGI(TAG, "[%05d] Temp: %.1f\xc2\xb0""C | Predicted: %s | "
                      "Logits: [%.2f, %.2f, %.2f, %.2f] | Latency: %lu us",
                 iteration++,
                 temp_celsius,
                 class_labels[predicted_class],
                 logit_0, logit_1, logit_2, logit_3,
                 inference_us);

        // Wait 1 second before next reading
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
