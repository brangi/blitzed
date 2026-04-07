// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "touch_sensor.h"

// Set to 1 for data collection mode (streams raw touch data over serial)
// Set to 0 for inference mode (real-time gesture classification)
#define MODE_DATA_COLLECTION 0

static const char *TAG = "blitzed";

#if MODE_DATA_COLLECTION

/**
 * Data collection mode: streams raw touch pad readings at 50Hz over serial.
 * Format: TOUCH,<timestamp_ms>,<pad0>,<pad1>,<pad2>,<pad3>
 *
 * A Python script on the host reads this stream and labels gesture windows.
 */
void app_main(void) {
    ESP_LOGI(TAG, "Blitzed Touch Gesture — Data Collection Mode");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());

    if (touch_sensor_init() != 0) {
        ESP_LOGE(TAG, "Touch sensor init failed");
        return;
    }

    // Print baselines and thresholds for the collection script
    for (int i = 0; i < NUM_TOUCH_PADS; i++) {
        printf("BASELINE,%d,%u,%u\n", i, touch_sensor_get_baseline(i),
               touch_sensor_get_threshold(i));
    }
    printf("READY\n");
    fflush(stdout);

    ESP_LOGI(TAG, "Streaming touch data at 50Hz. Touch GPIO 4, 12, 14, 27.");

    uint16_t readings[NUM_TOUCH_PADS];
    uint32_t start_ms = (uint32_t)(esp_timer_get_time() / 1000);

    while (1) {
        touch_sensor_read_all(readings);

        uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000) - start_ms;

        printf("TOUCH,%lu,%u,%u,%u,%u\n",
               (unsigned long)now_ms,
               readings[0], readings[1], readings[2], readings[3]);

        // 50Hz = 20ms between reads
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

#else // INFERENCE MODE

#include "feature_extract.h"
#include "blitzed_inference.h"
#include "blitzed_model_weights.h"

#define SAMPLE_RATE_HZ    50
#define SAMPLE_PERIOD_MS  (1000 / SAMPLE_RATE_HZ)
#define WINDOW_STEPS      75   // 1.5 seconds at 50Hz
#define INACTIVITY_STEPS  25   // 0.5 seconds of no activity to end window

static const char *gesture_labels[] = {
    "swipe_right", "swipe_left", "single_tap", "double_tap", "long_press"
};

void app_main(void) {
    ESP_LOGI(TAG, "Blitzed Touch Gesture Classifier");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

    if (touch_sensor_init() != 0) {
        ESP_LOGE(TAG, "Touch sensor init failed");
        return;
    }

    if (blitzed_model_init() != 0) {
        ESP_LOGE(TAG, "Model init failed");
        return;
    }

    ESP_LOGI(TAG, "Touch pins: GPIO 4, 12, 14, 27");
    ESP_LOGI(TAG, "Gestures: swipe_right, swipe_left, single_tap, double_tap, long_press");
    ESP_LOGI(TAG, "Waiting for touch...");

    uint16_t window[WINDOW_STEPS][NUM_TOUCH_PADS];
    uint16_t readings[NUM_TOUCH_PADS];
    int gesture_count = 0;

    while (1) {
        // Wait for any pad to be touched
        touch_sensor_read_all(readings);
        bool any_touched = false;
        for (int i = 0; i < NUM_TOUCH_PADS; i++) {
            if (touch_sensor_is_touched(readings[i], i)) {
                any_touched = true;
                break;
            }
        }

        if (!any_touched) {
            vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
            continue;
        }

        // Gesture started — collect window
        int steps = 0;
        int inactive_count = 0;

        while (steps < WINDOW_STEPS && inactive_count < INACTIVITY_STEPS) {
            touch_sensor_read_all(window[steps]);

            bool active = false;
            for (int i = 0; i < NUM_TOUCH_PADS; i++) {
                if (touch_sensor_is_touched(window[steps][i], i)) {
                    active = true;
                    break;
                }
            }

            if (active) {
                inactive_count = 0;
            } else {
                inactive_count++;
            }

            steps++;
            vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
        }

        // Extract features
        float features[NUM_FEATURES];
        extract_touch_features(window, steps, features);

        // Quantize features to INT8
        int8_t input[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            input[i] = blitzed_quantize_input(features[i]);
        }

        // Run inference
        int8_t output[NUM_CLASSES];
        uint64_t t_start = esp_timer_get_time();
        blitzed_model_predict(input, NUM_FEATURES, output, NUM_CLASSES);
        uint64_t t_end = esp_timer_get_time();
        uint32_t latency_us = (uint32_t)(t_end - t_start);

        int predicted = blitzed_argmax(output, NUM_CLASSES);

        // Find confidence (difference between top two logits)
        int8_t best = -128, second = -128;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (output[i] > best) {
                second = best;
                best = output[i];
            } else if (output[i] > second) {
                second = output[i];
            }
        }
        float confidence = blitzed_dequantize_output(best) - blitzed_dequantize_output(second);

        ESP_LOGI(TAG, "[%04d] Gesture: %-12s | Margin: %.2f | Latency: %lu us | Steps: %d",
                 gesture_count++,
                 gesture_labels[predicted],
                 confidence,
                 (unsigned long)latency_us,
                 steps);

        // Wait for all pads to be released before next detection
        while (1) {
            touch_sensor_read_all(readings);
            bool still_touched = false;
            for (int i = 0; i < NUM_TOUCH_PADS; i++) {
                if (touch_sensor_is_touched(readings[i], i)) {
                    still_touched = true;
                    break;
                }
            }
            if (!still_touched) break;
            vTaskDelay(pdMS_TO_TICKS(50));
        }
        vTaskDelay(pdMS_TO_TICKS(200));  // Debounce
    }
}

#endif // MODE_DATA_COLLECTION
