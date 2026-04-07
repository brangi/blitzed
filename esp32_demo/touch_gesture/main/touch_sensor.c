// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#include "touch_sensor.h"
#include "driver/touch_pad.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "touch";

// ESP32 touch pad numbers for our 4 pads
// Physical GPIO: 4, 12, 14, 27
static const touch_pad_t pad_map[NUM_TOUCH_PADS] = {
    TOUCH_PAD_NUM0,  // GPIO4
    TOUCH_PAD_NUM5,  // GPIO12
    TOUCH_PAD_NUM6,  // GPIO14
    TOUCH_PAD_NUM7,  // GPIO27
};

static uint16_t baselines[NUM_TOUCH_PADS];
static uint16_t thresholds[NUM_TOUCH_PADS];

int touch_sensor_init(void) {
    esp_err_t ret;

    ret = touch_pad_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "touch_pad_init failed: %s", esp_err_to_name(ret));
        return -1;
    }

    // Set reference voltage for charge/discharge cycle
    touch_pad_set_voltage(TOUCH_HVOLT_2V7, TOUCH_LVOLT_0V5, TOUCH_HVOLT_ATTEN_1V);

    // Configure each pad with no interrupt threshold
    for (int i = 0; i < NUM_TOUCH_PADS; i++) {
        ret = touch_pad_config(pad_map[i], 0);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "touch_pad_config(%d) failed: %s", i, esp_err_to_name(ret));
            return -1;
        }
    }

    // Calibrate: use raw reads (no filter) to get baseline
    ESP_LOGI(TAG, "Calibrating touch pads (do not touch)...");
    vTaskDelay(pdMS_TO_TICKS(200));

    for (int i = 0; i < NUM_TOUCH_PADS; i++) {
        uint32_t sum = 0;
        for (int s = 0; s < 5; s++) {
            uint16_t val;
            touch_pad_read(pad_map[i], &val);
            sum += val;
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        baselines[i] = (uint16_t)(sum / 5);
        // Touch causes values to DROP on ESP32 v1
        // Threshold at 70% of baseline
        thresholds[i] = (uint16_t)(baselines[i] * 0.7f);

        ESP_LOGI(TAG, "Pad %d (GPIO%d): baseline=%u, threshold=%u",
                 i, (int[]){4, 12, 14, 27}[i], baselines[i], thresholds[i]);
    }

    ESP_LOGI(TAG, "Touch calibration complete");
    return 0;
}

void touch_sensor_read_all(uint16_t out[NUM_TOUCH_PADS]) {
    for (int i = 0; i < NUM_TOUCH_PADS; i++) {
        touch_pad_read(pad_map[i], &out[i]);
    }
}

bool touch_sensor_is_touched(uint16_t raw, int pad_idx) {
    if (pad_idx < 0 || pad_idx >= NUM_TOUCH_PADS) return false;
    return raw < thresholds[pad_idx];
}

uint16_t touch_sensor_get_threshold(int pad_idx) {
    if (pad_idx < 0 || pad_idx >= NUM_TOUCH_PADS) return 0;
    return thresholds[pad_idx];
}

uint16_t touch_sensor_get_baseline(int pad_idx) {
    if (pad_idx < 0 || pad_idx >= NUM_TOUCH_PADS) return 0;
    return baselines[pad_idx];
}
