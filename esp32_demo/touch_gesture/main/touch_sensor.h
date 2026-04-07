// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#ifndef TOUCH_SENSOR_H
#define TOUCH_SENSOR_H

#include <stdint.h>
#include <stdbool.h>

#define NUM_TOUCH_PADS 4

// Touch pad GPIO mapping for ESP32-WROOM-32:
//   T0 = GPIO4,  T5 = GPIO12, T6 = GPIO14, T7 = GPIO27
// These are physically adjacent on the 30-pin devkit.

/**
 * Initialize 4 touch pads and calibrate baseline thresholds.
 * Reads each pad 20 times over ~1 second to establish untouched baseline.
 * Threshold is set to 70% of baseline (touch causes values to DROP on ESP32 v1).
 *
 * @return 0 on success, negative on error
 */
int touch_sensor_init(void);

/**
 * Read raw values from all 4 touch pads.
 *
 * @param out  Array of 4 uint16_t values (one per pad)
 */
void touch_sensor_read_all(uint16_t out[NUM_TOUCH_PADS]);

/**
 * Check if a pad is currently being touched.
 *
 * @param raw      Raw reading from the pad
 * @param pad_idx  Index 0-3 into the pad array
 * @return true if the pad reading is below the calibrated threshold
 */
bool touch_sensor_is_touched(uint16_t raw, int pad_idx);

/**
 * Get the calibrated threshold for a pad.
 *
 * @param pad_idx  Index 0-3 into the pad array
 * @return Threshold value (touch detected when raw < threshold)
 */
uint16_t touch_sensor_get_threshold(int pad_idx);

/**
 * Get the baseline (untouched) value for a pad.
 *
 * @param pad_idx  Index 0-3 into the pad array
 * @return Baseline value
 */
uint16_t touch_sensor_get_baseline(int pad_idx);

#endif // TOUCH_SENSOR_H
