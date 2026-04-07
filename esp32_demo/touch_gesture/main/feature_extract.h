// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#ifndef FEATURE_EXTRACT_H
#define FEATURE_EXTRACT_H

#include <stdint.h>

// Total number of features extracted from one gesture window.
// 16 per-pad features (4 pads x 4 features each) + 4 global features.
#define NUM_FEATURES 20

/**
 * Extract 20 normalized features from a raw touch sensor window.
 *
 * Computes per-pad temporal and intensity features plus global spatial
 * features. All output values are normalized to [0, 1] and are ready
 * to be passed directly to blitzed_quantize_input().
 *
 * Per-pad features (indices 0..15, 4 per pad, pad stride = 4):
 *   [pad*4 + 0] touch_duration   : fraction of steps pad was active
 *   [pad*4 + 1] touch_centroid   : normalized center-of-mass time of activity
 *   [pad*4 + 2] mean_intensity   : mean (baseline-raw)/baseline during contact
 *   [pad*4 + 3] touch_count      : distinct touch-release transitions / 5.0
 *
 * Global features (indices 16..19):
 *   [16] active_pin_count   : pads touched at all / 4.0
 *   [17] spatial_sweep      : direction correlation of sweep across pads [0,1]
 *   [18] total_duration     : mean per-pad duration across all pads
 *   [19] inter_onset_interval : mean inter-onset delay (normalized)
 *
 * Requires touch_sensor.h to be compiled in the same build unit.
 * Baselines and thresholds are read via touch_sensor_get_baseline() and
 * touch_sensor_get_threshold().
 *
 * @param window   Raw touch readings: window[step][pad], sampled at 50 Hz
 * @param n_steps  Number of time steps in the window (rows)
 * @param features Output array of NUM_FEATURES floats, all in [0, 1]
 */
void extract_touch_features(const uint16_t window[][4], int n_steps,
                             float features[NUM_FEATURES]);

#endif // FEATURE_EXTRACT_H
