// Copyright 2026 Gibran Rodriguez <brangi000@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#include "feature_extract.h"
#include "touch_sensor.h"

// Internal clamp helpers — avoid pulling in math.h for fminf/fmaxf
static inline float clamp_f(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void extract_touch_features(const uint16_t window[][4], int n_steps,
                             float features[NUM_FEATURES]) {
    if (!window || !features || n_steps <= 0) {
        // Zero-fill and return rather than crash in an ISR-deferred task
        for (int k = 0; k < NUM_FEATURES; k++) features[k] = 0.0f;
        return;
    }

    const float n_steps_f = (float)n_steps;

    // -------------------------------------------------------------------------
    // Per-pad accumulators
    // -------------------------------------------------------------------------
    float duration[4]    = {0};   // fraction of touched steps
    float centroid[4]    = {0};   // weighted center-of-mass time
    float intensity[4]   = {0};   // mean (baseline - raw) / baseline during touch
    float tc[4]          = {0};   // raw touch-release transition counts
    float onset[4];               // first touched step index (-1 = never)
    int   active[4]      = {0};   // was pad ever touched?

    for (int p = 0; p < 4; p++) onset[p] = -1.0f;

    // Per-pad, per-step state for transition detection
    int prev_touched[4] = {0};

    // First-pass: accumulate touched_steps, weighted_time, intensity_sum,
    // transitions, and onset time per pad
    int touched_count[4]    = {0};
    float centroid_num[4]   = {0};
    float intensity_sum[4]  = {0};

    for (int t = 0; t < n_steps; t++) {
        for (int p = 0; p < 4; p++) {
            uint16_t baseline  = touch_sensor_get_baseline(p);
            uint16_t threshold = touch_sensor_get_threshold(p);
            uint16_t raw       = window[t][p];

            // A pad is touched when its value drops below threshold (70% baseline)
            int is_touched = (raw < threshold) ? 1 : 0;

            if (is_touched) {
                touched_count[p]++;
                centroid_num[p]  += (float)t;

                // Intensity: how deep is the touch relative to baseline?
                //   (baseline - raw) / baseline clamped to [0, 1]
                float rel = 0.0f;
                if (baseline > 0) {
                    rel = clamp_f(((float)baseline - (float)raw) / (float)baseline,
                                  0.0f, 1.0f);
                }
                intensity_sum[p] += rel;

                if (onset[p] < 0.0f) {
                    onset[p] = (float)t;
                    active[p] = 1;
                }
            }

            // Touch-release transition: went from untouched -> touched
            if (is_touched && !prev_touched[p]) {
                tc[p] += 1.0f;
            }
            prev_touched[p] = is_touched;
        }
    }

    // Compute per-pad features
    for (int p = 0; p < 4; p++) {
        int idx = p * 4;

        // Feature 0: touch_duration — fraction of window that was active
        duration[p]       = (float)touched_count[p] / n_steps_f;
        features[idx + 0] = duration[p];

        // Feature 1: touch_centroid — time center-of-mass, normalized to [0,1]
        //   If never touched, default to 0.5 (middle of window)
        if (touched_count[p] > 0) {
            centroid[p]       = (centroid_num[p] / (float)touched_count[p]) / n_steps_f;
            features[idx + 1] = centroid[p];
        } else {
            centroid[p]       = 0.5f;
            features[idx + 1] = 0.5f;
        }

        // Feature 2: mean_intensity — average depth of contact during touch steps
        if (touched_count[p] > 0) {
            intensity[p]      = intensity_sum[p] / (float)touched_count[p];
            features[idx + 2] = intensity[p];
        } else {
            intensity[p]      = 0.0f;
            features[idx + 2] = 0.0f;
        }

        // Feature 3: touch_count — distinct press events, normalized by 5
        //   Divide by 5.0 so that 5 rapid taps maps to 1.0; clamp above 1.0
        features[idx + 3] = clamp_f(tc[p] / 5.0f, 0.0f, 1.0f);
    }

    // -------------------------------------------------------------------------
    // Global features (indices 16-19)
    // -------------------------------------------------------------------------

    // Feature 16: active_pin_count — how many pads fired at all / 4.0
    int num_active = 0;
    for (int p = 0; p < 4; p++) {
        if (active[p]) num_active++;
    }
    features[16] = (float)num_active / 4.0f;

    // Feature 17: spatial_sweep — direction-of-swipe correlation in [0, 1]
    //
    //   Only meaningful when >= 2 pads fired. Measures whether onset times
    //   progress monotonically across pad indices (swipe left-to-right vs
    //   right-to-left). Computes the Pearson-like correlation between
    //   pad_index and centroid time, then maps [-1, 1] -> [0, 1].
    //
    //   pad_deviation  = pad_idx - 1.5        (mean of 0,1,2,3 = 1.5)
    //   time_deviation = centroid - mean_centroid
    //   covariance     = sum(pad_dev * time_dev)   (active pads only)
    //   denominator    = num_active * 1.5 * max(std_centroid, 0.01)
    //   correlation    = cov / denom, clamped to [-1, 1]
    //   output         = (correlation + 1) / 2
    if (num_active < 2) {
        features[17] = 0.5f;
    } else {
        // Compute mean centroid over active pads
        float mean_centroid = 0.0f;
        for (int p = 0; p < 4; p++) {
            if (active[p]) mean_centroid += centroid[p];
        }
        mean_centroid /= (float)num_active;

        // Compute std deviation of centroid over active pads
        float var = 0.0f;
        for (int p = 0; p < 4; p++) {
            if (active[p]) {
                float d = centroid[p] - mean_centroid;
                var += d * d;
            }
        }
        float std_centroid = 0.0f;
        // Manual sqrt approximation via Newton's method to avoid libm dependency
        if (var > 0.0f) {
            float x = var;
            // Three Newton iterations are sufficient for this range
            x = (x + var / x) * 0.5f;
            x = (x + var / x) * 0.5f;
            x = (x + var / x) * 0.5f;
            std_centroid = x;
        }
        float denom_t = (std_centroid > 0.01f) ? std_centroid : 0.01f;

        float covariance = 0.0f;
        for (int p = 0; p < 4; p++) {
            if (active[p]) {
                float pad_dev  = (float)p - 1.5f;
                float time_dev = centroid[p] - mean_centroid;
                covariance += pad_dev * time_dev;
            }
        }

        float denom = (float)num_active * 1.5f * denom_t;
        float corr  = (denom > 0.0f) ? (covariance / denom) : 0.0f;
        corr = clamp_f(corr, -1.0f, 1.0f);
        features[17] = (corr + 1.0f) / 2.0f;
    }

    // Feature 18: total_duration — mean per-pad duration across all 4 pads
    float sum_duration = 0.0f;
    for (int p = 0; p < 4; p++) sum_duration += duration[p];
    features[18] = sum_duration / 4.0f;

    // Feature 19: inter_onset_interval — mean time between successive onsets
    //
    //   Collect onset times for active pads in pad-index order, compute
    //   mean successive difference, normalize by n_steps.
    //   If 0 or 1 pads active, = 0.
    if (num_active < 2) {
        features[19] = 0.0f;
    } else {
        // Gather onset times in pad order (pad index already defines spatial order)
        float onset_times[4];
        int   onset_count = 0;
        for (int p = 0; p < 4; p++) {
            if (active[p]) {
                onset_times[onset_count++] = onset[p];
            }
        }

        float ioi_sum = 0.0f;
        for (int i = 1; i < onset_count; i++) {
            float diff = onset_times[i] - onset_times[i - 1];
            // diff could be negative if a later pad fired first; take abs
            if (diff < 0.0f) diff = -diff;
            ioi_sum += diff;
        }
        float mean_ioi = ioi_sum / (float)(onset_count - 1);
        // Normalize by window length so result is in [0, 1]
        features[19] = clamp_f(mean_ioi / n_steps_f, 0.0f, 1.0f);
    }
}
