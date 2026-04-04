#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/i2c.h"

// Temperature sensor — ESP-IDF v5.x API
// Fall back gracefully for older IDF versions that lack this driver.
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#include "driver/temperature_sensor.h"
#define HAVE_TEMP_SENSOR_V5 1
#else
#define HAVE_TEMP_SENSOR_V5 0
#endif

#include "mpu6050.h"
#include "blitzed_inference.h"
#include "blitzed_model_weights.h"

static const char *TAG = "blitzed";

// ---- I2C bus configuration ----
#define I2C_MASTER_PORT     I2C_NUM_0
#define I2C_MASTER_SDA_IO   21      // GPIO 21 — SDA
#define I2C_MASTER_SCL_IO   22      // GPIO 22 — SCL
#define I2C_MASTER_FREQ_HZ  400000  // 400 kHz fast-mode

// ---- MPU6050 sampling ----
// 32 samples per RMS measurement ~= 32 ms at 1 kHz sample rate
#define ACCEL_RMS_SAMPLES   32

// ---- Inference benchmark ----
#define BENCHMARK_ITERATIONS 10000

// ---- Normalisation constants (must match training script) ----
// temperature: raw_celsius / 120.0
// acceleration: raw_g    / 12.0
#define TEMP_NORM_SCALE   120.0f
#define ACCEL_NORM_SCALE   12.0f

// ----------------------------------------------------------------
// I2C master init
// ----------------------------------------------------------------
static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_MASTER_SDA_IO,
        .scl_io_num       = I2C_MASTER_SCL_IO,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    esp_err_t ret = i2c_param_config(I2C_MASTER_PORT, &conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2c_param_config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = i2c_driver_install(I2C_MASTER_PORT, I2C_MODE_MASTER, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2c_driver_install failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "I2C master initialized: SDA=GPIO%d SCL=GPIO%d @ %d Hz",
             I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO, I2C_MASTER_FREQ_HZ);
    return ESP_OK;
}

// ----------------------------------------------------------------
// On-chip temperature sensor
// ESP-IDF v5 uses the temperature_sensor driver; older IDF exposes
// an internal ADC path.  We wrap both behind a single function so
// the rest of the code is version-agnostic.
// ----------------------------------------------------------------
#if HAVE_TEMP_SENSOR_V5
static temperature_sensor_handle_t s_temp_handle = NULL;

static esp_err_t temp_sensor_init(void)
{
    temperature_sensor_config_t config = TEMPERATURE_SENSOR_CONFIG_DEFAULT(10, 80);
    esp_err_t ret = temperature_sensor_install(&config, &s_temp_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "temperature_sensor_install failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = temperature_sensor_enable(s_temp_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "temperature_sensor_enable failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "On-chip temperature sensor initialized (ESP-IDF v5 API)");
    return ESP_OK;
}

static esp_err_t temp_sensor_read_celsius(float *out_celsius)
{
    return temperature_sensor_get_celsius(s_temp_handle, out_celsius);
}

#else // ESP-IDF < v5: use the legacy internal temperature sensor

#include "driver/temp_sensor.h"

static esp_err_t temp_sensor_init(void)
{
    temp_sensor_config_t config = TSENS_CONFIG_DEFAULT();
    esp_err_t ret = temp_sensor_set_config(config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "temp_sensor_set_config failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = temp_sensor_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "temp_sensor_start failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "On-chip temperature sensor initialized (legacy API)");
    return ESP_OK;
}

static esp_err_t temp_sensor_read_celsius(float *out_celsius)
{
    return temp_sensor_read_celsius(out_celsius);
}

#endif // HAVE_TEMP_SENSOR_V5

// ----------------------------------------------------------------
// Benchmark: measure min / mean / max inference latency
// ----------------------------------------------------------------
static void run_benchmark(int num_iterations)
{
    ESP_LOGI(TAG, "Starting benchmark: %d iterations", num_iterations);

    int8_t input[4]  = {0, 0, 0, 0};
    int8_t output[4];

    uint32_t min_us  = UINT32_MAX;
    uint32_t max_us  = 0;
    uint64_t total_us = 0;

    for (int i = 0; i < num_iterations; i++) {
        // Vary inputs slightly to prevent trivial cache shortcuts
        input[0] = (int8_t)( i        % 127);
        input[1] = (int8_t)((i +  43) % 127);
        input[2] = (int8_t)((i +  87) % 127);
        input[3] = (int8_t)((i + 131) % 127);

        uint64_t start = esp_timer_get_time();
        blitzed_model_predict(input, 4, output, 4);
        uint64_t end   = esp_timer_get_time();

        uint32_t latency_us = (uint32_t)(end - start);
        if (latency_us < min_us) min_us = latency_us;
        if (latency_us > max_us) max_us = latency_us;
        total_us += latency_us;
    }

    uint32_t mean_us = (uint32_t)(total_us / (uint64_t)num_iterations);

    ESP_LOGI(TAG, "Benchmark complete:");
    ESP_LOGI(TAG, "  Min latency:  %lu us", (unsigned long)min_us);
    ESP_LOGI(TAG, "  Mean latency: %lu us", (unsigned long)mean_us);
    ESP_LOGI(TAG, "  Max latency:  %lu us", (unsigned long)max_us);
    ESP_LOGI(TAG, "  Throughput:   %lu inferences/sec",
             (unsigned long)(1000000UL / (mean_us ? mean_us : 1)));
}

// ----------------------------------------------------------------
// app_main
// ----------------------------------------------------------------
void app_main(void)
{
    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, " Blitzed Predictive Maintenance Demo");
    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Model: Dense(4,32)+ReLU -> Dense(32,4)");
    ESP_LOGI(TAG, "Classes: healthy | warning | critical | shutdown_required");

    // ---- Initialize peripherals ----

    // I2C bus
    ESP_ERROR_CHECK(i2c_master_init());

    // MPU6050 accelerometer
    ESP_ERROR_CHECK(mpu6050_init(I2C_MASTER_PORT, MPU6050_ADDR_DEFAULT));

    // On-chip temperature sensor
    ESP_ERROR_CHECK(temp_sensor_init());

    // ---- Initialize inference engine ----
    if (blitzed_model_init() != 0) {
        ESP_LOGE(TAG, "Failed to initialize Blitzed model");
        return;
    }
    ESP_LOGI(TAG, "Blitzed inference engine ready");

    // ---- Startup benchmark ----
    run_benchmark(BENCHMARK_ITERATIONS);

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Starting real-time predictive maintenance loop (1 s interval)");
    ESP_LOGI(TAG, "");

    // ---- Main loop ----
    int8_t q_input[4];
    int8_t q_output[4];
    int iteration = 0;

    while (1) {
        // 1. Read on-chip temperature (°C)
        float temperature = 0.0f;
        esp_err_t temp_ret = temp_sensor_read_celsius(&temperature);
        if (temp_ret != ESP_OK) {
            ESP_LOGW(TAG, "Temperature read failed: %s", esp_err_to_name(temp_ret));
            temperature = 0.0f;  // safe fallback; model will still run
        }

        // 2. Read accelerometer RMS per axis over 32 samples
        float rms_x = 0.0f, rms_y = 0.0f, rms_z = 0.0f;
        esp_err_t accel_ret = mpu6050_read_accel_rms(&rms_x, &rms_y, &rms_z,
                                                      ACCEL_RMS_SAMPLES);
        if (accel_ret != ESP_OK) {
            ESP_LOGW(TAG, "Accel RMS read failed: %s", esp_err_to_name(accel_ret));
            // fallback: leave at 0.0g — model will still classify
        }

        // 3. Quantize 4 inputs
        //    Normalisation mirrors training: temp/120.0, accel/12.0
        q_input[0] = blitzed_quantize_input(temperature / TEMP_NORM_SCALE);
        q_input[1] = blitzed_quantize_input(rms_x      / ACCEL_NORM_SCALE);
        q_input[2] = blitzed_quantize_input(rms_y      / ACCEL_NORM_SCALE);
        q_input[3] = blitzed_quantize_input(rms_z      / ACCEL_NORM_SCALE);

        // 4. Run INT8 inference
        uint64_t inf_start = esp_timer_get_time();
        int ret = blitzed_model_predict(q_input, 4, q_output, 4);
        uint64_t inf_end   = esp_timer_get_time();
        uint32_t latency_us = (uint32_t)(inf_end - inf_start);

        if (ret != 0) {
            ESP_LOGE(TAG, "Inference failed with error %d", ret);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // 5. Decode result
        int predicted_class = blitzed_argmax(q_output, NUM_CLASSES);

        // 6. Log — structured format for easy serial parsing
        ESP_LOGI(TAG,
                 "[%05d] Temp: %5.1f C | Accel RMS: [%4.2f, %4.2f, %4.2f] g"
                 " | Status: %-18s | Latency: %lu us",
                 iteration,
                 temperature,
                 rms_x, rms_y, rms_z,
                 class_labels[predicted_class],
                 (unsigned long)latency_us);

        // 7. Elevate log level for actionable conditions
        if (predicted_class == 2) {
            // critical: elevated temp AND multi-axis vibration
            ESP_LOGW(TAG,
                     "[ALERT] CRITICAL condition detected — "
                     "temp=%.1f C rms=[%.2f %.2f %.2f] g. "
                     "Schedule maintenance.",
                     temperature, rms_x, rms_y, rms_z);
        } else if (predicted_class == 3) {
            // shutdown_required: extreme temp OR extreme vibration
            ESP_LOGE(TAG,
                     "[ALERT] SHUTDOWN REQUIRED — "
                     "temp=%.1f C rms=[%.2f %.2f %.2f] g. "
                     "Stop equipment immediately!",
                     temperature, rms_x, rms_y, rms_z);
        }

        iteration++;

        // 1 second sample interval
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
