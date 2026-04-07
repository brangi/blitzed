#include "mpu6050.h"
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "mpu6050";

// State retained after init for use in read functions
static i2c_port_t s_i2c_port = I2C_NUM_0;
static uint8_t s_addr = MPU6050_ADDR_DEFAULT;

// --- Internal helpers ---

static esp_err_t mpu6050_write_register(uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_write_to_device(
        s_i2c_port,
        s_addr,
        buf,
        sizeof(buf),
        pdMS_TO_TICKS(MPU6050_I2C_TIMEOUT_MS)
    );
}

static esp_err_t mpu6050_read_registers(uint8_t reg, uint8_t *data, size_t len) {
    return i2c_master_write_read_device(
        s_i2c_port,
        s_addr,
        &reg,
        1,
        data,
        len,
        pdMS_TO_TICKS(MPU6050_I2C_TIMEOUT_MS)
    );
}

// --- Public API ---

esp_err_t mpu6050_init(i2c_port_t port, uint8_t addr) {
    esp_err_t ret;

    s_i2c_port = port;
    s_addr = addr;

    // Configure I2C master bus
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = 21,          // GPIO 21 — ESP32 default SDA
        .scl_io_num = 22,          // GPIO 22 — ESP32 default SCL
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000, // 400kHz Fast Mode
    };

    ret = i2c_param_config(port, &conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = i2c_driver_install(port, I2C_MODE_MASTER, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // Verify device presence via WHO_AM_I register (should return 0x68)
    uint8_t who_am_i = 0;
    ret = mpu6050_read_registers(MPU6050_REG_WHO_AM_I, &who_am_i, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read WHO_AM_I: %s", esp_err_to_name(ret));
        return ret;
    }
    if (who_am_i != 0x68) {
        ESP_LOGE(TAG, "Unexpected WHO_AM_I: 0x%02X (expected 0x68)", who_am_i);
        return ESP_ERR_NOT_FOUND;
    }
    ESP_LOGI(TAG, "MPU6050 detected at address 0x%02X", addr);

    // Wake MPU6050 from sleep: clear SLEEP bit in PWR_MGMT_1
    // Also select internal 8MHz oscillator as clock source
    ret = mpu6050_write_register(MPU6050_REG_PWR_MGMT_1, 0x00);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to wake MPU6050: %s", esp_err_to_name(ret));
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10)); // Allow sensor to stabilize after wake

    // Set sample rate: SMPLRT_DIV = 7 => 1kHz / (1 + 7) = 125 Hz
    // For vibration analysis, 125Hz samples are sufficient for sub-60Hz fault frequencies
    ret = mpu6050_write_register(MPU6050_REG_SMPLRT_DIV, 0x07);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set sample rate: %s", esp_err_to_name(ret));
        return ret;
    }

    // Configure low-pass filter: CONFIG register, DLPF_CFG=2 => 94Hz bandwidth
    // Reduces high-frequency noise while preserving vibration signal content
    ret = mpu6050_write_register(MPU6050_REG_CONFIG, 0x02);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure DLPF: %s", esp_err_to_name(ret));
        return ret;
    }

    // Set accelerometer full-scale range to ±4g
    // Vibration amplitudes in industrial motors typically stay under 4g for normal operation
    ret = mpu6050_write_register(MPU6050_REG_ACCEL_CONFIG, MPU6050_ACCEL_FS_4G);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set accel range: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "MPU6050 initialized: ±4g range, ~125Hz sample rate, 94Hz DLPF");
    return ESP_OK;
}

esp_err_t mpu6050_read_accel(float *ax, float *ay, float *az) {
    if (!ax || !ay || !az) {
        return ESP_ERR_INVALID_ARG;
    }

    // Burst-read 6 bytes starting at ACCEL_XOUT_H (0x3B)
    // Register layout: XOUT_H, XOUT_L, YOUT_H, YOUT_L, ZOUT_H, ZOUT_L
    uint8_t raw[6];
    esp_err_t ret = mpu6050_read_registers(MPU6050_REG_ACCEL_XOUT_H, raw, 6);
    if (ret != ESP_OK) {
        return ret;
    }

    // Reconstruct signed 16-bit values (big-endian from sensor)
    int16_t raw_x = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t raw_y = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t raw_z = (int16_t)((raw[4] << 8) | raw[5]);

    // Convert to g using ±4g scale factor (8192 LSB/g)
    *ax = (float)raw_x / MPU6050_ACCEL_SCALE_4G;
    *ay = (float)raw_y / MPU6050_ACCEL_SCALE_4G;
    *az = (float)raw_z / MPU6050_ACCEL_SCALE_4G;

    return ESP_OK;
}

esp_err_t mpu6050_read_accel_rms(float *rms_x, float *rms_y, float *rms_z,
                                  int num_samples) {
    if (!rms_x || !rms_y || !rms_z || num_samples <= 0) {
        return ESP_ERR_INVALID_ARG;
    }

    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;
    double sum_sq_z = 0.0;

    for (int i = 0; i < num_samples; i++) {
        float ax, ay, az;
        esp_err_t ret = mpu6050_read_accel(&ax, &ay, &az);
        if (ret != ESP_OK) {
            return ret;
        }
        sum_sq_x += (double)ax * (double)ax;
        sum_sq_y += (double)ay * (double)ay;
        sum_sq_z += (double)az * (double)az;
    }

    *rms_x = (float)sqrt(sum_sq_x / num_samples);
    *rms_y = (float)sqrt(sum_sq_y / num_samples);
    *rms_z = (float)sqrt(sum_sq_z / num_samples);

    return ESP_OK;
}
