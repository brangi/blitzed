#include "mpu6050.h"
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "mpu6050";

// State held after init
static i2c_port_t s_i2c_port = I2C_NUM_0;
static uint8_t    s_dev_addr  = MPU6050_ADDR_DEFAULT;

// ---- low-level register helpers ----

static esp_err_t mpu6050_write_reg(uint8_t reg, uint8_t value)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (s_dev_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, value, true);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(s_i2c_port, cmd,
                                          pdMS_TO_TICKS(MPU6050_I2C_TIMEOUT_MS));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t mpu6050_read_regs(uint8_t reg, uint8_t *buf, size_t len)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();

    // Write register address
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (s_dev_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);

    // Repeated-start then read
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (s_dev_addr << 1) | I2C_MASTER_READ, true);
    if (len > 1) {
        i2c_master_read(cmd, buf, len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, buf + (len - 1), I2C_MASTER_NACK);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(s_i2c_port, cmd,
                                          pdMS_TO_TICKS(MPU6050_I2C_TIMEOUT_MS));
    i2c_cmd_link_delete(cmd);
    return ret;
}

// ---- public API ----

esp_err_t mpu6050_init(i2c_port_t port, uint8_t addr)
{
    s_i2c_port = port;
    s_dev_addr = addr;

    // Verify device identity
    uint8_t who_am_i = 0;
    esp_err_t ret = mpu6050_read_regs(MPU6050_REG_WHO_AM_I, &who_am_i, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "WHO_AM_I read failed: %s", esp_err_to_name(ret));
        return ret;
    }
    // WHO_AM_I returns bits 6:1 of the I2C address — 0x68 for the default device
    if (who_am_i != 0x68) {
        ESP_LOGW(TAG, "Unexpected WHO_AM_I: 0x%02X (expected 0x68)", who_am_i);
    }

    // Wake from sleep: clear SLEEP bit in PWR_MGMT_1, select internal 8 MHz oscillator
    ret = mpu6050_write_reg(MPU6050_REG_PWR_MGMT_1, 0x00);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "PWR_MGMT_1 write failed: %s", esp_err_to_name(ret));
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));  // allow oscillator to stabilise

    // Sample rate = gyro_rate / (1 + SMPLRT_DIV)
    // Gyro output rate with DLPF enabled = 1 kHz.  SMPLRT_DIV=0 -> 1 kHz.
    ret = mpu6050_write_reg(MPU6050_REG_SMPLRT_DIV, 0x00);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SMPLRT_DIV write failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // DLPF bandwidth ~94 Hz (CONFIG register DLPF_CFG=2) to reduce noise
    ret = mpu6050_write_reg(MPU6050_REG_CONFIG, 0x02);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CONFIG (DLPF) write failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // Accelerometer full-scale = +-4g (AFS_SEL=1)
    ret = mpu6050_write_reg(MPU6050_REG_ACCEL_CONFIG, MPU6050_ACCEL_FS_4G);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "ACCEL_CONFIG write failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "MPU6050 initialized: addr=0x%02X port=%d, +-4g range, 1kHz sample rate",
             addr, port);
    return ESP_OK;
}

esp_err_t mpu6050_read_accel(float *ax, float *ay, float *az)
{
    if (!ax || !ay || !az) {
        return ESP_ERR_INVALID_ARG;
    }

    // 6 bytes starting at ACCEL_XOUT_H: XH, XL, YH, YL, ZH, ZL
    uint8_t buf[6];
    esp_err_t ret = mpu6050_read_regs(MPU6050_REG_ACCEL_XOUT_H, buf, 6);
    if (ret != ESP_OK) {
        return ret;
    }

    // Reconstruct signed 16-bit values (big-endian)
    int16_t raw_x = (int16_t)((buf[0] << 8) | buf[1]);
    int16_t raw_y = (int16_t)((buf[2] << 8) | buf[3]);
    int16_t raw_z = (int16_t)((buf[4] << 8) | buf[5]);

    // Convert to g using +-4g scale factor (8192 LSB/g)
    *ax = (float)raw_x / MPU6050_ACCEL_SCALE_4G;
    *ay = (float)raw_y / MPU6050_ACCEL_SCALE_4G;
    *az = (float)raw_z / MPU6050_ACCEL_SCALE_4G;

    return ESP_OK;
}

esp_err_t mpu6050_read_accel_rms(float *rms_x, float *rms_y, float *rms_z,
                                  int num_samples)
{
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
            ESP_LOGW(TAG, "Sample %d failed: %s", i, esp_err_to_name(ret));
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
