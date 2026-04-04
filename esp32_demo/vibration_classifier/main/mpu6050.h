#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>
#include "driver/i2c.h"
#include "esp_err.h"

// MPU6050 I2C address (AD0 pin low)
#define MPU6050_ADDR_DEFAULT 0x68

// MPU6050 register addresses
#define MPU6050_REG_PWR_MGMT_1   0x6B
#define MPU6050_REG_SMPLRT_DIV   0x19
#define MPU6050_REG_CONFIG       0x1A
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B
#define MPU6050_REG_WHO_AM_I     0x75

// Accelerometer scale factors
// ±2g:  16384 LSB/g
// ±4g:   8192 LSB/g
// ±8g:   4096 LSB/g
// ±16g:  2048 LSB/g
#define MPU6050_ACCEL_SCALE_4G   8192.0f

// ACCEL_CONFIG register values for full-scale range
#define MPU6050_ACCEL_FS_2G  0x00
#define MPU6050_ACCEL_FS_4G  0x08
#define MPU6050_ACCEL_FS_8G  0x10
#define MPU6050_ACCEL_FS_16G 0x18

// I2C timing
#define MPU6050_I2C_TIMEOUT_MS 100

/**
 * Initialize the MPU6050 accelerometer.
 *
 * Configures the I2C master bus, wakes the MPU6050 from sleep,
 * sets ±4g accelerometer range, and configures 1kHz sample rate.
 *
 * @param port  I2C port number (I2C_NUM_0 or I2C_NUM_1)
 * @param addr  MPU6050 I2C address (MPU6050_ADDR_DEFAULT = 0x68)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t mpu6050_init(i2c_port_t port, uint8_t addr);

/**
 * Read instantaneous accelerometer values.
 *
 * Reads raw 16-bit registers for all three axes and converts
 * to g units using the ±4g scale factor (8192 LSB/g).
 *
 * @param ax  Pointer to store X-axis acceleration in g
 * @param ay  Pointer to store Y-axis acceleration in g
 * @param az  Pointer to store Z-axis acceleration in g
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t mpu6050_read_accel(float *ax, float *ay, float *az);

/**
 * Read RMS acceleration over multiple samples per axis.
 *
 * Takes num_samples readings, computes root-mean-square for each axis.
 * RMS captures both DC offset and AC vibration energy, making it
 * well-suited as a vibration severity indicator.
 *
 * @param rms_x       Pointer to store X-axis RMS in g
 * @param rms_y       Pointer to store Y-axis RMS in g
 * @param rms_z       Pointer to store Z-axis RMS in g
 * @param num_samples Number of samples to collect (recommended: 32-128)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t mpu6050_read_accel_rms(float *rms_x, float *rms_y, float *rms_z,
                                  int num_samples);

#endif // MPU6050_H
