#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>
#include "driver/i2c.h"
#include "esp_err.h"

// MPU6050 I2C address (AD0 pin low = 0x68, AD0 pin high = 0x69)
#define MPU6050_ADDR_DEFAULT  0x68
#define MPU6050_ADDR_ALT      0x69

// MPU6050 register map (relevant subset)
#define MPU6050_REG_SMPLRT_DIV    0x19   // Sample rate divider
#define MPU6050_REG_CONFIG        0x1A   // DLPF configuration
#define MPU6050_REG_ACCEL_CONFIG  0x1C   // Accelerometer configuration
#define MPU6050_REG_INT_ENABLE    0x38   // Interrupt enable
#define MPU6050_REG_ACCEL_XOUT_H  0x3B   // Accel X high byte (followed by XL, YH, YL, ZH, ZL)
#define MPU6050_REG_TEMP_OUT_H    0x41   // Temperature high byte
#define MPU6050_REG_PWR_MGMT_1    0x6B   // Power management 1 (sleep bit)
#define MPU6050_REG_WHO_AM_I      0x75   // Should read 0x68

// Accelerometer full-scale ranges
// AFS_SEL=0 -> +-2g  (16384 LSB/g)
// AFS_SEL=1 -> +-4g  ( 8192 LSB/g)
// AFS_SEL=2 -> +-8g  ( 4096 LSB/g)
// AFS_SEL=3 -> +-16g ( 2048 LSB/g)
#define MPU6050_ACCEL_FS_4G   0x08   // AFS_SEL=1, +-4g range
#define MPU6050_ACCEL_SCALE_4G  8192.0f  // LSB per g for +-4g range

// I2C timing
#define MPU6050_I2C_TIMEOUT_MS  100

/**
 * Initialize the MPU6050 accelerometer.
 *
 * Configures the I2C peripheral, wakes the MPU6050 from sleep, sets the
 * accelerometer to +-4g full-scale range, and sets a 1 kHz sample rate.
 *
 * @param port  I2C port number (I2C_NUM_0 or I2C_NUM_1)
 * @param addr  I2C address of the device (MPU6050_ADDR_DEFAULT = 0x68)
 * @return      ESP_OK on success, otherwise an esp_err_t error code
 */
esp_err_t mpu6050_init(i2c_port_t port, uint8_t addr);

/**
 * Read a single accelerometer sample.
 *
 * Raw register values are converted to g using the configured full-scale range.
 *
 * @param ax  Output: acceleration along X axis in g
 * @param ay  Output: acceleration along Y axis in g
 * @param az  Output: acceleration along Z axis in g
 * @return    ESP_OK on success, otherwise an esp_err_t error code
 */
esp_err_t mpu6050_read_accel(float *ax, float *ay, float *az);

/**
 * Compute per-axis RMS acceleration over a burst of samples.
 *
 * Reads num_samples consecutive accelerometer measurements and returns the
 * root-mean-square value for each axis.  This is the standard vibration
 * metric used by industrial condition-monitoring systems.
 *
 * @param rms_x       Output: RMS acceleration on X axis in g
 * @param rms_y       Output: RMS acceleration on Y axis in g
 * @param rms_z       Output: RMS acceleration on Z axis in g
 * @param num_samples Number of samples to average (16..256 recommended)
 * @return            ESP_OK on success, otherwise an esp_err_t error code
 */
esp_err_t mpu6050_read_accel_rms(float *rms_x, float *rms_y, float *rms_z,
                                  int num_samples);

#endif // MPU6050_H
