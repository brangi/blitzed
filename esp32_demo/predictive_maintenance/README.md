# Blitzed Predictive Maintenance Starter Kit

This demo is the first commercial reference workflow for Blitzed: combine ESP32 temperature and MPU6050 vibration measurements, run a small INT8 classifier locally, and emit actionable maintenance status without cloud inference.

## What it detects

The reference model classifies four states:

1. `healthy`
2. `warning`
3. `critical`
4. `shutdown_required`

The model is `Dense(4, 32) + ReLU + Dense(32, 4)` with 292 trainable parameters and four inputs:

```text
temperature / 120.0
rms_accel_x / 12.0
rms_accel_y / 12.0
rms_accel_z / 12.0
```

The normalization contract must remain identical in the training script and firmware.

## Hardware

- ESP32-WROOM-32 development board
- MPU6050 accelerometer
- ESP-IDF v5.x
- I2C wiring:
  - SDA → GPIO 21
  - SCL → GPIO 22
  - VCC/GND according to the MPU6050 breakout board

The temperature input uses the ESP32 internal temperature sensor. It is a chip-temperature signal, not an ambient or machine-surface measurement.

## Reproduce the model artifacts

From the repository root:

```bash
python tools/train_predictive_maintenance.py
```

The script uses a deterministic seed, keeps a held-out validation split separate from training/calibration, and writes these files to `main/`:

- `blitzed_model_weights.h` — generated INT8/INT32 model constants
- `blitzed_model_weights.bin` — raw weight export
- `blitzed_model_report.json` — reproducibility and validation report
- `blitzed_release_manifest.json` — release manifest with artifact sizes and SHA-256 checksums

For a temporary or custom artifact directory:

```bash
python tools/train_predictive_maintenance.py \
  --seed 42 \
  --samples 3000 \
  --epochs 1000 \
  --output-dir /tmp/blitzed-predictive-maintenance
```

### Train from recorded customer data

Use a CSV with this exact header:

```csv
temperature,rms_accel_x,rms_accel_y,rms_accel_z,label
38.2,0.42,0.51,0.39,healthy
52.7,2.81,0.66,0.71,warning
```

Labels may be `healthy`, `warning`, `critical`, `shutdown_required`, or integer IDs `0` through `3`. Every row must contain temperature in °C and RMS acceleration in g, using the same sensor/calibration definitions as the firmware.

Inspect dataset quality before training:

```bash
python tools/train_predictive_maintenance.py \
  --dataset data/machine-readings.csv \
  --quality-only \
  --output-dir /tmp/blitzed-customer-model
```

This writes `blitzed_dataset_quality.json` with class balance, feature statistics, duplicate rows, out-of-range readings, missing classes, warnings, and provenance metadata.

Every quality and model report records the UTC generation time, Git revision, training-script SHA-256, Python/NumPy versions, CLI parameters, and—when using a CSV—the dataset path and SHA-256. Keep the report with the exported firmware artifacts so a deployed model can be traced back to its exact inputs and code.

A normal training run treats the selected output directory as a release directory. The release manifest checksums the model header, binary weights, and validation report. Distribute those files together and verify the manifest before flashing or registering a model:

```bash
python3 tools/verify_predictive_maintenance_release.py \
  /tmp/blitzed-customer-model/blitzed_release_manifest.json
```

Verification fails if an artifact is missing, changed, has the wrong size, or uses an unsafe path outside the release directory.

Train and export from that dataset with:

```bash
python tools/train_predictive_maintenance.py \
  --dataset data/machine-readings.csv \
  --validation-split 0.2 \
  --seed 42 \
  --output-dir /tmp/blitzed-customer-model
```

The loader validates required columns, numeric values, finite readings, and labels. The split is deterministic and stratified by status where enough samples exist. The model report includes the quality results alongside split sizes, class metrics, float-vs-INT8 accuracy loss, model size, seed, and artifact paths.

## Build a verified customer release

Use the release command after collecting and reviewing representative data:

```bash
python3 tools/build_predictive_maintenance_release.py \
  --dataset data/machine-readings.csv \
  --output-dir releases/machine-v1 \
  --seed 42
```

It runs the quality gate, blocks unresolved warnings, trains the model, creates the release manifest, and verifies every artifact checksum. Review `blitzed_dataset_quality.json` before using `--allow-quality-warnings` for known and documented exceptions.

## Collect labeled sensor data

The firmware emits structured sensor lines during its normal loop. Capture a session for one known machine state and assign the **ground-truth** label for that session:

```bash
mkdir -p data
idf.py -p <serial-device> monitor | \
  python3 tools/collect_predictive_maintenance_data.py \
  --label healthy \
  --output data/machine-readings.csv
```

Stop that session, change the machine to a known `warning`, `critical`, or `shutdown_required` condition, and repeat with `--append`:

```bash
idf.py -p <serial-device> monitor | \
  python3 tools/collect_predictive_maintenance_data.py \
  --label warning \
  --output data/machine-readings.csv \
  --append
```

Saved monitor logs work too:

```bash
python3 tools/collect_predictive_maintenance_data.py \
  --input capture.log \
  --label critical \
  --output data/machine-readings.csv \
  --append
```

The collector is dependency-free and ignores the model's predicted `Status`; labels must reflect the actual machine condition. Collect representative samples across operating speeds, loads, sensor mounting positions, and both healthy and fault conditions before training.

## Build and flash

```bash
source ~/esp/esp-idf-v5.3/export.sh
cd esp32_demo/predictive_maintenance
idf.py set-target esp32
idf.py build
idf.py -p <serial-device> flash monitor
```

At startup, firmware runs a latency benchmark, then samples the sensors once per second. `critical` and `shutdown_required` states are emitted as elevated log messages. If either sensor read fails or returns invalid values, the firmware emits `Status: sensor_fault`, skips inference, and asks the operator to inspect the sensors; it never turns missing data into a healthy prediction.

## Commercialization status

This is a reference implementation, not a production safety system. The checked-in training data is synthetic and must be replaced or supplemented with labeled measurements from the target machine, sensor mounting, operating speeds, and fault modes before making reliability claims.

Before a customer deployment, validate:

- Sensor calibration and mounting repeatability.
- False-negative rate for critical and shutdown states.
- Recovery behavior after an MPU6050 or temperature sensor failure, including the `sensor_fault` path before allowing predictions to resume.
- Model performance across machines and operating conditions.
- ESP32 memory, latency, watchdog, and long-running stability.
- A human-reviewed maintenance response for every alert level.
