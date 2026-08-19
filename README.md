# ESP32 FastAPI Collector

This folder contains a small FastAPI app that receives labeled MPU6050 telemetry from the ESP32 sketch and appends each POST to [fastapi_api/data/sensor_data_labeled.csv](e:/esp32/fastapi_api/data/sensor_data_labeled.csv).

## KNN model

The ESP32 sketch uses [knn_model.h](knn_model.h), generated from `data/bicep.csv`. The squat dataset remains available but is currently excluded from the model. The model standardizes all 12 MPU6050 features and embeds 64 evenly spaced KNN prototypes per label in flash memory.

Regenerate the header after collecting new data:

```bash
python train_knn.py
```

The current model produces three labels: `bicep_bad`, `bicep_good`, and `bicep_idle_good`. The sketch prints one KNN prediction for each live buffer on Serial.

## Live monitor

The live monitor is a React/Vite frontend built into `fastapi_api/app/static`, which FastAPI serves at `/`. After changing the React source, rebuild it from the frontend directory:

```bash
cd fastapi_api/frontend
npm install
npm run build
```

For frontend-only development, run `npm run dev` from `fastapi_api/frontend`. Vite proxies `/api` requests to the FastAPI server at `http://127.0.0.1:8000`.

`model.ino` owns the firmware `setup()` and `loop()` plus the KNN adapter. `esp32_dual_mpu6050_post.ino` supplies the sensor collection, buffering, and upload functions. Keep both files in the same Arduino sketch folder; Arduino combines `.ino` tabs when compiling.

Each upload includes a majority-vote prediction under `prediction`. Start the API and open `http://127.0.0.1:8000/` to view the live exercise, posture, confidence, and ESP32 connection state:

```bash
cd fastapi_api
pip install -r requirements.txt
python main.py
```

The ESP32 must be able to reach the computer at `SERVER_URL`, and the browser must reach the computer running FastAPI. The dashboard marks the device offline after three seconds without a prediction update.

## Run

```bash
cd fastapi_api
pip install -r requirements.txt
python main.py
```

You can also run it with Uvicorn from the project root:

```bash
cd fastapi_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## ESP32 endpoint

Set the ESP32 `SERVER_URL` to:

```text
http://YOUR_PC_IP:8000/api/sensor-data
```

## Payload shape

The endpoint expects:

- `exercise`
- `posture`
- `user`
- `samples[]`
- `samples[].t`
- `samples[].ax1`, `ay1`, `az1`, `gx1`, `gy1`, `gz1`
- `samples[].ax2`, `ay2`, `az2`, `gx2`, `gy2`, `gz2`
