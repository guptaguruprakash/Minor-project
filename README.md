# ESP32 FastAPI Collector

This folder contains a small FastAPI app that receives labeled MPU6050 telemetry from the ESP32 sketch and appends each POST to [fastapi_api/data/sensor_data_labeled.csv](e:/esp32/fastapi_api/data/sensor_data_labeled.csv).

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
