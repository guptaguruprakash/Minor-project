# ESP32 FastAPI Collector

This folder contains a small FastAPI app that receives MPU6050 telemetry from the ESP32 sketch and appends each POST to `data/telemetry.jsonl`.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

If you prefer running from the project root, this also works:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## ESP32 endpoint

Set the ESP32 `API_URL` to:

```text
http://YOUR_PC_IP:8080/api/telemetry
```

## Payload shape

The endpoint expects:

- `exercise`
- `posture`
- `user`
- `samples[]`
- `samples[].timestamp`
- `samples[].sensor1.ax`, `ay`, `az`, `gx`, `gy`, `gz`
- `samples[].sensor2.ax`, `ay`, `az`, `gx`, `gy`, `gz`
