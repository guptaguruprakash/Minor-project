"""
Smart Gym Assistant - FastAPI Labeled Sensor Data Receiver
------------------------------------------------------------
Receives batched sensor samples from the ESP32 uploader, tagged with an
exercise name, posture label, and user name, then appends them to a CSV file.

Expected JSON payload:

{
    "exercise": "squat",
    "posture": "good",
    "user": "Birendra",
    "samples": [
        {
            "t": 12345,
            "ax1": 0.1, "ay1": 0.2, "az1": 9.8, "gx1": 0.01, "gy1": 0.02, "gz1": 0.03,
            "ax2": 0.1, "ay2": 0.2, "az2": 9.8, "gx2": 0.01, "gy2": 0.02, "gz2": 0.03
        }
    ]
}

Run with:
        pip install fastapi uvicorn pydantic
        uvicorn main:app --host 0.0.0.0 --port 5000
"""

import csv
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
CSV_HEADERS = [
    "received_at", "timestamp",
    "user", "exercise", "posture", "label",
    "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
]
DEFAULT_CSV_FILE = "sensor_data_labeled.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_csv_exists()
    yield


app = FastAPI(title="Smart Gym Labeled Sensor Data API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(BASE_DIR, "static", "assets")),
    name="react-assets",
)

# A lock so concurrent POSTs never interleave writes to the same file
csv_lock = threading.Lock()
live_lock = threading.Lock()
latest_live_prediction = {
    "online": False,
    "mode": "unknown",
    "exercise": "unknown",
    "posture": "unknown",
    "label": "unknown",
    "confidence": 0.0,
    "timestamp": 0,
    "samples": 0,
    "repetitions": 0,
    "received_at": None,
}
live_clients: set[WebSocket] = set()


def get_csv_path() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, DEFAULT_CSV_FILE)


def get_csv_files() -> List[str]:
    csv_path = get_csv_path()
    if os.path.exists(csv_path):
        return [csv_path]
    return []


class SensorSample(BaseModel):
    t: int
    ax1: float
    ay1: float
    az1: float
    gx1: float
    gy1: float
    gz1: float
    ax2: float
    ay2: float
    az2: float
    gx2: float
    gy2: float
    gz2: float


class SensorBatch(BaseModel):
    exercise: str
    posture: str
    user: str
    samples: List[SensorSample]
    mode: Optional[str] = "data_collection"
    prediction: Optional[dict] = None


async def broadcast_live_status():
    with live_lock:
        status = dict(latest_live_prediction)

    disconnected = []
    for client in tuple(live_clients):
        try:
            await client.send_json(status)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        live_clients.discard(client)


@app.websocket("/ws/live")
async def live_status_socket(websocket: WebSocket):
    await websocket.accept()
    live_clients.add(websocket)
    with live_lock:
        status = dict(latest_live_prediction)
    await websocket.send_json(status)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        live_clients.discard(websocket)


def ensure_csv_exists():
    """Create the shared CSV with a header row if it doesn't exist yet."""
    csv_path = get_csv_path()
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    normalized_rows = [
        [cell.lstrip("\ufeff") for cell in row]
        for row in rows
    ]
    cleaned_rows = [
        row for index, row in enumerate(normalized_rows)
        if index == 0 or row != CSV_HEADERS
    ]

    if not cleaned_rows or cleaned_rows[0] != CSV_HEADERS or cleaned_rows != rows:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            writer.writerows(cleaned_rows[1:])


@app.post("/api/sensor-data")
async def receive_sensor_data(batch: SensorBatch):
    if not batch.samples:
        raise HTTPException(status_code=400, detail="No samples received")

    received_at = datetime.now(timezone.utc).isoformat()
    label = f"{batch.exercise}_{batch.posture}"  # e.g. "squat_good", "pushup_bad"
    mode = batch.mode or "data_collection"
    is_collection_mode = mode == "data_collection"

    csv_path = get_csv_path()

    if is_collection_mode:
        with csv_lock:
            ensure_csv_exists()
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for s in batch.samples:
                    writer.writerow([
                        received_at, s.t,
                        batch.user, batch.exercise, batch.posture, label,
                        s.ax1, s.ay1, s.az1, s.gx1, s.gy1, s.gz1,
                        s.ax2, s.ay2, s.az2, s.gx2, s.gy2, s.gz2,
                    ])

    with live_lock:
        # FIX: Any valid ESP32 batch is a heartbeat; prediction is optional.
        latest_live_prediction.update({
                "online": True,
                "mode": mode,
                "received_at": received_at,
        })
        if batch.prediction:
            latest_live_prediction.update({
                "exercise": str(batch.prediction.get("exercise", "unknown")),
                "posture": str(batch.prediction.get("posture", "unknown")),
                "label": str(batch.prediction.get("label", "unknown")),
                "confidence": float(batch.prediction.get("confidence", 0.0)),
                "timestamp": int(batch.prediction.get("timestamp", 0)),
                "samples": int(batch.prediction.get("samples", len(batch.samples))),
            })
        else:
            # FIX: Do not leave an old live prediction visible during data collection.
            latest_live_prediction.update({
                "exercise": "data collection",
                "posture": "recording",
                "label": "raw samples",
                "confidence": 0.0,
                "timestamp": 0,
                "samples": len(batch.samples),
            })

    await broadcast_live_status()

    return {
        "status": "success",
        "samples_stored": len(batch.samples) if is_collection_mode else 0,
        "user": batch.user,
        "exercise": batch.exercise,
        "posture": batch.posture,
        "label": label,
        "file": os.path.basename(csv_path),
        "prediction": batch.prediction,
    }


@app.get("/api/live-status")
def get_live_status():
    with live_lock:
        status = dict(latest_live_prediction)

    received_at = status["received_at"]
    if received_at:
        received_time = datetime.fromisoformat(received_at)
        status["online"] = (datetime.now(timezone.utc) - received_time).total_seconds() < 3.0
    return status


@app.get("/api/live-prediction")
async def receive_live_prediction(
    mode: str = Query("live"),
    exercise: str = Query("unknown"),
    posture: str = Query("unknown"),
    label: str = Query("unknown"),
    confidence: float = Query(0.0, ge=0.0, le=1.0),
    timestamp: int = Query(0),
    samples: int = Query(0, ge=0),
    repetitions: int = Query(0, ge=0),
):
    """Receive only a local ESP32 KNN result; no sensor data is stored."""
    received_at = datetime.now(timezone.utc).isoformat()
    with live_lock:
        latest_live_prediction.update({
            "online": True,
            "mode": mode,
            "exercise": exercise,
            "posture": posture,
            "label": label,
            "confidence": confidence,
            "timestamp": timestamp,
            "samples": samples,
            "repetitions": repetitions,
            "received_at": received_at,
        })
    await broadcast_live_status()
    return {"status": "success", "mode": mode, "posture": posture}


@app.get("/api/sensor-data/summary")
def get_label_summary():
    """
    Row counts per label - lets you see at a glance whether your training
    set is balanced across exercises/postures before you train the KNN model.
    """
    csv_files = get_csv_files()
    if not csv_files:
        return {"total": 0, "by_label": {}}

    counts = {}
    total = 0
    skipped = 0
    for csv_path in csv_files:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "label" not in reader.fieldnames:
                skipped += 1
                continue
            for row in reader:
                label = (row.get("label") or "").strip()
                if not label:
                    skipped += 1
                    continue
                counts[label] = counts.get(label, 0) + 1
                total += 1

    return {"total": total, "by_label": counts, "skipped": skipped}


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/live-repetitions.js", response_class=FileResponse)
def live_repetitions_script():
    return FileResponse(os.path.join(BASE_DIR, "static", "live-repetitions.js"), media_type="application/javascript")