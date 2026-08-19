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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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

# A lock so concurrent POSTs never interleave writes to the same file
csv_lock = threading.Lock()
live_lock = threading.Lock()
latest_live_prediction = {
    "online": False,
    "exercise": "unknown",
    "posture": "unknown",
    "label": "unknown",
    "confidence": 0.0,
    "timestamp": 0,
    "samples": 0,
    "received_at": None,
}


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
    prediction: Optional[dict] = None


def ensure_csv_exists():
    """Create the shared CSV with a header row if it doesn't exist yet."""
    csv_path = get_csv_path()
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


@app.post("/api/sensor-data")
def receive_sensor_data(batch: SensorBatch):
    if not batch.samples:
        raise HTTPException(status_code=400, detail="No samples received")

    received_at = datetime.now(timezone.utc).isoformat()
    label = f"{batch.exercise}_{batch.posture}"  # e.g. "squat_good", "pushup_bad"

    csv_path = get_csv_path()

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

    if batch.prediction:
        with live_lock:
            latest_live_prediction.update({
                "online": True,
                "exercise": str(batch.prediction.get("exercise", "unknown")),
                "posture": str(batch.prediction.get("posture", "unknown")),
                "label": str(batch.prediction.get("label", "unknown")),
                "confidence": float(batch.prediction.get("confidence", 0.0)),
                "timestamp": int(batch.prediction.get("timestamp", 0)),
                "samples": int(batch.prediction.get("samples", len(batch.samples))),
                "received_at": datetime.now(timezone.utc).isoformat(),
            })

    return {
        "status": "success",
        "samples_stored": len(batch.samples),
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
    for csv_path in csv_files:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                counts[label] = counts.get(label, 0) + 1
                total += 1

    return {"total": total, "by_label": counts}


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))