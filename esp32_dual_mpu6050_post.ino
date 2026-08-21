/*
  Smart Gym Assistant - Dual-Core Sensor Logger + POST Uploader
  ---------------------------------------------------------------
  Hardware: ESP32 + 2x MPU6050 (I2C addresses 0x68 and 0x69, set via AD0 pin)

  Design:
    - Core 0 : "Collector" task -> reads both sensors at a fixed sample rate,
               writes samples into whichever buffer is currently "active".
    - Core 1 : "Uploader" task  -> waits until a buffer is marked FULL,
               serializes it to JSON, and POSTs it over WiFi.

    Two buffers (ping-pong / double buffering) are used so the collector
    NEVER has to wait for the network call to finish:
      - While buffer A is being uploaded, the collector fills buffer B.
      - When buffer B fills up, it's handed off and the collector switches
        back to buffer A (which by then should be free again).

  Libraries required (Arduino Library Manager):
    - Adafruit MPU6050
    - Adafruit Unified Sensor
    - ArduinoJson
    - WiFi (built-in for ESP32)
    - HTTPClient (built-in for ESP32)
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_SH110X.h>

#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_RESET -1
#define OLED_I2C_ADDR 0x3C

// ---------------- USER CONFIG ----------------
const char* WIFI_SSID     = "Home_Wifi";
const char* WIFI_PASSWORD = "P@ssword75";
const char* SERVER_URL    = "http://192.168.18.105:8000/api/sensor-data"; // FastAPI endpoint
// FIX: Live mode publishes only the KNN result, never the sensor batch.
const char* LIVE_RESULT_URL = "http://192.168.18.105:8000/api/live-prediction";

#define SENSOR1_ADDR 0x68   // AD0 pin LOW
#define SENSOR2_ADDR 0x69   // AD0 pin HIGH
#define BUZZER_PIN   25     // Passive buzzer signal pin
#define BAD_POSTURE_TONE_HZ 2000
#define BAD_POSTURE_TONE_MS 150
#define BAD_POSTURE_BEEP_COOLDOWN_MS 1500

#define BUFFER_SIZE   50     // samples per buffer before a POST is triggered
#define SAMPLE_RATE_MS 20    // ~50 Hz sampling

// ---------------- DATA COLLECTION LABELS ----------------
// Set these before each recording session (e.g. via Serial input, a physical
// switch, or by re-flashing) so every batch is tagged with what it is.
// This is what turns raw sensor readings into supervised training data
// for the KNN / Random Forest model.
//   EXERCISE_NAME : "squat" | "pushup" | "bicep_curl"
//   POSTURE_LABEL : "good"  | "bad"
String EXERCISE_NAME = "squat";
String POSTURE_LABEL = "good";
String USER_NAME = "bibek";
// FIX: Keep the operating mode separate from network connectivity.
String OPERATION_MODE = "live";

// ---------------- OLED DISPLAY ----------------
Adafruit_SH1106G oled(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
bool oledReady = false;
volatile unsigned long totalSamplesCollected = 0;
unsigned long lastOledUpdateMs = 0;
volatile bool collectionEnabled = false;
volatile bool postFailed = false;
volatile bool predictionAvailable = false;
volatile uint8_t latestPrediction = 0;
volatile uint8_t latestPredictionConfidence = 0;
volatile unsigned long predictionVersion = 0;
RepCounter repCounter;
unsigned long lastBadPostureBeepMs = 0;
unsigned long lastBroadcastPredictionVersion = 0;

// ---------------- DATA STRUCTURES ----------------
struct SensorSample {
  unsigned long timestamp;
  float ax1, ay1, az1, gx1, gy1, gz1;   // sensor 1 (accel + gyro)
  float ax2, ay2, az2, gx2, gy2, gz2;   // sensor 2 (accel + gyro)
};

#include "knn_model.h"
#include "RepCounter.h"

uint8_t predictSensorFeatures(const float* features);
const char* knnExerciseName(uint8_t label);
const char* knnPostureName(uint8_t label);

uint8_t predictSensorSample(const SensorSample& sample) {
  const float features[SmartGymKnn::FEATURE_COUNT] = {
    sample.ax1, sample.ay1, sample.az1, sample.gx1, sample.gy1, sample.gz1,
    sample.ax2, sample.ay2, sample.az2, sample.gx2, sample.gy2, sample.gz2,
  };
  return predictSensorFeatures(features);
}

// Double buffers
SensorSample bufferA[BUFFER_SIZE];
SensorSample bufferB[BUFFER_SIZE];

// Buffer bookkeeping
volatile bool useBufferA   = true;   // which buffer the collector is currently filling
volatile int  indexA       = 0;
volatile int  indexB       = 0;
volatile unsigned long collectionGeneration = 0;

// Handshake between cores
SemaphoreHandle_t bufferMutex;        // protects the shared bookkeeping above
QueueHandle_t      uploadQueue;       // holds "ready to upload" buffer descriptors
WebServer webServer(80);
WebSocketsServer webSocket = WebSocketsServer(81);
struct UploadJob {
  SensorSample* data;
  int count;
  unsigned long generation;
};

// ---------------- SENSOR OBJECTS ----------------
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;

// FIX: Declare functions explicitly because they are called before their definitions.
void updateOLED(unsigned long collectedCount);
void applySessionConfig();
void resetCollection();
void handleRoot();
void handleStart();
void handleLive();
void handleCollect();
void handlePause();
void handleReset();
void handleStatus();
void setupWebServer();
void webSocketEvent(uint8_t clientNumber, WStype_t type, uint8_t* payload, size_t length);
String currentPredictionJson();
void broadcastPredictionIfUpdated();
void checkSerialForLabelUpdate();
void beepForBadPosture();
void collectorTask(void* pvParameters);
void uploaderTask(void* pvParameters);
void updateKnnPrediction(SensorSample* data, int count);
void publishLivePrediction(int count);
float motionSignal(const SensorSample& sample);
bool isActiveExerciseLabel(uint8_t label);
void setRepCounterExercise(uint8_t label);
void postBuffer(SensorSample* data, int count);
bool setupSensors();
bool setupOLED();
void connectWiFi();

// ---------------- WIFI SETUP ----------------
void showOLEDMessage(const String& line1, const String& line2 = "", const String& line3 = "", uint8_t textSize = 1) {
  if (!oledReady) {
    return;
  }

  oled.clearDisplay();
  oled.setTextColor(SH110X_WHITE);
  oled.setTextSize(textSize);
  oled.setCursor(0, 0);
  oled.println(line1);

  if (!line2.isEmpty()) {
    oled.setTextSize(1);
    oled.setCursor(0, 18);
    oled.println(line2);
  }

  if (!line3.isEmpty()) {
    oled.setCursor(0, 36);
    oled.println(line3);
  }

  oled.display();
}

void connectWiFi() {
  if (oledReady) {
    showOLEDMessage("Connecting", WIFI_SSID, "Please wait...");
  }

  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    dots++;

    if (oledReady && dots % 4 == 0) {
      showOLEDMessage("Connecting", WIFI_SSID, String("Attempt ") + dots);
    }
  }

  String ip = WiFi.localIP().toString();
  Serial.println("\nWiFi connected. IP: " + ip);

  if (oledReady) {
    showOLEDMessage("WiFi Connected", ip, "Starting up...", 1);
    delay(1000);
    updateOLED(totalSamplesCollected);
  }
}

String getCollectionState() {
  return collectionEnabled ? "running" : "paused";
}

// ---------------- SENSOR SETUP ----------------
bool setupSensors() {
  if (!mpu1.begin(SENSOR1_ADDR)) {
    Serial.println("Failed to find MPU6050 #1");
    return false;
  }
  if (!mpu2.begin(SENSOR2_ADDR)) {
    Serial.println("Failed to find MPU6050 #2");
    return false;
  }

  for (Adafruit_MPU6050* m : {&mpu1, &mpu2}) {
    m->setAccelerometerRange(MPU6050_RANGE_8_G);
    m->setGyroRange(MPU6050_RANGE_500_DEG);
    m->setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  Serial.println("Both MPU6050 sensors initialized.");
  return true;
}

// ---------------- OLED SETUP ----------------
bool setupOLED() {
  if (!oled.begin(OLED_I2C_ADDR, true)) {
    Serial.println("OLED init failed");
    return false;
  }

  oledReady = true;
  oled.clearDisplay();
  oled.setTextColor(SH110X_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println("Smart Gym Assistant");
  oled.println("Starting...");
  oled.display();
  return true;
}

void updateOLED(unsigned long collectedCount) {
  if (!oledReady) {
    return;
  }

  unsigned long now = millis();
  if (now - lastOledUpdateMs < 250) {
    return;
  }
  lastOledUpdateMs = now;

  oled.clearDisplay();
  oled.setTextColor(SH110X_WHITE);
  if (OPERATION_MODE == "live") {
    oled.setTextSize(1);
    oled.setCursor(0, 0);
    oled.println("LIVE PREDICTION");
    oled.setTextSize(2);
    oled.setCursor(0, 18);
    if (predictionAvailable) {
      oled.println(knnExerciseName(latestPrediction));
      oled.setCursor(0, 38);
      oled.println(knnPostureName(latestPrediction));
    } else {
      oled.println("WAITING");
    }
    oled.setTextSize(1);
    oled.setCursor(0, 56);
    if (predictionAvailable) {
      oled.print(latestPredictionConfidence);
      oled.print("%");
      oled.setCursor(72, 56);
      oled.print("R:");
      oled.print(repCounter.repetitions());
    }
  } else {
    oled.setTextSize(1);
    oled.setCursor(0, 0);
    oled.println("Data Collected");
    oled.setTextSize(2);
    oled.setCursor(0, 18);
    oled.println(collectedCount);
    oled.setTextSize(1);
    oled.setCursor(0, 48);
    if (postFailed) {
      oled.print("POST FAIL");
    } else if (!collectionEnabled) {
      oled.print("PAUSED");
    } else {
      oled.print("RECORDING");
    }
  }
  oled.display();
}

void applySessionConfig() {
  Serial.println("Session updated -> user: " + USER_NAME + ", exercise: " + EXERCISE_NAME + ", posture: " + POSTURE_LABEL);
}

void resetCollection() {
  collectionEnabled = false;
  repCounter.reset();

  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  indexA = 0;
  indexB = 0;
  useBufferA = true;
  totalSamplesCollected = 0;
  collectionGeneration++;
  xQueueReset(uploadQueue);
  xSemaphoreGive(bufferMutex);

  postFailed = false;
  predictionAvailable = false;
  predictionVersion++;
  Serial.println("Collection reset; paused at sample 0.");
}

void handleRoot() {
  String html = "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>";
  html += "<title>Smart Gym Assistant</title><style>";
  html += ":root{--bg:#081311;--panel:#10231f;--panel2:#142d27;--line:#285046;--text:#edf8f1;--muted:#91aaa0;--lime:#c9f36b;--mint:#66e0bd;--red:#ff756d;--shadow:0 22px 60px #0008}*{box-sizing:border-box}body{margin:0;min-width:320px;color:var(--text);font-family:Arial,sans-serif;background:radial-gradient(circle at 85% 5%,#2b755733,transparent 30%),linear-gradient(145deg,#122d26,var(--bg) 52%);min-height:100vh}body:before{content:'';position:fixed;inset:0;z-index:-1;opacity:.18;background-image:linear-gradient(#fff1 1px,transparent 1px),linear-gradient(90deg,#fff1 1px,transparent 1px);background-size:42px 42px}.app{width:min(1080px,92vw);margin:auto;padding:34px 0 42px}.topbar{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--line);padding-bottom:24px;gap:20px}.brand{display:flex;align-items:center;gap:14px}.mark{display:grid;place-items:center;width:42px;height:42px;border:1px solid var(--lime);color:var(--lime);font-size:20px;font-weight:bold;transform:rotate(-8deg)}.eyebrow,.label,.meta,.mode{font-size:11px;letter-spacing:.14em;text-transform:uppercase}.eyebrow,.label{color:var(--muted)}h1{margin:5px 0 0;font-size:clamp(27px,5vw,48px);letter-spacing:-1.5px}h1 span{color:var(--lime)}.connection{text-align:right;color:var(--muted);font-size:12px}.connection strong{color:var(--mint);display:block;margin-bottom:7px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--red);margin-right:7px}.dot.online{background:var(--mint);box-shadow:0 0 0 5px #66e0bd1c}.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:22px}.card{border:1px solid var(--line);background:linear-gradient(145deg,#17342dE6,#0d1d1aE6);border-radius:10px;padding:25px;box-shadow:var(--shadow)}.card-head{display:flex;justify-content:space-between;align-items:center}.pulse{width:10px;height:10px;border-radius:50%;background:var(--lime);box-shadow:0 0 0 7px #c9f36b18}.value{margin:58px 0 48px;font-size:clamp(42px,9vw,92px);line-height:.85;letter-spacing:-4px;text-transform:capitalize;overflow-wrap:anywhere}.posture{min-height:286px;display:flex;flex-direction:column;justify-content:space-between}.posture .value{color:var(--lime);text-transform:uppercase;margin:54px 0 25px}.posture.bad .value{color:var(--red)}.meta{color:var(--muted);display:flex;justify-content:space-between;gap:14px}.meta strong{color:var(--text);font-weight:normal}.track{height:8px;background:#07100e;margin-top:12px;overflow:hidden}.track span{display:block;height:100%;width:0;background:var(--lime);transition:width .35s}.posture.bad .track span{background:var(--red)}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-top:17px}.button{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:11px 15px;color:var(--text);text-decoration:none;background:#17352e;font-size:12px;text-transform:uppercase;letter-spacing:.08em}.button.primary{color:#102016;background:var(--lime);border-color:var(--lime);font-weight:bold}.recording{margin-top:17px}.recording form{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.recording label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.recording input{display:block;width:100%;margin-top:7px;border:1px solid var(--line);border-radius:5px;background:#091613;color:var(--text);padding:11px}.recording button{grid-column:1/-1;cursor:pointer}.status{margin-top:17px;color:var(--muted);font-size:12px}.status b{color:var(--text)}@media(max-width:700px){.topbar{align-items:start;flex-direction:column}.connection{text-align:left}.hero{grid-template-columns:1fr}.recording form{grid-template-columns:1fr}.value{margin:45px 0 40px}}";
  html += "</style></head><body><main class='app'><header class='topbar'><div class='brand'><div class='mark'>SG</div><div><div class='eyebrow'>Dual MPU6050 / ESP32</div><h1>Smart Gym <span>Assistant</span></h1></div></div><div class='connection'><strong><i class='dot' id='dot'></i><span id='connection'>Connecting</span></strong><span class='mode'>Mode / " + OPERATION_MODE + "</span></div></header>";
  html += "<section class='hero'><article class='card'><div class='card-head'><span class='label'>Detected exercise</span><i class='pulse'></i></div><div class='value' id='live-exercise'>";
  html += predictionAvailable ? knnExerciseName(latestPrediction) : "waiting";
  html += "</div><div class='meta'><span>Classifier</span><strong>ESP32 KNN</strong></div></article><article class='card posture' id='posture-card'><div><span class='label'>Posture signal</span><div class='value' id='live-posture'>";
  html += predictionAvailable ? knnPostureName(latestPrediction) : "waiting";
  html += "</div></div><div><div class='meta'><span>Confidence</span><strong><span id='live-confidence'>" + String(latestPredictionConfidence) + "</span>%</strong></div><div class='track'><span id='confidence-track'></span></div><div class='meta rep-meta'><span>Repetitions</span><strong id='live-repetitions'>" + String(repCounter.repetitions()) + "</strong></div></div></article></section>";
  html += "<div class='status'>System status: <b>" + getCollectionState() + "</b></div><nav class='toolbar'><a class='button primary' href='/live'>Live mode</a><a class='button' href='/collect'>Data collection</a>";
  if (OPERATION_MODE == "data_collection") {
    html += "<a class='button' href='/pause'>Pause</a><a class='button' href='/reset'>Reset</a></nav><section class='card recording'><div class='label'>Recording session</div><form action='/start' method='get'><label>Exercise<input name='exercise' value='" + EXERCISE_NAME + "'></label><label>Posture<input name='posture' value='" + POSTURE_LABEL + "'></label><label>User<input name='user' value='" + USER_NAME + "'></label><button class='button primary' type='submit'>Start recording</button></form></section>";
  } else {
    html += "</nav>";
  }
  html += "<script>let socket;function setConnection(online){document.getElementById('dot').className=online?'dot online':'dot';document.getElementById('connection').textContent=online?'ESP32 live':'Reconnecting'}function connect(){socket=new WebSocket('ws://'+location.hostname+':81/');socket.onopen=()=>setConnection(true);socket.onclose=()=>{setConnection(false);setTimeout(connect,1500)};socket.onerror=()=>socket.close();socket.onmessage=e=>{const d=JSON.parse(e.data),confidence=Math.round((d.confidence||0)*100),posture=document.getElementById('live-posture').textContent=d.posture||'unknown';document.getElementById('live-exercise').textContent=d.exercise||'unknown';document.getElementById('live-confidence').textContent=confidence;document.getElementById('live-repetitions').textContent=d.repetitions||0;document.getElementById('confidence-track').style.width=confidence+'%';document.getElementById('posture-card').classList.toggle('bad',posture==='bad')}}connect();</script></main></body></html>";
  webServer.send(200, "text/html", html);
}

void handleStart() {
  if (webServer.hasArg("exercise")) {
    EXERCISE_NAME = webServer.arg("exercise");
  }
  if (webServer.hasArg("posture")) {
    POSTURE_LABEL = webServer.arg("posture");
  }
  if (webServer.hasArg("user")) {
    USER_NAME = webServer.arg("user");
  }

  applySessionConfig();
  repCounter.reset();
  postFailed = false;
  collectionEnabled = true;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Collection started");
}

void handleLive() {
  OPERATION_MODE = "live";
  repCounter.reset();
  postFailed = false;
  collectionEnabled = true;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Live mode enabled");
}

void handleCollect() {
  // FIX: Select data collection mode without starting sampling or POST uploads.
  OPERATION_MODE = "data_collection";
  postFailed = false;
  collectionEnabled = false;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Data collection mode enabled");
}

void handlePause() {
  collectionEnabled = false;
  repCounter.reset();
  postFailed = false;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Collection paused");
}

void handleReset() {
  resetCollection();
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Collection reset");
}

void handleStatus() {
  String json = "{";
  json += "\"collection\":\"" + getCollectionState() + "\",";
  json += "\"mode\":\"" + OPERATION_MODE + "\",";
  json += "\"samples_collected\":" + String(totalSamplesCollected) + ",";
  json += "\"user\":\"" + USER_NAME + "\",";
  json += "\"exercise\":\"" + EXERCISE_NAME + "\",";
  json += "\"posture\":\"" + POSTURE_LABEL + "\",";
  json += "\"knn_posture\":\"";
  json += predictionAvailable ? knnPostureName(latestPrediction) : "unknown";
  json += "\",";
  json += "\"knn_confidence\":" + String(latestPredictionConfidence) + ",";
  json += "\"repetitions\":" + String(repCounter.repetitions()) + ",";
  json += "\"post_failed\":" + String(postFailed ? "true" : "false");
  json += "}";
  webServer.send(200, "application/json", json);
}

void setupWebServer() {
  webServer.on("/", handleRoot);
  webServer.on("/start", handleStart);
  webServer.on("/live", handleLive);
  webServer.on("/collect", handleCollect);
  webServer.on("/pause", handlePause);
  webServer.on("/reset", handleReset);
  webServer.on("/status", handleStatus);
  webServer.begin();
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

String currentPredictionJson() {
  DynamicJsonDocument doc(512);
  doc["exercise"] = predictionAvailable ? knnExerciseName(latestPrediction) : "unknown";
  doc["posture"] = predictionAvailable ? knnPostureName(latestPrediction) : "unknown";
  doc["label"] = predictionAvailable ? SmartGymKnn::labelName(latestPrediction) : "unknown";
  doc["confidence"] = predictionAvailable ? latestPredictionConfidence / 100.0f : 0.0f;
  doc["repetitions"] = repCounter.repetitions();
  String payload;
  serializeJson(doc, payload);
  return payload;
}

void webSocketEvent(uint8_t clientNumber, WStype_t type, uint8_t* payload, size_t length) {
  (void)payload;
  (void)length;
  if (type == WStype_CONNECTED) {
    String predictionJson = currentPredictionJson();
    webSocket.sendTXT(clientNumber, predictionJson);
  }
}

void broadcastPredictionIfUpdated() {
  if (predictionVersion != lastBroadcastPredictionVersion) {
    String predictionJson = currentPredictionJson();
    webSocket.broadcastTXT(predictionJson);
    lastBroadcastPredictionVersion = predictionVersion;
  }
}

// ---------------- SERIAL LABEL CONTROL ----------------
// While recording, type a command like:  SET squat good
// or:                                    SET pushup bad
// into the Serial Monitor to switch labels between sets, no reflash needed.
void checkSerialForLabelUpdate() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("SET ")) {
      int firstSpace  = line.indexOf(' ', 4);
      if (firstSpace > 0) {
        String exercise = line.substring(4, firstSpace);
        String posture  = line.substring(firstSpace + 1);
        exercise.trim();
        posture.trim();

        if ((exercise == "squat" || exercise == "pushup" || exercise == "bicep_curl") &&
            (posture == "good" || posture == "bad")) {
          EXERCISE_NAME = exercise;
          POSTURE_LABEL = posture;
          Serial.println("Label updated -> exercise: " + EXERCISE_NAME + ", posture: " + POSTURE_LABEL);
        } else {
          Serial.println("Invalid label. Use: SET <squat|pushup|bicep_curl> <good|bad>");
        }
      }
    }
  }
}

// =================================================================
// CORE 0 TASK: Collect sensor data continuously
// =================================================================
void collectorTask(void* pvParameters) {
  sensors_event_t a1, g1, temp1;
  sensors_event_t a2, g2, temp2;

  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    if (!collectionEnabled) {
      updateOLED(totalSamplesCollected);
      vTaskDelay(pdMS_TO_TICKS(200));
      continue;
    }

    checkSerialForLabelUpdate(); // non-blocking; only acts if new input is waiting

    mpu1.getEvent(&a1, &g1, &temp1);
    mpu2.getEvent(&a2, &g2, &temp2);

    SensorSample sample;
    sample.timestamp = millis();
    sample.ax1 = a1.acceleration.x; sample.ay1 = a1.acceleration.y; sample.az1 = a1.acceleration.z;
    sample.gx1 = g1.gyro.x;         sample.gy1 = g1.gyro.y;         sample.gz1 = g1.gyro.z;
    sample.ax2 = a2.acceleration.x; sample.ay2 = a2.acceleration.y; sample.az2 = a2.acceleration.z;
    sample.gx2 = g2.gyro.x;         sample.gy2 = g2.gyro.y;         sample.gz2 = g2.gyro.z;

    // Critical section: write into the active buffer and check for "full"
    xSemaphoreTake(bufferMutex, portMAX_DELAY);

    SensorSample* fullBuffer = nullptr;
    int fullCount = 0;
    unsigned long generationSnapshot = collectionGeneration;

    if (useBufferA) {
      bufferA[indexA++] = sample;
      if (indexA >= BUFFER_SIZE) {
        fullBuffer = bufferA;
        fullCount  = BUFFER_SIZE;
        indexA = 0;
        useBufferA = false;   // switch collection to buffer B
      }
    } else {
      bufferB[indexB++] = sample;
      if (indexB >= BUFFER_SIZE) {
        fullBuffer = bufferB;
        fullCount  = BUFFER_SIZE;
        indexB = 0;
        useBufferA = true;    // switch collection back to buffer A
      }
    }

    totalSamplesCollected++;
    unsigned long countSnapshot = totalSamplesCollected;

    xSemaphoreGive(bufferMutex);

    updateOLED(countSnapshot);

    // Hand the full buffer off to the uploader task (non-blocking for the collector)
    if (fullBuffer != nullptr) {
      UploadJob job = { fullBuffer, fullCount, generationSnapshot };
      // If the queue is somehow still full (uploader falling behind), drop the
      // oldest pending job rather than stalling collection.
      if (xQueueSend(uploadQueue, &job, 0) != pdTRUE) {
        UploadJob discard;
        xQueueReceive(uploadQueue, &discard, 0);
        xQueueSend(uploadQueue, &job, 0);
        Serial.println("Warning: uploader falling behind, dropped oldest batch.");
      }
    }

    // Maintain a steady sample rate regardless of jitter above
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_RATE_MS));
  }
}

// =================================================================
// CORE 1 TASK: Upload full buffers via HTTP POST
// =================================================================
void uploaderTask(void* pvParameters) {
  UploadJob job;

  for (;;) {
    // Blocks here until the collector hands off a full buffer -
    // this never touches or slows down data collection.
    if (xQueueReceive(uploadQueue, &job, portMAX_DELAY) == pdTRUE) {
      if (job.generation == collectionGeneration) {
        // FIX: Live mode performs local KNN inference and never sends a POST.
        if (OPERATION_MODE == "live") {
          updateKnnPrediction(job.data, job.count);
        } else {
          postBuffer(job.data, job.count);
        }
      } else {
        Serial.println("Skipped upload from a previous collection session.");
      }
    }
  }
}

void beepForBadPosture() {
  const bool badPosture = latestPrediction == SmartGymKnn::LABEL_BICEP_BAD ||
                          latestPrediction == SmartGymKnn::LABEL_SQUAT_BAD;
  if (!badPosture) {
    noTone(BUZZER_PIN);
    return;
  }

  const unsigned long now = millis();
  if (now - lastBadPostureBeepMs >= BAD_POSTURE_BEEP_COOLDOWN_MS) {
    tone(BUZZER_PIN, BAD_POSTURE_TONE_HZ, BAD_POSTURE_TONE_MS);
    lastBadPostureBeepMs = now;
  }
}

float motionSignal(const SensorSample& sample) {
  const float acceleration1 = sqrtf(sample.ax1 * sample.ax1 + sample.ay1 * sample.ay1 + sample.az1 * sample.az1);
  const float acceleration2 = sqrtf(sample.ax2 * sample.ax2 + sample.ay2 * sample.ay2 + sample.az2 * sample.az2);
  const float gyro1 = sqrtf(sample.gx1 * sample.gx1 + sample.gy1 * sample.gy1 + sample.gz1 * sample.gz1);
  const float gyro2 = sqrtf(sample.gx2 * sample.gx2 + sample.gy2 * sample.gy2 + sample.gz2 * sample.gz2);
  return acceleration1 + acceleration2 + (gyro1 + gyro2) * 57.29578f;
}

bool isActiveExerciseLabel(uint8_t label) {
  return label != SmartGymKnn::LABEL_BICEP_IDLE &&
         label != SmartGymKnn::LABEL_SQUAT_IDLE;
}

void setRepCounterExercise(uint8_t label) {
  const RepExercise exercise = label == SmartGymKnn::LABEL_BICEP_BAD ||
                                label == SmartGymKnn::LABEL_BICEP_GOOD ||
                                label == SmartGymKnn::LABEL_BICEP_IDLE
                                    ? REP_EXERCISE_BICEP
                                    : REP_EXERCISE_SQUAT;
  if (repCounter.exercise() != exercise) {
    repCounter.reset();
  }
  repCounter.setExercise(exercise);
}

void updateKnnPrediction(SensorSample* data, int count) {
  uint8_t predictionVotes[SmartGymKnn::LABEL_COUNT] = {};
  for (int i = 0; i < count; i++) {
    const uint8_t samplePrediction = predictSensorSample(data[i]);
    ++predictionVotes[samplePrediction];
    setRepCounterExercise(samplePrediction);
    repCounter.update(motionSignal(data[i]), data[i].timestamp,
                      isActiveExerciseLabel(samplePrediction));
  }

  uint8_t prediction = 0;
  for (uint8_t label = 1; label < SmartGymKnn::LABEL_COUNT; label++) {
    if (predictionVotes[label] > predictionVotes[prediction]) {
      prediction = label;
    }
  }

  const uint8_t winningVotes = predictionVotes[prediction];
  const float predictionConfidence = count > 0 ? static_cast<float>(winningVotes) / count : 0.0f;
  latestPrediction = prediction;
  latestPredictionConfidence = static_cast<uint8_t>(predictionConfidence * 100.0f);
  predictionAvailable = true;
  postFailed = false;
  predictionVersion++;
  beepForBadPosture();
  Serial.printf("KNN: %s / %s (%.0f%%)\n",
                knnExerciseName(prediction), knnPostureName(prediction), predictionConfidence * 100.0f);
  publishLivePrediction(count);
}

void publishLivePrediction(int count) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, live result not published.");
    return;
  }

  String url = String(LIVE_RESULT_URL) + "?mode=live";
  // FIX: Publish the exercise predicted by KNN, not the collection label.
  url += "&exercise=" + String(knnExerciseName(latestPrediction));
  url += "&posture=" + String(knnPostureName(latestPrediction));
  url += "&label=" + String(SmartGymKnn::labelName(latestPrediction));
  url += "&confidence=" + String(latestPredictionConfidence / 100.0f, 3);
  url += "&repetitions=" + String(repCounter.repetitions());
  url += "&timestamp=" + String(count > 0 ? millis() : 0);
  url += "&samples=" + String(count);

  HTTPClient http;
  http.begin(url);
  const int httpCode = http.GET();
  if (httpCode >= 200 && httpCode < 300) {
    Serial.println("Live KNN result published.");
  } else {
    Serial.printf("Live result publish failed: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

void postBuffer(SensorSample* data, int count) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping POST.");
    postFailed = true;
    collectionEnabled = false;
    return;
  }

  Serial.printf("Data collection: uploaded %d labeled samples\n", count);

  // Build JSON payload
  DynamicJsonDocument doc(16384); // adjust size if BUFFER_SIZE grows
  doc["exercise"] = EXERCISE_NAME;   // e.g. "squat", "pushup", "bicep_curl"
  doc["posture"]  = POSTURE_LABEL;   // "good" or "bad"
  doc["user"]     = USER_NAME;
  // FIX: Send the selected mode so the dashboard can show it independently of online state.
  doc["mode"]      = OPERATION_MODE;
  JsonArray samples = doc.createNestedArray("samples");

  for (int i = 0; i < count; i++) {
    JsonObject s = samples.createNestedObject();
    s["t"]   = data[i].timestamp;
    s["ax1"] = data[i].ax1; s["ay1"] = data[i].ay1; s["az1"] = data[i].az1;
    s["gx1"] = data[i].gx1; s["gy1"] = data[i].gy1; s["gz1"] = data[i].gz1;
    s["ax2"] = data[i].ax2; s["ay2"] = data[i].ay2; s["az2"] = data[i].az2;
    s["gx2"] = data[i].gx2; s["gy2"] = data[i].gy2; s["gz2"] = data[i].gz2;
  }

  String payload;
  serializeJson(doc, payload);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(payload);

  if (httpCode >= 200 && httpCode < 300) {
    Serial.printf("POST sent (%d samples) -> HTTP %d\n", count, httpCode);
    postFailed = false;
  } else {
    Serial.printf("POST failed: %s\n", http.errorToString(httpCode).c_str());
    postFailed = true;
    collectionEnabled = false;
  }

  http.end();
}

// ---------------- SETUP ----------------
// FIX: Use the Arduino entry point so the linker can find setup().
void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin();
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  if (!setupOLED()) {
    Serial.println("OLED unavailable; continuing without display.");
  }

  connectWiFi();

  setupWebServer();

  if (!setupSensors()) {
    Serial.println("Sensor init failed. Halting.");
    while (true) delay(1000);
  }

  bufferMutex = xSemaphoreCreateMutex();
  uploadQueue = xQueueCreate(2, sizeof(UploadJob)); // small queue: only need 1-2 pending batches
  // FIX: Boot remains paused; the user must explicitly press Start Collection.
  collectionEnabled = false;

  // Pin collector to Core 0, uploader to Core 1
  xTaskCreatePinnedToCore(
      collectorTask, "CollectorTask", 4096, NULL, 2, NULL, 0);

  xTaskCreatePinnedToCore(
      uploaderTask, "UploaderTask", 8192, NULL, 1, NULL, 1);

  // Nothing else runs in loop(); everything happens in the two tasks above.
}

// FIX: Use the Arduino entry point so the runtime continuously services the WebServer.
void loop() {
  webServer.handleClient();
  webSocket.loop();
  broadcastPredictionIfUpdated();
  vTaskDelay(pdMS_TO_TICKS(10));
}
