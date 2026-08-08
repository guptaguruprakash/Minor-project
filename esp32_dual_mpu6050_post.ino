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
const char* WIFI_SSID     = "Civil_Engineers";
const char* WIFI_PASSWORD = "Dogladosh@123";
const char* SERVER_URL    = "http://192.168.18.130:8000/api/sensor-data"; // FastAPI endpoint

#define SENSOR1_ADDR 0x68   // AD0 pin LOW
#define SENSOR2_ADDR 0x69   // AD0 pin HIGH

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
String USER_NAME = "Birendra";

// ---------------- OLED DISPLAY ----------------
Adafruit_SH1106G oled(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
bool oledReady = false;
volatile unsigned long totalSamplesCollected = 0;
unsigned long lastOledUpdateMs = 0;
volatile bool collectionEnabled = false;
volatile bool postFailed = false;

// ---------------- DATA STRUCTURES ----------------
struct SensorSample {
  unsigned long timestamp;
  float ax1, ay1, az1, gx1, gy1, gz1;   // sensor 1 (accel + gyro)
  float ax2, ay2, az2, gx2, gy2, gz2;   // sensor 2 (accel + gyro)
};

// Double buffers
SensorSample bufferA[BUFFER_SIZE];
SensorSample bufferB[BUFFER_SIZE];

// Buffer bookkeeping
volatile bool useBufferA   = true;   // which buffer the collector is currently filling
volatile int  indexA       = 0;
volatile int  indexB       = 0;

// Handshake between cores
SemaphoreHandle_t bufferMutex;        // protects the shared bookkeeping above
QueueHandle_t      uploadQueue;       // holds "ready to upload" buffer descriptors
WebServer webServer(80);

struct UploadJob {
  SensorSample* data;
  int count;
};

// ---------------- SENSOR OBJECTS ----------------
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;

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
    delay(10000);
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
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println("Data Collected");
  oled.setTextSize(2);
  oled.setCursor(0, 18);
  oled.println(collectedCount);
  oled.setTextSize(1);
  oled.setCursor(0, 48);
  oled.print(postFailed ? "POST FAIL " : (collectionEnabled ? "RUN " : "PAUSE "));
  oled.print(EXERCISE_NAME);
  oled.display();
}

void applySessionConfig() {
  Serial.println("Session updated -> user: " + USER_NAME + ", exercise: " + EXERCISE_NAME + ", posture: " + POSTURE_LABEL);
}

void handleRoot() {
  String html = "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>";
  html += "<title>ESP32 Collector</title></head><body style='font-family:Arial;padding:20px'>";
  html += "<h2>ESP32 Collector</h2>";
  html += "<p>Status: <b>" + getCollectionState() + "</b></p>";
  html += "<p>Collected samples: <b>" + String(totalSamplesCollected) + "</b></p>";
  html += "<form action='/start' method='get' style='display:grid;gap:10px;max-width:320px'>";
  html += "<label>Exercise name<input name='exercise' value='" + EXERCISE_NAME + "' style='width:100%'></label>";
  html += "<label>Posture label<input name='posture' value='" + POSTURE_LABEL + "' style='width:100%'></label>";
  html += "<label>User<input name='user' value='" + USER_NAME + "' style='width:100%'></label>";
  html += "<button type='submit' style='padding:12px 18px'>Start Collection</button>";
  html += "</form>";
  html += "<p><a href='/pause'><button style='padding:12px 18px'>Pause Collection</button></a></p>";
  html += "<p><a href='/status'>View Status JSON</a></p>";
  html += "</body></html>";
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
  postFailed = false;
  collectionEnabled = true;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Collection started");
}

void handlePause() {
  collectionEnabled = false;
  postFailed = false;
  webServer.sendHeader("Location", "/", true);
  webServer.send(302, "text/plain", "Collection paused");
}

void handleStatus() {
  String json = "{";
  json += "\"collection\":\"" + getCollectionState() + "\",";
  json += "\"samples_collected\":" + String(totalSamplesCollected) + ",";
  json += "\"user\":\"" + USER_NAME + "\",";
  json += "\"exercise\":\"" + EXERCISE_NAME + "\",";
  json += "\"posture\":\"" + POSTURE_LABEL + "\",";
  json += "\"post_failed\":" + String(postFailed ? "true" : "false");
  json += "}";
  webServer.send(200, "application/json", json);
}

void setupWebServer() {
  webServer.on("/", handleRoot);
  webServer.on("/start", handleStart);
  webServer.on("/pause", handlePause);
  webServer.on("/status", handleStatus);
  webServer.begin();
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
      UploadJob job = { fullBuffer, fullCount };
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
void postBuffer(SensorSample* data, int count);

void uploaderTask(void* pvParameters) {
  UploadJob job;

  for (;;) {
    // Blocks here until the collector hands off a full buffer -
    // this never touches or slows down data collection.
    if (xQueueReceive(uploadQueue, &job, portMAX_DELAY) == pdTRUE) {
      postBuffer(job.data, job.count);
    }
  }
}

void postBuffer(SensorSample* data, int count) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping POST.");
    postFailed = true;
    collectionEnabled = false;
    return;
  }

  // Build JSON payload
  DynamicJsonDocument doc(16384); // adjust size if BUFFER_SIZE grows
  doc["exercise"] = EXERCISE_NAME;   // e.g. "squat", "pushup", "bicep_curl"
  doc["posture"]  = POSTURE_LABEL;   // "good" or "bad"
  doc["user"]     = USER_NAME;
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
void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin();

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
  collectionEnabled = false;

  // Pin collector to Core 0, uploader to Core 1
  xTaskCreatePinnedToCore(
      collectorTask, "CollectorTask", 4096, NULL, 2, NULL, 0);

  xTaskCreatePinnedToCore(
      uploaderTask, "UploaderTask", 8192, NULL, 1, NULL, 1);

  // Nothing else runs in loop(); everything happens in the two tasks above.
}

void loop() {
  webServer.handleClient();
  vTaskDelay(pdMS_TO_TICKS(10));
}
