  #include <WiFi.h>
  #include <HTTPClient.h>
  #include <WebServer.h>
  #include <WebSocketsServer.h>
  #include <ArduinoJson.h>
  #include <Wire.h>
  #include <Adafruit_MPU6050.h>
  #include <Adafruit_Sensor.h>
  #include <Adafruit_SH110X.h>
  #include "freertos/ringbuf.h"
  #include "config.h"

  #define OLED_WIDTH 128
  #define OLED_HEIGHT 64
  #define OLED_RESET -1
  #define OLED_I2C_ADDR 0x3C

  // ---------------- USER CONFIG ----------------
  #define SENSOR1_ADDR 0x68   // AD0 pin LOW
  #define SENSOR2_ADDR 0x69   // AD0 pin HIGH
  #define BUZZER_PIN   25     // Passive buzzer signal pin
  #define BAD_POSTURE_TONE_HZ 2000
  #define BAD_POSTURE_TONE_MS 150
  #define BAD_POSTURE_BEEP_COOLDOWN_MS 1500

  struct SensorSample {
    unsigned long timestamp;
    float ax1, ay1, az1, gx1, gy1, gz1;   // sensor 1 (accel + gyro)
    float ax2, ay2, az2, gx2, gy2, gz2;   // sensor 2 (accel + gyro)
  };

  #define WINDOW_SIZE             50     // complete collection batch size
  #define PREDICTION_WINDOW_SIZE  40     // recent labels used for live voting
  #define RING_BUFFER_SIZE        (WINDOW_SIZE * 3 * (sizeof(SensorSample) + 8))
  #define UPLOAD_QUEUE_DEPTH      8      // completed batches retained while POST runs
  #define QUEUE_SEND_TIMEOUT_MS   100
  #define LIVE_SNAPSHOT_INTERVAL_MS 500
  #define MAX_UPLOAD_ATTEMPTS     1
  #define HTTP_CONNECT_TIMEOUT_MS 2000
  #define HTTP_RESPONSE_TIMEOUT_MS 3000
  #define SAMPLE_RATE_MS 20    // ~50 Hz sampling
  #define WIFI_CONNECT_TIMEOUT_MS 10000
  #define MOTION_THRESHOLD 3.0f
  #define EXERCISE_INACTIVITY_TIMEOUT_MS 1200

  enum OperationMode : uint8_t {
    MODE_LIVE,
    MODE_DATA_COLLECTION
  };

  volatile OperationMode operationMode = MODE_LIVE;
  char exerciseName[24] = "squat";
  char postureLabel[12] = "good";
  char userName[32] = "bibek";

  const char* operationModeName() {
    return operationMode == MODE_LIVE ? "live" : "data_collection";
  }

  bool copyText(char* destination, size_t destinationSize, const String& source,
                bool* sanitized = nullptr) {
    if (sanitized != nullptr) {
      *sanitized = false;
    }

    String safeText;
    safeText.reserve(source.length());
    for (size_t i = 0; i < source.length(); i++) {
      const char character = source[i];
      const bool allowed = character >= 32 && character != '\'' &&
                          character != '"' && character != '<' && character != '>';
      if (allowed) {
        safeText += character;
      } else if (sanitized != nullptr) {
        *sanitized = true;
      }
    }

    if (destinationSize == 0) {
      return safeText.length() > 0;
    }
    strncpy(destination, safeText.c_str(), destinationSize - 1);
    destination[destinationSize - 1] = '\0';
    return safeText.length() >= destinationSize;
  }

  // ---------------- OLED DISPLAY ----------------
  Adafruit_SH1106G oled(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
  bool oledReady = false;
  volatile unsigned long totalSamplesCollected = 0;
  unsigned long lastOledUpdateMs = 0;
  volatile bool collectionEnabled = false;
  volatile bool uploadHalted = false;
  volatile bool postFailed = false;
  volatile bool predictionAvailable = false;
  volatile uint8_t latestPrediction = 0;
  volatile uint8_t latestPredictionConfidence = 0;
  volatile unsigned long predictionVersion = 0;
  volatile unsigned long lastMotionAt = 0;
  unsigned long lastBadPostureBeepMs = 0;
  unsigned long lastBroadcastPredictionVersion = 0;
  uint8_t predictionWindow[PREDICTION_WINDOW_SIZE] = {};
  int predictionWindowCount = 0;
  int predictionWindowIndex = 0;
  volatile unsigned long collectionGeneration = 0;
  volatile bool pipelineResetRequested = false;
  volatile bool pauseRequested = false;
  volatile bool pauseFinalized = false;

  #include "knn_model.h"
  #include "RepCounter.h"

  RepCounter repCounter;

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

  RingbufHandle_t sensorRingBuffer;
  QueueHandle_t uploadQueue;
  struct UploadJob {
    SensorSample data[WINDOW_SIZE];
    int count;
    unsigned long generation;
    char exercise[sizeof(exerciseName)];
    char posture[sizeof(postureLabel)];
    char user[sizeof(userName)];
  };
  UploadJob retryJob;
  volatile bool retryJobPending = false;
  volatile bool retryRequested = false;
  WebServer webServer(80);
  WebSocketsServer webSocket = WebSocketsServer(81);

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
  void processingTask(void* pvParameters);
  void networkTask(void* pvParameters);
  void processKnnSample(const SensorSample& sample);
  void resetPredictionWindow();
  void publishLivePrediction(int count);
  float motionSignal(const SensorSample& sample);
  bool isActiveExerciseLabel(uint8_t label);
  void setRepCounterExercise(uint8_t label);
  bool postBuffer(const UploadJob& job);
  bool setupSensors();
  bool setupOLED();
  bool connectWiFi();
  bool exerciseStarted(unsigned long timestamp);
  float exerciseMotionSignal(const SensorSample& sample);

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

  bool connectWiFi() {
    if (oledReady) {
      showOLEDMessage("Connecting", WIFI_SSID, "Please wait...");
    }

    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const unsigned long startMs = millis();
    int dots = 0;
    while (WiFi.status() != WL_CONNECTED) {
      delay(300);
      Serial.print(".");
      dots++;

      if (millis() - startMs >= WIFI_CONNECT_TIMEOUT_MS) {
        Serial.println("\nWiFi unavailable; continuing in offline live mode.");
        if (oledReady) {
          showOLEDMessage("WiFi Offline", "Live mode", "OLED prediction only");
          delay(1000);
        }
        return false;
      }

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

    return true;
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
    if (operationMode == MODE_LIVE) {
      oled.setTextSize(1);
      oled.setCursor(0, 0);
      oled.println("LIVE PREDICTION");
      oled.setTextSize(2);
      oled.setCursor(0, 18);
      if (!exerciseStarted(millis())) {
        oled.println("NOT STARTED");
      } else if (predictionAvailable) {
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
    Serial.printf("Session updated -> user: %s, exercise: %s, posture: %s\n",
                  userName, exerciseName, postureLabel);
  }

  void resetCollection() {
    collectionEnabled = false;
    uploadHalted = false;
    pauseRequested = false;
    pauseFinalized = false;

    totalSamplesCollected = 0;
    collectionGeneration++;
    pipelineResetRequested = true;

    postFailed = false;
    predictionAvailable = false;
    lastMotionAt = 0;
    predictionVersion++;
    Serial.println("Collection reset; paused at sample 0.");
  }

  void handleRoot() {
    String html = "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<title>Smart Gym Assistant</title><style>";
    html += ":root{--bg:#081311;--panel:#10231f;--panel2:#142d27;--line:#285046;--text:#edf8f1;--muted:#91aaa0;--lime:#c9f36b;--mint:#66e0bd;--red:#ff756d;--shadow:0 22px 60px #0008}*{box-sizing:border-box}body{margin:0;min-width:320px;color:var(--text);font-family:Arial,sans-serif;background:radial-gradient(circle at 85% 5%,#2b755733,transparent 30%),linear-gradient(145deg,#122d26,var(--bg) 52%);min-height:100vh}body:before{content:'';position:fixed;inset:0;z-index:-1;opacity:.18;background-image:linear-gradient(#fff1 1px,transparent 1px),linear-gradient(90deg,#fff1 1px,transparent 1px);background-size:42px 42px}.app{width:min(1080px,92vw);margin:auto;padding:34px 0 42px}.topbar{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--line);padding-bottom:24px;gap:20px}.brand{display:flex;align-items:center;gap:14px}.mark{display:grid;place-items:center;width:42px;height:42px;border:1px solid var(--lime);color:var(--lime);font-size:20px;font-weight:bold;transform:rotate(-8deg)}.eyebrow,.label,.meta,.mode{font-size:11px;letter-spacing:.14em;text-transform:uppercase}.eyebrow,.label{color:var(--muted)}h1{margin:5px 0 0;font-size:clamp(27px,5vw,48px);letter-spacing:-1.5px}h1 span{color:var(--lime)}.connection{text-align:right;color:var(--muted);font-size:12px}.connection strong{color:var(--mint);display:block;margin-bottom:7px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--red);margin-right:7px}.dot.online{background:var(--mint);box-shadow:0 0 0 5px #66e0bd1c}.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:22px}.card{border:1px solid var(--line);background:linear-gradient(145deg,#17342dE6,#0d1d1aE6);border-radius:10px;padding:25px;box-shadow:var(--shadow)}.card-head{display:flex;justify-content:space-between;align-items:center}.pulse{width:10px;height:10px;border-radius:50%;background:var(--lime);box-shadow:0 0 0 7px #c9f36b18}.value{margin:58px 0 48px;font-size:clamp(42px,9vw,92px);line-height:.85;letter-spacing:-4px;text-transform:capitalize;overflow-wrap:anywhere}.posture{min-height:286px;display:flex;flex-direction:column;justify-content:space-between}.posture .value{color:var(--lime);text-transform:uppercase;margin:54px 0 25px}.posture.bad .value{color:var(--red)}.meta{color:var(--muted);display:flex;justify-content:space-between;gap:14px}.meta strong{color:var(--text);font-weight:normal}.track{height:8px;background:#07100e;margin-top:12px;overflow:hidden}.track span{display:block;height:100%;width:0;background:var(--lime);transition:width .35s}.posture.bad .track span{background:var(--red)}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-top:17px}.button{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:11px 15px;color:var(--text);text-decoration:none;background:#17352e;font-size:12px;text-transform:uppercase;letter-spacing:.08em}.button.primary{color:#102016;background:var(--lime);border-color:var(--lime);font-weight:bold}.recording{margin-top:17px}.recording form{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.recording label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.recording input{display:block;width:100%;margin-top:7px;border:1px solid var(--line);border-radius:5px;background:#091613;color:var(--text);padding:11px}.recording button{grid-column:1/-1;cursor:pointer}.status{margin-top:17px;color:var(--muted);font-size:12px}.status b{color:var(--text)}@media(max-width:700px){.topbar{align-items:start;flex-direction:column}.connection{text-align:left}.hero{grid-template-columns:1fr}.recording form{grid-template-columns:1fr}.value{margin:45px 0 40px}}";
    html += "</style></head><body><main class='app'><header class='topbar'><div class='brand'><div class='mark'>SG</div><div><div class='eyebrow'>Dual MPU6050 / ESP32</div><h1>Smart Gym <span>Assistant</span></h1></div></div><div class='connection'><strong><i class='dot' id='dot'></i><span id='connection'>Connecting</span></strong><span class='mode'>Mode / ";
    html += operationModeName();
    html += "</span></div></header>";
    html += "<section class='hero'><article class='card'><div class='card-head'><span class='label'>Detected exercise</span><i class='pulse'></i></div><div class='value' id='live-exercise'>";
    html += operationMode == MODE_LIVE && exerciseStarted(millis()) && predictionAvailable
          ? knnExerciseName(latestPrediction)
          : operationMode == MODE_LIVE ? "exercise not started" : "unknown";
    html += "</div><div class='meta'><span>Classifier</span><strong>ESP32 KNN</strong></div></article><article class='card posture' id='posture-card'><div><span class='label'>Posture signal</span><div class='value' id='live-posture'>";
    html += predictionAvailable ? knnPostureName(latestPrediction) : "waiting";
    html += "</div></div><div><div class='meta'><span>Confidence</span><strong><span id='live-confidence'>" + String(latestPredictionConfidence) + "</span>%</strong></div><div class='track'><span id='confidence-track'></span></div><div class='meta rep-meta'><span>Repetitions</span><strong id='live-repetitions'>" + String(repCounter.repetitions()) + "</strong></div></div></article></section>";
    html += "<div class='status'>System status: <b>" + getCollectionState() + "</b></div><nav class='toolbar'><a class='button primary' href='/live'>Live mode</a><a class='button' href='/collect'>Data collection</a>";
    if (operationMode == MODE_DATA_COLLECTION) {
      html += "<a class='button' href='/pause'>Pause</a><a class='button' href='/reset'>Reset</a></nav><section class='card recording'><div class='label'>Recording session</div><form action='/start' method='get'><label>Exercise<input name='exercise' value='";
      html += exerciseName;
      html += "'></label><label>Posture<input name='posture' value='";
      html += postureLabel;
      html += "'></label><label>User<input name='user' value='";
      html += userName;
      html += "'></label><button class='button primary' type='submit'>Start recording</button></form></section>";
    } else {
      html += "</nav>";
    }
    html += "<script>let socket;function setConnection(online){document.getElementById('dot').className=online?'dot online':'dot';document.getElementById('connection').textContent=online?'ESP32 live':'Reconnecting'}function connect(){socket=new WebSocket('ws://'+location.hostname+':81/');socket.onopen=()=>setConnection(true);socket.onclose=()=>{setConnection(false);setTimeout(connect,1500)};socket.onerror=()=>socket.close();socket.onmessage=e=>{const d=JSON.parse(e.data),confidence=Math.round((d.confidence||0)*100),posture=document.getElementById('live-posture').textContent=d.posture||'unknown';document.getElementById('live-exercise').textContent=d.exercise||'unknown';document.getElementById('live-confidence').textContent=confidence;document.getElementById('live-repetitions').textContent=d.repetitions||0;document.getElementById('confidence-track').style.width=confidence+'%';document.getElementById('posture-card').classList.toggle('bad',posture==='bad')}}connect();</script></main></body></html>";
    webServer.send(200, "text/html", html);
  }

  void handleStart() {
    operationMode = MODE_DATA_COLLECTION;
    bool fieldTruncated = false;
    bool fieldSanitized = false;
    bool fieldChanged;

    if (webServer.hasArg("exercise")) {
      fieldTruncated |= copyText(exerciseName, sizeof(exerciseName), webServer.arg("exercise"), &fieldChanged);
      fieldSanitized |= fieldChanged;
    }
    if (webServer.hasArg("posture")) {
      fieldTruncated |= copyText(postureLabel, sizeof(postureLabel), webServer.arg("posture"), &fieldChanged);
      fieldSanitized |= fieldChanged;
    }
    if (webServer.hasArg("user")) {
      fieldTruncated |= copyText(userName, sizeof(userName), webServer.arg("user"), &fieldChanged);
      fieldSanitized |= fieldChanged;
    }
    if (fieldTruncated) {
      Serial.println("Warning: one or more session fields were truncated to fit firmware limits.");
    }
    if (fieldSanitized) {
      Serial.println("Warning: unsafe characters were removed from one or more session fields.");
    }

    if (WiFi.status() != WL_CONNECTED) {
      collectionEnabled = false;
      postFailed = true;
      webServer.send(503, "text/plain", "Collection unavailable: WiFi is disconnected");
      return;
    }

    applySessionConfig();
    // Start a new session without retrying the previous unacknowledged batch.
    UploadJob discardedJob;
    while (xQueueReceive(uploadQueue, &discardedJob, 0) == pdTRUE) {
      // Remove batches belonging to the previous recording session.
    }
    retryJobPending = false;
    retryRequested = false;
    resetCollection();
    postFailed = false;
    collectionEnabled = true;
    webServer.sendHeader("Location", "/", true);
    webServer.send(302, "text/plain", "Collection started");
  }

  void handleLive() {
    resetCollection();
    retryRequested = false;
    operationMode = MODE_LIVE;
    postFailed = false;
    collectionEnabled = true;
    webServer.sendHeader("Location", "/", true);
    webServer.send(302, "text/plain", "Live mode enabled");
  }

  void handleCollect() {
    resetCollection();
    retryRequested = false;
    operationMode = MODE_DATA_COLLECTION;
    postFailed = false;
    collectionEnabled = false;
    webServer.sendHeader("Location", "/", true);
    webServer.send(302, "text/plain", "Data collection mode enabled");
  }

  void handlePause() {
    collectionEnabled = false;
    pauseRequested = true;
    pauseFinalized = false;
    postFailed = false;
    webServer.sendHeader("Location", "/", true);
    webServer.send(302, "text/plain", "Collection stopping after buffered data is posted");
  }

  void handleReset() {
    resetCollection();
    webServer.sendHeader("Location", "/", true);
    webServer.send(302, "text/plain", "Collection reset");
  }

  void handleStatus() {
    String json = "{";
    json += "\"collection\":\"" + getCollectionState() + "\",";
    json += "\"mode\":\"";
    json += operationModeName();
    json += "\",";
    json += "\"samples_collected\":" + String(totalSamplesCollected) + ",";
    json += "\"user\":\"";
    json += userName;
    json += "\",";
    json += "\"exercise\":\"";
    json += exerciseName;
    json += "\",";
    json += "\"posture\":\"";
    json += postureLabel;
    json += "\",";
    json += "\"knn_posture\":\"";
    json += exerciseStarted(millis()) && predictionAvailable ? knnPostureName(latestPrediction) : "unknown";
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
    JsonDocument doc;
    const bool started = exerciseStarted(millis());
    const bool livePrediction = operationMode == MODE_LIVE && started && predictionAvailable;
    doc["exercise"] = livePrediction ? knnExerciseName(latestPrediction)
                                      : operationMode == MODE_LIVE ? "exercise not started" : "unknown";
    doc["posture"] = livePrediction ? knnPostureName(latestPrediction) : "unknown";
    doc["label"] = livePrediction ? SmartGymKnn::labelName(latestPrediction) : "unknown";
    doc["confidence"] = livePrediction ? latestPredictionConfidence / 100.0f : 0.0f;
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
    static char command[48];
    static size_t commandLength = 0;

    uint8_t processedCharacters = 0;
    while (Serial.available() > 0 && processedCharacters < 8) {
      processedCharacters++;
      const char character = static_cast<char>(Serial.read());
      if (character == '\n' || character == '\r') {
        if (commandLength == 0) {
          continue;
        }
        command[commandLength] = '\0';
        String line(command);
        line.trim();
        commandLength = 0;

        if (line.startsWith("SET ")) {
          int firstSpace = line.indexOf(' ', 4);
          if (firstSpace > 0) {
            String exercise = line.substring(4, firstSpace);
            String posture = line.substring(firstSpace + 1);
            exercise.trim();
            posture.trim();

            if ((exercise == "squat" || exercise == "pushup" || exercise == "bicep_curl") &&
                (posture == "good" || posture == "bad")) {
              copyText(exerciseName, sizeof(exerciseName), exercise);
              copyText(postureLabel, sizeof(postureLabel), posture);
              Serial.printf("Label updated -> exercise: %s, posture: %s\n",
                            exerciseName, postureLabel);
            } else {
              Serial.println("Invalid label. Use: SET <squat|pushup|bicep_curl> <good|bad>");
            }
          }
        }
      } else if (commandLength < sizeof(command) - 1) {
        command[commandLength++] = character;
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
    bool wasCollecting = false;

    for (;;) {
      if (!collectionEnabled || pauseRequested) {
        wasCollecting = false;
        vTaskDelay(pdMS_TO_TICKS(200));
        continue;
      }

      if (uploadHalted) {
        wasCollecting = false;
        vTaskDelay(pdMS_TO_TICKS(50));
        continue;
      }

      if (!wasCollecting) {
        // Do not catch up periods elapsed while collection was paused.
        lastWake = xTaskGetTickCount();
        wasCollecting = true;
      }

      if (operationMode == MODE_DATA_COLLECTION && WiFi.status() != WL_CONNECTED) {
        collectionEnabled = false;
        postFailed = true;
        Serial.println("WiFi disconnected; data collection stopped.");
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

      if (operationMode == MODE_DATA_COLLECTION) {
        // Block instead of dropping a sample when the processing task is busy.
        const BaseType_t sent = xRingbufferSend(
            sensorRingBuffer, &sample, sizeof(sample), portMAX_DELAY);
        if (sent != pdTRUE) {
          postFailed = true;
          Serial.println("Collection paused: sensor ring buffer unavailable; retrying sample.");
          continue;
        }
      } else {
        BaseType_t sent = xRingbufferSend(sensorRingBuffer, &sample, sizeof(sample), 0);
        if (sent == pdTRUE) {
          totalSamplesCollected++;
          vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_RATE_MS));
          continue;
        }

        // Live mode favors responsiveness: discard the oldest sample if needed.
        size_t discardedSize = 0;
        void* discarded = xRingbufferReceive(sensorRingBuffer, &discardedSize, 0);
        if (discarded != nullptr) {
          vRingbufferReturnItem(sensorRingBuffer, discarded);
        }
        xRingbufferSend(sensorRingBuffer, &sample, sizeof(sample), 0);
      }

      totalSamplesCollected++;
      // Maintain a steady sample rate regardless of jitter above
      vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_RATE_MS));
    }
  }

  // =================================================================
  // CORE 1 TASK: Process samples and form collection windows
  // =================================================================
  void processingTask(void* pvParameters) {
    SensorSample collectionWindow[WINDOW_SIZE];
    int collectionCount = 0;
    bool lastLiveMode = operationMode == MODE_LIVE;
    unsigned long lastGeneration = collectionGeneration;

    for (;;) {
      if (pipelineResetRequested) {
        size_t discardedSize = 0;
        void* discarded = nullptr;
        while ((discarded = xRingbufferReceive(sensorRingBuffer, &discardedSize, 0)) != nullptr) {
          vRingbufferReturnItem(sensorRingBuffer, discarded);
        }
        collectionCount = 0;
        resetPredictionWindow();
        repCounter.reset();
        pipelineResetRequested = false;
      }

      const bool liveMode = operationMode == MODE_LIVE;
      if (liveMode != lastLiveMode || lastGeneration != collectionGeneration) {
        collectionCount = 0;
        resetPredictionWindow();
        repCounter.reset();
        lastLiveMode = liveMode;
        lastGeneration = collectionGeneration;
      }

      if (pauseRequested && !pauseFinalized) {
        size_t pendingSize = 0;
        SensorSample* pending = nullptr;
        while ((pending = static_cast<SensorSample*>(
                   xRingbufferReceive(sensorRingBuffer, &pendingSize, 0))) != nullptr) {
          if (pendingSize == sizeof(SensorSample) && collectionCount < WINDOW_SIZE) {
            collectionWindow[collectionCount++] = *pending;
          }
          vRingbufferReturnItem(sensorRingBuffer, pending);
        }

        if (collectionCount > 0) {
          UploadJob finalJob;
          memcpy(finalJob.data, collectionWindow,
                 collectionCount * sizeof(SensorSample));
          finalJob.count = collectionCount;
          finalJob.generation = collectionGeneration;
          strncpy(finalJob.exercise, exerciseName, sizeof(finalJob.exercise));
          finalJob.exercise[sizeof(finalJob.exercise) - 1] = '\0';
          strncpy(finalJob.posture, postureLabel, sizeof(finalJob.posture));
          finalJob.posture[sizeof(finalJob.posture) - 1] = '\0';
          strncpy(finalJob.user, userName, sizeof(finalJob.user));
          finalJob.user[sizeof(finalJob.user) - 1] = '\0';
          if (xQueueSendToBack(uploadQueue, &finalJob,
                               pdMS_TO_TICKS(QUEUE_SEND_TIMEOUT_MS)) == pdTRUE) {
            collectionCount = 0;
          } else {
            uploadHalted = true;
            postFailed = true;
            collectionCount = 0;
          }
        }
        pauseFinalized = true;
      }

      size_t sampleSize = 0;
      SensorSample* sample = static_cast<SensorSample*>(
          xRingbufferReceive(sensorRingBuffer, &sampleSize, pdMS_TO_TICKS(100)));
      if (sample == nullptr) {
        continue;
      }

      if (pipelineResetRequested) {
        vRingbufferReturnItem(sensorRingBuffer, sample);
        continue;
      }

      if (uploadHalted && operationMode == MODE_DATA_COLLECTION) {
        vRingbufferReturnItem(sensorRingBuffer, sample);
        continue;
      }

      if (sampleSize == sizeof(SensorSample)) {
        if (liveMode) {
          processKnnSample(*sample);
        } else {
          collectionWindow[collectionCount++] = *sample;
          if (collectionCount == WINDOW_SIZE) {
            UploadJob job;
            memcpy(job.data, collectionWindow, sizeof(collectionWindow));
            job.count = WINDOW_SIZE;
            job.generation = collectionGeneration;
            strncpy(job.exercise, exerciseName, sizeof(job.exercise));
            job.exercise[sizeof(job.exercise) - 1] = '\0';
            strncpy(job.posture, postureLabel, sizeof(job.posture));
            job.posture[sizeof(job.posture) - 1] = '\0';
            strncpy(job.user, userName, sizeof(job.user));
            job.user[sizeof(job.user) - 1] = '\0';
            // The active collection window is now independent of the queued job.
            // The queue owns the completed batch until the network task gets a 2xx.
            // Send to the back: completed batches are uploaded oldest first.
            if (xQueueSendToBack(uploadQueue, &job,
                                 pdMS_TO_TICKS(QUEUE_SEND_TIMEOUT_MS)) != pdTRUE) {
              postFailed = true;
              uploadHalted = true;
              collectionEnabled = false;
              pipelineResetRequested = true;
              collectionCount = 0;
              vRingbufferReturnItem(sensorRingBuffer, sample);
              continue;
            }
            collectionCount = 0;
          }
        }
      }
      vRingbufferReturnItem(sensorRingBuffer, sample);
    }
  }

  // =================================================================
  // CORE 1 TASK: Run Wi-Fi requests independently from sensing and ML
  // =================================================================
  void networkTask(void* pvParameters) {
    UploadJob job;
    unsigned long lastSnapshotMs = 0;

    for (;;) {
      const bool retryAvailable = retryJobPending && retryRequested;
      const bool queuedBatchAvailable = !uploadHalted &&
                    uxQueueMessagesWaiting(uploadQueue) > 0;
      if (pauseRequested && pauseFinalized && !retryJobPending &&
          !uploadHalted && uxQueueMessagesWaiting(uploadQueue) == 0) {
        collectionEnabled = false;
      }
      if (!uploadHalted && (retryAvailable || queuedBatchAvailable)) {
        bool haveJob = false;
        bool wasRetryJob = false;
        if (retryAvailable) {
          job = retryJob;
          retryRequested = false;
          haveJob = true;
          wasRetryJob = true;
        } else if (xQueueReceive(uploadQueue, &job, pdMS_TO_TICKS(100)) == pdTRUE) {
          haveJob = true;
        }

        if (haveJob) {
          if (uploadHalted) {
            continue;
          }

          if (job.generation != collectionGeneration) {
            continue;
          }

          // Keep the dequeued batch until it is uploaded successfully. A
          // failed request must not silently lose labeled training samples.
          int uploadAttempts = 0;
          bool uploaded = false;
          while (uploadAttempts < MAX_UPLOAD_ATTEMPTS) {
            uploadAttempts++;
            if (postBuffer(job)) {
              if (wasRetryJob) {
                retryJobPending = false;
              }
                if (operationMode == MODE_DATA_COLLECTION && !pauseRequested &&
                  job.generation == collectionGeneration) {
                collectionEnabled = true;
              }
              postFailed = false;
              uploaded = true;
              break;
            }
            postFailed = true;
            Serial.printf("POST attempt %d/%d failed; clearing unposted data.\n",
                          uploadAttempts, MAX_UPLOAD_ATTEMPTS);
          }
          if (!uploaded) {
            UploadJob discardedJob;
            while (xQueueReceive(uploadQueue, &discardedJob, 0) == pdTRUE) {
              // Discard all batches after the failed batch as requested.
            }
            retryJobPending = false;
            retryRequested = false;
            if (operationMode == MODE_DATA_COLLECTION) {
              pipelineResetRequested = true;
              uploadHalted = true;
              collectionEnabled = false;
              postFailed = true;
              Serial.println("Upload failed; collection stopped and all unposted data was cleared.");
            }
          }
        }
      } else {
        const unsigned long now = millis();
        if (now - lastSnapshotMs >= LIVE_SNAPSHOT_INTERVAL_MS) {
          lastSnapshotMs = now;
          if (predictionAvailable) {
            publishLivePrediction(1);
          }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
      }
    }
  }

  void beepForBadPosture() {
    static bool previousBadPosture = false;
    const bool badPosture = latestPrediction == SmartGymKnn::LABEL_BICEP_BAD ||
                            latestPrediction == SmartGymKnn::LABEL_SQUAT_BAD;
    if (!badPosture) {
      if (previousBadPosture) {
        noTone(BUZZER_PIN);
      }
      previousBadPosture = false;
      return;
    }

    const unsigned long now = millis();
    if (now - lastBadPostureBeepMs >= BAD_POSTURE_BEEP_COOLDOWN_MS) {
      tone(BUZZER_PIN, BAD_POSTURE_TONE_HZ, BAD_POSTURE_TONE_MS);
      lastBadPostureBeepMs = now;
    }
    previousBadPosture = true;
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
          label != SmartGymKnn::LABEL_PUSHUP_IDLE &&
          label != SmartGymKnn::LABEL_SQUAT_IDLE;
  }

  void setRepCounterExercise(uint8_t label) {
    const RepExercise exercise = label == SmartGymKnn::LABEL_BICEP_BAD ||
                                  label == SmartGymKnn::LABEL_BICEP_GOOD ||
                                  label == SmartGymKnn::LABEL_BICEP_IDLE
                                      ? REP_EXERCISE_BICEP
                                      : label == SmartGymKnn::LABEL_PUSHUP_BAD ||
                                        label == SmartGymKnn::LABEL_PUSHUP_GOOD ||
                                        label == SmartGymKnn::LABEL_PUSHUP_IDLE
                                          ? REP_EXERCISE_PUSHUP
                                          : REP_EXERCISE_SQUAT;
    if (repCounter.exercise() != exercise) {
      repCounter.reset();
    }
    repCounter.setExercise(exercise);
  }

  void resetPredictionWindow() {
    memset(predictionWindow, 0, sizeof(predictionWindow));
    predictionWindowCount = 0;
    predictionWindowIndex = 0;
  }

  float exerciseMotionSignal(const SensorSample& sample) {
    const float acceleration1 = sqrtf(sample.ax1 * sample.ax1 + sample.ay1 * sample.ay1 + sample.az1 * sample.az1);
    const float acceleration2 = sqrtf(sample.ax2 * sample.ax2 + sample.ay2 * sample.ay2 + sample.az2 * sample.az2);
    const float gyro1 = sqrtf(sample.gx1 * sample.gx1 + sample.gy1 * sample.gy1 + sample.gz1 * sample.gz1);
    const float gyro2 = sqrtf(sample.gx2 * sample.gx2 + sample.gy2 * sample.gy2 + sample.gz2 * sample.gz2);
    return fabsf(acceleration1 - 9.80665f) + fabsf(acceleration2 - 9.80665f) +
           (gyro1 + gyro2) * 57.29578f;
  }

  bool exerciseStarted(unsigned long timestamp) {
    return lastMotionAt != 0 && timestamp - lastMotionAt <= EXERCISE_INACTIVITY_TIMEOUT_MS;
  }

  void processKnnSample(const SensorSample& sample) {
    if (operationMode != MODE_LIVE) {
      return;
    }

    if (exerciseMotionSignal(sample) >= MOTION_THRESHOLD) {
      lastMotionAt = sample.timestamp;
    } else if (!exerciseStarted(sample.timestamp)) {
      if (predictionAvailable) {
        predictionAvailable = false;
        latestPredictionConfidence = 0;
        resetPredictionWindow();
        predictionVersion++;
      }
      return;
    }

    const uint8_t samplePrediction = predictSensorSample(sample);
    predictionWindow[predictionWindowIndex] = samplePrediction;
    predictionWindowIndex = (predictionWindowIndex + 1) % PREDICTION_WINDOW_SIZE;
    if (predictionWindowCount < PREDICTION_WINDOW_SIZE) {
      predictionWindowCount++;
    }

    setRepCounterExercise(samplePrediction);
    repCounter.update(motionSignal(sample), sample.timestamp,
                      isActiveExerciseLabel(samplePrediction));

    uint8_t predictionVotes[SmartGymKnn::LABEL_COUNT] = {};
    for (int i = 0; i < predictionWindowCount; i++) {
      ++predictionVotes[predictionWindow[i]];
    }

    uint8_t prediction = 0;
    for (uint8_t label = 1; label < SmartGymKnn::LABEL_COUNT; label++) {
      if (predictionVotes[label] > predictionVotes[prediction]) {
        prediction = label;
      }
    }

    const uint8_t winningVotes = predictionVotes[prediction];
    const float predictionConfidence = static_cast<float>(winningVotes) / predictionWindowCount;
    latestPrediction = prediction;
    latestPredictionConfidence = static_cast<uint8_t>(predictionConfidence * 100.0f);
    predictionAvailable = true;
    postFailed = false;
    predictionVersion++;
    beepForBadPosture();
    Serial.printf("KNN: %s / %s (%.0f%%)\n",
                  knnExerciseName(prediction), knnPostureName(prediction), predictionConfidence * 100.0f);
  }

  void publishLivePrediction(int count) {
    if (operationMode != MODE_LIVE) {
      return;
    }

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
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(HTTP_RESPONSE_TIMEOUT_MS);
    http.begin(url);
    const int httpCode = http.GET();
    if (httpCode >= 200 && httpCode < 300) {
      Serial.println("Live KNN result published.");
    } else {
      Serial.printf("Live result publish failed: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }

  bool postBuffer(const UploadJob& job) {
    const SensorSample* data = job.data;
    const int count = job.count;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi not connected, skipping POST.");
      postFailed = true;
      return false;
    }

    Serial.printf("Data collection: attempting POST for %d labeled samples\n", count);

    // Build JSON payload
    JsonDocument doc;
    doc["exercise"] = job.exercise;   // e.g. "squat", "pushup", "bicep_curl"
    doc["posture"]  = job.posture;   // "good" or "bad"
    doc["user"]     = job.user;
    // FIX: Send the selected mode so the dashboard can show it independently of online state.
    doc["mode"]      = "data_collection";
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
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(HTTP_RESPONSE_TIMEOUT_MS);
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(payload);

    if (httpCode >= 200 && httpCode < 300) {
      Serial.printf("POST sent (%d samples) -> HTTP %d\n", count, httpCode);
      postFailed = false;
      http.end();
      return true;
    } else {
      Serial.printf("POST failed: %s\n", http.errorToString(httpCode).c_str());
      postFailed = true;
    }

    http.end();
    return false;
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

      sensorRingBuffer = xRingbufferCreate(RING_BUFFER_SIZE, RINGBUF_TYPE_NOSPLIT);
      // Completed jobs remain available while the network task serializes and posts
      // an older job. The processing task only clears its active window after enqueue.
      uploadQueue = xQueueCreate(UPLOAD_QUEUE_DEPTH, sizeof(UploadJob));
      if (sensorRingBuffer == nullptr || uploadQueue == nullptr) {
      Serial.println("Failed to create FreeRTOS buffers. Halting.");
      while (true) delay(1000);
      }
    // Start paused when Wi-Fi is available; offline mode keeps local KNN active.
    collectionEnabled = WiFi.status() != WL_CONNECTED;

      // Pin collection to Core 0 and processing/network work to Core 1.
    xTaskCreatePinnedToCore(
        collectorTask, "CollectorTask", 4096, NULL, 2, NULL, 0);

    xTaskCreatePinnedToCore(
        processingTask, "ProcessingTask", 8192, NULL, 2, NULL, 1);

      xTaskCreatePinnedToCore(
        networkTask, "NetworkTask", 8192, NULL, 1, NULL, 1);

    // Nothing else runs in loop(); everything happens in the two tasks above.
  }

  // FIX: Use the Arduino entry point so the runtime continuously services the WebServer.
  void loop() {
    webServer.handleClient();
    webSocket.loop();
    broadcastPredictionIfUpdated();
    updateOLED(totalSamplesCollected);
    vTaskDelay(pdMS_TO_TICKS(10));
  }
