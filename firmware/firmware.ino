/*  ESP32‑CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include "FS.h"
#include "SD_MMC.h"
#include "secrets.h"           
#include "sd_read_write.h"

/* ----------  Pin mapping ---------- */
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5
#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM     9
#define Y4_GPIO_NUM     8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

const int photoInterval = 10000; // 10 seconds

static QueueHandle_t   frameQueue   = nullptr; // frames for SD writer
static SemaphoreHandle_t sdMutex     = nullptr; // protect SD card
static SemaphoreHandle_t cameraMutex = nullptr; // exclusive camera access
static SemaphoreHandle_t inferenceTrigger = nullptr; // user-requested photo
static TaskHandle_t streamTaskHandle = nullptr; // streaming task had

static WebServer server(80);

/* forward declarations */
static void initCamera();
static void connectWifi();
static void initWebServer();
static void startMjpegTask(WiFiClient client);
static void streamTask(void *pvParameters);
static void inferenceTask(void *pvParameters);

static void cameraCaptureTask(void *pvParameters);
static void sdWriterTask(void *pvParameters);

/* ----------------------------------------------------------------------- */
void setup() {
  Serial.begin(115200);
  delay(500);
  initCamera();
  connectWifi();

  sdmmcInit();
  removeDir(SD_MMC, "/camera");
  createDir(SD_MMC, "/camera");

  initWebServer();

  frameQueue = xQueueCreate(5, sizeof(camera_fb_t*));
  sdMutex = xSemaphoreCreateMutex();
  cameraMutex = xSemaphoreCreateMutex();
  inferenceTrigger = xSemaphoreCreateBinary();

  xTaskCreatePinnedToCore(cameraCaptureTask,
                          "CamCap",
                          4096,
                          nullptr,
                          2,
                          nullptr,
                          0);

  xTaskCreatePinnedToCore(sdWriterTask,
                          "SDWrite",
                          8192,
                          nullptr,
                          2,
                          nullptr,
                          1);

  xTaskCreatePinnedToCore(inferenceTask,
                          "Infer",
                          8192,
                          nullptr,
                          2,
                          nullptr,
                          0);
  
}

void loop() {
  server.handleClient();
}

static void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.fb_count     = 3; 

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
}

static void connectWifi()
{
  if (WiFi.isConnected()) return;

  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(MY_SSID, MY_PASS);

  const unsigned long timeout = 20000;
  unsigned long start = millis();
  Serial.print("Connecting to Wi‑Fi");
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeout) {
    delay(500);
    Serial.print('.');
  }

  Serial.println();
  if (WiFi.isConnected()) {
    Serial.println("Wi‑Fi connected");
        Serial.print("Camera IP address: ");
    Serial.println(WiFi.localIP());
  } else {
        Serial.println("Wi‑Fi failed to connect");
  }
}

static void initWebServer() {
  server.on("/", HTTP_GET, []() {
    const char html[] PROGMEM = R"=====(
      <!DOCTYPE html>
      <html>
        <head><title>ESP32‑CAM Live</title></head>
        <body>
          <h1>Camera Live Stream</h1>
          <img src="/stream" style="width:100%;max-width:640px;">
          <p>
            <a href="/snapshot">Snapshot</a> |
            <a href="#" onclick="saveToSD()">Save to SD</a>
          </p>

          <script>
          function saveToSD() {
            fetch('/save')
              .then(response => {
                if(response.ok) alert("Capture queued!");
                else alert("Camera busy!");
              });
          }
          </script>
        </body>
      </html>
    )=====";
    server.send(200, "text/html", html);
  });

  server.on("/snapshot", HTTP_GET, []() {
    if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
      server.send(503, "text/plain", "Camera Busy");
      return;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    xSemaphoreGive(cameraMutex);
    if (!fb) {
      server.send(500, "text/plain", "Capture failed");
      return;
    }

    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Length", String(fb->len));
    server.client().write(fb->buf, fb->len);
    esp_camera_fb_return(fb);
  });

  server.on("/stream", HTTP_GET, []() {
    WiFiClient client = server.client(); 
    if (!client) {
      server.send(500, "text/plain", "No client");
      return;
    }

    if (streamTaskHandle) {
      vTaskDelete(streamTaskHandle);
      streamTaskHandle = nullptr;
    }

    // Allocate the client on the heap and pass the pointer to the task.
    WiFiClient *pClient = new WiFiClient(std::move(client));
    BaseType_t ok = xTaskCreatePinnedToCore(
            streamTask,
            "MjpegStream",
            6144,
            pClient,
            2,
            &streamTaskHandle,
            1); 
    if (ok != pdPASS) {
      delete pClient;
      server.send(500, "text/plain", "Failed to start stream task");
      return;
    }

    server.sendHeader("Cache-Control", "no-cache");
    server.send(200, "text/plain", "Streaming started");
  });

  server.on("/save", HTTP_GET, []() {
    BaseType_t res = xSemaphoreGive(inferenceTrigger);
    
    String html;
    if (res == pdTRUE) {
      html = R"=====(
        <script>
          alert("Capture queued successfully!");
          window.location.href = "/";
        </script>
      )=====";
    } else {
      html = R"=====(
        <script>
          alert("Error: Capture already in progress. Please wait.");
          window.location.href = "/";
        </script>
      )=====";
    }
    
    server.send(200, "text/html", html);
  });

  server.onNotFound([]() { server.send(404, "text/plain", "Not Found"); });
  server.begin();
  Serial.println("HTTP server started – port 80");
}

static void streamTask(void *pvParameters) {
  WiFiClient *pClient = static_cast<WiFiClient*>(pvParameters);
  WiFiClient client = std::move(*pClient);
  delete pClient;               

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println(); 

  while (client.connected()) {
    // get new frame
    if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
      // Camera busy – skip this loop iteration.
      continue;
    }
    camera_fb_t *fb = esp_camera_fb_get();
    xSemaphoreGive(cameraMutex);

    if (!fb) {
      // Something went wrong – just continue to keep the connection alive.
      continue;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: ");
    client.println(fb->len);
    client.println(); // blank line before JPEG data
    client.write(fb->buf, fb->len);
    client.println(); // end of part

    esp_camera_fb_return(fb);

    vTaskDelay(pdMS_TO_TICKS(30)); // ~30fps
  }

  client.stop();
  Serial.println("[Stream] client disconnected – task ending");
  streamTaskHandle = nullptr;
  vTaskDelete(NULL);
}

static void inferenceTask(void *pvParameters) {
  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    // Wait for the next interval.
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(photoInterval));

    if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
      Serial.println("[Inference] could not lock camera – skip this interval");
      continue;
    }
    camera_fb_t *fb = esp_camera_fb_get();
    xSemaphoreGive(cameraMutex);
    if (!fb) {
      Serial.println("[Inference] camera capture failed");
      continue;
    }

    // POST request to C++ server
    HTTPClient http;
    http.setConnectTimeout(5000);
    http.setTimeout(5000);
    http.begin(POST_URL);
    http.addHeader("Content-Type", "image/jpeg");
    int rc = http.POST(fb->buf, fb->len);
    Serial.printf("[Inference] POST rc=%d, size=%u bytes\n", rc, fb->len);
    http.end();

    esp_camera_fb_return(fb);
  }
}

static void cameraCaptureTask(void *pvParameters) {
  for (;;) {
    if (xSemaphoreTake(inferenceTrigger, portMAX_DELAY) != pdTRUE) continue;

    if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
      Serial.println("CamTask: could not lock camera");
      continue;
    }

    camera_fb_t *fb = esp_camera_fb_get();

    xSemaphoreGive(cameraMutex);

    if (!fb) {
      Serial.println("CamTask: capture failed");
      continue;
    }

    if (xQueueSend(frameQueue, &fb, pdMS_TO_TICKS(1000)) != pdTRUE) {
      Serial.println("CamTask: frameQueue full -> dropping frame");
      esp_camera_fb_return(fb);
    } else {
      Serial.println("CamTask: frame queued for SD");
    }
  }
}

static void sdWriterTask(void *pvParameters) {
  for (;;) {
    camera_fb_t *fb = nullptr;
    if (xQueueReceive(frameQueue, &fb, portMAX_DELAY) != pdTRUE) continue;

    if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
      Serial.println("SDTask: could not lock SD card");
      esp_camera_fb_return(fb);
      continue;
    }

    int photo_index = readFileNum(SD_MMC, "/camera");
    if (photo_index == -1) {
      Serial.println("Save to sd card failed.");
      xSemaphoreGive(sdMutex);
      esp_camera_fb_return(fb);
      continue;
    }

    String path = "/camera/" + String(photo_index) + ".jpg";
    writejpg(SD_MMC, path.c_str(), fb->buf, fb->len);
    Serial.printf("SDTask: %u bytes written to %s\n", fb->len, path.c_str());

    xSemaphoreGive(sdMutex);
    esp_camera_fb_return(fb);
  }
}
