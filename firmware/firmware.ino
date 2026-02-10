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
unsigned long lastPhotoTime = 0;

/* --------------------------  FreeRTOS objects  ------------------- */
typedef struct {
  camera_fb_t *camera_frame_buffer;          // pointer to the camera frame buffer
  uint32_t     timestamp;                  // when it was captured (ms since boot)
} frame_item_t;

static QueueHandle_t   frameQueue   = nullptr; // frames for SD writer
static SemaphoreHandle_t sdMutex     = nullptr; // protect SD card
static SemaphoreHandle_t cameraMutex = nullptr; // exclusive camera access
static SemaphoreHandle_t captureSem  = nullptr; // user-requested photo

static WebServer server(80);

/* forward declarations */
static void initCamera();
static void connectWifi();
static void sendPhotoToServer();
static void initWebServer();
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

  frameQueue = xQueueCreate(5, sizeof(frame_item_t));
  sdMutex = xSemaphoreCreateMutex();
  cameraMutex = xSemaphoreCreateMutex();
  captureSem = xSemaphoreCreateBinary();

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
}

/* ----------------------------------------------------------------------- */
void loop() {
  if (millis() - lastPhotoTime >= photoInterval) {
    lastPhotoTime = millis();
    sendPhotoToServer();
  }
  server.handleClient();
}

/* ----------------------------------------------------------------------- */
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


#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

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

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif
}

/* ----------------------------------------------------------------------- */
static void connectWifi()
{
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }

    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(MY_SSID, MY_PASS);

    const unsigned long timeout = 20000;
    unsigned long start = millis();
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

/* ----------------------------------------------------------------------- */
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
        <a href="/save">Save to SD</a>
        </p>
      </body>
    </html>
)=====";
    server.send(200, "text/html", html);
  });

  server.on("/snapshot", HTTP_GET, []() {
    if (xSemaphoreTake(cameraMutex, 0) != pdTRUE) {
        Serial.println("Camera busy - lock denied.");
        server.send(503, "text/plain", "Camera Busy");
        return;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      server.send(500, "text/plain", "Capture failed");
      xSemaphoreGive(cameraMutex);
      return;
    }

    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Length", String(fb->len));
    server.client().write(fb->buf, fb->len);
    esp_camera_fb_return(fb);
    xSemaphoreGive(cameraMutex);
  });

  server.on("/stream", HTTP_GET, []() {
    WiFiClient client = server.client();
    String header = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    client.print(header);
    while (client.connected()) {
        if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
            break;
        }
        camera_fb_t *fb = esp_camera_fb_get();
        xSemaphoreGive(cameraMutex);
        if (!fb) {
            break;
        }
        client.print("--frame\r\n");
        client.print("Content-Type: image/jpeg\r\n");
        client.print("Content-Length: ");
        client.print(fb->len);
        client.print("\r\n\r\n");
        client.write(fb->buf, fb->len);
        client.print("\r\n");
        esp_camera_fb_return(fb);
        delay(100); 
    }
  });

  server.on("/save", HTTP_GET, []() {
    if (xSemaphoreGive(captureSem) == pdTRUE) {
        server.send(200, "application/json", "{\"status\":\"capture_queued\"}");
    } else {
        server.send(429, "application/json", "{\"error\":\"capture_already_in_progress\"}");
    }
  });

  server.onNotFound([]() {
    server.send(404, "text/plain", "Not Found");
  });

  server.begin();
  Serial.println("HTTP server started – listen on port 80");
}

/* ----------------------------------------------------------------------- */
static void sendPhotoToServer () {

  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
    if (WiFi.status() != WL_CONNECTED) {
      return;
    }
  }

  if (xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
    Serial.println("sendPhotoToServer: could not lock camera");
    return;
  }

  camera_fb_t *camera_frame_buffer = esp_camera_fb_get();

  if (!camera_frame_buffer) {
    Serial.println("sendPhotoToServer: camera capture failed");
    xSemaphoreGive(cameraMutex);
    return;
  }

  HTTPClient http;
  http.begin(POST_URL);
  http.addHeader("Content-Type", "image/jpeg");
  int httpResponseCode = http.POST(camera_frame_buffer->buf, camera_frame_buffer->len);
  Serial.printf("POST code: %d, size: %u bytes\n", httpResponseCode, camera_frame_buffer->len);
  http.end();

  esp_camera_fb_return(camera_frame_buffer);
  xSemaphoreGive(cameraMutex);
}

/* ----------------------------------------------------------------------- */
static void cameraCaptureTask(void *pvParameters) {
  for (;;) {
    if (xSemaphoreTake(captureSem, portMAX_DELAY) != pdTRUE) continue;

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

    frame_item_t item;
    item.camera_frame_buffer = fb;
    item.timestamp = millis();

    if (xQueueSend(frameQueue, &item, pdMS_TO_TICKS(1000)) != pdTRUE) {
      Serial.println("CamTask: frameQueue full -> dropping frame");
      esp_camera_fb_return(fb);
    } else {
      Serial.println("CamTask: frame queued for SD");
    }
  }
}

/* ----------------------------------------------------------------------- */
static void sdWriterTask(void *pvParameters) {
  for (;;) {
    frame_item_t item;
    if (xQueueReceive(frameQueue, &item, portMAX_DELAY) != pdTRUE) continue;

    if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
      Serial.println("SDTask: could not lock SD card");
      esp_camera_fb_return(item.camera_frame_buffer);
      continue;
    }

    int photo_index = readFileNum(SD_MMC, "/camera");
    if (photo_index == -1) {
      Serial.println("Save to sd card failed.");
      xSemaphoreGive(sdMutex);
      esp_camera_fb_return(item.camera_frame_buffer);
      continue;
    }

    String path = "/camera/" + String(photo_index) + ".jpg";
    writejpg(SD_MMC, path.c_str(), item.camera_frame_buffer->buf, item.camera_frame_buffer->len);

    Serial.printf("SDTask: %u bytes written to %s\n", item.camera_frame_buffer->len, path);
    xSemaphoreGive(sdMutex);
    esp_camera_fb_return(item.camera_frame_buffer);
  }
}
