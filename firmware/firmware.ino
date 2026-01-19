/*  ESP32‑CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "secrets.h"

/* ----------  Pin mapping (keep only what your board uses) ---------- */
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM    9
#define Y4_GPIO_NUM    8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13

const int photoInterval = 10000; // 10 seconds
unsigned long lastPhotoTime = 0;

static void initCamera();
static void connectWifi();
static void sendPhoto();

void setup() {
  initCamera();
  connectWifi();
}

void loop() {
  if (millis() - lastPhotoTime >= photoInterval) {
    lastPhotoTime = millis();
    sendPhoto();
  }
}

static void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.frame_size = FRAMESIZE_SXGA; 
      config.jpeg_quality = 10;
      config.fb_count = 1;
    } else {
      config.frame_size = FRAMESIZE_SVGA;
      config.jpeg_quality = 12;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)

#endif

}

static void connectWifi()
{
    // If we already have a connection – great.
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }

    // Make sure the driver is in STA mode and clear any old credentials.
    WiFi.disconnect(true);          // erase NVS stored credentials
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);           // keep the radio awake
    WiFi.begin(MY_SSID, MY_PASS);

    // Wait up to 20 s for a successful association.
    const unsigned long timeout = 20000;   // ms
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeout) {
        delay(500);
    }

}



static void sendPhoto () {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
    if (WiFi.status() != WL_CONNECTED) {
        return;
    }
  }

  camera_fb_t * camera_frame_buffer = NULL;
  camera_frame_buffer = esp_camera_fb_get();

  if (!camera_frame_buffer) {
    return;
  }
  HTTPClient http;
  http.begin(POST_URL);
  http.addHeader("Content-Type", "image/jpeg");
  // send post request
  int httpResponseCode = http.POST(camera_frame_buffer->buf, camera_frame_buffer->len);
  // receive server response

  // Release the frame buffer to avoid memory leaks
  esp_camera_fb_return(camera_frame_buffer);
  // Close connection
  http.end(); 
}
