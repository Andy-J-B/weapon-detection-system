#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>


// ===========================
// Select camera model in board_config.h
// ===========================
#include "board_config.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "Kevin 5G";
const char *password = "Cookie1234";

const char *POST_URL = "";

// Set interval for photo taking in ms
const int photoInterval = 10000; // 10 seconds
unsigned long lastPhotoTime = 0;

// function declarations
void setupLedFlash();
void connectWifi();
void sendPhoto();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

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
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    // if PSRAM IC present, init with the best quality/resolution for processing.
    if (psramFound()) {
      // HIGH QUALITY, HIGH RESOLUTION:
      // SXGA (1280x1024) is a great balance for detailed object detection.
      config.frame_size = FRAMESIZE_SXGA; 
      config.jpeg_quality = 10;   // High quality (lower number = better quality, larger file)
      config.fb_count = 1;        // 1 frame buffer is sufficient for synchronous snapshot-send cycle
      
      // CRITICAL: Remove/Avoid setting grab_mode to CAMERA_GRAB_LATEST.
      // The default grab_mode (CAMERA_GRAB_WHEN_EMPTY) is better for single, synchronous snapshots.
      
    } else {
      // Limit the frame size when PSRAM is not available (DRAM only)
      // SVGA (800x600) is typically the highest safe resolution without PSRAM
      config.frame_size = FRAMESIZE_SVGA;
      config.jpeg_quality = 12;           // Drop quality slightly to reduce size further
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  
  } else {
    // Best option for face detection/recognition
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


#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  connectWifi();
}

void loop() {
  // Do nothing. Everything is done in another task by the web server
  delay(10000);
}

void connectWifi () {
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");



  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void sendPhoto () {
  if (Wifi.status() != WL_CONNECTED) {
    Serial.println("Wifi is disconnected, will try to reconnect.");
    connectWifi();
  }
  // Capture the image
  Serial.println("Capturing the image ...");

  camera_fb_t * camera_frame_buffer = NULL;
  camera_frame_buffer = esp_camera_fb_get();

  if (!camera_frame_buffer) {
    Serial.println("Capuring image failed!");
    Serial.println("No frame buffer created, exiting program...");
    return;
  }

  Serial.printf("Photo captured. Size: %u bytes\n", camera_frame_buffer->len);

  HttpClient http;

  // construct http post
  http.begin(POST);
  http.addHeader("Content-Type", "image/jpeg");

  Serial.print("Sending the POST request to :");
  Serial.println(POST_URL);

  // send post request
  int httpResponseCode = http.POST(camera_frame_buffer->buf, camera_frame_buffer->len);
  // receive server response
  if (httpResponseCode > 0) {
    Serial.printf("HTTP POST Success! Response code: %d\n", httpResponseCode);
    String payload = http.getString();
    Serial.print("Server Response: ");
    Serial.println(payload);
  } else {
    Serial.printf("HTTP POST Failed! Error code: %d\n", httpResponseCode);
  }

  // Release the frame buffer to avoid memory leaks
  esp_camera_fb_return(fb);

  // Close connection
  http.end();
  
}
