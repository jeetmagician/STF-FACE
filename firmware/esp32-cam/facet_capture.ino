/*
 * Facet capture device - AI-Thinker ESP32-CAM
 *
 * On a button press, captures one JPEG frame and POSTs it to
 * POST /api/database-search/device-capture on the Facet backend. The
 * backend runs the search and appends the result to its in-memory capture
 * log; the Facet web app's /live page polls that log and shows the result
 * there - this device has no display of its own.
 *
 * Deliberately manual-trigger only. Wiring this to a PIR motion sensor or a
 * timer instead of a physical button turns an on-demand lookup into
 * unattended, continuous capture of whoever happens to walk past - see
 * docs/ETHICS.md in the main repo for why that is out of scope for this
 * tool as shipped.
 *
 * Setup: copy secrets.h.example to secrets.h and fill in your WiFi
 * credentials and the backend's LAN address. See README.md in this folder
 * for wiring, board settings and flashing instructions.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "secrets.h"

// Button: one leg to this pin, the other to GND. GPIO13 has no
// boot-strapping role on the AI-Thinker board, unlike GPIO0/2/12/15.
#define BUTTON_PIN 13

// The AI-Thinker board's onboard white LED, used here only as a status
// flash - one blink for "capturing", two for "sent OK", five for "failed".
#define FLASH_LED_PIN 4

// Standard AI-Thinker ESP32-CAM camera pin map.
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

void blinkFlash(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(onMs);
    digitalWrite(FLASH_LED_PIN, LOW);
    if (i < times - 1) delay(offMs);
  }
}

bool setupCamera() {
  camera_config_t config = {};
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
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    // SVGA (800x600) gives the detector a real face to work with without
    // producing an unnecessarily large upload over WiFi.
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  return esp_camera_init(&config) == ESP_OK;
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP: ");
  Serial.println(WiFi.localIP());
}

bool captureAndSend() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    return false;
  }

  const String boundary = "FacetCaptureBoundary7331";
  String head = "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"device_id\"\r\n\r\n" +
                String(DEVICE_ID) + "\r\n"
                "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  size_t totalLen = head.length() + fb->len + tail.length();
  uint8_t *body = (uint8_t *)malloc(totalLen);
  if (!body) {
    Serial.println("Out of memory building the upload body");
    esp_camera_fb_return(fb);
    return false;
  }
  memcpy(body, head.c_str(), head.length());
  memcpy(body + head.length(), fb->buf, fb->len);
  memcpy(body + head.length() + fb->len, tail.c_str(), tail.length());

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  if (strlen(API_KEY) > 0) {
    http.addHeader("X-API-Key", API_KEY);
  }

  int status = http.POST(body, totalLen);
  String response = http.getString();

  free(body);
  esp_camera_fb_return(fb);
  http.end();

  Serial.printf("HTTP %d\n", status);
  Serial.println(response);
  return status == 200;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!setupCamera()) {
    Serial.println("Camera init failed - check board selection in Tools > Board");
  }

  connectWiFi();
  Serial.println("Ready. Press the button to capture and search.");
}

void loop() {
  static bool lastPressed = false;
  bool pressed = digitalRead(BUTTON_PIN) == LOW;

  if (pressed && !lastPressed) {
    delay(30); // debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      Serial.println("Button pressed - capturing");
      blinkFlash(1, 60, 0);

      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi dropped, reconnecting...");
        connectWiFi();
      }

      bool ok = captureAndSend();
      blinkFlash(ok ? 2 : 5, 100, 100);
    }
  }
  lastPressed = pressed;
  delay(20);
}
