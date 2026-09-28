# Facet capture device (ESP32-CAM)

A button press captures one photo and sends it to the Facet backend's
`/api/database-search/device-capture` endpoint, which searches it against
your configured database folder and logs the result. Watch the result on
the web app's **Live capture** page (`/live`) - the device itself has no
screen.

Deliberately manual-trigger only. See the note at the top of
`facet_capture.ino` and [docs/ETHICS.md](../../docs/ETHICS.md) before
changing that.

## Hardware

- An **AI-Thinker ESP32-CAM** board (the common ~$6-8 one with the OV2640
  camera). Other ESP32-CAM variants will need different camera pin numbers.
- A **USB-to-serial (FTDI) programmer** - the board has no USB port of its
  own.
- A **momentary push button**, plus two jumper wires.
- A proper **5V power supply** (500mA+) for normal operation. The FTDI
  programmer's own 3.3V/5V rail is usually too weak once WiFi and the camera
  are both active - browning out mid-capture is the most common "it doesn't
  work" symptom with this board.

### Wiring

| ESP32-CAM | FTDI programmer |
|---|---|
| 5V | 5V |
| GND | GND |
| U0R | TX |
| U0T | RX |

| Button |  |
|---|---|
| One leg | GPIO13 |
| Other leg | GND |

To **flash**: connect GPIO0 to GND, then power up or press reset - this
puts the board in flashing mode. After a successful upload, disconnect
GPIO0 from GND and reset again to run normally.

## Arduino IDE setup

1. File → Preferences → Additional Board Manager URLs, add:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
2. Tools → Board → Boards Manager → install "esp32" (Espressif Systems).
3. Tools → Board → select **AI Thinker ESP32-CAM**.
4. Tools → Partition Scheme → **Huge APP (3MB No OTA/1MB SPIFFS)** - the
   camera + WiFi + HTTP client libraries need more than the default app
   partition.
5. Tools → Port → select the FTDI programmer's serial port.

## Configure and flash

```bash
cd firmware/esp32-cam
cp secrets.h.example secrets.h
# edit secrets.h: WiFi credentials, SERVER_URL, DEVICE_ID
```

`SERVER_URL` must point at the backend's **LAN IP**, not `localhost` -
`localhost` on the ESP32 means the ESP32 itself. Find your server's LAN IP:

```bash
ipconfig getifaddr en0   # Wi-Fi, on macOS
```

Then start the backend so it actually accepts connections from other
devices on the network (by default it only listens on `127.0.0.1`, i.e.
only itself):

```bash
cd backend && source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Make sure `ENABLE_DATABASE_SEARCH=true` and `DATABASE_SEARCH_DIR` are set
(see the main [README](../../README.md#database-search-one-to-many-off-by-default)),
and that the ESP32-CAM and your computer are on the **same WiFi network**.
If your Mac's firewall blocks incoming connections, allow them for the
`uvicorn`/Python process, or for port 8000.

Open `facet_capture.ino` in the Arduino IDE and upload it (GPIO0 to GND
while flashing, as above). Open the Serial Monitor at 115200 baud to watch
connection and capture logs.

## Using it

1. Open the Facet web app's **Live capture** page (`/live`) on your
   computer or another device on the same network.
2. Press the button on the ESP32-CAM.
3. Within a few seconds, the result appears on `/live`: a thumbnail of the
   captured face, the closest match from your database folder (if any),
   and its similarity score.

The original captured photo is never stored anywhere - only a small
rendered thumbnail and the search result are kept, and only in memory
(cleared on backend restart). If several faces are in frame, the device
automatically uses the most prominent one - there is no way to disambiguate
from the device itself.

## Security note

If this device will be reachable by anyone else on your network, set
`API_KEY` in `backend/.env` and put the same value in `secrets.h` - without
it, anyone on the LAN can call the capture endpoint.
