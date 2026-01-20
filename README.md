# Weapon Detection System

An integrated edge-to-cloud security solution utilizing an **ESP32-CAM** for real-time image acquisition and a **C++ Boost.Asio** server for high-performance **YOLOv5** inference. The system balances periodic monitoring with on-demand local storage, ensuring both automated detection and manual evidence logging.

## 🚀 Features

### Firmware (ESP32-CAM)

- **Multi-Tasking Architecture:** Leverages FreeRTOS to manage concurrent operations across both ESP32 cores.
- **Periodic Inference:** Automatically captures and POSTs JPEG frames to the detection server every 10 seconds.
- **On-Demand SD Storage:** Trigger-based image saving to the local MicroSD card via a dedicated HTTP endpoint (`/save`).
- **Concurrency Control:** Implements **Binary Semaphores** for task synchronization and **Mutexes** for thread-safe access to the Camera and SD-MMC peripherals.
- **Zero-Copy SD Writing:** Direct writing from the camera buffer to the SD card to minimize memory overhead.

### Server (C++ / OpenCV / Boost)

- **Asynchronous I/O:** Built with `Boost.Asio` to handle multiple concurrent HTTP POST requests efficiently.
- **YOLOv5 Inference:** Utilizes the OpenCV DNN module to run inference on a custom-trained `best.onnx` model.
- **Real-time Processing:** Decodes JPEG streams and identifies weapon classes (e.g., knives, handguns) with configurable confidence thresholds.
- **Robust Parsing:** Custom HTTP header parsing and error handling for reliable communication with edge devices.

---

## 📂 Project Structure

```text
.
├── firmware                 # ESP32-CAM Arduino Source
│   ├── firmware.ino         # Main logic (FreeRTOS tasks & WiFi)
├── server                   # C++ Backend Source
│   ├── src/main.cpp         # Boost.Asio server & OpenCV Inference
│   └── best.onnx            # Trained YOLOv5 model
└── models                   # Training scripts and datasets

```

---

## 🛠️ Installation & Setup

### 1. Firmware Configuration

1. Open `firmware/firmware.ino` in the Arduino IDE.
2. Install the **ESP32** board support package.
3. Create a `secrets.h` file in the firmware folder:

```cpp
#define MY_SSID "Your_WiFi_Name"
#define MY_PASS "Your_WiFi_Password"
#define POST_URL "http://your-server-ip:8080"

```

4. Select **AI Thinker ESP32-CAM** as the board and upload.

### 2. Server Build Requirements

Ensure you have the following dependencies installed:

- **Boost C++ Libraries** (>= 1.66)
- **OpenCV** (with DNN module)
- **CMake** (>= 3.10)

### 3. Compiling the Server

```bash
cd server
mkdir build
cd build
cmake ..
cmake --build .
./wds_program

```

---

## 🖥️ Usage

1. **Start the Server:** Run the compiled `wds_program`. It will listen on port `8080` by default.
2. **Power the ESP32-CAM:** Once connected to WiFi, it will begin sending images for inference.
3. **Manual Save:** Access the ESP32's IP address in your browser:

- `http://<esp-ip>/` — Status page.
- `http://<esp-ip>/save` — Triggers the camera to save a high-resolution frame to the SD card.

---

## 🧠 Technical Overview

### FreeRTOS Task Distribution

- **Core 0 (CamCap):** Dedicated to on-demand user requests. It waits for a semaphore from the WebServer and queues frames for the SD card.
- **Core 1 (SDWrite):** Handles high-latency I/O operations (writing to SD) without blocking the main loop or network POSTs.

### Inference Pipeline

The server receives a raw JPEG buffer, decodes it into a `cv::Mat`, and performs the following:

1. **Blob Conversion:** Normalizes pixels to `[0, 1]` and resizes to `640x640`.
2. **Forward Pass:** Runs the ONNX model via `net.forward()`.
3. **Post-Processing:** Scans detections for specific `weaponClassIds` and applies confidence filtering.
