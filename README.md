# 📏 Real-Time Object Width Measurement with YOLOv8 Segmentation

> 🚀 **Real-time computer vision system for detecting, segmenting, and measuring object width in centimeters : deployed on a Raspberry Pi 5 with a live camera feed.**

Built during my internship, this project combines **YOLOv8 object detection, YOLOv8 segmentation, ONNX Runtime, OpenCV, and Raspberry Pi 5** to perform pixel-level object measurement on resource-constrained hardware.

The system detects an object inside a predefined **Region of Interest (ROI)**, generates a pixel-precise segmentation mask, extracts its maximum continuous horizontal width, and converts that measurement into centimeters using a calibrated camera setup.

---

## 🔍 Overview

The core challenge was not simply running YOLO on a Raspberry Pi — it was making **segmentation-based measurement fast enough for real-time use**.

Instead of running the computationally expensive segmentation model on every frame, the system uses a **two-stage inference pipeline**:

```text
Live Camera Feed
       │
       ▼
  ROI Extraction
       │
       ▼
YOLOv8 Detection ──────► No object → Continue scanning
       │
       │ Object found
       ▼
YOLOv8 Segmentation
       │
       ▼
Binary Segmentation Mask
       │
       ▼
Maximum Continuous
Horizontal Width
       │
       ▼
Pixel → Centimeter
Calibration
       │
       ▼
Real-World Width
       │
       ▼
Live Dashboard + Saved Result
```

This architecture reduces unnecessary segmentation inference while retaining pixel-level measurement accuracy.

---

## ✨ Key Features

* 🎯 Real-time object detection using **YOLOv8**
* 🧩 Pixel-level object segmentation using **YOLOv8 Segmentation**
* 🔄 Two-stage **detect → segment** inference pipeline
* 🎥 ROI-based camera processing
* 📏 Maximum continuous horizontal width extraction from segmentation masks
* 📐 Pixel-to-centimeter calibration
* ⚡ ONNX model export and validation
* 🖥️ Raspberry Pi 5 deployment
* 📊 Live camera dashboard
* 🔧 Linux-based headless deployment over SSH
* 🛡️ COCO class filtering to avoid detecting people as measurement targets

---

## 🧠 How It Works

### Stage 1 — Detection

A lightweight YOLOv8 detection model continuously scans the ROI.

Its job is simply to answer:

> **"Is there an object here that should be measured?"**

Detection is significantly cheaper than segmentation, allowing it to run continuously without wasting compute on empty frames.

### Stage 2 — Segmentation

Once an object is detected, the segmentation model runs on that frame and generates a **pixel-level mask**.

The mask is then analyzed to calculate the object's actual horizontal width.

```text
Detection
   │
   ├── No object ──► Keep scanning
   │
   └── Object found
           │
           ▼
      Segmentation
           │
           ▼
       Measurement
```

---

## 🔄 Why Two Models?

Running segmentation on every frame would be unnecessarily expensive on Raspberry Pi-class hardware.

The pipeline therefore separates **detection** from **measurement**:

```text
┌──────────────────────┐
│  YOLOv8 Detection    │
│  Lightweight / Fast  │
└──────────┬───────────┘
           │
           │ Object found
           ▼
┌──────────────────────┐
│ YOLOv8 Segmentation  │
│ Pixel-level mask     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Width Measurement    │
│ Pixel → Centimeter   │
└──────────────────────┘
```

This avoids repeatedly running the expensive segmentation model while maintaining accurate measurements.

---

## 📏 Width Measurement Algorithm

A bounding-box width is not always an accurate representation of an object's physical width.

For irregularly shaped objects, the bounding box can include significant background:

```text
Bounding Box
┌─────────────────────┐
│        ███          │
│      ███████        │
│    ███████████      │
│      ███████        │
└─────────────────────┘
```

Instead, this project analyzes the **segmentation mask itself**.

For every row of the binary mask:

1. Find continuous runs of object pixels.
2. Measure the length of each run.
3. Keep the longest run for that row.
4. Track the maximum run across all rows.

For example:

```text
000011111111000

    <------>

    8 pixels
```

The largest continuous horizontal run across the entire mask becomes the measured object width.

This allows the measurement to follow the object's actual segmented silhouette rather than simply using the bounding-box width.

---

## 🎯 Pixel-to-Centimeter Calibration

The measured width initially exists in pixels.

To convert it into a real-world measurement:

```text
PIXELS_PER_CM = pixel_width / known_width_cm

object_width_cm = measured_pixel_width / PIXELS_PER_CM
```

For example:

```text
Known object width = 20 cm
Measured pixel width = 392 px

PIXELS_PER_CM = 392 / 20
              = 19.6 px/cm
```

The calibration factor depends on:

* Camera
* Camera resolution
* Camera-to-object distance
* Lens characteristics
* ROI configuration

Therefore, calibration should be performed for the physical setup in which the system will operate.

---

## 🛡️ Handling False Detections

The models use standard **COCO pretrained weights**, which contain 80 object classes.

One practical issue is that a person can appear in the ROI while holding the target object.

Since `person` is COCO class `0`, the pipeline explicitly excludes it from both detection and segmentation:

```python
classes = [c for c in range(80) if c != 0]
```

This prevents the system from incorrectly treating a person's hand, arm, or body as the measurement target.

---

## ⚡ ONNX Optimization

Running the original PyTorch models directly on the Raspberry Pi introduced unnecessary inference overhead.

The models were therefore exported to **ONNX** and executed using **ONNX Runtime**.

The export process uses:

* 640 × 640 input resolution
* `simplify=True`
* `dynamic=True`

The exported model is immediately loaded into ONNX Runtime and tested with a dummy inference pass.

This provides two benefits:

1. ⚡ Reduced inference overhead compared with the original PyTorch pipeline.
2. ✅ Export errors are detected before deployment to the Raspberry Pi.

The complete export and validation workflow is available in:

```bash
python scripts/export_to_onnx.py
```

---

## 📊 Example Measurement

A successful measurement produces output similar to:

```text
Initializing...
Camera default resolution: 1280 x 720
Using ROI size: 300x300

--- Searching for an object... Timer started. ---

Object found: cup.
Running segmentation automatically...

Maximum Continuous Horizontal Width: 392 pixels
Calculated width: 20.00 cm

Object detected AND segmented!
Total time for measurement: 0.87 seconds.

[INFO] Measurement saved as 'width_measurement.png'
```

The resulting dashboard provides:

```text
┌──────────────────────────────┬─────────────────────┐
│                              │ Detection Status    │
│       LIVE CAMERA FEED       │                     │
│            + ROI             │ Segmentation        │
│                              │                     │
│                              │ Width: 20.00 cm     │
│                              │                     │
└──────────────────────────────┴─────────────────────┘
```

---

## 📁 Project Structure

```text
WidthRaspberrypi5/
│
├── src/
│   └── WidthCalculation.py
│       └── Main detection, segmentation,
│           measurement, and camera loop
│
├── scripts/
│   └── export_to_onnx.py
│       └── Model export + ONNX validation
│
├── requirements.txt
└── README.md
```

Model weights (`.pt`, `.onnx`) are intentionally excluded from version control to keep the repository lightweight.

---

## 🛠️ Installation

### Option 1 — Local Development

```bash
git clone https://github.com/AarushiSharma1515/WidthRaspberrypi5.git

cd WidthRaspberrypi5

pip install -r requirements.txt

python src/WidthCalculation.py
```

### Option 2 — Raspberry Pi 5

Update the system:

```bash
sudo apt update
sudo apt upgrade -y
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy onnxruntime
```

For Raspberry Pi Camera Module:

```bash
sudo apt install -y python3-picamera2
```

Clone the repository:

```bash
git clone https://github.com/AarushiSharma1515/WidthRaspberrypi5.git

cd WidthRaspberrypi5

python src/WidthCalculation.py
```

---

## 🖥️ Raspberry Pi Deployment

The system was deployed on a **Raspberry Pi 5 running Linux** and operated alongside a live camera feed.

Useful commands during deployment:

### Find Raspberry Pi IP

```bash
hostname -I
```

### Check connected cameras

```bash
libcamera-hello --list-cameras
ls /dev/video*
```

### Monitor CPU and temperature

```bash
top
vcgencmd measure_temp
```

### Keep the application running after disconnecting SSH

```bash
nohup python3 src/WidthCalculation.py &
```

---

## 🎯 Calibration

The `PIXELS_PER_CM` parameter must be calibrated for the intended camera setup.

### Calibration procedure

1. Place an object with a known physical width at the measurement distance.
2. Run the application.
3. Record the detected pixel width.
4. Calculate:

```text
PIXELS_PER_CM =
    measured_pixel_width / known_width_cm
```

5. Update the calibration constant in the application.

For accurate measurements, the camera position and object distance should remain consistent with the calibration setup.

---

## 🎮 Controls

| Key | Action                                        |
| --- | --------------------------------------------- |
| `q` | Quit application                              |
| `n` | Reset detection mode and measure a new object |

---

## 🐛 Troubleshooting

| Problem                     | Possible Solution                                         |
| --------------------------- | --------------------------------------------------------- |
| Camera does not open        | Check the camera index used by `cv2.VideoCapture()`       |
| No object detected          | Lower the YOLO confidence threshold                       |
| Measurements are inaccurate | Recalibrate `PIXELS_PER_CM`                               |
| Low FPS                     | Use YOLO nano models and/or reduce ROI size               |
| Camera not detected         | Check `/dev/video*` and Pi camera configuration           |
| High CPU usage              | Reduce input resolution, ROI size, or inference frequency |

---

## 🧩 Engineering Challenges

### 1. Real-Time Inference on Edge Hardware

The Raspberry Pi 5 has considerably fewer computational resources than a desktop GPU.

The solution was to:

* Use lightweight YOLO models
* Restrict processing to an ROI
* Export models to ONNX
* Avoid unnecessary segmentation inference
* Monitor CPU and temperature during deployment

### 2. Accurate Width Extraction

Bounding boxes provide coarse object dimensions but can include significant background for irregular shapes.

Using the segmentation mask allows the measurement to follow the object's actual visible silhouette.

### 3. Detection Reliability

A live camera feed can contain irrelevant objects, especially people holding the target object.

Explicit class filtering prevents the `person` class from becoming a measurement target.

### 4. Deployment Reliability

The system was not only developed locally but deployed to a headless Raspberry Pi environment.

This required handling:

* Linux environments
* Camera interfaces
* SSH-based deployment
* Python dependencies
* ONNX Runtime
* CPU/thermal monitoring
* Long-running processes

---

## 📈 Performance

The optimized pipeline was designed around the constraints of Raspberry Pi 5 hardware.

A representative measurement run:

```text
Detection + Segmentation + Measurement
Total measurement time: ~0.87 seconds
```

The key optimization was reducing how often the segmentation model runs rather than attempting to make segmentation run continuously.

For further benchmarking, the pipeline can track:

```text
Detection FPS
Segmentation latency
End-to-end measurement latency
CPU utilization
Temperature
ROI size
Input resolution
ONNX vs PyTorch inference time
```

---

## 💡 What I Learned

### Computer Vision

* Object detection vs. segmentation trade-offs
* Binary mask processing
* Pixel-level geometric measurements
* ROI-based vision pipelines
* Camera calibration

### Edge AI

* Deploying ML models on Raspberry Pi 5
* ONNX model conversion and validation
* Inference optimization under CPU constraints
* Monitoring resource usage during real-time inference

### Systems Engineering

* Linux-based deployment
* SSH and headless systems
* Camera device management
* Long-running processes
* Debugging hardware/software integration issues

---

## 🔮 Future Improvements

Potential improvements include:

* 📊 Multi-frame measurement averaging for greater stability
* 📐 Perspective correction for non-front-facing objects
* 🎯 Automatic camera calibration
* 📏 Distance-aware pixel-to-centimeter conversion
* ⚡ Hardware acceleration where available
* 🔄 Continuous measurement mode
* 🧭 More robust object tracking between frames
* 🧪 Automated performance benchmarking
* 📦 Support for additional measurement axes

---

## 🧰 Tech Stack

| Category             | Technology                |
| -------------------- | ------------------------- |
| Language             | Python                    |
| Computer Vision      | OpenCV                    |
| Detection            | YOLOv8                    |
| Segmentation         | YOLOv8-Seg                |
| Model Optimization   | ONNX                      |
| Runtime              | ONNX Runtime              |
| Numerical Processing | NumPy                     |
| Hardware             | Raspberry Pi 5            |
| OS                   | Linux                     |
| Camera               | USB / Raspberry Pi Camera |
| Deployment           | SSH / Headless Linux      |

---

## 🔗 Repository

**GitHub:**
https://github.com/AarushiSharma1515/WidthRaspberrypi5

---

## 🚀 Summary

This project demonstrates an end-to-end **edge computer vision measurement system**, from live camera capture and object detection to segmentation, geometric analysis, calibration, optimization, and real-world deployment.

The main engineering approach was to combine:

**Lightweight Detection → Selective Segmentation → Mask-Based Measurement → Calibration → ONNX Inference**

to make pixel-level object measurement practical on **Raspberry Pi-class hardware**.
