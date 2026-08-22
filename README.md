# Real-Time Object Width Measurement with YOLOv8 Segmentation

A real-time computer vision system that detects an object, segments it, and measures its physical width in centimeters — running live on a Raspberry Pi 5.

Built during my internship. It combines YOLOv8 detection, YOLOv8 segmentation, ONNX Runtime, and OpenCV to do pixel-level width measurement on hardware that doesn't have a GPU to spare.

The system watches a defined Region of Interest (ROI), detects when an object enters it, segments that object into a pixel mask, extracts its maximum horizontal width from the mask, and converts that into a real-world measurement using a calibrated pixel-to-cm ratio.

---

## Why this was hard

Running YOLO on a Raspberry Pi isn't the hard part anymore — the hard part is making *segmentation-based* measurement fast enough to be usable in real time. Segmentation is expensive, and Pi-class hardware doesn't have much room to spare.

So the pipeline is split into two stages instead of running one expensive model on every frame:

```
Live Camera Feed
      │
      ▼
 ROI Extraction
      │
      ▼
YOLOv8 Detection ──► No object found → keep scanning
      │
      │ object found
      ▼
YOLOv8 Segmentation
      │
      ▼
Binary Segmentation Mask
      │
      ▼
Max Continuous Horizontal Width
      │
      ▼
Pixel → cm Calibration
      │
      ▼
Live Dashboard + Saved Result
```

Detection runs continuously since it's cheap. Segmentation only runs once something is actually in frame. That split is the main reason this stays usable on Pi hardware instead of grinding to a crawl.

---

## How the width is actually measured

A bounding box isn't a good proxy for physical width — for anything that isn't a perfect rectangle, the box includes a lot of background:

```
┌─────────────────────┐
│        ███          │
│      ███████        │
│    ███████████      │
│      ███████        │
└─────────────────────┘
```

So instead of using the box, the system works directly on the segmentation mask. For every row in the binary mask, it finds the longest continuous run of object pixels, then tracks the maximum run across all rows:

```
000011111111000
    <------>
    8 pixels
```

That maximum run is the measured width — it follows the actual silhouette of the object rather than a rectangle drawn around it.

**Pixel → cm conversion:**
```
PIXELS_PER_CM = measured_pixel_width / known_width_cm
```
For example, a 20cm object measuring 392px gives `PIXELS_PER_CM = 19.6`. This ratio is specific to the physical setup — camera, resolution, distance from lens to object, ROI size — so it has to be recalibrated whenever any of those change.

---

## Accuracy — preliminary results from the live Pi 5 deployment

I ran three trials against known-width objects, measured on the actual Raspberry Pi 5 deployment (not a desktop simulation):

| Object | Actual Width | Measured Width | Error |
|---|---|---|---|
| Object A | 6.5 cm | 6.38 cm | 1.8% |
| Object B | 7.5 cm | 7.55 cm | 0.7% |
| Object C | 6.5 cm | 6.33 cm | 2.6% |

Average error across these trials: **~1.7%**.

This is a small sample, so I'm treating it as an early signal, not a validated accuracy claim — more trials across different object types, distances, and lighting conditions are needed before I'd call this a proven number. It's on the list under Future Improvements below.

---

## Handling false positives

The models use standard COCO pretrained weights (80 classes). One real problem: a person's hand or arm often ends up inside the ROI while they're holding the object being measured. Since `person` is COCO class 0, the pipeline excludes it explicitly:

```python
classes = [c for c in range(80) if c != 0]
```

This stops the system from measuring someone's hand instead of the object in it.

---

## ONNX export

Running the raw PyTorch models directly on the Pi added inference overhead I didn't need. Both models are exported to ONNX and run through ONNX Runtime instead — 640×640 input, `simplify=True`, `dynamic=True`. The export script immediately runs a dummy inference pass against the exported model, so export errors get caught before the model ever reaches the Pi, not after.

```bash
python scripts/export_to_onnx.py
```

---

## Example run

```
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

---

## Project structure

```
WidthRaspberrypi5/
│
├── WidthCalculation.py      # main detection + segmentation + measurement loop
├── segmentRef.py            # segmentation reference/helper logic
├── test_detection.py        # detection-only test script
├── test_yolo.py             # YOLO model sanity checks
├── scripts/
│   └── export_to_onnx.py    # model export + ONNX validation
├── requirements.txt
└── README.md
```

Model weights (`.pt`, `.onnx`) are intentionally excluded from version control — they're regenerated locally via the export script rather than committed, to keep the repo lightweight.

---

## Running it

**Local development:**
```bash
git clone https://github.com/AarushiSharma1515/WidthRaspberrypi5.git
cd WidthRaspberrypi5
pip install -r requirements.txt
python WidthCalculation.py
```

**Raspberry Pi 5:**
```bash
sudo apt update && sudo apt upgrade -y
pip install ultralytics opencv-python numpy onnxruntime

# for the Pi camera module
sudo apt install -y python3-picamera2

git clone https://github.com/AarushiSharma1515/WidthRaspberrypi5.git
cd WidthRaspberrypi5
python WidthCalculation.py
```

**Deployment notes:**
```bash
hostname -I                          # find Pi's IP
libcamera-hello --list-cameras       # check connected cameras
vcgencmd measure_temp                # check thermals under load
nohup python3 WidthCalculation.py &  # keep running after SSH disconnects
```

---

## Calibration

`PIXELS_PER_CM` is specific to your camera + distance + ROI setup and needs to be set for wherever this is actually deployed:

1. Place an object of known width at the intended measurement distance.
2. Run the app and record the measured pixel width.
3. `PIXELS_PER_CM = measured_pixel_width / known_width_cm`
4. Update the constant in the script.

Keep the camera position and object distance consistent with whatever you calibrated against — this is the single biggest source of error if it drifts.

---

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `n` | Reset and measure a new object |

---

## Troubleshooting

| Problem | Try this |
|---|---|
| Camera won't open | Check the camera index in `cv2.VideoCapture()` |
| Nothing detected | Lower the YOLO confidence threshold |
| Measurements look off | Recalibrate `PIXELS_PER_CM` for your current setup |
| Low FPS | Use the nano model variants, shrink the ROI |
| High CPU load | Reduce input resolution or inference frequency |

---

## What was actually hard about this

**Real-time inference on the Pi.** Nowhere near the compute of a desktop GPU, so the whole design leans on lightweight models, ROI-restricted processing, ONNX instead of raw PyTorch, and skipping segmentation whenever nothing's in frame.

**Getting width right, not just detected.** Bounding boxes are a bad proxy for physical width on anything irregularly shaped — the segmentation-mask approach exists specifically to fix that.

**Keeping detections clean.** A live feed picks up more than the target object — mainly whoever's holding it. Explicit class filtering handles that.

**Actually deploying it.** Headless Linux, SSH, camera device handling, dependency management on ARM, thermal monitoring, long-running processes without a display attached — running it in a notebook and running it on a live Pi turned out to be two very different problems.

---

## What's next

- More accuracy trials — different objects, distances, lighting, and a real standard deviation instead of 3 data points
- Multi-frame averaging to smooth out per-frame noise
- Perspective correction for objects that aren't front-facing
- Automatic camera calibration instead of a manual constant
- Distance-aware pixel-to-cm conversion (right now it assumes fixed distance)
- Hardware acceleration where available on the Pi

---

## Tech stack

| Category | Tools |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Detection | YOLOv8 |
| Segmentation | YOLOv8-Seg |
| Optimization | ONNX / ONNX Runtime |
| Hardware | Raspberry Pi 5 |
| OS | Linux |
| Deployment | SSH / headless |
