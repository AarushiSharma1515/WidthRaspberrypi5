Object Width Measurement using YOLOv8 Segmentation
This project uses YOLOv8 (Ultralytics) object detection and segmentation to detect an object in real-time via webcam, segment it, and calculate its horizontal width both in pixels and centimeters (after calibration).
It features:
- ONNX-based YOLO models for faster inference.
- Real-time dashboard with detection status, FPS counter, and segmented view.
- Automatic object measurement upon detection.
- ROI (Region of Interest)–based detection for better performance.
- Pixel-to-centimeter conversion (with calibration).
How It Works
1. Detection Mode – Continuously searches for the largest object in the ROI using a YOLOv8 detection model.
2. Segmentation Mode – Once an object is found, the YOLOv8 segmentation model is run on it and finds the longest continuous horizontal run of non-zero pixels in the segmented mask.
3. Measurement – Calculates object width in pixels and converts to centimeters using a PIXELS_PER_CM calibration factor.
4. Dashboard View – Left: Live camera feed with ROI box. Top-right: Detection status. Bottom-right: Segmentation result and measured width.
Project Structure
.

├── main.py                # Main script with detection, segmentation, and measurement logic

├── yolov8n.onnx           # YOLOv8 detection model (ONNX format)

├── yolov8n-seg.onnx       # YOLOv8 segmentation model (ONNX format)

├── yolov8n-seg.pt         # Original PyTorch segmentation model

└── width_measurement.png  # Saved measurement image after successful detection
Installation
1️⃣ Clone the repository:
git clone https://github.com/yourusername/object-width-measurement.git
cd object-width-measurement
2️⃣ Install dependencies:
pip install ultralytics opencv-python numpy
3️⃣ Export YOLO models to ONNX (if not already exported):
model = YOLO("yolov8n-seg.pt")
model.export(format="onnx", opset=12, simplify=True)
Calibration
The variable PIXELS_PER_CM determines the pixel-to-centimeter conversion. You must calibrate it for accurate results.

Steps:
1. Place a known-sized object (e.g., a ruler) at the fixed distance your camera will measure from.
2. Run the script.
3. Measure the object width in pixels (printed in console).
4. Compute: PIXELS_PER_CM = pixel_width / known_object_width_in_cm
5. Update the constant in the script.
Usage
Run the script:
python main.py
Controls:
- q → Quit the application.
- n → Reset to detection mode for measuring a new object.
Example Output
Console Output Example:
Initializing...
Camera default resolution: 1280 x 720
Using ROI size: 300x300

--- Searching for an object... Timer started. ---
Object found: cup. Running segmentation automatically...
Maximum Continuous Horizontal Width: 392 pixels
Calculated width: 20.00 cm
Object detected AND segmented! Total time for measurement: 0.87 seconds.
[INFO] Measurement saved as 'width_measurement.png'
Troubleshooting
- Cannot open camera → Check your webcam index in cv2.VideoCapture(0).
- Inaccurate measurements → Re-calibrate PIXELS_PER_CM for your setup.
- No object detected → Lower the YOLO detection conf threshold.
- Slow FPS → Use yolov8n models or resize ROI.

