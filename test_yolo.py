from ultralytics import YOLO

# Load the YOLOv8n PyTorch model
model = YOLO('yolov8n.pt')

# Export to ONNX format (with simplified graph for OpenCV compatibility)
model.export(format='onnx', simplify=True)
from ultralytics import YOLO

# Load the pretrained PyTorch segmentation model
model = YOLO("yolov8n-seg.pt")

# Export the model to ONNX format with opset 12
success = model.export(format="onnx", opset=12)

# Verify the export
if success:
    print(f"Export successful: {success}")
else:
    print("Export failed.")