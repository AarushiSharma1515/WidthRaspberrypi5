import cv2
import numpy as np
import time
import onnxruntime as ort
import os

# === Configuration ===
MODEL_PATH = "yolov8n.onnx"
NAMES_FILE = "coco.names"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_WIDTH, INPUT_HEIGHT = 640, 640

# === Load Class Names ===
if not os.path.exists(NAMES_FILE):
    raise FileNotFoundError(f"'{NAMES_FILE}' not found.")
with open(NAMES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load ONNX Runtime session ===
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# === Initialize Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_height, image_width = frame.shape[:2]

    # Preprocess
    img_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    blob = blob.astype(np.float32)
    # Run ONNX inference
    outputs = session.run(None, {input_name: blob})[0]
    outputs = outputs[0].T  # (84, 8400) → (8400, 84)

    boxes, confidences, class_ids = [], [], []
    for row in outputs:
        cx, cy, w, h = row[:4]
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESHOLD:
            left = int((cx - w / 2) * image_width / INPUT_WIDTH)
            top = int((cy - h / 2) * image_height / INPUT_HEIGHT)
            width = int(w * image_width / INPUT_WIDTH)
            height = int(h * image_height / INPUT_HEIGHT)

            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Detection - ONNXRuntime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
