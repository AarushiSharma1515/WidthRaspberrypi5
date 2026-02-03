import logging
import cv2
import numpy as np
import time
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")

# Export the model to ONNX format
success = model.export(format="onnx", opset=12, simplify=True) 
print(f"Export successful: {success}")
# --- Constants and Global State ---
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Define application states (simplified)
STATE_DETECTING = 0
STATE_SHOWING_RESULT = 1

# --- CALIBRATION CONSTANT ---
# You NEED to calibrate this value.
PIXELS_PER_CM = 19.6 # this is specifically if the object is kept at 20.5cm
if PIXELS_PER_CM <= 0:
    print("WARNING: PIXELS_PER_CM is not calibrated. Width in CM will be inaccurate.")

# --- Helper Functions ---

def initialize():
    """Initializes BOTH YOLO models and the camera."""
    print("Initializing...")
    # Load the DETECTION model as ONNX
    yolo_detect = YOLO("yolov8n.onnx", verbose=False) # Changed to ONNX
    # Keep the SEGMENTATION model as PT unless you also export it to ONNX
    yolo_seg = YOLO("yolov8n-seg.onnx", verbose=False) 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      print("Cannot open camera")
      return None, None, None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera default resolution: {int(width)} x {int(height)}")

    return yolo_detect, yolo_seg, cap

def find_longest_continuous_run(row):
    """
    Finds the start, end, and length of the longest continuous run of non-zero pixels in a 1D array.
    """
    max_len = 0
    max_start = -1
    max_end = -1
    current_len = 0
    current_start = -1

    for i, pixel in enumerate(row):
        if pixel > 0:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i - 1
            current_len = 0

    if current_len > max_len:
        max_len = current_len
        max_start = current_start
        max_end = len(row) - 1

    return max_len, max_start, max_end

def calculate_width_from_preview(yolo_seg_model, preview_img):
    """
    Runs segmentation and calculates the longest continuous horizontal width of the object.
    Returns pixel width and segmented image.
    """
    seg_res = yolo_seg_model(preview_img, conf=0.4)[0]
    if not seg_res.masks:
        return None, None

    segmented_image = np.zeros_like(preview_img)
    # Ensure best_idx is safe if there are no boxes detected by seg_res.boxes
    if len(seg_res.boxes) == 0:
        return None, None
        
    best_idx = np.argmax(seg_res.boxes.conf.cpu().numpy()) # Ensure numpy conversion for argmax
    contour = seg_res.masks.xy[best_idx].astype(np.int32)
    cv2.drawContours(segmented_image, [contour], -1, (0, 255, 100), cv2.FILLED)

    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Handle cases where contour might be empty or invalid after conversion
    if contour.shape[0] < 2: # Need at least 2 points for min/max calculation
        return None, None

    topmost_y = contour[:, 1].min()
    bottommost_y = contour[:, 1].max()

    overall_max_width = 0
    best_y, best_x_start, best_x_end = 0, 0, 0

    # Ensure y range is valid
    if topmost_y > bottommost_y: # Should not happen if contour is valid, but good to check
        return None, None

    for y in range(topmost_y, bottommost_y + 1):
        # Ensure y is within binary image bounds
        if y < 0 or y >= binary.shape[0]:
            continue
        row_width, row_start, row_end = find_longest_continuous_run(binary[y, :])
        if row_width > overall_max_width:
            overall_max_width = row_width
            best_y, best_x_start, best_x_end = y, row_start, row_end
    
    if overall_max_width > 0:
        cv2.line(segmented_image, (best_x_start, best_y), (best_x_end, best_y), (255, 0, 0), 2)
        print(f"Maximum Continuous Horizontal Width: {overall_max_width} pixels")

    return overall_max_width, segmented_image


def draw_dashboard(frame, roi_coords, state, segmented_img, pixel_width, cm_width, fps, cls_name=None):
    """Creates and returns the main dashboard view, adjusted to hide the locked preview."""
    FULL_W, FULL_H = frame.shape[1], frame.shape[0]
    ROI_W, ROI_H = roi_coords[2], roi_coords[3]
    
    DASH_W = FULL_W + ROI_W
    DASH_H = max(FULL_H, ROI_H * 2) 

    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    dash[0:FULL_H, 0:FULL_W] = frame # Place the main camera feed on the left

    roi_x0, roi_y0, _, _ = roi_coords
    roi_color = (0, 0, 255)  # Red for DETECTING
    if state == STATE_SHOWING_RESULT:
        roi_color = (0, 255, 0)   # Green for RESULT
    cv2.rectangle(dash, (roi_x0, roi_y0), (roi_x0 + ROI_W, roi_y0 + ROI_H), roi_color, 2)
    cv2.putText(dash, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Placeholder for the top right quadrant during DETECTION
    if state == STATE_DETECTING:
        dummy_top_right = np.ones((ROI_H, ROI_W, 3), dtype=np.uint8) * 60
        cv2.putText(dummy_top_right, "Searching...", (50, ROI_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        dash[0:ROI_H, FULL_W:FULL_W+ROI_W] = dummy_top_right
        cv2.putText(dash, "Detection Status", (FULL_W + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Always display the segmented image if available in the bottom right quadrant
    if segmented_img is not None:
        dash[ROI_H:ROI_H*2, FULL_W:FULL_W+ROI_W] = segmented_img

        # Display both pixel and CM width
        width_text = f"Width: {int(pixel_width)} px"
        if cm_width is not None and PIXELS_PER_CM > 0:
            width_text += f" | {cm_width:.2f} cm"
        
        label = f"{cls_name if cls_name else 'Object'} | {width_text}"
        cv2.putText(dash, label, (FULL_W + 5, ROI_H + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        # Placeholder for the bottom right quadrant if no segmented image
        dummy_bottom_right = np.ones((ROI_H, ROI_W, 3), dtype=np.uint8) * 30
        cv2.putText(dummy_bottom_right, "Segmentation Result", (10, ROI_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        dash[ROI_H:ROI_H*2, FULL_W:FULL_W+ROI_W] = dummy_bottom_right

    return dash

# --- Main Application Logic ---
def main():
    # yolo_detect, yolo_seg, cap = initialize()
    if not yolo_detect:
        return

    # Configuration
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera.")
        return
    FULL_H, FULL_W = frame.shape[:2]

    ROI_W, ROI_H = 300, 300
    roi_x0 = (FULL_W - ROI_W) // 2
    roi_y0 = (FULL_H - ROI_H) // 2
    roi_coords = (roi_x0, roi_y0, ROI_W, ROI_H)

    # Create a dummy frame and draw the initial dashboard
    dummy_frame = np.ones_like(frame) * 30  # Gray background
    initial_dashboard = draw_dashboard(dummy_frame, roi_coords, STATE_DETECTING, None, None, None, 0) # Pass None for cm_width initially

    # Show the dashboard before anything else prints
    cv2.imshow("Dashboard", initial_dashboard)
    cv2.waitKey(1)  # Needed to render the window

    # Now print everything else
    print(f"Using ROI size: {ROI_W}x{ROI_H}")

    state = STATE_DETECTING
    preview, segmented_image, pixel_width, cm_width = None, None, None, None # Initialize cm_width
    cls_name = None 
    prev_time = 0
    
    detection_cycle_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        now = time.time()
        fps = 1 / (now - prev_time) if prev_time != 0 else 0
        prev_time = now

        full_frame = frame.copy()
        roi = full_frame[roi_y0:roi_y0+ROI_H, roi_x0:roi_x0+ROI_W]

        if state == STATE_DETECTING:
            if detection_cycle_start_time is None:
                detection_cycle_start_time = time.time()
                print("\n--- Searching for an object... Timer started. ---")
            yolo_start_time= time.time()
            # Changed confidence threshold here
            res = yolo_detect(roi, conf=0.25, verbose=False)[0] 
            yolo_stop_time=time.time()
            print(f"time for detection:{yolo_stop_time - yolo_start_time:.2f} seconds") 
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                classes = res.boxes.cls.cpu().numpy().astype(int)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(np.argmax(areas))
                cls_name = yolo_detect.names[int(classes[idx])]
   
                preview = roi.copy()
                print(f"Object found: {cls_name}. Running segmentation automatically...")
                
                segmentation_start_time = time.time()
                pixel_width, segmented_image = calculate_width_from_preview(yolo_seg, preview)
                segmentation_stop_time = time.time()
                print(f"Time for segmentation: {segmentation_stop_time - segmentation_start_time:.2f} seconds")

                if pixel_width:
                    # Calculate CM width here
                    if PIXELS_PER_CM > 0:
                        cm_width = pixel_width / PIXELS_PER_CM
                        print(f"Calculated width: {cm_width:.2f} cm")
                    else:
                        cm_width = None
                        print("WARNING: PIXELS_PER_CM is not set, cannot calculate CM width.")

                    total_detection_time = time.time() - detection_cycle_start_time
                    print(f"✅ Object detected AND segmented! Total time for measurement: {total_detection_time:.2f} seconds.")
                    cv2.imwrite("width_measurement.png", segmented_image)
                    print("INFO] Measurement saved as 'width_measurement.png'")
                    print("  Press 'n' to detect a new object.")
                    state = STATE_SHOWING_RESULT
                    detection_cycle_start_time = None 
                else:
                    print("Could not find an object to segment. Resuming detection.")
                    preview = None 
                    segmented_image = None
                    pixel_width = None
                    cm_width = None # Clear CM width on failure
                    cls_name = None 
        
        dashboard = draw_dashboard(frame, roi_coords, state, segmented_image, pixel_width, cm_width, fps, cls_name=cls_name)

        cv2.imshow("Dashboard", dashboard)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('n'):
            state = STATE_DETECTING
            preview, segmented_image, pixel_width, cm_width = None, None, None, None # Reset cm_width
            cls_name = None 
                
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    yolo_detect, yolo_seg, cap = initialize()
    main()
