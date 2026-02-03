import logging
import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- Constants and Global State ---
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Define application states
STATE_DETECTING = 0
STATE_PREVIEW_LOCKED = 1  # New state: Object is stable, waiting for lock command
STATE_WAIT_CONFIRM = 2    # Preview is locked, waiting for segmentation command
STATE_SHOWING_RESULT = 3

# --- Helper Functions ---

def initialize():
    """Initializes BOTH YOLO models and the camera."""
    print("Initializing...")
    yolo_detect = YOLO("yolov8n.pt", verbose=False)
    yolo_seg = YOLO("yolov8n-seg.pt", verbose=False)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      print("Cannot open camera")
      return None, None, None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera default resolution: {int(width)} x {int(height)}")

    return yolo_detect, yolo_seg, cap

def check_object_stability(yolo_model, res, prev_center, prev_cls, move_thresh):
    """Checks if the detected object is stable in the frame."""
    if len(res.boxes) == 0:
        return None, None, None, False

    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    classes = res.boxes.cls.cpu().numpy().astype(int)
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
     
    x1, y1, x2, y2 = boxes[idx]
    curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    curr_cls = classes[idx]

    is_stable = False
    if prev_cls is not None and curr_cls == prev_cls:
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        if np.hypot(dx, dy) <= move_thresh:
            is_stable = True
    
    return curr_center, curr_cls, yolo_model.names[int(curr_cls)], is_stable

def calculate_width_from_preview(yolo_seg_model, preview_img):
    """Runs segmentation on an image and calculates pixel width."""
    seg_res = yolo_seg_model(preview_img, conf=0.4)[0]
    if not seg_res.masks:
        return None, None

    boxes = seg_res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = np.argmax(areas)
    
    contour = seg_res.masks.xy[idx].astype(np.int32)
    rect = cv2.minAreaRect(contour)
    pixel_width = min(rect[1])

    segmented_image = np.zeros_like(preview_img)
    cv2.drawContours(segmented_image, [contour], -1, (0, 255, 100), cv2.FILLED)
    
    return pixel_width, segmented_image

def draw_dashboard(frame, roi_coords, state, preview, segmented_img, width, fps):
    """Creates and returns the main dashboard view."""
    FULL_W, FULL_H = frame.shape[1], frame.shape[0]
    ROI_W, ROI_H = roi_coords[2], roi_coords[3]
    DASH_W = FULL_W + ROI_W
    DASH_H = max(FULL_H, ROI_H * 2)
    
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    dash[0:FULL_H, 0:FULL_W] = frame
    
    roi_x0, roi_y0, _, _ = roi_coords
    roi_color = (0, 0, 255)  # Red for DETECTING
    if state == STATE_PREVIEW_LOCKED:
        roi_color = (255, 255, 0) # Cyan for READY TO LOCK
    elif state == STATE_WAIT_CONFIRM:
        roi_color = (0, 255, 255) # Yellow for WAITING
    elif state == STATE_SHOWING_RESULT:
        roi_color = (0, 255, 0)   # Green for RESULT
    cv2.rectangle(dash, (roi_x0, roi_y0), (roi_x0 + ROI_W, roi_y0 + ROI_H), roi_color, 2)
    cv2.putText(dash, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if preview is not None:
        dash[0:ROI_H, FULL_W:FULL_W+ROI_W] = preview
        cv2.putText(dash, "Locked Preview", (FULL_W + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if segmented_img is not None:
        dash[ROI_H:ROI_H*2, FULL_W:FULL_W+ROI_W] = segmented_img
        cv2.putText(dash, "Segmented Object", (FULL_W + 5, ROI_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if width:
            cv2.putText(dash, f"Width: {int(width)} px", (FULL_W + 5, ROI_H + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    return dash

# --- Main Application Logic ---
def main():
    yolo_detect, yolo_seg, cap = initialize()
    if not yolo_detect:
        return

    # Configuration
    ret, frame = cap.read()
    if not ret:
        print("Could not read from camera")
        return
    FULL_H, FULL_W = frame.shape[:2]  

    ROI_W, ROI_H = 300, 300
    roi_x0 = (FULL_W - ROI_W) // 2
    roi_y0 = (FULL_H - ROI_H) // 2
    roi_coords = (roi_x0, roi_y0, ROI_W, ROI_H)

    roi = frame[roi_y0:roi_y0+ROI_H, roi_x0:roi_x0+ROI_W]
    print(f"Detected ROI size: {roi.shape[1]}x{roi.shape[0]}")

    state = STATE_DETECTING
    stable_thresh, move_thresh = 1.5, 10.0
    stable_start, prev_center, prev_cls = None, None, None
    preview, segmented_image, pixel_width = None, None, None
    prev_time = 0
    detection_cycle_start_time = None

    dummy_preview = np.ones((ROI_H, ROI_W, 3), dtype=np.uint8) * 60
    cv2.putText(dummy_preview, "Detection", (50, ROI_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
              print("\n--- Searching for a stable object... Timer started. ---")

          # The frame limit has been removed, detection now runs continuously

        #   res = yolo_detect(roi, conf=0.4, verbose=False)[0]
              yolo_start_time= time.time()
            # Changed confidence threshold here
              res = yolo_detect(roi, conf=0.25, verbose=False)[0] 
              yolo_stop_time=time.time()
              print(f"time for detection:{yolo_stop_time - yolo_start_time}") 
          center, cls, cls_name, is_stable = check_object_stability(yolo_detect, res, prev_center, prev_cls, move_thresh)

          if is_stable:
              stable_start = stable_start or now
              if (now - stable_start >= stable_thresh):
                  total_detection_time = time.time() - detection_cycle_start_time
                  print(f"✅ Stable object found! Final time to lock preview: {total_detection_time:.2f} seconds.")
                  
                  detection_cycle_start_time = None
                  preview = roi.copy()
                  print(f"Object found: {cls_name}. Press 'y' to lock preview.")
                  state = STATE_WAIT_CONFIRM
          else:
              stable_start = now if center is not None else None
          prev_center, prev_cls = center, cls

        current_preview = preview if preview is not None else dummy_preview
        dashboard = draw_dashboard(full_frame, roi_coords, state, current_preview, segmented_image, pixel_width, fps)
        
        cv2.imshow("Dashboard", dashboard)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
                
        if state == STATE_WAIT_CONFIRM:
            if key == ord('y'):
                print("Running segmentation...")
                pixel_width, segmented_image = calculate_width_from_preview(yolo_seg, preview)
                if pixel_width:
                    cv2.imwrite("green_mask.png",segmented_image)
                    print("INFO] Green mask saved as 'green_mask.png")
                    print(f"  -> Calculated Pixel Width: {int(pixel_width)} px")
                    print("  Press 'n' to detect a new object.")
                    state = STATE_SHOWING_RESULT
                else:
                    print("Could not find an object. Resuming detection.")
                    state = STATE_DETECTING
                    preview = None
            elif key == ord('n'):
                state = STATE_DETECTING
                preview, stable_start, prev_center, prev_cls = None, None, None, None
                
        elif state == STATE_SHOWING_RESULT:
            if key == ord('n'):
                state = STATE_DETECTING
                preview, segmented_image, pixel_width = None, None, None
                stable_start, prev_center, prev_cls = None, None, None

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()