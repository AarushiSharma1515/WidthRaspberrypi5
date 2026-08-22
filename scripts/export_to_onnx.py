from ultralytics import YOLO
import os

YOLO_MODEL_PATH = 'yolov8n.pt' # a pretrained yolo model download the weights automatically if not present in current directory

INPUT_IMG_SIZE = [640, 640]
SIMPLIFY_ONNX = True

DYNAMIC_INPUTS = True

try:
    print(f"[{os.path.basename(__file__)}] Loading YOLO model from: {YOLO_MODEL_PATH}")
    # Load the YOLO model
    model = YOLO(YOLO_MODEL_PATH)
    print(f"[{os.path.basename(__file__)}] YOLO model loaded successfully.")

    print(f"\n[{os.path.basename(__file__)}] Exporting model to ONNX format...")
    print(f"  - Input Model: {YOLO_MODEL_PATH}")
    print(f"  - Target Format: ONNX")
    print(f"  - Input Image Size: {INPUT_IMG_SIZE}")
    print(f"  - Simplify ONNX Graph: {SIMPLIFY_ONNX}")
    print(f"  - Dynamic Inputs: {DYNAMIC_INPUTS}")

    import time

    start_time = time.time()

    export_result = model.export(
        format="onnx",
        imgsz=INPUT_IMG_SIZE,
        simplify=SIMPLIFY_ONNX,
        dynamic=DYNAMIC_INPUTS,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{os.path.basename(__file__)}] Export success {elapsed_time:.2f}s, saved as '{export_result}'")
        

    print(f"\n[{os.path.basename(__file__)}] ONNX model successfully exported to: {export_result}")
    print(f"[{os.path.basename(__file__)}] Conversion complete!")

except FileNotFoundError:
    print(f"[{os.path.basename(__file__)}] Error: YOLO model file not found at '{YOLO_MODEL_PATH}'.")
    print(f"[{os.path.basename(__file__)}] Please check the path or ensure the pre-trained model can be downloaded.")
except Exception as e:
    print(f"[{os.path.basename(__file__)}] An error occurred during conversion: {e}")

print(f"\n[{os.path.basename(__file__)}] --- Verifying ONNX model with ONNX Runtime ")
try:
    import onnxruntime as ort
    import numpy as np

    if not os.path.exists(export_result):
        print(f"[{os.path.basename(__file__)}] Error: Exported ONNX file not found at '{export_result}'. Skipping verification.")
    else:
        print(f"[{os.path.basename(__file__)}] Loading ONNX model: {export_result}")
        
        session = ort.InferenceSession(export_result, providers=['CPUExecutionProvider'])

        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        input_name = input_info.name
        output_name = output_info.name
        input_shape = input_info.shape

        print(f"[{os.path.basename(__file__)}] ONNX Model Input Name: {input_name}, Shape: {input_shape}, Type: {input_info.type}")
        print(f"[{os.path.basename(__file__)}] ONNX Model Output Name: {output_name}, Shape: {output_info.shape}, Type: {output_info.type}")

        dummy_input_shape = [1, 3] + list(INPUT_IMG_SIZE)
        dummy_input = np.random.rand(*dummy_input_shape).astype(np.float32)

        print(f"[{os.path.basename(__file__)}] Running dummy inference with input shape: {dummy_input.shape}")
        onnx_output = session.run([output_name], {input_name: dummy_input})

        print(f"[{os.path.basename(__file__)}] Dummy inference successful. ONNX Output shape: {onnx_output[0].shape}")
        print(f"[{os.path.basename(__file__)}] ONNX model verified!")

except ImportError:
    print(f"[{os.path.basename(__file__)}] ONNX Runtime not fully installed or accessible. Cannot perform verification.")
except Exception as e:
    print(f"[{os.path.basename(__file__)}] An error occurred during ONNX model verification: {e}")

print(f"\n[{os.path.basename(__file__)}] Script finished.")

