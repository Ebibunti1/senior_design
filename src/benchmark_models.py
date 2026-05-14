from ultralytics import YOLO
import cv2
import time
import os

VIDEO_PATH = "test_video.mp4"
MODEL_PATHS = {
    "PyTorch .pt": "best.pt",
    "ONNX .onnx": "best.onnx",
    "TensorRT .engine": "best.engine"
}

for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: {model_path} not found")
        continue

    print(f"\nTesting model: {model_name}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        continue

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model(frame, verbose=False)
        frame_count += 1

    cap.release()

    total_time = time.time() - start_time
    fps = frame_count / total_time if total_time > 0 else 0

    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
