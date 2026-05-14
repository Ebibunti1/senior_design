from ultralytics import YOLO
import cv2
import os
import time

MODEL_PATH = "best.pt"
VIDEO_PATH = "test_video.mp4"
OUTPUT_DIR = "results/video_detection"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

output_path = f"{OUTPUT_DIR}/detected_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time if total_time > 0 else 0

print("Video detection completed.")
print(f"Frames processed: {frame_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average FPS: {fps:.2f}")
print(f"Output saved to: {output_path}")
