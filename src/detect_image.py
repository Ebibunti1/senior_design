from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "best.pt"
IMAGE_PATH = "test_image.jpg"
OUTPUT_DIR = "results/image_detection"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)

for result in results:
    result.save(filename=f"{OUTPUT_DIR}/image_result.jpg")

print("Image detection completed.")
print(f"Result saved to: {OUTPUT_DIR}/image_result.jpg")
