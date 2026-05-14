from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "best.pt"
IMAGE_PATH = "test_image.jpg"
OUTPUT_DIR = "results/metamorphic_resize"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Original image
original_results = model(image)
original_results[0].save(filename=f"{OUTPUT_DIR}/original_result.jpg")

# Resized image
resized_image = cv2.resize(image, (640, 640))
cv2.imwrite(f"{OUTPUT_DIR}/resized_image.jpg", resized_image)
resized_results = model(resized_image)
resized_results[0].save(filename=f"{OUTPUT_DIR}/resized_result.jpg")

print("Original image detections:")
for box in original_results[0].boxes:
    print(box)

print("\nResized image detections:")
for box in resized_results[0].boxes:
    print(box)
