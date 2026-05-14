from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "best.pt"
IMAGE_PATH = "test_image.jpg"
OUTPUT_DIR = "results/metamorphic_brightness"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Original image
original_results = model(image)
original_results[0].save(filename=f"{OUTPUT_DIR}/original_result.jpg")

# Brightness-adjusted image
bright_image = cv2.convertScaleAbs(image, alpha=1.1, beta=20)
cv2.imwrite(f"{OUTPUT_DIR}/brightness_adjusted.jpg", bright_image)
bright_results = model(bright_image)
bright_results[0].save(filename=f"{OUTPUT_DIR}/brightness_result.jpg")

print("Original image detections:")
for box in original_results[0].boxes:
    print(box)

print("\nBrightness-adjusted image detections:")
for box in bright_results[0].boxes:
    print(box)
