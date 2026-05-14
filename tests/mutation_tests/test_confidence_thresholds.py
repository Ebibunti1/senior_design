from ultralytics import YOLO
import os

MODEL_PATH = "best.pt"
TEST_IMAGES = {
    "T1_barrel": "barrel_test.jpg",
    "T2_pothole": "pothole_test.jpg",
    "T3_sign": "sign_test.jpg",
}
CONFIDENCE_THRESHOLDS = {
    "M1_low_conf_0_10": 0.10,
    "M2_medium_conf_0_50": 0.50,
    "M3_high_conf_0_75": 0.75,
}
OUTPUT_DIR = "results/mutation_confidence"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

for mutant_id, conf_value in CONFIDENCE_THRESHOLDS.items():
    print(f"\nRunning {mutant_id} with confidence threshold {conf_value}")
    mutant_dir = f"{OUTPUT_DIR}/{mutant_id}"
    os.makedirs(mutant_dir, exist_ok=True)

    for test_id, image_path in TEST_IMAGES.items():
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        results = model(image_path, conf=conf_value)
        output_file = f"{mutant_dir}/{test_id}_result.jpg"
        results[0].save(filename=output_file)

        print(f"\nTest Case: {test_id}")
        print(f"Image: {image_path}")
        print(f"Saved result: {output_file}")
        for box in results[0].boxes:
            print(box)
