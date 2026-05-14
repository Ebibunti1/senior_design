from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "best.pt"
TEST_IMAGES = {
    "T1_barrel": "barrel_test.jpg",
    "T2_pothole": "pothole_test.jpg",
    "T3_sign": "sign_test.jpg",
}
OUTPUT_DIR = "results/mutation_image_transformations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

for test_id, image_path in TEST_IMAGES.items():
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    dark_image = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
    resized_image = cv2.resize(image, (640, 640))

    mutants = {
        "M4_grayscale": grayscale,
        "M5_dark": dark_image,
        "M6_resized": resized_image,
    }

    for mutant_name, mutant_image in mutants.items():
        print(f"\nRunning {mutant_name} on {test_id}")
        mutant_dir = f"{OUTPUT_DIR}/{mutant_name}"
        os.makedirs(mutant_dir, exist_ok=True)

        input_file = f"{mutant_dir}/{test_id}_input.jpg"
        output_file = f"{mutant_dir}/{test_id}_result.jpg"
        cv2.imwrite(input_file, mutant_image)

        results = model(mutant_image)
        results[0].save(filename=output_file)

        print(f"Saved transformed input: {input_file}")
        print(f"Saved detection result: {output_file}")
        for box in results[0].boxes:
            print(box)
