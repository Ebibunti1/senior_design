from ultralytics import YOLO

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

# Export to ONNX
model.export(format="onnx")

# Export to TensorRT engine
# This should be done on the Jetson/device where the engine will be used
model.export(format="engine")

print("Model export completed.")
