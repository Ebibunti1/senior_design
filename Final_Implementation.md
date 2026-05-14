# Final Implementation

My final implementation focused on the perception/object detection portion of the IGVC project. This included dataset preparation, labeling, YOLOv8 model training, model export, and testing on recorded image/video data.

The implementation scripts are located in the `src/` folder.

- `detect_image.py`: Run YOLOv8 on a single image and save results.
- `detect_video.py`: Run YOLOv8 on a video and save an annotated output.
- `export_model.py`: Export trained weights to ONNX and TensorRT.
- `benchmark_models.py`: Compare model formats on video input.

The model files are stored in the `models/` folder if included in the repository.
