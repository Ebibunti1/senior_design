# Design Document

## Project Scope

This document describes the design for my perception/object detection work in the IGVC senior design project.

My work focused on building a YOLOv8-based object detection pipeline for detecting IGVC-related objects such as barrels, potholes, and signs.

## System Design Overview

The object detection pipeline follows this general flow:

```text
Camera / Recorded Video
	↓
Frame Extraction / Image Input
	↓
YOLOv8 Object Detection Model
	↓
Bounding Boxes + Class Labels + Confidence Scores
	↓
Detection Results for Testing / Integration
```

## Main Design Components

Component	Description
Dataset Collection	Image and video data were gathered for IGVC-related objects.
Dataset Labeling	Objects such as barrels, potholes, and signs were labeled for YOLOv8 training.
YOLOv8 Training	A YOLOv8 model was trained using the labeled dataset.
Model Testing	The trained model was tested on recorded video data to evaluate detection performance.
Detection Output	The model outputs bounding boxes, class labels, and confidence scores for detected objects.

## Object Detection Model

The perception system uses a YOLOv8-based object detection model. YOLOv8 was selected because it supports real-time object detection and can detect multiple object classes in image or video frames.

The model was trained to detect:

Barrels
Potholes
Signs

For each detected object, the model outputs:

Object class
Confidence score
Bounding box coordinates

## Open-Closed Principle Discussion

The design supports the Open-Closed Principle because the object detection pipeline can be extended without rewriting the entire system.

For example, if the team needs to detect a new object class later, such as cones or additional obstacle types, the dataset and model can be updated or retrained while keeping the same general pipeline structure.

The input process, model inference process, and output format can remain mostly the same. This means the system is open for extension through new training data, new object classes, or improved model weights, but closed for major modification to the overall software structure.

## Design Support for Future Requirements

This design can support future requirements by allowing:

Additional object classes to be added through dataset updates
Improved model performance through retraining
Different YOLO model versions to be tested
Detection results to be used by integration or navigation components
Testing on different videos, lighting conditions, and field environments

## Summary

The design provides a clear object detection pipeline for the IGVC perception system. It supports dataset preparation, YOLOv8 training, model testing, and detection output generation. The design is also flexible enough to support future improvements without requiring major changes to the system structure.

