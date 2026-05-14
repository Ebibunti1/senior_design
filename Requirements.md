# Requirements Document

## Project Scope

This document includes the new or modified requirements related to my perception/object detection work for the IGVC senior design project.

My contribution focused on gathering data, labeling the dataset, training a YOLOv8 object detection model, and testing the model for IGVC-related objects.

## Perception / Object Detection Requirements

| Requirement ID | Requirement | Description | Verification Method | Status |
|---|---|---|---|---|
| PER.FR.1 | Obstacle Detection Output | The perception system shall detect relevant IGVC obstacles and objects, including barrels, potholes, and signs. | Model testing on recorded video data | Satisfied |
| PER.FR.2 | Detection Output for Navigation | The perception system shall provide object detection outputs that can be used by the navigation/system integration team. | Detection output review / integration support | In Progress |
| SW.R2 | Camera Image Input | The system shall use camera image/video input for object detection. | Testing with recorded video data | Satisfied |
| SW.R4 | YOLOv8 Object Detection | The system shall use a YOLOv8-based object detection model to detect IGVC-related objects in real time. | YOLOv8 training and testing | Satisfied |
| PER.CON.1 | Lighting Robustness | The object detection model should perform under different outdoor lighting conditions, including sunlight and overcast conditions. | Dataset testing / field-style video testing | Open |

## Requirement Notes

My main responsibility was related to the perception/object detection portion of the IGVC project. This included gathering image and video data, labeling the dataset, training a YOLOv8 object detection model, and testing the trained model.

The YOLOv8 model was trained to detect IGVC-related objects such as barrels, potholes, and signs. After training, I tested the model on recorded video data and evaluated its detection performance.

The full system integration and navigation behavior depend on the perception outputs being available for use by the rest of the system.

