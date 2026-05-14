# Traceability Information

## Overview

This document maps each perception/object detection requirement to the design, implementation, and verification work completed for the IGVC senior design project.

## Quick Map

- PER.FR.1 -> YOLOv8 pipeline -> trained model -> video-based testing
- PER.FR.2 -> detection output format -> inference output -> output review
- SW.R2 -> camera/video input flow -> recorded frames -> video-based testing
- SW.R4 -> YOLOv8 train + infer -> trained model -> performance evaluation
- PER.CON.1 -> outdoor data coverage -> labeled dataset -> lighting-condition tests

## Traceability Matrix

| Req ID | Requirement (short) | Design | Implementation | Verification |
|---|---|---|---|---|
| PER.FR.1 | Detect barrels, potholes, signs | YOLOv8 pipeline | Trained model + labeled data | Tested on recorded video data |
| PER.FR.2 | Provide outputs for integration | Output format defined | YOLOv8 inference output | Output review for usability |
| SW.R2 | Camera/video input used | Input pipeline | Recorded frames as input | Tested using recorded video data |
| SW.R4 | YOLOv8 real-time detection | YOLOv8 train + infer | Trained model | Model testing and performance eval |
| PER.CON.1 | Lighting robustness | Outdoor-style dataset | Labeled dataset + training | Tested on varied lighting data |

## Notes

My work focused on the perception/object detection portion of the IGVC project, including dataset gathering, labeling, YOLOv8 training, and model testing.

The full system integration and navigation behavior depend on using the object detection outputs produced by the perception system.

