# CS 4910 Final Submission

**Student:** Ebisa Bunti  
**Course:** CS 4910 Senior Design  
**Project:** IGVC Object Detection / Perception  

## Overview

This repository contains my final submission materials for CS 4910. My work focused on the object detection/perception portion of the IGVC senior design project.

The perception system uses camera-based object detection to identify important objects for autonomous navigation, including obstacles such as barrels, potholes, and signs.

## Submission Contents

This repository includes the required final submission materials:

1. Test Report
2. Requirements Document
3. Design Document
4. Final Implementation
5. Traceability Information

## Repository Structure

```text
CS4910_Final_Submission/
|
|-- README.md
|-- Requirements.md
|-- Design.md
|-- Traceability.md
|-- Test_Report.md
|-- Final_Implementation.md
|-- requirements.txt
|
|-- src/
|   |-- detect_image.py
|   |-- detect_video.py
|   |-- export_model.py
|   `-- benchmark_models.py
|
|-- models/
|   `-- trained model files (if included)
|
|-- tests/
|   |-- metamorphic_tests/
|   `-- mutation_tests/
|
`-- results/
    `-- test outputs, screenshots, and supporting results
```

## Running the Test Scripts

Install dependencies:

```bash
pip install -r requirements.txt
```

Run metamorphic tests:

```bash
python tests/metamorphic_tests/test_brightness.py
python tests/metamorphic_tests/test_resize.py
```

Run mutation tests:

```bash
python tests/mutation_tests/test_confidence_thresholds.py
python tests/mutation_tests/test_image_transformations.py
```

The scripts save output images and detection results in the results/ folder.


