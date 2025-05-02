# Bounding Box Utilities Documentation

## Overview

This directory contains utilities for processing bounding boxes, particularly for mapping them from local image coordinates to a global coordinate system. This is useful for deduplicating overlapping detections and standardizing object annotations based on real-world spatial data.

## Directory Structure

```
.
├─bbox_transformations.py
├─bbox_utils.py
├─connectors.py
└─io_utils.py
```

### bbox_transformations.py
Implements key classes for transforming bounding box coordinates from local image-based coordinates to global real-world coordinates.

#### **Key Classes**
1. **BBoxFilter**
   - Identifies and deduplicates overlapping bounding boxes by considering image fields of view (FOVs) and using Intersection over Union (IoU) metrics.
   - Selects the best bounding box based on camera proximity and deduplicates redundant detections.

2. **BBoxMapper**
   - Maps bounding boxes from image space to global coordinates using metadata from Structure from Motion (SfM) outputs.
   - Uses Metashape to perform coordinate transformations.

3. **GlobalToLocalMapper**
   - Maps global bounding boxes back to image coordinates for reference.
   - Uses camera transformation matrices and chunk information to perform reverse mapping.

### bbox_utils.py
Contains utility functions and helper classes for bounding box operations.

#### **Key Functions**
1. **bb_iou**
   - Computes the Intersection over Union (IoU) between two bounding boxes.
   - Used for filtering and selecting the best bounding box during deduplication.

2. **generate_hash**
   - Generates a unique hash for bounding boxes based on image ID and bounding box ID.
   - Ensures uniqueness when comparing overlapping bounding boxes.

### connectors.py
Provides interfaces for reading data from SfM outputs and converting annotation files into structured bounding box objects.

#### **Key Classes**
1. **SfMComponents**
   - Reads CSV files from the autoSfM pipeline.
   - Merges camera reference and field-of-view data into a structured DataFrame.

2. **BBoxComponents**
   - Converts bounding box coordinates from annotation files into structured objects.
   - Uses a user-defined reader function to flexibly support different annotation formats (XML, JSON, CSV).
   - Computes camera parameters such as pixel dimensions, focal lengths, and orientation angles from metadata.

### io_utils.py
Handles input/output operations related to reading bounding box annotations from XML and YOLO CSV formats.

#### **Key Classes**
1. **ParseXML**
   - Reads bounding box annotations from XML files.
   - Extracts object coordinates and class labels.

2. **ParseYOLOCsv**
   - Reads YOLO-style bounding box annotations from CSV files.
   - Parses coordinates, object classes, and classifier confidence scores.
   - Filters images based on predefined criteria and ensures compatibility with full-resolution images.

## Usage

1. **Reading Bounding Boxes**
   - Use `ParseXML` or `ParseYOLOCsv` to read annotations from XML or CSV files.
   
2. **Transforming Bounding Boxes**
   - Use `BBoxMapper` to convert local bounding box coordinates into global real-world coordinates.
   - Use `GlobalToLocalMapper` to revert global coordinates back to image space if needed.

3. **Deduplicating Bounding Boxes**
   - Use `BBoxFilter` to identify overlapping bounding boxes and select the most accurate detection.
   - Adjust IoU thresholds (`FOV_IOU_THRESH` and `BBOX_OVERLAP_THRESH`) for fine-tuning overlap detection.

4. **Integrating with SfM Data**
   - Use `SfMComponents` to extract relevant metadata from autoSfM CSV outputs.
   - Use `BBoxComponents` to structure bounding boxes into Python objects for further processing.

## Dependencies

- `numpy`
- `pandas`
- `cv2`
- `Metashape`
- `scipy`
- `tqdm`

## Future Enhancements
- Add support for additional annotation formats.
- Improve performance of bounding box deduplication.
- Expand support for different SfM pipelines beyond autoSfM.