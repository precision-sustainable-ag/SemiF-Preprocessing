# Codebase Overview

## Project Structure

The project contains the following files:

```
.
├─ callbacks.py
├─ config_utils.py
├─ dataframe.py
├─ estimation.py
├─ logger.py
├─ metashape_utils.py
└─ resize.py
```

## File Descriptions

### callbacks.py
Contains a simple progress percentage callback function:

```py
    def percentage_callback(progress_percent: float) -> None:
        print(f"{progress_percent}% done.")
```

### config_utils.py
Handles configuration utilities including:
- Directory creation and management.
- Parsing YAML configuration files.
- Setting up paths for processing.
- Checking the existence of necessary data files and directories.

### dataframe.py
Implements a lightweight DataFrame class for handling tabular data without requiring pandas. Key features include:
- Parsing lists of dictionaries into structured data.
- Saving content to CSV.
- Retrieving data efficiently.

### estimation.py
Contains utilities for:
- Computing camera statistics using Metashape.
- Calculating camera field of view.
- Handling coordinate transformations.

### logger.py
Implements a basic logging system that writes log messages both to the console and a log file.

### metashape_utils.py
Implements utilities for working with Agisoft Metashape, including:
- Loading and managing Metashape projects.
- Detecting markers.
- Aligning photos.
- Exporting camera reference and GCP reference data.
- Generating depth maps, dense clouds, and orthomosaics.

### resize.py
Handles image resizing and mask creation, including:
- Scaling down images and masks.
- Detecting missing data.
- Masking images based on color thresholds.
- Using multiprocessing for efficient resizing operations.

## Dependencies
- Python 3.x
- OpenCV
- NumPy
- Metashape (Agisoft)
- PIL (Pillow)
- YAML
- Logging utilities
- Multiprocessing

## Usage
This codebase is designed to be used in an image processing pipeline. Configuration is managed via YAML files, and each module provides specialized functionality for processing and handling images, metadata, and spatial data.


## Contribution Guidelines
- Follow PEP8 coding standards.
- Use docstrings for all functions.
- Log key steps and errors.
- Optimize performance for large image datasets.

## License
This project follows an open-source license. See LICENSE file for details.

