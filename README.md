# SemiF-Preprocessing

This repository implements a multi-stage image processing pipeline for handling raw camera images. The pipeline supports:

- **Copying raw images** from network storage (“lockers”) in parallel.
- **Detecting and processing ColorChecker charts** to generate color correction matrices.
- **Converting raw images to DNG** format using the [pidng](https://github.com/ajkumar25/pidng) library.
- **Developing images** by converting DNG files to JPEGs using RawTherapee.

The project leverages [Hydra](https://hydra.cc/) for configuration management.

---


## Project Structure

```
.
├─data
│  ├─semifield-upload         
│     └─*.RAW
│  ├─semifield-developed
      ├─ccm
      │  └─NC_123123123.txt # Created during colorchecker.py. This is the first data product to be created in the semifield-developed
   │  ├─temp_dngs # Created during raw2dng.py. These need to be removed after dng2jpg.py
   │  └─images # Created during dng2jpg.py. This is the final product.
│       
├─conf
│  ├─config.yaml         # Main Hydra configuration file (defines tasks, parameters, file masks, etc.)
│  └─paths
│      └─default.yaml    # Directory and NFS path settings
└─src
   ├─colorchecker.py     # Detects ColorChecker charts, computes & saves color correction matrices
   ├─copy_from_lockers.py# Copies raw files from network storage in parallel
   ├─dng2jpg.py          # Develops images (converts DNG files to JPEGs using RawTherapee)
   ├─raw2dng.py         # Converts raw images to DNG format using pidng
   └─utils
       └─utils.py       # Utility functions (e.g., find_lts_dir to locate batch directories)
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up a conda environment and install dependencies:**

   
3. **Install and run Rawtherapee for inspecting results:**
    ```bash
   bash scripts/download_rawtherapee.sh
   ./RawTherapee_5.8.AppImage
   ```
---

## Configuration

- **Global Settings:**  
  The `conf/config.yaml` file contains the main settings such as batch identifiers, task order, test mode parameters, and file masks.

---

## Scripts

Each stage of the pipeline is implemented as a standalone script. You can run them individually or integrate them into a larger workflow.

### Copying Raw Files

The script `copy_from_lockers.py` copies raw image files from pre-defined network storage locations to local storage. It uses parallel processing with a thread pool for efficiency.

### Generating Color Correction Matrices

The `colorchecker.py` script detects a ColorChecker chart in raw images and computes a color correction matrix (CCM) using OpenCV. The CCM is saved as a text file for later use during raw conversion.


### Converting RAW to DNG

The `raw2dng.py` script converts raw binary image data into the standardized DNG format. It leverages the CCM (if available) to calibrate colors and embeds necessary metadata.


### Developing DNG Images to JPEG

The `dng2jpg.py` script calls RawTherapee via the command line to convert DNG files into JPEG images. It also has an option to update file permissions post-development.

---

