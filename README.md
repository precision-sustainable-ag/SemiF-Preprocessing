# SemiF-Preprocessing
Repo to setup and perform preprocessing of SemiField images. 
Preprocessing includes color calibration using a color checker chart and running detection on plants.


## Environment Setup
To set up the necessary dependencies for running the preprocessing pipeline:
```bash
conda env create -f ./environment.yaml
conda activate semif_preprocessing
```
**Note**: The project also requires `rawtherapee-cli` which is 
auto-installed when needed. Alternatively, it can be manually installed 
using `scripts/validate_rawtherapee.sh`. 

## Execution
1. Setup environment using aforementioned steps.
2. Edit `conf/config.yaml`
   * `batch_id`: batch you want to preprocess
   * `tasks/copy_from_lockers`: copy raw files from LTS to 
   `./data/<lts_location>/semifield-upload/<batch_id>` (comment out if 
     already downloaded)
   * `tasks/raw2jpg`: convert local raw files to jpegs stored in LTS: 
     `<lts_location>/semifield-developed-images/<batch_id>`
   * `tasks/raw2png` and `tasks/png2jpg` can be left commented out unless 
     you need to run these separately
   * `raw2png/remove_raws`: delete local raw files when each file is 
     converted to pngs
   * `raw2jpg/remove_pngs`: delete local png files when each png is 
     converted to jpeg
3. Execute `python main.py` in the root directory

## Project Structure
```text
.
├── conf
│   ├── ccm     # config files to define color correction matrix (ccm)
│   ├── config.yaml
│   ├── hydra
│   └── paths
├── data
│   └── semifield-utils
│       └── image_development
│           ├── color_matrix        # ccm files saved as numpy arrays 
│           └── dev_profiles        # rawtherapee pp3 profile
├── main.py                         # main entry point for omegaconf
├── scripts
│   └── validate_rawtherapee.sh     # script to validate rawtherapee installation
└── src
    ├── archive                     # archived code for future reference
    ├── copy_from_lockers.py        # copy raw files from NFS to local storage
    ├── raw2jpg.py                  # convert downloaded raw image to jpeg (raw2png + png2jpg processing)
    ├── png2jpg.py                  # convert png files to jpeg
    ├── raw2png.py                  # convert raw images to png
    └── utils
        ├── calculate_ccm.py        # utils to calculate and save ccm based on yaml config 
        ├── debayer.py
        ├── preprocess.py           # collection static methods for image preprocessing
        └── utils.py                # common util functions
```