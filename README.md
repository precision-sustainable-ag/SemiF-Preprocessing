# SemiF-Preprocessing
Repo to setup and perform preprocessing of SemiField images. 
Preprocessing includes color calibration using a color checker chart and running detection on plants.


### **Image Inspection Tool**  

#### **Overview**
This script provides an interactive tool for inspecting and categorizing images that have been preprocessed (color corrected). Users can label images based on predefined quality control categories and generate a **CSV report**. The tool pulls from a directory of sample images in LTS and saves the CSV report in the LTS batch folderr. The tool reminds users to create issues in github and offers an option to **review flagged images** before submitting issues.

---

#### **How It Works**
1. **Loads sample images** from a specified batch directory.
2. **Displays images one at a time** for quality assessment.
3. **Allows users to classify images** using keyboard shortcuts:
   - `1` - **Pass ✅**
   - `2` - **Preprocessing Quality 🎨** (artifacts, exposure, etc.)
   - `3` - **Potting Area Cleanliness 🧹** (messy surroundings)
   - `4` - **Non-Target Weeds 🌿** (unwanted weeds in the potting area)
   - `5` - **Plant Spacing 🌱** (overlapping or incorrect plant/pot spacing)
   - `0` - **Other 📝** (miscellaneous issues)
   - `q` - **Quit ❌** (exit the review process)
4. **Saves the results** to a CSV file.
5. **Flagged images (anything not marked as "Pass")** can be reviewed at the end, allowing users to take screenshots.
6. **Prompts users to submit a GitHub issue** for flagged images.


---

#### **Running the Script**
To run the script, you can use **Hydra for configuration management**:

**Set the config in (`conf/config.yaml`):**
```yaml
tasks:
 - inspect_imagess
batch_id: "NC_2025-02-21"
```


```bash
python main.py
```

or run from the command line:

```bash
python main.py tasks=[inspect_images] batch_id=NC_2025-02-21
```
Note: change the "batch_id" field to whichever batch id you've been assigned.

---

#### **Keyboard Controls**
| Key | Label | Description |
|-----|-------|-------------|
| `1` | Pass ✅ | No issues detected |
| `2` | Preprocessing Quality 🎨 | Artifacts, exposure, color correction issues |
| `3` | Potting Area Cleanliness 🧹 | Messy with lots of residue or cluttered potting area |
| `4` | Non-Target Weeds 🌿 | Presence of unwanted weeds in the pots or on the landscape fabric |
| `5` | Plant Spacing 🌱 | Plants too close or overlapping |
| `0` | Other 📝 | Any other issue not covered by the above |
| `q` | Quit ❌ | Exit the review process |

---

#### **Output**
- **Results CSV File:**  
  A CSV file is saved in the batch directory, named:  
  ```plaintext
  {batch_id}_preprocessing_inspection_results.csv
  ```
  Example:
  ```csv
  BatchID,ImageID,Selection,Timestamp,User,LTSLocation
  NC_2025-02-21,NC_179435674,Pass,2024-03-07 14:23:01,user123,lonterm_images2
  NC_2025-02-21,NC_179435789,Preprocessing Quality,2024-03-07 14:23:01,user123,lonterm_images2
  ```
- **GitHub Issue Prompt:**  
  If flagged images exist, users are encouraged to submit an issue:
  ```plaintext
  📌 After taking screenshots, submit an issue on GitHub:
  🔗 https://github.com/precision-sustainable-ag/SemiF-Preprocessing/issues
  ```
  Suggested issue title:
  ```plaintext
  {batch_id} preprocessing inspection: X flagged images
  ```

---

#### **Notes**
- Ensure **X11 or X410 forwarding** is properly configured if running on a remote server.
- The tool automatically **resumes** labeling if interrupted.
- Flagged images will **display their category labels in the terminal** when reviewed a second time.

---

## Environment Setup
To set up the necessary dependencies for running the preprocessing pipeline:
```bash
conda env create -f ./environment.yaml
conda activate semif_prep
```
**Note**: The project requires the most recent version of PiDNG which can be installed by running `python pip install  git+https://github.com/schoolpost/PiDNG.git`

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
    ├── raw2jpg.py                  # convert downloaded raw image to jpeg (raw2png + png2jpg processing)
    ├── png2jpg.py                  # convert png files to jpeg
    ├── raw2png.py                  # convert raw images to png
    └── utils
        ├── copy_from_lockers.py    # copy raw files from NFS to local storage
        ├── calculate_ccm.py        # utils to calculate and save ccm based on yaml config 
        ├── debayer.py              # demosaic image to manually record colorchecker values
        ├── preprocess.py           # collection static methods for image preprocessing
        └── utils.py                # common util functions
```
