# Chapter 2: Configuration Management (Hydra)

In [Chapter 1: Pipeline Orchestration](01_pipeline_orchestration_.md), we learned how `SemiF-Preprocessing` acts like a "General Contractor" to manage a sequence of image processing tasks. We saw that the orchestrator (`main.py`) uses a "blueprint" – a configuration – to know what to do. Now, let's open up that blueprint and see how it's managed!

Imagine you've just run the entire image processing pipeline for a batch of images from a field called "Field_A_sunny_day". The results look good, but you notice the brightness in the final JPG images is a bit too high. You want to re-run just the image conversion step with a different brightness setting. Or, perhaps next week you get data from "Field_B_cloudy_day", which is stored in a completely different folder and might need different processing parameters.

How do you tell the `SemiF-Preprocessing` factory to use different settings for these different scenarios without rewriting the Python code every time? This is where **Configuration Management** comes in, and in our project, we use a powerful tool called **Hydra**.

## What's the Big Deal with Configuration Management?

Think of `SemiF-Preprocessing` as a sophisticated image processing factory. This factory has many machines (our software modules for tasks like image correction, 3D modeling, etc.). Each machine has various dials and switches (parameters) that control how it operates:
*   Input/output folder paths
*   Image resize percentages
*   Color correction profiles
*   Which specific tasks to run (e.g., "just do color correction today")

If these settings were hard-coded directly into the Python scripts, changing anything would mean editing the code. This is risky (you might break something!) and inefficient, especially if you want to run the same pipeline with many different settings for different batches or experiments.

**Configuration Management** solves this by:
1.  **Externalizing Settings:** Keeping all settings outside the main Python code, in separate files.
2.  **Flexibility:** Allowing you to easily change settings for different runs.
3.  **Organization:** Structuring settings logically, making them easier to understand and manage.
4.  **Reproducibility:** Letting you save and version-control your "recipes" (configurations) for different processing jobs.

**Hydra** is the tool we use for this. It's like the **master control panel** for our entire image processing factory. You can set all the dials and switches on this panel *before* starting a production run. Different arrangements of these dials (configurations) can even be saved as presets.

## Meet Hydra: Your Project's Control Panel

Hydra helps us manage all settings using simple text files written in a format called **YAML** (which stands for "YAML Ain't Markup Language" – it's designed to be human-readable).

### The Central Blueprint: `conf/config.yaml`

The main configuration file for our project is typically `conf/config.yaml`. Let's look at a simplified piece of it, similar to what we saw in Chapter 1, but now with a bit more detail:

```yaml
# conf/config.yaml (simplified snippet)

# ... (Hydra-specific settings, usually at the top) ...

############################################
# Main configuration for the current batch
############################################
batch_id: MD_2025-05-02 # Which image batch to process
season: cool_season_covers_2024_2025
modes:
  # - sync      # Example: Download data (commented out, so it won't run)
  - correct   # Run image correction
  - asfm      # Run 3D modeling
  # - label     # Example: Run object labeling (commented out)
############################################

# Settings for the raw to JPG conversion step
raw2jpg:
  resize_factor: 0.25  # Make JPGs 25% of original size
  remove_dngs: true    # Delete intermediate DNG files

# ... (other settings for paths, AutoSfM, etc.) ...
```
*   `batch_id` and `modes` are familiar from Chapter 1. They tell the pipeline *which* data to process and *which major tasks* to run.
*   `raw2jpg`: This is a **group** of settings specifically for the image conversion task.
    *   `resize_factor: 0.25`: This tells the "raw2jpg" machine to resize images to 25% of their original dimensions.
    *   `remove_dngs: true`: This tells it to clean up intermediate files.

With Hydra, you can change these values directly in the `config.yaml` file before running the pipeline. For example, if you wanted smaller JPGs, you could change `resize_factor` to `0.1`.

### Overriding Settings from the Command Line

What if you want to try a different `resize_factor` for just *one* run, without editing `config.yaml` permanently? Hydra makes this super easy! You can "override" settings directly from your terminal command:

```bash
# Run the pipeline, but for this run, use a resize_factor of 0.5
python main.py raw2jpg.resize_factor=0.5

# You can also change the batch_id and modes
python main.py batch_id=TX_EXPERIMENT_01 modes=[correct] raw2jpg.resize_factor=0.1
```
*   `raw2jpg.resize_factor=0.5`: This tells Hydra to use `0.5` for `resize_factor` *inside* the `raw2jpg` group, just for this specific execution. The value in `config.yaml` remains unchanged for future runs.
*   Notice the `.` (dot) notation: `group_name.parameter_name`. This is how Hydra lets you target specific settings.

Command-line overrides are very powerful for quick experiments or when running automated scripts for many batches with slight variations.

### How Python Code Sees These Settings: The `cfg` Object

In [Chapter 1: Pipeline Orchestration](01_pipeline_orchestration_.md), we saw a `cfg` object being passed around in `main.py` and to task functions. This `cfg` object is created by Hydra and contains all the configuration values.

The magic happens with a special piece of Python code called a **decorator** in `main.py`:

```python
# main.py (simplified snippet showing Hydra's entry point)
import hydra
from omegaconf import DictConfig # The type of 'cfg'

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 'cfg' now contains all settings from config.yaml and command-line overrides!
    log.info(f"Processing batch: {cfg.batch_id}")
    log.info(f"Resize factor for raw2jpg: {cfg.raw2jpg.resize_factor}")
    
    # ... (the rest of the pipeline orchestration logic from Chapter 1) ...
    # run_single_batch(cfg) 
```
*   `@hydra.main(...)`: This line "decorates" our `main` function. It tells Hydra:
    *   `config_path="conf"`: Look for configuration files in a folder named `conf` (relative to `main.py`).
    *   `config_name="config"`: The main configuration file is `config.yaml`.
*   When you run `python main.py ...`, Hydra automatically:
    1.  Reads `conf/config.yaml`.
    2.  Applies any command-line overrides.
    3.  Bundles all these settings into the `cfg` object.
    4.  Calls your `main` function, passing this `cfg` object to it.

Inside your Python code, you can then access settings like attributes of an object:
*   `cfg.batch_id` gives you the value of `batch_id`.
*   `cfg.raw2jpg.resize_factor` gives you the `resize_factor` from the `raw2jpg` group.

This makes it very clean to use configuration values throughout your project.

## Organizing Your Factory's Controls: Hydra's Key Features

A real factory control panel isn't just one giant board; it's often organized into sections. Hydra allows for similar organization.

### 1. Configuration Groups and Defaults

Imagine your `config.yaml` file getting very, very long with settings for paths, image correction, 3D modeling, reporting, etc. It would become hard to manage!

Hydra allows you to break down your configuration into multiple files, organized into **groups**. Look at the top of the actual `conf/config.yaml`:

```yaml
# conf/config.yaml (snippet showing defaults)
defaults:
  - _self_        # Means "include settings from this file itself"
  - paths: default # Load path settings from a 'default.yaml' in 'paths' group
  - exif: default  # Load exif settings from a 'default.yaml' in 'exif' group
  - asfm: default  # Load AutoSfM settings from 'default.yaml' in 'asfm' group
  # ... and so on for other groups
```
This `defaults` list tells Hydra how to compose the final configuration:
*   `paths: default`: This instruction tells Hydra: "Go look in a subfolder named `conf/paths/` for a file named `default.yaml`. Load the settings from that file and make them available in our `cfg` object under the key `paths`."

So, if `conf/paths/default.yaml` contains:
```yaml
# conf/paths/default.yaml (simplified snippet)
workdir: /home/user/SemiF-Preprocessing # Project's working directory
data_dir: ${paths.workdir}/data/        # Main data directory
log_dir: ${paths.workdir}/.logs/
```
Then, in your Python code, you can access these as:
*   `cfg.paths.workdir`
*   `cfg.paths.data_dir`

This is like having separate, smaller control panels for "Path Settings," "EXIF Data Settings," "AutoSfM Settings," etc., and Hydra assembles them all together based on the `defaults` list in the main `config.yaml`. This keeps your configurations modular and easier to navigate. You can learn more about how paths are managed in [Chapter 3: Data Synchronization and Path Management](03_data_synchronization_and_path_management_.md).

The `${paths.workdir}` syntax you see is called **interpolation**. It means "take the value of `workdir` from within the `paths` group and substitute it here." So `data_dir` automatically becomes `/home/user/SemiF-Preprocessing/data/`.

### 2. Accessing Configuration in Task Modules

When the main orchestrator in `main.py` calls a specific task module (like the image correction module), it passes the same `cfg` object. This means each task module has access to all the settings it might need.

For example, a hypothetical `correct_images` function inside the `correct.py` module might look like this:

```python
# src/correct.py (conceptual snippet)
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

def main(cfg: DictConfig): # Hydra's cfg object is passed here!
    log.info(f"Starting image correction for batch: {cfg.batch_id}")
    
    input_image_folder = f"{cfg.paths.batch_dir}/raw_images" # Example path construction
    output_image_folder = f"{cfg.paths.batch_dir}/corrected_images"
    
    resize_factor = cfg.raw2jpg.resize_factor # Get the resize factor
    
    log.info(f"Images will be resized by: {resize_factor*100}%")
    # ... actual image correction logic using these parameters ...
    log.info("Image correction complete.")
```
This task function can directly use `cfg.batch_id`, `cfg.paths.batch_dir` (which itself might be built using `batch_id` like `${paths.developed_dir}/${batch_id}`), and `cfg.raw2jpg.resize_factor` without needing them to be passed as individual arguments. All settings come neatly packaged in `cfg`.

### 3. Automatic Output Directories

When you run an experiment, it's good practice to save logs and outputs in a unique directory for that specific run. Hydra helps with this too!

In `conf/config.yaml`, you might see something like:
```yaml
# conf/config.yaml (snippet for Hydra's run directory)
hydra:
  run:
    dir: ${paths.log_dir}/${season}/${batch_id}/${job_now}
  output_subdir: ${hydra.run.dir}/hydra
```
*   `job_now: &nowdir ${now:%Y-%m-%d}/${now:%H:%M:%S}`: This defines a variable `job_now` that includes the current date and time.
*   `hydra:run:dir: ...`: This tells Hydra to create a new directory for each run. The path is constructed using other configuration values like `paths.log_dir`, `season`, `batch_id`, and the `job_now` timestamp.

For example, a run for `batch_id: MD_2025-05-02` on August 15th, 2024, at 2:30 PM might automatically save its logs and Hydra's own internal files into a path like:
`/home/user/SemiF-Preprocessing/.logs/cool_season_covers_2024_2025/MD_2025-05-02/2024-08-15/14:30:00/`

This is incredibly helpful for:
*   Keeping logs from different runs separate.
*   Reproducing results, as you know exactly which configuration was used for which output (Hydra saves the effective configuration in this directory).

## Solving Our Use Cases with Hydra

Let's revisit our earlier scenarios:

1.  **Adjusting JPG brightness (or `resize_factor`):**
    *   **Option A (Permanent Change):** Edit `conf/config.yaml`, find the `raw2jpg` section, and change `resize_factor: 0.25` to `resize_factor: 0.1`. Save the file. The next time you run `python main.py`, it will use 0.1.
    *   **Option B (Temporary Change for One Run):** Don't edit any files. Simply run:
        ```bash
        python main.py raw2jpg.resize_factor=0.1
        ```

2.  **Processing a new batch ("Field_B_cloudy_day") from a different location:**
    *   Many paths in `SemiF-Preprocessing` are already set up to use the `batch_id`. For instance, `paths.batch_dir` is defined as `${paths.developed_dir}/${batch_id}` in `conf/paths/default.yaml`.
    *   So, often, just changing the `batch_id` is enough:
        ```bash
        python main.py batch_id=Field_B_cloudy_day
        ```
        This command will make Hydra use "Field_B_cloudy_day" wherever `${batch_id}` is referenced in the configuration, effectively changing many input/output paths automatically.
    *   If "Field_B_cloudy_day" also needs a different `resize_factor` and only the `correct` mode:
        ```bash
        python main.py batch_id=Field_B_cloudy_day modes=[correct] raw2jpg.resize_factor=0.3
        ```
    *   For more complex changes (e.g., if the base `developed_dir` itself needs to change for Field B), you could:
        1.  Create a new configuration group file, say `conf/paths/field_b_setup.yaml`, with the specific path settings for Field B.
        2.  Then, you could tell Hydra to use this specific path configuration from the command line:
            ```bash
            python main.py batch_id=Field_B_cloudy_day paths=field_b_setup
            ```
            This temporarily overrides the `paths: default` from `conf/config.yaml` for this run.

## Under the Hood: How Hydra Assembles Your Configuration

When you run `python main.py batch_id=NEW_BATCH raw2jpg.resize_factor=0.1`, here's a simplified view of what Hydra does behind the scenes:

```mermaid
sequenceDiagram
    participant User
    participant "main.py (@hydra.main)"
    participant HydraCore as Hydra Core
    participant "config.yaml"
    participant "paths/default.yaml (via defaults)"
    participant "asfm/default.yaml (via defaults)"

    User->>"main.py (@hydra.main)": Executes `python main.py batch_id=NEW_BATCH raw2jpg.resize_factor=0.1`
    "main.py (@hydra.main)"->>HydraCore: Initialization (triggered by @hydra.main decorator)
    HydraCore->>"config.yaml": Load base config (config_name="config")
    "config.yaml"-->>HydraCore: Returns content (batch_id, modes, raw2jpg settings, defaults list)
    
    Note over HydraCore: Processing 'defaults' list from config.yaml...
    HydraCore->>"paths/default.yaml (via defaults)": Load 'paths' group
    "paths/default.yaml (via defaults)"-->>HydraCore: Returns path-specific settings
    
    HydraCore->>"asfm/default.yaml (via defaults)": Load 'asfm' group
    "asfm/default.yaml (via defaults)"-->>HydraCore: Returns ASFM-specific settings
    
    Note over HydraCore: All base and default group files loaded.
    HydraCore->>HydraCore: Merge all loaded configurations into one structure.
    HydraCore->>HydraCore: Apply command-line overrides (batch_id=NEW_BATCH, raw2jpg.resize_factor=0.1)
    
    HydraCore-->>"main.py (@hydra.main)": Provides final 'cfg' object
    "main.py (@hydra.main)"->>"main.py (@hydra.main)": Calls decorated main_function(cfg) with the complete configuration
```

1.  **You run the script** with command-line arguments.
2.  The `@hydra.main` decorator in `main.py` hands control to **Hydra Core**.
3.  Hydra Core reads the main **`config.yaml`** file (because `@hydra.main` specified `config_path="conf", config_name="config"`).
4.  It looks at the `defaults` list in `config.yaml`. For each entry (like `paths: default` or `asfm: default`), it loads the corresponding YAML file (e.g., `conf/paths/default.yaml`, `conf/asfm/default.yaml`).
5.  Hydra **merges** all these pieces: the settings from `config.yaml` itself, plus all the settings from the files loaded via the `defaults` list. It intelligently combines them into a single, structured configuration. If there are overlapping settings, there are rules for which one wins (often, more specific ones or later ones in the list).
6.  Then, Hydra applies any **command-line overrides**. These always take the highest precedence. So, `batch_id=NEW_BATCH` will replace whatever `batch_id` was in `config.yaml`, and `raw2jpg.resize_factor=0.1` will replace the value from `config.yaml`.
7.  Finally, Hydra creates the `cfg` object containing this complete, final configuration and passes it to your `main` function in `main.py`.

This process ensures that your script always gets a consistent and complete set of parameters, assembled from various organized files and customizable on the fly.

## Conclusion

You've now learned how `SemiF-Preprocessing` uses **Hydra** to manage all its settings, acting as a flexible and powerful "master control panel." Key takeaways:

*   **Settings are External:** Configurations live in human-readable **YAML files** (mainly `conf/config.yaml` and files in its subdirectories), not in the Python code.
*   **Structured and Organized:** Settings can be grouped into different files (e.g., for `paths`, `asfm`) and composed together.
*   **Easy Overrides:** You can change settings for a single run directly from the **command line** without editing files.
*   **Accessible in Code:** The `@hydra.main` decorator loads everything into a `cfg` object, making settings easily usable in your Python functions (e.g., `cfg.batch_id`, `cfg.raw2jpg.resize_factor`).
*   **Reproducibility:** Hydra helps manage output directories and saves the configuration for each run, making your work more traceable.

Understanding Hydra is key to customizing and running the `SemiF-Preprocessing` pipeline for different datasets and experimental needs. It gives you fine-grained control over the "image processing factory" without needing to be a Python programming expert.

Now that we understand how settings for things like file paths are managed, let's dive deeper into how the project actually handles the data on disk and synchronizes it from various sources.

Next up: [Chapter 3: Data Synchronization and Path Management](03_data_synchronization_and_path_management_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
