# Chapter 4: Image Processing Task Module

In [Chapter 3: Data Synchronization and Path Management](03_data_synchronization_and_path_management_.md), we learned about the "logistics and warehousing" of our `SemiF-Preprocessing` project – how it finds data, keeps helper files updated, and stores results. We saw how data is moved to the "factory floor" for processing. Now, let's explore the "departments" on that factory floor: the **Image Processing Task Modules**.

Imagine our image processing pipeline as a large factory. This factory has different departments, each specializing in one stage of production:
*   One department for **cleaning** raw materials (e.g., correcting raw images).
*   One for **assembling** parts (e.g., creating 3D models from images).
*   One for **applying labels** (e.g., identifying and labeling plants in images).
*   And one for **packaging and shipping** (e.g., generating reports and delivering final results).

In `SemiF-Preprocessing`, these "departments" are called **Image Processing Task Modules**.

## What is an Image Processing Task Module?

An **Image Processing Task Module** is a logical grouping of related sub-tasks that together achieve a specific, major stage in the overall image preprocessing workflow.

Think of it like this:
*   **Module Name (Department Name):** `correct`, `asfm` (Auto Structure from Motion), `label`, `deliver`.
*   **Purpose (Department's Job):**
    *   `correct`: To perform initial image corrections, like converting from RAW format and adjusting colors.
    *   `asfm`: To create 3D models or orthomosaics from the images.
    *   `label`: To identify objects (like plants) in the images and draw bounding boxes.
    *   `deliver`: To package results, generate reports, and move data to final storage.
*   **Structure:** Each module has:
    *   Its own main entry point (a specific Python script, e.g., `src/correct.py`).
    *   A defined set of operations or *sub-tasks* it performs in sequence.

This modular approach helps keep our complex image processing pipeline organized, understandable, and manageable. Each "department" focuses on its specific responsibilities.

## How Task Modules are Activated in the Pipeline

Remember from [Chapter 1: Pipeline Orchestration](01_pipeline_orchestration_.md) how we tell the `main.py` script (our "General Contractor") which major stages to run? We do this using the `modes` list in our configuration file (`conf/config.yaml`).

Here's a snippet from `conf/config.yaml`:
```yaml
# conf/config.yaml (snippet)

# ... other settings ...

modes:
  # - sync      # This module would handle data synchronization
  - correct   # Activate the 'correct' image processing task module
  - asfm      # Activate the 'asfm' (3D modeling) task module
  - label     # Activate the 'label' (object detection) task module
  - deliver   # Activate the 'deliver' (results packaging) task module

# ... other settings ...
```
When `main.py` runs, it looks at this `modes` list. For each entry (like `correct`, `asfm`, etc.), it calls the corresponding Image Processing Task Module. If `correct` is listed before `label`, the image correction module will complete its work before the labeling module begins.

The main orchestrator (`main.py`) has a sort of "phonebook" (we called it `TASK_REGISTRY` in Chapter 1) that maps these mode names to the actual Python scripts for each module. For example:
*   `"correct"` maps to the main function in `src/correct.py`.
*   `"label"` maps to the main function in `src/label.py`.

## Inside a Task Module: A Mini-Pipeline of Sub-Tasks

Now, let's step inside one of these "departments." An Image Processing Task Module usually doesn't just do one single thing. It often has its own sequence of smaller, more specific steps, which we call **sub-tasks**.

For example, the `correct` module (our "image cleaning department") might need to:
1.  Convert images from the camera's RAW format to a more common format like DNG, then to JPG ([Chapter 5: Image File Conversion (RAW to JPG)](05_image_file_conversion__raw_to_jpg_.md)).
2.  Update the image metadata (EXIF data) with new information ([Chapter 6: EXIF Data Management](06_exif_data_management_.md)).

How does the module know which sub-tasks to run? Again, this is defined in our `conf/config.yaml` file, but in a different section, usually under `tasks`:

```yaml
# conf/config.yaml (snippet)

# ... (modes list as shown above) ...

tasks: # Defines sub-tasks for each module (mode)
  sync:
    - sync_from_remote 
  correct:
    - raw2jpg       # First sub-task for 'correct' module
    - update_exif   # Second sub-task for 'correct' module
  asfm:
    - autosfm       # Sub-task for 'asfm' module
  label:
    - detect_plants
    - merge_overlapping_bboxes
    - remap_labels
    - assign_species
  deliver:
    - inspect_images
    - move_data
    - report
```
When the main orchestrator (`main.py`) calls, for instance, the `correct` module, the `correct` module's code will look at `cfg.tasks.correct` (where `cfg` is our Hydra configuration object from [Chapter 2: Configuration Management (Hydra)](02_configuration_management__hydra__.md)). This gives it the list `['raw2jpg', 'update_exif']`. The `correct` module then runs these sub-tasks in that order.

### Example: The `src/correct.py` Module

Let's peek at a simplified version of what the `src/correct.py` file (the entry point for the `correct` module) might look like. This script is responsible for managing the sub-tasks of image correction.

```python
# src/correct.py (simplified for understanding)
import logging
from omegaconf import DictConfig # For accessing configuration

# Import the functions that perform the actual sub-tasks
from src.tasks.correct_utils.raw2jpg import main as raw2jpg_sub_task_function
from src.tasks.correct_utils.update_exif import main as update_exif_sub_task_function

log = logging.getLogger(__name__)

# A "phonebook" for sub-tasks within the 'correct' module
SUB_TASK_REGISTRY = {
    "raw2jpg": raw2jpg_sub_task_function,
    "update_exif": update_exif_sub_task_function,
}

# This 'main' function of the 'correct' module is called by the main orchestrator (main.py)
def main(cfg: DictConfig) -> None:
    log.info(f"Starting 'correct' module for batch: {cfg.batch_id}")
    
    # Get the list of sub-tasks for 'correct' from the config 
    # (e.g., ['raw2jpg', 'update_exif'])
    sub_tasks_to_run_list = cfg.tasks.correct 
    
    for sub_task_name in sub_tasks_to_run_list:
        if sub_task_name in SUB_TASK_REGISTRY:
            log.info(f"--- Running 'correct' sub-task: {sub_task_name} ---")
            
            # Look up the actual function for this sub-task
            actual_sub_task_to_call = SUB_TASK_REGISTRY[sub_task_name]
            
            # Call the sub-task function, passing the configuration
            actual_sub_task_to_call(cfg) 
            
            log.info(f"--- Finished 'correct' sub-task: {sub_task_name} ---")
        else:
            log.error(f"Unknown sub-task in 'correct' module: {sub_task_name}")
            # In a real scenario, this might stop the process or raise an error
            
    log.info(f"'Correct' module finished for batch: {cfg.batch_id}")

# Note: The actual src/correct.py file has a @hydra.main decorator.
# This allows it to be run as a standalone script (e.g., python src/correct.py)
# for testing or specific tasks. When called by the main.py orchestrator,
# this main(cfg) function is directly invoked with the already loaded 'cfg'.
```
In this simplified example:
1.  The `main(cfg)` function is the entry point for the `correct` module. It's called by the main pipeline orchestrator (`main.py`).
2.  It gets its list of sub-tasks to perform (e.g., `raw2jpg`, `update_exif`) from `cfg.tasks.correct`.
3.  It has its own `SUB_TASK_REGISTRY` to map sub-task names to the actual Python functions that do the work (like `raw2jpg_sub_task_function`).
4.  It loops through the list of sub-tasks and calls them one by one, passing along the `cfg` object so each sub-task has access to all necessary configurations.

Files like `src/label.py` and `src/deliver.py` have a similar structure: a `main(cfg)` function that takes the configuration, looks up its assigned sub-tasks from `cfg.tasks.<module_name>`, and runs them in sequence using its own internal sub-task registry.

## The Big Picture: How It All Connects

Let's visualize how the main orchestrator, a task module, and its sub-tasks interact when you run the pipeline to process a batch, specifically asking for the `correct` mode:

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator (main.py)
    participant Config (e.g., config.yaml)
    participant Correct_Module (e.g., src/correct.py)
    participant Raw2JPG_SubTask (in correct_utils)
    participant UpdateEXIF_SubTask (in correct_utils)

    User->>Orchestrator (main.py): Run pipeline with batch_id='XYZ', modes=['correct']
    Orchestrator (main.py)->>Config (e.g., config.yaml): Load all settings (including modes, tasks.correct)
    Config (e.g., config.yaml)-->>Orchestrator (main.py): 'cfg' object created
    
    Note over Orchestrator (main.py): Found 'correct' in cfg.modes.
    Orchestrator (main.py)->>Correct_Module (e.g., src/correct.py): Call its main(cfg) function
    
    Correct_Module (e.g., src/correct.py)->>Config (e.g., config.yaml): Access cfg.tasks.correct (e.g., ['raw2jpg', 'update_exif'])
    Note over Correct_Module (e.g., src/correct.py): Now knows sub-tasks to run.
    
    Correct_Module (e.g., src/correct.py)->>Raw2JPG_SubTask (in correct_utils): Call main(cfg) for 'raw2jpg'
    Note right of Raw2JPG_SubTask (in correct_utils): Performs RAW to JPG conversion.
    Raw2JPG_SubTask (in correct_utils)-->>Correct_Module (e.g., src/correct.py): 'raw2jpg' sub-task done
    
    Correct_Module (e.g., src/correct.py)->>UpdateEXIF_SubTask (in correct_utils): Call main(cfg) for 'update_exif'
    Note right of UpdateEXIF_SubTask (in correct_utils): Updates image metadata.
    UpdateEXIF_SubTask (in correct_utils)-->>Correct_Module (e.g., src/correct.py): 'update_exif' sub-task done
    
    Correct_Module (e.g., src/correct.py)-->>Orchestrator (main.py): 'correct' module processing finished
    Orchestrator (main.py)-->>User: Pipeline for 'correct' mode complete.
```

1.  **You (User)** start the pipeline, specifying you want to run the `correct` mode.
2.  The **Orchestrator (`main.py`)** loads the **Configuration (`config.yaml`)** into a `cfg` object.
3.  The Orchestrator sees `correct` in the `cfg.modes` list and calls the `main(cfg)` function of the **Correct_Module (`src/correct.py`)**.
4.  The **Correct_Module** looks at `cfg.tasks.correct` to get its list of sub-tasks (e.g., `['raw2jpg', 'update_exif']`).
5.  It then calls the first sub-task, **Raw2JPG_SubTask**, which does its work (e.g., converting images).
6.  Once that's done, it calls the next sub-task, **UpdateEXIF_SubTask**, which does its work (e.g., updating metadata).
7.  After all its sub-tasks are complete, the **Correct_Module** finishes and returns control to the **Orchestrator**.

## Why This Modular Structure?

Organizing the pipeline into these Task Modules, each with its own sub-tasks, offers several benefits:

*   **Clarity and Organization:** The entire complex process is broken down into manageable, understandable stages ("departments") and further into specific steps (sub-tasks).
*   **Focus:** If you're working on improving plant detection (part of the `label` module), you don't need to get into the details of RAW image conversion (part of the `correct` module).
*   **Flexibility:**
    *   You can easily change the order of major processing stages by reordering the `modes` list in `config.yaml`.
    *   You can skip an entire stage by commenting out a mode (e.g., `# - asfm`).
    *   Within a module, you can change the order of sub-tasks or skip some by modifying the list under `cfg.tasks.<module_name>`.
*   **Reusability:** A well-defined module (like `correct`) could potentially be reused in other pipelines if needed.
*   **Easier Debugging:** If something goes wrong, it's often easier to pinpoint which module or even which sub-task is causing the issue.

## Conclusion

You've now learned about **Image Processing Task Modules** in `SemiF-Preprocessing`. These are like specialized departments in our image processing factory, each responsible for a distinct stage of the workflow (e.g., `correct`, `asfm`, `label`, `deliver`).

Key takeaways:
*   Task Modules are activated based on the `modes` list in `conf/config.yaml`.
*   Each module (e.g., `src/correct.py`) has an entry point (its `main(cfg)` function) called by the main pipeline orchestrator.
*   Modules manage their own sequence of *sub-tasks*, which are also defined in `conf/config.yaml` under the `tasks:` section (e.g., `cfg.tasks.correct`).
*   This modular design makes the pipeline organized, flexible, and easier to manage.

Now that we understand how these "departments" operate by running a series of sub-tasks, let's dive into one of the most common and fundamental sub-tasks you'll encounter: converting image files from their raw camera format into a more usable format like JPG. This is often the very first step within the `correct` module.

Next up: [Chapter 5: Image File Conversion (RAW to JPG)](05_image_file_conversion__raw_to_jpg_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)
