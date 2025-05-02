import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
from src.tasks.correct_utils.raw2jpg import main as raw2jpg 
from src.tasks.correct_utils.update_exif import main as update_exif

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "raw2jpg": raw2jpg,
    "update_exif": update_exif,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting image correction tasks...")
    
    task_dict = cfg.tasks.correct

    for task, enabled in task_dict.items():

        if enabled:
            log.info(f"Running task {task}")

            if task in TASK_REGISTRY:
                log.info(f"Running task {task}")
                try:
                    TASK_REGISTRY[task](cfg)
                except Exception as e:
                    log.error(f"Error running task {task}: {e}")
                    raise

            else:
                log.error(f"Task {task} not found in correction task registry")
                raise ValueError(f"Task {task} not found in correction task registry")
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    log.info("Image correction complete.")

    return

if __name__ == "__main__":
    main()