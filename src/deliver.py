import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
from src.tasks.deliver_utils.inspect_images import main as inspect_images
from src.tasks.deliver_utils.move_data import main as move_data
from src.tasks.deliver_utils.report import main as report

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "inspect_images": inspect_images,
    "move_data": move_data,
    "report": report,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting moving data and reporting tasks...")
    
    task_dict = cfg.tasks.deliver

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
                log.error(f"Task {task} not found in delvier task registry")
                raise ValueError(f"Task {task} not found in deliver task registry")
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    
    log.info("Moving data and reporting complete.")
    return

if __name__ == "__main__":
    main()