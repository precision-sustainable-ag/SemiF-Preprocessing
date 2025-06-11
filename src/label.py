import logging
import hydra
from omegaconf import DictConfig

# Import the task functions
from src.tasks.label_utils.detect_plants import main as detect_plants
from src.tasks.label_utils.merge_overlapping_bboxes import main as merge_overlapping_bboxes
from src.tasks.label_utils.remap_labels import main as remap_labels
from src.tasks.label_utils.assign_species import main as assign_species
from src.tasks.label_utils.assign_rates import main as assign_rates

from src.utils.artifact_utils import artifact_updater

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "detect_plants": artifact_updater("detect_plants")(detect_plants),
    "merge_overlapping_bboxes": artifact_updater("merge_overlapping_bboxes")(merge_overlapping_bboxes),
    "remap_labels": artifact_updater("remap_labels")(remap_labels),
    "assign_species": artifact_updater("assign_species")(assign_species),
    "assign_rates": artifact_updater("assign_rates")(assign_rates),
}


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting labeling tasks...")
    
    task_dict = cfg.tasks.label

    for task in task_dict:

        if task in TASK_REGISTRY:
            log.info(f"Running task {task}")
            try:
                TASK_REGISTRY[task](cfg)
            except Exception as e:
                log.error(f"Error running task {task}: {e}")
                raise

        else:
            log.error(f"Task {task} not found in labeling task registry")
            raise ValueError(f"Task {task} not found in labeling task registry")
    
    log.info("Labeling complete.")
    return

if __name__ == "__main__":
    main()