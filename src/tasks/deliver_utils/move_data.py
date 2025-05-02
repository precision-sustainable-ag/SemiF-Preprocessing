import logging
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.utils.utils import find_lts_dir
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

class CleanUpLocalTemp:
    """Downloads a specific image batch identified by batch_id."""
    def __init__(self, cfg, batch_id: str):
        self.cfg = cfg
        self.batch_id = batch_id

        self.local_batch_dir = Path(cfg.paths.batch_dir)
        self.lts_batch_dir = find_lts_dir(batch_id, cfg.paths.lts_locations, local=False, developed=True, dngs=False, jpgs=True) / "semifield-developed-images" / batch_id
        self.setup_src_dst_paths()

    def setup_src_dst_paths(self):
        # Source paths
        self.src_metadata = self.local_batch_dir / "metadata"
        self.src_cam_references = Path(self.cfg.paths.refs)
        self.src_log_path = Path(HydraConfig.get().runtime.output_dir) / f"{self.cfg.batch_id}.log"
        self.src_inspection_dir = Path(self.cfg.paths.inspection_dir)
        
        # Destination paths
        self.dst_metadata = self.lts_batch_dir / "metadata"
        self.dst_metadata.mkdir(parents=True, exist_ok=True)

        self.dst_cam_references = self.lts_batch_dir / "reference"
        self.dst_cam_references.mkdir(parents=True, exist_ok=True)

        self.dst_inspection_dir = self.lts_batch_dir / "inspection"
    
    def move_data(self):
        """Move the metadata and inspection results to the LTS directory."""
        # Move metadata
        if self.src_metadata.exists():
            shutil.copytree(str(self.src_metadata), str(self.dst_metadata), dirs_exist_ok=True)
            log.info(f"Copied {self.src_metadata} to {self.dst_metadata}")
        else:
            log.warning(f"Metadata directory {self.src_metadata} does not exist. Skipping move.")

        # Move camera references
        if self.src_cam_references.exists():
            shutil.copytree(str(self.src_cam_references), str(self.dst_cam_references), dirs_exist_ok=True)
            log.info(f"Copied {self.src_cam_references} to {self.dst_cam_references}")
        else:
            log.warning(f"Camera references directory {self.src_cam_references} does not exist. Skipping move.")

        # Move inspection results
        if self.src_inspection_dir.exists():
            # Copy the log file to the inspection directory first
            shutil.copy(str(self.src_log_path), str(self.src_inspection_dir))
            # Copy the inspection directory to the LTS directory
            shutil.copytree(str(self.src_inspection_dir), str(self.dst_inspection_dir), dirs_exist_ok=True)
            log.info(f"Copied {self.src_inspection_dir} to {self.dst_inspection_dir}")
        else:
            log.warning(f"Inspection results file {self.src_inspection_dir} does not exist. Skipping move.")
        

    def can_remove_local_dir(self):
        """ Check that all the files in the temp directories are in the LTS directories. """
        can_remove_local_dir = False
        temp_metadata = [mdata for mdata in self.local_batch_dir.glob("metadata/*.json")]
        
        lts_metadata = [mdata for mdata in self.lts_batch_dir.glob("metadata/*.json")]
        
        lts_cam_references_exists = len(list(self.dst_cam_references.glob("*.csv"))) == 5
        lts_inspection_results_exists = self.dst_inspection_dir.exists()
        
        metadata_diff = len(temp_metadata) - len(lts_metadata)
        if metadata_diff > 0 or not lts_cam_references_exists or not lts_inspection_results_exists:
            if metadata_diff:
                log.error(f"Metadata files in local batch directory not found in LTS directory: {metadata_diff}")
            if not lts_cam_references_exists:
                log.error(f"Camera references not found in LTS directory: {self.dst_cam_references}")
            if not lts_inspection_results_exists:
                log.error(f"Inspection results not found in LTS directory: {self.dst_inspection_dir}")
            log.error(f"Local batch directory {self.local_batch_dir} cannot be removed.")
            can_remove_local_dir = False
        else:
            log.info(f"All files in temp directories are in the LTS directories for batch {self.batch_id}.")
            can_remove_local_dir = True
        return can_remove_local_dir
    
    def cleanup_temp(self):
        """Remove the batch_id folder from the temp directory."""
        # Move the data to the LTS directory
        self.move_data()
        log.info(f"Moved data for batch {self.batch_id} to LTS directory.")
        # Check if we can remove the local directory
        can_remove = self.can_remove_local_dir()
        
        # Remove the local batch directory if all files are in the LTS directory
        if not can_remove:
            log.error(f"Cannot remove temp directories for batch {self.batch_id}. Exiting.")
            return
        else:
            try:
                # Remove all the subfoolders and their contents except the inspection folder
                local_batch_dir_contents = self.local_batch_dir.glob("*")
                for item in local_batch_dir_contents:
                    if item.is_dir() and item.name != "inspection":
                        shutil.rmtree(item)
                        log.info(f"Removed {item}.")
                    elif item.is_file():
                        item.unlink()
                        log.info(f"Removed {item}.")
                
                # Remove only the prediction images folder in the inspection directory
                prediction_images = self.local_batch_dir / "inspection" / "prediction_images"
                shutil.rmtree(prediction_images)
                log.info(f"Removed {prediction_images}.")
                
            except Exception as e:
                log.error(f"Failed to remove temp directories for batch {self.batch_id}: {e}")
                return

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting cleanup of local temp directories.")
    batch_id = cfg.batch_id
    try:
        cleaner = CleanUpLocalTemp(cfg, batch_id)
        cleaner.cleanup_temp()
        log.info("Finished cleanup of local temp directories.")
    except Exception as e:
        log.error(f"Error during cleanup: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    
