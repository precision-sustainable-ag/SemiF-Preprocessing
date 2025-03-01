import numpy as np
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from concurrent.futures import as_completed, ProcessPoolExecutor
from src.utils.preprocess import Preprocessor
from src.utils.utils import find_lts_dir, find_raw_dir

log = logging.getLogger(__name__)

class BatchProcessor:
    """Coordinates the overall batch processing workflow, including file management,
    image processing, and downscaling."""

    def __init__(self, cfg: DictConfig):
        """Initializes the batch processor with configuration settings.
        
        Args:
            cfg (DictConfig): Configuration object containing paths and processing parameters.
        """
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.lts_dir = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=False)
        self.local_data_dir = Path(self.cfg.paths.data_dir)
        self.src_dir, self.output_dir = self.setup_paths()
    
    def setup_paths(self):
        """Sets up and validates required directories for processing.
        
        Returns:
            tuple: Paths to source, raw, output, and downscaled directories.
        """
        src_dir = find_raw_dir(self.local_data_dir, self.batch_id, self.lts_dir)
        
        if src_dir is None:
            log.error(f"No RAW directory found for batch {self.batch_id}.")
            raise FileNotFoundError("No RAW directory found for batch.")
        
        output_dir = Path(self.cfg.paths.data_dir) / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "pngs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the color matrix for color correction
        ccm_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.ccm
        self.color_matrix_path = Path(self.cfg.paths.image_development, "color_matrix", ccm_name + ".npz")
        
        if not self.color_matrix_path.exists():
            log.warning(f"Color matrix {self.color_matrix_path} not found. Using default color matrix.")
            self.color_matrix_path = Path(self.cfg.paths.image_development, "color_matrix", "default.npz")
        log.info(f"Using color matrix: {self.color_matrix_path}")
        
        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
        return src_dir, output_dir

    def load_transformation_matrix(self) -> np.ndarray:
        """Loads the color transformation matrix from file.
        
        Returns:
            np.ndarray: The loaded transformation matrix.
        """
        color_matrix_path = Path(self.color_matrix_path)
        if not color_matrix_path.exists():
            raise FileNotFoundError(f"Color matrix file {color_matrix_path} not found.")
        with np.load(color_matrix_path) as data:
            return data["matrix"]

    def get_raw_files(self):
        """Get all RAW files from the source directory."""
        all_raw_files = list(self.src_dir.glob("*.RAW")) + list(self.src_dir.glob("*.raw"))
        # Determine the largest file size to filter out incomplete or corrupted files
        # max_file_size = max(f.stat().st_size for f in all_raw_files)
        # raw_files = sorted([f for f in all_raw_files if f.stat().st_size == max_file_size])
        # raw_files = sorted(all_raw_files)
        log.info(f"Processing {len(all_raw_files)} RAW files.")
        return all_raw_files
    
    def process_files(self, transformation_matrix):
        """Processes RAW image files by applying demosaicing and color correction.
        
        Args:
            transformation_matrix (np.ndarray): Transformation matrix for color correction.
        """
        raw_files = sorted(self.get_raw_files())

        # Optionally filter files by specific names.
        # raw_files = [f for f in raw_files if f.stem in ["NC_1740166530"]] 
        if not raw_files:
            log.info("No raw files found.")
            return
   
        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = [
                executor.submit(Preprocessor.process_image, raw_file, self.cfg, transformation_matrix, self.output_dir)
                for raw_file in raw_files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except ValueError as e:
                    log.error(f"Error processing file: {e}")
                except KeyboardInterrupt:
                    log.info("Batch processing interrupted.")
                    executor.shutdown(wait=False, cancel_futures=True)
        
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for batch image processing."""
    log.info(f"Batch preprocessing started for batch {cfg.batch_id}.")
    batch_processor = BatchProcessor(cfg)
    transformation_matrix = batch_processor.load_transformation_matrix()
    batch_processor.process_files(transformation_matrix)
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
