"""
This script handles the conversion of RAW image files to JPGs for a given batch. It:
- Converts RAW files to DNG using a custom pipeline.
- Converts DNGs to JPGs using RawTherapee.
- Saves resized sample images for verification.
- Optionally deletes intermediate DNG files.
"""

import logging
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

from src.dng2jpg import DNGToJpgConverter
from src.raw2dng import RawToDNGConverter
from src.utils.utils import find_lts_dir, find_raw_dir

log = logging.getLogger(__name__)

class Raw2Jpg:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = self.cfg.batch_id
        

        # Sampling and cleanup settings
        self.sample_count = cfg.raw2jpg.jpg_samples
        self.sample_resize_factor = cfg.raw2jpg.resize_factor
        self.remove_dngs = cfg.raw2jpg.remove_dngs
        self.max_workers = cfg.max_workers

        self._initialize_paths()
    
    def _initialize_paths(self):
        """Initialize and create all necessary directory paths and files."""
        # Local data directory
        self.local_data_dir = Path(self.cfg.paths.data_dir)
        # LTS directory
        self.lts_locations = self.cfg.paths.lts_locations
        self.lts_dir = Path(find_lts_dir(self.batch_id, self.lts_locations, local=False))
        # Developed DNG (local) and JPG (LTS) directory
        self.developed_dng_dir = self.local_data_dir / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "dngs"
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)
        self.lts_jpg_dst = self.lts_dir / "semifield-developed-images" / self.batch_id / "images"
        self.lts_jpg_dst.mkdir(parents=True, exist_ok=True)
        # Local JPG sample directory
        self.lts_sample_dir = self.lts_dir / "semifield-developed-images" / self.batch_id / "preprocessing_samples"
        self.lts_sample_dir.mkdir(parents=True, exist_ok=True) if self.sample_count > 0 else None
        # File masks
        self.file_masks = self.cfg.file_masks
        # Image Development paths (.pp3)
        self.rt_pp3_name = f"{self.cfg.rt_pp3_name}.pp3"
        self.local_pp3_path = Path(self.cfg.paths.image_development) / "dev_profiles" / self.rt_pp3_name
        self.profiles_backup = self.cfg.paths.img_dev_lts_bkp
        # RT validation script
        self.validate_rt_cli_script = Path(self.cfg.paths.scripts) / "validate_rawtherapee.sh"
        self.setup_profiling_paths()
        # CCM Path
        self.ccm_name = f"{self.cfg.ccm_name}.npy"
        self.local_ccm_path = Path(self.cfg.paths.image_development) / "color_matrices" / self.ccm_name

        

    def setup_profiling_paths(self) -> None:
        """
        Ensures RawTherapee profile and validation script exist, copying from backup if needed.
        """
        if not self.local_pp3_path.exists():
            self.local_pp3_path.parent.mkdir(parents=True, exist_ok=True)
            log.warning(f"RawTherapee profile not found locally, copying from {self.profiles_backup}")
            pp3_backup_path = Path(self.profiles_backup) / "dev_profiles" / self.rt_pp3_name
            shutil.copy(pp3_backup_path, self.local_pp3_path)
        # sanity check
        if not self.local_pp3_path.exists():
            log.error(f"RawTherapee profile not found")
            raise FileNotFoundError(f"RawTherapee profile still not found: {self.local_pp3_path.name}")

        if not self.validate_rt_cli_script.exists():
            log.error(
                f"RawTherapee CLI validation script not found: {self.validate_rt_cli_script.name}")
            raise FileNotFoundError(
                f"RawTherapee CLI validation script not found: {self.validate_rt_cli_script.name}")

    def get_raw_files(self) -> list[tuple[Path, bool]]:
        """
        Retrieve all raw image files for the batch and sample a subset for quality check.

        Returns:
            list[tuple[Path, bool]]: List of file paths and flags for sample resizing.
        """
        self.raw_dir = find_raw_dir(self.local_data_dir, self.batch_id, self.lts_dir)
        raw_files = [file for file_mask in self.file_masks.raw_files for file in self.raw_dir.glob(f"*{file_mask}")]
        sampled_files = set(random.sample(raw_files, min(len(raw_files), self.sample_count)))
        raw_files = [(file, file in sampled_files) for file in raw_files]
        log.info(f"Found {len(raw_files)} RAW files.")
        return raw_files
    
    def remove_local_dng(self, dng_file: Path) -> None:
        """
        Remove a DNG file after conversion, if flagged.

        Args:
            dng_file (Path): Path to the DNG file.
        """
        if dng_file.suffix.lower() == ".dng":
            dng_file.unlink()
            log.debug(f"Removed {dng_file.name}")
        else:
            log.warning(f"Cannot remove {dng_file} as it is not a DNG file.")

    def save_lts_sample(self, original_path: str, output_path:str) -> None:
        """Save a low-resolution JPG sample for manual inspection."""
        img = cv2.imread(original_path)

        # Save with minimal quality for small file size
        img = cv2.resize(img, (0, 0), fx=self.sample_resize_factor, fy=self.sample_resize_factor)
        cv2.imwrite(output_path, img)
        return

    def convert_raw_to_jpg(self, raw_file_tuple: tuple) -> bool:
        """
        Full pipeline for converting RAW → DNG → JPG.

        Args:
            raw_file_tuple (tuple): (Path, bool) indicating file and whether it's a sample.

        Returns:
            bool: True if the conversion succeeded, else False.
        """
        
        raw_file, to_inspect = raw_file_tuple
        try:
            # Convert RAW to DNG
            raw2dng = RawToDNGConverter(self.cfg.dng_tags, self.batch_id, self.lts_dir, self.developed_dng_dir, self.local_ccm_path)
            raw_data = raw2dng.load_raw_image(raw_file)
            dng_tags = raw2dng.configure_dng_tags()
            dng_file = raw2dng.convert_to_dng(raw_data, dng_tags, raw_file)
            log.debug(f"Converted RAW to DNG: {raw_file.name} -> {dng_file.name}")
            
            # Convert DNG to JPG
            jpg_output_path = self.lts_jpg_dst / f"{dng_file.stem}.jpg"
            dng2jpg = DNGToJpgConverter(dng_file, jpg_output_path, self.local_pp3_path, self.validate_rt_cli_script)
            rt_cli = dng2jpg.validate_rawtherapee()
            is_converted = dng2jpg.convert(rt_cli)
            
            # Save a low-quality sample for inspection
            if is_converted:
                log.info(f"Converted RAW to JPG: {raw_file.name} -> {jpg_output_path.name}")
                if to_inspect:
                    lts_sample_path = self.lts_sample_dir / f"{dng_file.stem}.jpg"
                    self.save_lts_sample(str(jpg_output_path), str(lts_sample_path))
                    log.debug(f"Saved low-quality sample: {lts_sample_path.name}")
            else:
                log.warning(f"JPG conversion failed: {dng_file.name}")
            
            # Clean up by removing the temporary DNG file
            if self.remove_dngs:
                self.remove_local_dng(dng_file)
            
            return is_converted
        
        except Exception as e:
            log.exception(f"Failed to convert {raw_file.name}: {e}")
            return False
        

    def process_files(self) -> None:
        """
        Process all RAW files by converting them to JPG using multiprocessing.
        """
        raw_files = self.get_raw_files()

        # Validate RawTherapee profile and CLI script
        dng2jpg = DNGToJpgConverter(None, None, self.local_pp3_path, self.validate_rt_cli_script)
        rt_cli = dng2jpg.validate_rawtherapee()
        if not rt_cli:
            log.error("RawTherapee validation failed. Conversion aborted.")
            return

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.convert_raw_to_jpg, args): args for
                       args in raw_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result:
                        log.warning(f"Conversion failed for: {futures[future][0].name}")
                except Exception as e:
                    log.error(f"Exception occurred during processing {futures[future][0].name}: {e}")

            # Ensure all processes finish before proceeding
            executor.shutdown(wait=True)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for RAW to JPG conversion."""
    log.info(f"Starting RAW to JPG conversion for batch: {cfg.batch_id}")
    converter = Raw2Jpg(cfg)
    converter.process_files()
    log.info("RAW to JPG conversion completed.")


if __name__ == "__main__":
    main()
