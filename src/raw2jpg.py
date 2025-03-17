import logging
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
import random
import hydra
from omegaconf import DictConfig
from src.raw2dng import RawToDNGConverter
from src.dng2jpg import DNGToJpgConverter
from src.utils.utils import find_lts_dir, find_raw_dir
import cv2
import shutil

log = logging.getLogger(__name__)

class Raw2Jpg:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.batch_id = self.cfg.batch_id
        self._initialize_paths()

        # Sampling and cleanup settings
        self.sample_count = cfg.raw2jpg.jpg_samples
        self.sample_resize_factor = cfg.raw2jpg.resize_factor
        self.remove_dngs = cfg.raw2jpg.remove_dngs
        self.max_workers = cfg.max_workers
    
    def _initialize_paths(self):
        # Local data directory
        self.local_data_dir = Path(self.cfg.paths.data_dir)
        # LTS directory
        self.lts_locations = self.cfg.paths.lts_locations
        self.lts_dir = find_lts_dir(self.batch_id, self.lts_locations, local=False)
        # PP3 and RT validation script
        self.pp3_path = Path(self.cfg.paths.image_development) / "dev_profiles" / f"{self.cfg.rt_pp3_name}.pp3"
        self.validate_rt_cli_script = Path(self.cfg.paths.scripts) / "validate_rawtherapee.sh"
        # Developed DNG (local) and JPG (LTS) directory
        self.developed_dng_dir = self.local_data_dir / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "dngs"
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)
        self.lts_jpg_dst = self.lts_dir / "semifield-developed-images" / self.batch_id / "images"
        self.lts_jpg_dst.mkdir(parents=True, exist_ok=True)
        # Local JPG sample directory
        self.local_sample_dir = self.local_data_dir / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "sample_images"
        self.local_sample_dir.mkdir(parents=True, exist_ok=True)
        # File masks
        self.file_masks = self.cfg.file_masks
        # Image Development paths
        self.rt_pp3_name = f"{self.cfg.rt_pp3_name}.pp3"
        self.local_pp3_path = Path(self.cfg.paths.image_development) / "dev_profiles" / self.rt_pp3_name
        self.profiles_backup = self.cfg.paths.img_dev_lts_bkp
        # RT validation script
        self.validate_rt_cli_script = Path(self.cfg.paths.scripts) / "validate_rawtherapee.sh"
        self.setup_profiling_paths()
        

    def setup_profiling_paths(self) -> None:
        """
        Sets up paths for the RawTherapee profile and validation script.
        """
        if not self.local_pp3_path.exists():
            self.pp3_path.parent.mkdir(parents=True, exist_ok=True)
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
        Get a list of RAW files to be converted to JPG.
        Returns:
            list[tuple[Path, bool]]: List of tuples containing the Path to the RAW file and a boolean indicating whether the file should be sampled.
        """
        self.raw_dir = find_raw_dir(self.local_data_dir, self.batch_id, self.lts_dir)
        raw_files = [file for file_mask in self.file_masks.raw_files for file in self.raw_dir.glob(f"*{file_mask}")]
        sampled_files = set(random.sample(raw_files, min(len(raw_files), self.sample_count)))
        raw_files = [(file, file in sampled_files) for file in raw_files]
        log.info(f"Found {len(raw_files)} RAW files.")
        return raw_files
    
    def remove_local_dng(self, dng_file: Path) -> None:
        """
        Removes a PNG file after conversion to JPG.
        
        Args:
            png_file (Path): Path to the PNG file to be removed.
        """
        if dng_file.suffix.lower() == ".dng":
            dng_file.unlink()
            log.info(f"Removed {dng_file.name}")
        else:
            log.warning(f"Cannot remove {dng_file} as it is not a DNG file.")

    def save_local_sample(self, original_path: str, output_path:str) -> None:
        """Save a fast, low-quality version for process verification"""
        img = cv2.imread(original_path)

        # Save with minimal quality for small file size
        img = cv2.resize(img, (0, 0), fx=self.sample_resize_factor, fy=self.sample_resize_factor)
        cv2.imwrite(output_path, img)
        return

    def convert_raw_to_jpg(self, raw_file_tuple: tuple) -> bool:
        """
        Convert a RAW file to a JPG file. Converts the RAW file to DNG first, then to JPG using Rawtherapee.
        Returns:
            bool: True if the conversion was successful, False otherwise.
        """
        # Convert RAW to DNG
        raw_file, to_inspect = raw_file_tuple
        raw2dng = RawToDNGConverter(self.cfg.dng_tags, self.batch_id, self.lts_dir, self.developed_dng_dir)
        raw_data = raw2dng.load_raw_image(raw_file)
        dng_tags = raw2dng.configure_dng_tags()
        dng_file = raw2dng.convert_to_dng(raw_data, dng_tags, raw_file)
        log.info(f"Converted RAW to DNG: {raw_file.name} -> {dng_file.name}")
        
        # Convert DNG to JPG
        jpg_output_path = self.lts_jpg_dst / f"{dng_file.stem}.jpg"
        dng2jpg = DNGToJpgConverter(dng_file, jpg_output_path, self.pp3_path, self.validate_rt_cli_script)
        rt_cli = dng2jpg.validate_rawtherapee()
        is_converted = dng2jpg.convert(rt_cli)
        
        # Save a low-quality sample for inspection
        if is_converted:
            log.info(f"Converted DNG to JPG: {dng_file.name} -> {jpg_output_path.name}")
            if to_inspect:
                local_sample_path = self.local_sample_dir / f"{dng_file.stem}.jpg"
                self.save_local_sample(str(jpg_output_path), str(local_sample_path))
                log.info(f"Saved low-quality sample: {local_sample_path.name}")
        else:
            log.warning(f"Failed to convert {dng_file.name} to JPG")
        
        # Clean up by removing the temporary DNG file
        if self.remove_dngs:
            self.remove_local_dng(dng_file)
        
        return is_converted

    def process_files(self) -> None:
        raw_files = self.get_raw_files()

        # Validate RawTherapee profile and CLI script
        dng2jpg = DNGToJpgConverter(None, None, self.pp3_path, self.validate_rt_cli_script)
        rt_cli = dng2jpg.validate_rawtherapee()
        if not rt_cli:
            log.error("RawTherapee validation failed. Exiting.")
            return

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.convert_raw_to_jpg, args): args for
                       args in raw_files}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error processing {futures[future]}: {e}")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for RAW to JPG conversion."""
    log.info(f"Starting RAW to JPG conversion for batch {cfg.batch_id}.")
    converter = Raw2Jpg(cfg)
    converter.process_files()
    log.info("RAW to JPG conversion completed.")


if __name__ == "__main__":
    main()
