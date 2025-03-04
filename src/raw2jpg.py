import logging
import random
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from src.png2jpg import PngToJpgConverter
from src.utils.preprocess import Preprocessor
from src.utils.utils import find_lts_dir, find_raw_dir

log = logging.getLogger(__name__)


class Raw2Jpg:

    def __init__(self, cfg: DictConfig):
        """
        Initializes the RAW to JPG converter with the provided configuration.
        
        Args:
            cfg (DictConfig): Hydra configuration object containing paths and processing parameters.
        """
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.remove_pngs = cfg.raw2jpg.remove_pngs
        self.local_data_dir = Path(self.cfg.paths.data_dir)
        self.max_workers = cfg.max_workers
        self.file_masks = cfg.file_masks

        # Locate long-term storage directory
        self.lts_dir = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations,
                                    local=False)

        # Set up image input and output directories
        (self.src_dir, self.temp_png_output_dir, self.jpg_output_dir,
         self.local_sample_dir) = self.setup_image_paths()

        # Set up profiling and color matrix paths
        self.pp3_path, self.validate_rt_cli_script = self.setup_profiling_paths()
        self.color_matrix_path = self.setup_color_matrix_path()

        # Load the color transformation matrix
        self.transformation_matrix = Preprocessor.load_transformation_matrix(
            self.color_matrix_path)

        # Sample paramaters
        self.sample_count = cfg.raw2jpg.jpg_samples
        self.sample_quality = cfg.raw2jpg.jpg_quality

    def setup_image_paths(self) -> tuple[Path, Path, Path, Path]:
        """
        Sets up paths for the RAW source directory, temporary PNG output directory, and final JPG output directory.
        
        Returns:
            tuple[Path, Path, Path, Path]: Source RAW directory, temporary PNG
            directory, JPG output directory, local directory to save samples.
        """
        src_raw_dir = find_raw_dir(self.local_data_dir, self.batch_id,
                                   self.lts_dir)
        temp_png_dir = Path(
            self.cfg.paths.data_dir) / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "pngs"
        temp_png_dir.mkdir(parents=True, exist_ok=True)
        jpg_output_dir = Path(
            self.lts_dir) / "semifield-developed-images" / self.batch_id / "images"
        jpg_output_dir.mkdir(parents=True, exist_ok=True)
        local_sample_dir = (Path(
            self.cfg.paths.data_dir) / self.lts_dir.name / "semifield-developed-images" / self.batch_id / "sample_jpgs")
        local_sample_dir.mkdir(parents=True, exist_ok=True)
        return src_raw_dir, temp_png_dir, jpg_output_dir, local_sample_dir

    def setup_profiling_paths(self) -> tuple[Path, Path]:
        """
        Sets up paths for the RawTherapee profile and validation script.
        
        Returns:
            tuple[Path, Path]: Paths to the RawTherapee profile (.pp3) and CLI validation script.
        """
        pp3_path = Path(
            self.cfg.paths.image_development) / "dev_profiles" / f"{self.cfg.png2jpg.rt_pp3_name}.pp3"
        if not pp3_path.exists():
            log.error(f"RawTherapee profile not found: {pp3_path}")
            raise FileNotFoundError(
                f"RawTherapee profile not found: {pp3_path}")

        validate_rt_cli_script = Path(
            self.cfg.paths.scripts) / "validate_rawtherapee.sh"
        if not validate_rt_cli_script.exists():
            log.error(
                f"RawTherapee CLI validation script not found: {validate_rt_cli_script}")
            raise FileNotFoundError(
                f"RawTherapee CLI validation script not found: {validate_rt_cli_script}")

        return pp3_path, validate_rt_cli_script

    def setup_color_matrix_path(self) -> Path:
        """
        Sets up the path to the color matrix file required for color correction.
        
        Returns:
            Path: Path to the color matrix file.
        """
        ccm_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.ccm
        color_matrix_path = Path(self.cfg.paths.image_development,
                                 "color_matrix", ccm_name + ".npz")
        if not color_matrix_path.exists():
            log.error(f"Color matrix file {color_matrix_path} not found.")
            raise FileNotFoundError(
                f"Color matrix file {color_matrix_path} not found.")

        return color_matrix_path

    def get_raw_files(self) -> list[Path]:
        """
        Retrieves RAW image files from the source directory.
        
        Returns:
            list[Path]: List of RAW image file paths.
        """
        raw_files = []
        for file_mask in self.file_masks.raw_files:
            raw_files.extend(list(self.src_dir.glob(f"*{file_mask}")))
        log.info(f"Found {len(raw_files)} RAW files for processing.")
        return raw_files

    def remove_local_png(self, png_file: Path) -> None:
        """
        Removes a PNG file after conversion to JPG.
        
        Args:
            png_file (Path): Path to the PNG file to be removed.
        """
        if png_file.suffix.lower() == ".png":
            png_file.unlink()
            log.info(f"Removed {png_file.name}")
        else:
            log.warning(f"Cannot remove {png_file} as it is not a PNG file.")

    def save_local_sample(self, original_path: str, output_path:str) -> None:
        """Save a fast, low-quality version for process verification"""
        img = cv2.imread(original_path)

        # Save with minimal quality for small file size
        cv2.imwrite(output_path, img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.sample_quality,
                     cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        return

    def convert_raw_to_jpg(self, args: tuple[Path, bool]) -> bool:
        """
        Converts a RAW image to a JPG format with preprocessing steps.
        
        Args:
            args(raw_file (Path), to_sample (bool)): Path to the RAW file,
            true if image should be saved for inspection
        
        Returns:
            bool: True if the conversion was successful, False otherwise.
        """

        # Apply preprocessing steps
        raw_file, to_inspect = args
        raw_array = Preprocessor.load_raw_image(raw_file, self.cfg)
        demosaiced = Preprocessor.demosaic_image(raw_array)
        gamma_corrected = Preprocessor.apply_gamma_correction(demosaiced,
                                                              gamma=1.1)  # gamma correction
        corrected_img = Preprocessor.apply_transformation_matrix(
            gamma_corrected, self.transformation_matrix)
        corrected_img = np.clip(corrected_img * 65535.0, 0, 65535).astype(
            np.uint16)  # clipping pixel values to 16-bit integer range
        corrected_image_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
        # Save the corrected image as a PNG file
        png_file = self.temp_png_output_dir / f"{raw_file.stem}.png"
        Preprocessor.save_image(corrected_image_bgr, png_file)

        # Convert the temporary PNG file to JPG
        output_jpg_path = self.jpg_output_dir / f"{raw_file.stem}.jpg"
        png2jpg_conv = PngToJpgConverter(png_file, output_jpg_path,
                                         self.pp3_path,
                                         self.validate_rt_cli_script)
        is_converted = png2jpg_conv.convert(png2jpg_conv.validate_rawtherapee())

        if is_converted:
            log.info(f"Successfully converted {png_file.name} to JPG")
            if to_inspect:
                sample_path = self.local_sample_dir / f"{png_file.stem}.jpg"
                self.save_local_sample(str(output_jpg_path), str(sample_path))
                log.info(f"Saved low quality sample for inspection: {sample_path}")
        else:
            log.warning(f"Failed to convert {png_file.name} to JPG")

        # Clean up the temporary PNG file
        if self.remove_pngs:
            self.remove_local_png(png_file)

        return is_converted

    def process_files(self) -> None:
        """
        Processes all RAW files by converting them to JPG using multiprocessing.
        """
        raw_files = self.get_raw_files()
        if not raw_files:
            log.warning("No RAW files found for processing.")
            return
        if len(raw_files) > self.sample_count:
            sampled_files = set(random.sample(raw_files, self.sample_count))
            raw_files = [(file, file in sampled_files) for file in raw_files]
        else:
            log.warning(f"Not enough images to sample, saving all")
            raw_files = [(file, True) for file in raw_files]

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
