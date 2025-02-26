import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import random
from utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class FileManager:
    """Handles file-related operations like copying and filtering."""

    @staticmethod
    def get_sampled_files(raw_files: list, sample_size: int, strategy: str = "random") -> list:
        """Samples a subset of files from a directory based on a given strategy."""
        if sample_size and len(raw_files) > sample_size:
            if strategy == "random":
                return random.sample(raw_files, sample_size)
            elif strategy == "first":
                return raw_files[:sample_size]
            elif strategy == "last":
                return raw_files[-sample_size:]
            elif strategy == "middle":
                mid = len(raw_files) // 2
                half_size = sample_size // 2
                return raw_files[mid - half_size:mid + half_size]
            else:
                raise ValueError(f"Unknown sample strategy: {strategy}")
        return raw_files


class ImageProcessor:
    """Handles image processing tasks like demosaicing and color correction."""

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Adjusts the gamma of a 16-bit image using a lookup table.
        """
        log.info(f"Adjusting gamma of image of shape: {image.shape}")
        inv_gamma = 1.0 / gamma
        table = ((np.arange(0, 65536, dtype=np.float32) / 65535.0) ** inv_gamma) * 65535.0
        table = table.astype(np.uint16)
        return table[image]

    @staticmethod    
    def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        # Ensure image is in [0, 1]
        corrected = np.power(image, 1 / gamma)
        return corrected
    
    @staticmethod
    def apply_log_compression(image: np.ndarray) -> np.ndarray:
        # Add a small constant to avoid log(0) issues
        epsilon = 1e-6
        compressed = np.log1p(image + epsilon) / np.log1p(1 + epsilon)
        return compressed

    @staticmethod
    def apply_transformation_matrix(source_img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """Apply a transformation matrix to the source image to correct its color space."""
        if transformation_matrix.shape != (9, 9):
            log.error("Transformation matrix must be a 9x9 matrix.")
            return None

        if source_img.ndim != 3:
            log.error("Source image must be an RGB image.")
            return None
        
        log.info(f"Applying transformation matrix to image of shape: {source_img.shape}")
        # Extract color channel coefficients from transformation matrix
        red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)

        # Normalize the source image to the range [0, 1]
        source_dtype = source_img.dtype
        max_val = np.iinfo(source_dtype).max if source_dtype.kind == 'u' else 1.0
        source_flt = source_img.astype(np.float64) / max_val
        # Normalize and compress dynamic range before transformation
        # source_compressed = ImageProcessor.apply_gamma_correction(source_flt, gamma=1.05)
        # source_compressed = ImageProcessor.apply_log_compression(source_flt)
        # tonemapReinhard = cv2.createTonemapReinhard(gamma=gamma)
        # source_compressed = tonemapReinhard.process(source_flt.astype(np.float32))
        source_r, source_g, source_b = cv2.split(source_flt)
        
        # Compute powers of source image
        source_b2, source_b3 = source_b**2, source_b**3
        source_g2, source_g3 = source_g**2, source_g**3
        source_r2, source_r3 = source_r**2, source_r**3
        
        # Compute color transformation
        b = (source_r * blue[0] + source_g * blue[1] + source_b * blue[2] +
            source_r2 * blue[3] + source_g2 * blue[4] + source_b2 * blue[5] +
            source_r3 * blue[6] + source_g3 * blue[7] + source_b3 * blue[8])
        
        g = (source_r * green[0] + source_g * green[1] + source_b * green[2] +
            source_r2 * green[3] + source_g2 * green[4] + source_b2 * green[5] +
            source_r3 * green[6] + source_g3 * green[7] + source_b3 * green[8])
        
        r = (source_r * red[0] + source_g * red[1] + source_b * red[2] +
            source_r2 * red[3] + source_g2 * red[4] + source_b2 * red[5] +
            source_r3 * red[6] + source_g3 * red[7] + source_b3 * red[8])

        corrected_img = cv2.merge([r, g, b])
        
        return corrected_img

    
    @staticmethod
    def demosaic_image(raw_file: Path, cfg: DictConfig):
        """Demosaics a RAW image file using bilinear interpolation."""
        log.info(f"Demosaicing: {raw_file}")
        im_height, im_width = cfg.colorchecker.height, cfg.colorchecker.width

        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))

        # demosaiced = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")        
        demosaiced = cv2.cvtColor(nparray, cv2.COLOR_BayerBG2RGB_EA)
        demosaiced = demosaiced.astype(np.float64) / 65535.0
        
        return demosaiced

    @staticmethod
    def resize_image(image: np.ndarray, downscale_factor: float):
        """Downscales an image file by a given factor."""
        height, width = image.shape[:2]
        new_height = int(height * downscale_factor)
        new_width = int(width * downscale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Path):
        """Saves an image to disk."""
        # cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # WRite as PNG
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        log.info(f"Saved: {output_path}")
    
    @staticmethod
    def remove_local_raw(local_raw_path: Path):
        """Removes an image file from disk."""
        # Sanity check
        if "research-project" in str(local_raw_path) or "screberg" in str(local_raw_path):
            log.warning("Refusing to remove file from LTS research-project directory.")
            return
        local_raw_path.unlink()
        log.info(f"Removed raw image: {local_raw_path}")

    @staticmethod
    def process_image(raw_file: Path, cfg: DictConfig, transformation_matrix, output_dir: Path):
        log.info(f"Processing: {raw_file}")

        # Demosaic
        demosaiced_rgb = ImageProcessor.demosaic_image(raw_file, cfg)
        
        # Apply color correction
        corrected_img = ImageProcessor.apply_transformation_matrix(demosaiced_rgb, transformation_matrix) 
        corrected_img = corrected_img * 65535.0

        corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)
        
        corrected_image_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
        

        # Convert to 8-bit or 16-bit RGB
        # rgb_bit_image = (corrected_image_rgb * 255).astype(np.uint8) if cfg.inspect_v31.bit_depth == 8 else (corrected_image_rgb * 65535).astype(np.uint16)

        
        # Save the final image
        ImageProcessor.save_image(corrected_image_bgr, output_dir / f"{raw_file.stem}.png")

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
        self.src_dir, self.output_dir = self.setup_paths()

    def setup_paths(self):
        """Sets up and validates required directories for processing.
        
        Returns:
            tuple: Paths to source, raw, output, and downscaled directories.
        """
        self.lts_dir_name = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=True).name
        
        src_dir = Path(self.cfg.paths.data_dir, self.lts_dir_name, "semifield-upload", self.batch_id)
        output_dir = Path(self.cfg.paths.data_dir) / self.lts_dir_name / "semifield-developed-images" / self.batch_id / "pngs_iteration3"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.color_matrix_path = Path(self.cfg.paths.image_development, "color_matrix", self.cfg.quick_process.ccm_name + ".npz")
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

    def process_files(self, transformation_matrix):
        """Processes RAW image files by applying demosaicing and color correction.
        
        Args:
            transformation_matrix (np.ndarray): Transformation matrix for color correction.
        """
        all_raw_files = list(self.src_dir.glob("*.RAW"))
        log.info(f"Found {len(all_raw_files)} RAW files.")
        
        # Determine the largest file size to filter out incomplete or corrupted files
        max_file_size = max(f.stat().st_size for f in all_raw_files)
        raw_files = sorted([f for f in all_raw_files if f.stat().st_size == max_file_size])

        if not raw_files:
            log.info("No new files to process.")
            return
        
        log.info(f"Processing {len(raw_files)} RAW files.")

        multiproc = True

        if not multiproc:
            for raw_file in raw_files:
                ImageProcessor.process_image(raw_file, self.cfg, transformation_matrix, self.output_dir)
        else:
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(ImageProcessor.process_image, raw_file, self.cfg, transformation_matrix, self.output_dir)
                    for raw_file in raw_files
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except ValueError as e:
                        log.error(f"Error processing file: {e}")
                    except KeyboardInterrupt:
                        log.info("Batch processing interrupted.")

    def run(self):
        """Executes the full batch processing workflow."""
        transformation_matrix = self.load_transformation_matrix()
        self.process_files(transformation_matrix)
        
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for batch image processing."""
    random.seed(42)
    batch_processor = BatchProcessor(cfg)
    batch_processor.run()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
