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
from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)

def print_image_stats(image: np.ndarray, label: str = "Image"):
    if image.size == 0:
        print(f"{label} is empty.")
    else:
        print(f"{label} - dtype: {image.dtype}, range: [{np.min(image)}, {np.max(image)}]")

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
    def generate_s_curve_lut(size: int = 256, contrast: float = 5.0, pivot: float = 0.5) -> np.ndarray:
        """
        Generates a 1D S-curve LUT to enhance image contrast.
        
        The LUT is generated using a tanh function that maps input values in [0, 1] to output values in [0, 1].
        A higher contrast value steepens the S-curve, boosting mid-tones.

        Parameters:
            size (int): Number of entries in the LUT (default: 256).
            contrast (float): Controls the steepness of the curve. Increase to boost contrast.
            pivot (float): The midpoint around which contrast is enhanced (default: 0.5).

        Returns:
            np.ndarray: The generated LUT as a 1D numpy array.
        """
        x = np.linspace(0, 1, size)
        # Create an S-curve using tanh. The normalization ensures the output spans [0, 1].
        lut = 0.5 * (np.tanh(contrast * (x - pivot)) / np.tanh(contrast * (pivot)) + 1)
        return lut
    
    @staticmethod
    def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Applies a lookup table (LUT) to an image.

        This function maps the input image's pixel values using the provided LUT.
        It assumes that the input image is normalized to the range [0, 1] and that 
        the LUT is a 1D array representing the mapping of these normalized values.

        Parameters:
            image (np.ndarray): The input image with values in [0, 1].
            lut (np.ndarray): A 1D numpy array representing the lookup table. 
                            For example, a LUT with 256 entries for an 8-bit mapping.

        Returns:
            np.ndarray: The image after applying the LUT, still normalized to [0, 1].
        """
        # Determine the number of points in the LUT
        lut_size = len(lut)
        
        # Create a domain for the LUT (values from 0 to 1)
        lut_domain = np.linspace(0, 1, lut_size)
        
        # Use numpy's interpolation to map each pixel value in the image through the LUT
        # np.interp operates element-wise on the image
        adjusted_image = np.interp(image, lut_domain, lut)
        
        return adjusted_image
    @staticmethod
    def apply_gain_and_white_balance(raw_array: np.ndarray, gains: tuple, pattern: str = "BGGR") -> np.ndarray:
        """
        Applies white balance gains to a raw Bayer image.

        This function multiplies each pixel by the corresponding gain for its color
        channel based on the Bayer pattern. It should be applied before demosaicing.

        Parameters:
            raw_array (np.ndarray): A 2D numpy array representing the raw Bayer image (e.g., dtype=np.uint16).
            gains (tuple): A tuple of three floats (gain_R, gain_G, gain_B). 
                        Gains should be provided relative to the sensor's response.
            pattern (str): The Bayer pattern of the raw data. Supported options are "BGGR", "RGGB", "GRBG", "GBRG".
                        Default is "BGGR".

        Returns:
            np.ndarray: The white-balanced raw image as a numpy array of the same dtype as raw_array.
        """
        # Ensure the input is a 2D array
        if raw_array.ndim != 2:
            raise ValueError("raw_array must be a 2D Bayer image.")
        
        # Convert to float for processing
        wb_array = raw_array.astype(np.float32)
        
        gain_r, gain_g, gain_b = gains
        height, width = wb_array.shape

        # Create boolean masks for each channel
        mask_r = np.zeros_like(wb_array, dtype=bool)
        mask_g = np.zeros_like(wb_array, dtype=bool)
        mask_b = np.zeros_like(wb_array, dtype=bool)
        
        pattern = pattern.upper()
        if pattern == "BGGR":
            # BGGR pattern: 
            # Row 0: [B, G, B, G, ...]
            # Row 1: [G, R, G, R, ...]
            mask_b[0::2, 0::2] = True  # Blue pixels: even rows, even cols
            mask_g[0::2, 1::2] = True  # Green pixels: even rows, odd cols
            mask_g[1::2, 0::2] = True  # Green pixels: odd rows, even cols
            mask_r[1::2, 1::2] = True  # Red pixels: odd rows, odd cols
        elif pattern == "RGGB":
            # RGGB pattern:
            # Row 0: [R, G, R, G, ...]
            # Row 1: [G, B, G, B, ...]
            mask_r[0::2, 0::2] = True  # Red pixels: even rows, even cols
            mask_g[0::2, 1::2] = True  # Green pixels: even rows, odd cols
            mask_g[1::2, 0::2] = True  # Green pixels: odd rows, even cols
            mask_b[1::2, 1::2] = True  # Blue pixels: odd rows, odd cols
        elif pattern == "GRBG":
            # GRBG pattern:
            # Row 0: [G, R, G, R, ...]
            # Row 1: [B, G, B, G, ...]
            mask_g[0::2, 0::2] = True  # Green pixels: even rows, even cols
            mask_r[0::2, 1::2] = True  # Red pixels: even rows, odd cols
            mask_b[1::2, 0::2] = True  # Blue pixels: odd rows, even cols
            mask_g[1::2, 1::2] = True  # Green pixels: odd rows, odd cols
        elif pattern == "GBRG":
            # GBRG pattern:
            # Row 0: [G, B, G, B, ...]
            # Row 1: [R, G, R, G, ...]
            mask_g[0::2, 0::2] = True  # Green pixels: even rows, even cols
            mask_b[0::2, 1::2] = True  # Blue pixels: even rows, odd cols
            mask_r[1::2, 0::2] = True  # Red pixels: odd rows, even cols
            mask_g[1::2, 1::2] = True  # Green pixels: odd rows, odd cols
        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        # Apply the gains to the corresponding pixels
        wb_array[mask_r] *= gain_r
        wb_array[mask_g] *= gain_g
        wb_array[mask_b] *= gain_b

        # Clip to valid range (assuming 16-bit data)
        wb_array = np.clip(wb_array, 0, 65535)

        return wb_array.astype(raw_array.dtype)

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
        # Normalize and compress dynamic range before transformation
        # Ensure image is in [0, 1]
        image_dtype = image.dtype
        max_val = np.iinfo(image_dtype).max if image_dtype.kind == 'u' else 1.0
        image = image.astype(np.float64) / max_val
        gamma_corrected = np.power(image, 1 / gamma)
        return gamma_corrected
    
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
        
        # Extract color channel coefficients from transformation matrix
        red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)

        # Normalize the source image to the range [0, 1]
        source_dtype = source_img.dtype
        max_val = np.iinfo(source_dtype).max if source_dtype.kind == 'u' else 1.0
        # source_flt = source_img.astype(np.float64) / max_val
        # Normalize and compress dynamic range before transformation
        # source_compressed = ImageProcessor.apply_gamma_correction(source_flt, gamma=1.05)
        # source_compressed = ImageProcessor.apply_log_compression(source_flt)
        # tonemapReinhard = cv2.createTonemapReinhard(gamma=gamma)
        # source_compressed = tonemapReinhard.process(source_flt.astype(np.float32))
        source_r, source_g, source_b = cv2.split(source_img)
        
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
    def load_raw_image(raw_file: Path, cfg: DictConfig):
        """Loads a RAW image file and applies white balance gains."""
        log.info(f"Loading: {raw_file}")
        im_height, im_width = cfg.colorchecker.height, cfg.colorchecker.width

        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))

        return nparray
    
    @staticmethod
    def demosaic_image(nparray: Path):
        """Demosaics a RAW image file using bilinear interpolation."""
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
    def process_image(raw_file: Path, cfg: DictConfig, transformation_matrix, output_dir: Path, gains):
        log.info(f"Processing: {raw_file}")

        # Load the RAW image
        raw_array = ImageProcessor.load_raw_image(raw_file, cfg)
        # Apply white balance
        wb_array = ImageProcessor.apply_gain_and_white_balance(raw_array, gains=gains, pattern="BGGR")
        print_image_stats(wb_array, "White balanced")
        # Demosaic
        demosaiced = ImageProcessor.demosaic_image(wb_array)
        print_image_stats(demosaiced, "Demosaiced RGB")
        
        # Apply gamma correction
        gamma_corrected = ImageProcessor.apply_gamma_correction(demosaiced, gamma=1.1)
        print_image_stats(gamma_corrected, "Gamma corrected")
        
        # Apply color correction
        corrected_img = ImageProcessor.apply_transformation_matrix(gamma_corrected, transformation_matrix) 
        print_image_stats(corrected_img, "Color corrected")
        
        corrected_img = corrected_img * 65535.0

        corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)
        
        corrected_image_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
        

        # Convert to 8-bit or 16-bit RGB
        # rgb_bit_image = (corrected_image_rgb * 255).astype(np.uint8) if cfg.inspect_v31.bit_depth == 8 else (corrected_image_rgb * 65535).astype(np.uint16)

        
        # Save the final image
        r_gain, g_gain, b_gain = gains
        raw_file_name = f"{raw_file.stem}_R{r_gain:.2f}_G{g_gain:.2f}_B{b_gain:.2f}.png"
        ImageProcessor.save_image(corrected_image_bgr, output_dir / raw_file_name)
        
        raw_file_crop_name = f"{raw_file.stem}_R{r_gain:.2f}_G{g_gain:.2f}_B{b_gain:.2f}_crop_grey.png"
        cropped_image = corrected_image_bgr[8322:8685, 8568:8880]
        ImageProcessor.save_image(cropped_image, output_dir / raw_file_crop_name)

        raw_file_crop_name = f"{raw_file.stem}_R{r_gain:.2f}_G{g_gain:.2f}_B{b_gain:.2f}_crop_lightgrey.png"
        cropped_image = corrected_image_bgr[8315:8653, 7598:7940]
        ImageProcessor.save_image(cropped_image, output_dir / raw_file_crop_name)

        raw_file_crop_name = f"{raw_file.stem}_R{r_gain:.2f}_G{g_gain:.2f}_B{b_gain:.2f}_crop_white.png"
        cropped_image = corrected_image_bgr[8310:8613, 7110:7440]
        ImageProcessor.save_image(cropped_image, output_dir / raw_file_crop_name)

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

        self.color_matrix_path = Path(self.cfg.paths.image_development, "color_matrix", self.cfg.raw2png.ccm_name + ".npz")
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
        
        # Define your white balance gains
        gain = (1.10, 1.0, 1.0)

        if not multiproc:
            for raw_file in raw_files:
                ImageProcessor.process_image(raw_file, self.cfg, transformation_matrix, self.output_dir, gain)
        else:
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(ImageProcessor.process_image, raw_file, self.cfg, transformation_matrix, self.output_dir, gain)
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
