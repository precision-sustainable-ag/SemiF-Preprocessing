import numpy as np
import cv2
import logging
from pathlib import Path
from omegaconf import DictConfig
from src.utils.utils import log_image_stats

log = logging.getLogger(__name__)



class Preprocessor:
    """
    A collection of static methods for various image processing tasks such as 
    lookup table generation, white balance, gamma/log corrections, color correction, 
    and handling of RAW image files.

    All methods are implemented as static methods because they do not require shared state.
    """

    # ---------------------------
    # Lookup Table (LUT) Methods
    # ---------------------------
    @staticmethod
    def load_transformation_matrix(color_matrix_path: Path) -> np.ndarray:
        """Loads the color transformation matrix from file.
        
        Returns:
            np.ndarray: The loaded transformation matrix.
        """
        # Load the color matrix for color correction
        with np.load(color_matrix_path) as data:
            return data["matrix"]
        
    @staticmethod
    def generate_s_curve_lut(size: int = 256, contrast: float = 5.0, pivot: float = 0.5) -> np.ndarray:
        """
        Generates a 1D S-curve lookup table (LUT) to enhance image contrast.
        
        The LUT is created using a hyperbolic tangent function that maps input values
        in the range [0, 1] to output values in the same range. A higher contrast value
        makes the S-curve steeper, enhancing mid-tone contrasts.

        Parameters:
            size (int): Number of entries in the LUT (default: 256).
            contrast (float): Steepness of the curve; higher values boost contrast.
            pivot (float): The midpoint for contrast enhancement (default: 0.5).

        Returns:
            np.ndarray: A 1D numpy array representing the LUT.
        """
        x = np.linspace(0, 1, size)
        lut = 0.5 * (np.tanh(contrast * (x - pivot)) / np.tanh(contrast * pivot) + 1)
        return lut

    @staticmethod
    def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Applies a lookup table (LUT) to an image.

        This function remaps each pixel value in the input image using the provided LUT.
        It assumes that the image is normalized to the range [0, 1].

        Parameters:
            image (np.ndarray): Input image with values in [0, 1].
            lut (np.ndarray): A 1D numpy array representing the LUT.

        Returns:
            np.ndarray: The image after applying the LUT, still normalized to [0, 1].
        """
        lut_size = len(lut)
        lut_domain = np.linspace(0, 1, lut_size)
        adjusted_image = np.interp(image, lut_domain, lut)
        return adjusted_image

    # ---------------------------
    # White Balance & Color Correction
    # ---------------------------
    @staticmethod
    def apply_gain_and_white_balance(raw_array: np.ndarray, gains: tuple, pattern: str = "BGGR") -> np.ndarray:
        """
        Applies white balance gains to a raw Bayer image.

        Each pixel is multiplied by the corresponding gain for its color channel,
        as determined by the specified Bayer pattern. This should be applied before demosaicing.

        Parameters:
            raw_array (np.ndarray): A 2D numpy array representing the raw Bayer image.
            gains (tuple): A tuple of three floats (gain_R, gain_G, gain_B).
            pattern (str): Bayer pattern of the raw data (e.g., "BGGR", "RGGB", "GRBG", "GBRG").

        Returns:
            np.ndarray: The white-balanced image, with the same dtype as the input.

        Raises:
            ValueError: If the input image is not 2D or the Bayer pattern is unsupported.
        """
        if raw_array.ndim != 2:
            raise ValueError("raw_array must be a 2D Bayer image.")
        
        wb_array = raw_array.astype(np.float32)
        gain_r, gain_g, gain_b = gains

        # Create boolean masks for each color channel based on the Bayer pattern.
        mask_r = np.zeros_like(wb_array, dtype=bool)
        mask_g = np.zeros_like(wb_array, dtype=bool)
        mask_b = np.zeros_like(wb_array, dtype=bool)

        pattern = pattern.upper()
        if pattern == "BGGR":
            mask_b[0::2, 0::2] = True  # Blue: even rows, even cols
            mask_g[0::2, 1::2] = True  # Green: even rows, odd cols
            mask_g[1::2, 0::2] = True  # Green: odd rows, even cols
            mask_r[1::2, 1::2] = True  # Red: odd rows, odd cols
        elif pattern == "RGGB":
            mask_r[0::2, 0::2] = True  # Red: even rows, even cols
            mask_g[0::2, 1::2] = True  # Green: even rows, odd cols
            mask_g[1::2, 0::2] = True  # Green: odd rows, even cols
            mask_b[1::2, 1::2] = True  # Blue: odd rows, odd cols
        elif pattern == "GRBG":
            mask_g[0::2, 0::2] = True  # Green: even rows, even cols
            mask_r[0::2, 1::2] = True  # Red: even rows, odd cols
            mask_b[1::2, 0::2] = True  # Blue: odd rows, even cols
            mask_g[1::2, 1::2] = True  # Green: odd rows, odd cols
        elif pattern == "GBRG":
            mask_g[0::2, 0::2] = True  # Green: even rows, even cols
            mask_b[0::2, 1::2] = True  # Blue: even rows, odd cols
            mask_r[1::2, 0::2] = True  # Red: odd rows, even cols
            mask_g[1::2, 1::2] = True  # Green: odd rows, odd cols
        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        # Apply the gains to each channel.
        wb_array[mask_r] *= gain_r
        wb_array[mask_g] *= gain_g
        wb_array[mask_b] *= gain_b

        # Clip the values to a 16-bit range.
        wb_array = np.clip(wb_array, 0, 65535)
        return wb_array.astype(raw_array.dtype)

    @staticmethod
    def apply_transformation_matrix(source_img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """
        Applies a color transformation matrix to correct the color space of an RGB image.

        The transformation uses polynomial (up to cubic) terms on each color channel.

        Parameters:
            source_img (np.ndarray): Input RGB image.
            transformation_matrix (np.ndarray): A 9x9 matrix containing transformation coefficients.

        Returns:
            np.ndarray: The color-corrected image, or None if input validations fail.
        """
        if transformation_matrix.shape != (9, 9):
            log.error("Transformation matrix must be a 9x9 matrix.")
            return None

        if source_img.ndim != 3:
            log.error("Source image must be an RGB image.")
            return None

        # Split the transformation matrix into channel-specific coefficients.
        red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)
        source_r, source_g, source_b = cv2.split(source_img)

        # Compute the polynomial terms (up to third power) for each channel.
        source_r2, source_r3 = source_r ** 2, source_r ** 3
        source_g2, source_g3 = source_g ** 2, source_g ** 3
        source_b2, source_b3 = source_b ** 2, source_b ** 3

        # Apply the transformation for each channel.
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
        log_image_stats(corrected_img, "Color corrected")
        return corrected_img

    # ---------------------------
    # Gamma and Log Corrections
    # ---------------------------
    @staticmethod
    def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Applies gamma correction using a power-law transformation.

        The image is normalized (if necessary) before the correction is applied.

        Parameters:
            image (np.ndarray): Input image. For unsigned types, expected to be in [0, 1].
            gamma (float): Gamma value (e.g., 1.1 or 2.2).

        Returns:
            np.ndarray: Gamma-corrected image.
        """
        image_dtype = image.dtype
        max_val = np.iinfo(image_dtype).max if image_dtype.kind == 'u' else 1.0
        image = image.astype(np.float64) / max_val
        gamma_corrected = np.power(image, 1 / gamma)
        log_image_stats(gamma_corrected, "Gamma corrected")
        return gamma_corrected

    @staticmethod
    def apply_log_compression(image: np.ndarray) -> np.ndarray:
        """
        Applies logarithmic compression to an image to compress its dynamic range.

        Parameters:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: The log-compressed image.
        """
        epsilon = 1e-6
        compressed = np.log1p(image + epsilon) / np.log1p(1 + epsilon)
        return compressed

    @staticmethod
    def apply_robust_gamma_transformation(image: np.ndarray, gamma: float, robust_percentile: float = 99) -> np.ndarray:
        """
        Applies a robust gamma transformation that first normalizes the image using a robust
        maximum (based on a specified percentile) and then applies gamma correction.

        Steps:
            1. Clip negative values.
            2. Normalize the image to [0, 1] using the robust maximum.
            3. Apply gamma correction.

        Parameters:
            image (np.ndarray): The image to be corrected (post-color transformation).
            gamma (float): The gamma value for correction.
            robust_percentile (float): Percentile for determining the normalization maximum (default: 99).

        Returns:
            np.ndarray: Gamma-corrected image normalized to [0, 1].
        """
        image_clipped = np.clip(image, 0, None)
        robust_max = np.percentile(image_clipped, robust_percentile)
        if robust_max <= 0:
            robust_max = 1.0  # Avoid division by zero.
        image_normalized = np.clip(image_clipped / robust_max, 0, 1)
        gamma_corrected = np.power(image_normalized, 1 / gamma)
        return gamma_corrected

    # ---------------------------
    # RAW Image Handling and I/O
    # ---------------------------
    @staticmethod
    def load_raw_image(raw_file: Path, cfg: DictConfig) -> np.ndarray:
        """
        Loads a RAW image from a file and reshapes it based on provided configuration.

        Parameters:
            raw_file (Path): Path to the RAW image file.
            cfg (DictConfig): Configuration object with 'raw2png.height' and 'raw2png.width'.

        Returns:
            np.ndarray: The loaded RAW image as a 2D numpy array.
        """
        im_height, im_width = cfg.raw2png.height, cfg.raw2png.width
        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))
        log_image_stats(nparray, "RAW")
        return nparray

    @staticmethod
    def demosaic_image(raw_array: np.ndarray) -> np.ndarray:
        """
        Demosaics a RAW Bayer image using bilinear interpolation.

        Parameters:
            raw_array (np.ndarray): White-balanced RAW Bayer image.

        Returns:
            np.ndarray: The demosaiced image normalized to [0, 1].
        """
        demosaiced = cv2.cvtColor(raw_array, cv2.COLOR_BayerBG2RGB_EA)
        demosaiced = demosaiced.astype(np.float64) / 65535.0
        log_image_stats(demosaiced, "Demosaiced")
        return demosaiced

    @staticmethod
    def resize_image(image: np.ndarray, downscale_factor: float) -> np.ndarray:
        """
        Downscales an image by a given factor.

        Parameters:
            image (np.ndarray): Input image.
            downscale_factor (float): Factor by which to reduce the image dimensions.

        Returns:
            np.ndarray: The resized image.
        """
        height, width = image.shape[:2]
        new_height = int(height * downscale_factor)
        new_width = int(width * downscale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    @staticmethod
    def save_image(image: np.ndarray, output_path: Path):
        """
        Saves an image to disk in PNG format.

        Parameters:
            image (np.ndarray): The image to save.
            output_path (Path): Destination file path.
        """
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        log.info(f"Saved: {output_path}")

    @staticmethod
    def remove_local_raw(local_raw_path: Path):
        """
        Removes a RAW image file from disk with a safety check against specific directories.

        Parameters:
            local_raw_path (Path): The path to the RAW image file.
        """
        if "research-project" in str(local_raw_path) or "screberg" in str(local_raw_path):
            log.warning("Refusing to remove file from LTS research-project directory.")
            return
        local_raw_path.unlink()
        log.info(f"Removed raw image: {local_raw_path}")

    # ---------------------------
    # Full Image Processing Pipeline
    # ---------------------------
    @staticmethod
    def process_image(raw_file: Path, cfg: DictConfig, transformation_matrix: np.ndarray, output_dir: Path):
        """
        Executes the full image processing pipeline, which includes:
        
            1. Loading the RAW image.
            2. Applying white balance gains.
            3. Demosaicing.
            4. Applying gamma correction.
            5. Applying color correction.
            6. Converting and saving the final image.

        Parameters:
            raw_file (Path): Path to the RAW image file.
            cfg (DictConfig): Configuration object containing image parameters.
            transformation_matrix (np.ndarray): A 9x9 color transformation matrix.
            output_dir (Path): Directory where the processed image will be saved.
        """
        log.info(f"Processing: {raw_file}")
        gains = (1.0, 1.0, 1.0)

        # 1. Load the RAW image.
        raw_array = Preprocessor.load_raw_image(raw_file, cfg)
        log_image_stats(raw_array, "RAW")
        # 2. Apply white balance.
        # wb_array = Preprocessor.apply_gain_and_white_balance(raw_array, gains=gains, pattern="BGGR")
        # log_image_stats(wb_array, "White balanced")

        # 3. Demosaic the image.
        demosaiced = Preprocessor.demosaic_image(raw_array)
        log_image_stats(demosaiced, "Demosaiced RGB")
        
        # 4. Apply gamma correction.
        gamma_corrected = Preprocessor.apply_gamma_correction(demosaiced, gamma=1.1)
        log_image_stats(gamma_corrected, "Gamma corrected")
        
        # 5. Apply color correction.
        corrected_img = Preprocessor.apply_transformation_matrix(gamma_corrected, transformation_matrix)
        log_image_stats(corrected_img, "Color corrected")

        # Scale back to 16-bit range and convert from RGB to BGR.
        corrected_img = corrected_img * 65535.0
        corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)
        corrected_image_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)

        # 6. Save the final image.
        raw_file_name = f"{raw_file.stem}.png"
        Preprocessor.save_image(corrected_image_bgr, output_dir / raw_file_name)
        
        if cfg.raw2png.remove_raws:
            Preprocessor.remove_local_raw(raw_file)