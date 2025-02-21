import logging
from pathlib import Path
from typing import Tuple, List, Union

import cv2
import numpy as np
import hydra
from omegaconf import DictConfig

from utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class ColorChecker:
    def __init__(self, cfg: DictConfig, local_batch_folder: Path) -> None:

        self.cfg = cfg
        self.batch_id: str = self.cfg.batch_id
        self.raw_files_mask: List[str] = self.cfg.file_masks.raw_files
        self.task_cfg = self.cfg.colorchecker
        self.uploads_folder: Path = local_batch_folder

    def list_raw_images(self) -> List[Path]:
        """
        List all raw image files in the uploads folder that match the specified file masks.
        """
        raw_files: List[Path] = []
        # Loop over each file mask and collect matching files
        for file_mask in self.raw_files_mask:
            raw_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        return sorted(raw_files)

    @staticmethod
    def preprocess_raw_image(image_path: Union[str, Path], height: int, width: int) -> np.ndarray:
        """
        Convert a raw image file to a numpy array and reshape it.
        """
        # Read the raw file into a numpy array and reshape based on provided dimensions
        raw_image = np.fromfile(str(image_path), dtype=np.uint16).astype(np.uint16)
        raw_image = raw_image.reshape(height, width)
        return raw_image

    @staticmethod
    def color_checker_exists(raw_image: np.ndarray) -> Tuple[bool, Union[np.ndarray, None]]:
        """
        Check if a ColorChecker chart exists in the raw image.

        The raw image is first converted from Bayer pattern to a BGR image and then analyzed
        using OpenCV's ColorChecker detection module.

        Args:
            raw_image (np.ndarray): The raw image data (assumed to be preprocessed to the correct shape).

        Returns:
            Tuple[bool, Union[np.ndarray, None]]:
                - A boolean indicating if the color checker was found.
                - The processed BGR image if found, else None.
        """
        try:
            # Convert Bayer pattern raw image to BGR format
            bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerBG2BGR)
            # Normalize the image to 8-bit values
            bgr_image = (bgr_image / 256).astype(np.uint8)
            # Create and use the ColorChecker detector
            ccm_detector = cv2.mcc.CCheckerDetector.create()
            color_checker_exists = ccm_detector.process(bgr_image, cv2.ccm.COLORCHECKER_Macbeth, 1)
            # Clean up the detector
            del ccm_detector
            return color_checker_exists, bgr_image
        except Exception as e:
            log.error(e)
            return False, None

    @staticmethod
    def process_ccm_image(gamma: int, ccm_bgr_img: np.ndarray) -> cv2.ccm.ColorCorrectionModel:
        """
        Generate a color correction model (CCM) from an image containing a ColorChecker chart.

        Args:
            gamma (int): Gamma value to be used for linearization.
            ccm_bgr_img (np.ndarray): The BGR image that contains the ColorChecker chart.

        Returns:
            cv2.ccm.ColorCorrectionModel: The computed color correction model.

        Raises:
            ValueError: If the number of detected ColorChecker charts is not exactly one.
        """
        # Create a ColorChecker detector and process the image
        ccm_detector = cv2.mcc.CCheckerDetector.create()
        ccm_detector.process(ccm_bgr_img, cv2.ccm.COLORCHECKER_Macbeth, 1)

        # Retrieve the list of detected ColorCheckers
        color_checkers = ccm_detector.getListColorChecker()

        # Ensure exactly one ColorChecker is detected
        if len(color_checkers) != 1:
            raise ValueError(
                f"Unexpected checker count, detected ({len(color_checkers)}), expected (1)"
            )

        log.info(f"Detected {len(color_checkers)} color checkers")
        # Use the first (and only) detected ColorChecker
        checker = color_checkers[0]
        # Retrieve the sRGB values for the ColorChecker chart
        chart_sRGB = checker.getChartsRGB()
        
        # Reshape and normalize the sRGB values
        src = chart_sRGB[:, 1].copy().reshape(24, 1, 3)
        src /= 255.0

        # Create the Color Correction Model using the normalized sRGB values
        ccm_model = cv2.ccm.ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
        ccm_model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        ccm_model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        ccm_model.setInitialMethod(cv2.ccm.INITIAL_METHOD_LEAST_SQUARE)
        ccm_model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        ccm_model.setLinearGamma(gamma)
        ccm_model.setLinearDegree(3)
        ccm_model.setSaturatedThreshold(0, 0.98)

        # Run the model to compute the CCM
        ccm_model.run()
        ccm_model.getCCM()

        # Clean up resources
        del ccm_detector, ccm_bgr_img
        return ccm_model

    def save_ccm(self, ccm_model: cv2.ccm.ColorCorrectionModel, img_name: str) -> Path:
        """
        Save the computed color correction matrix to a text file.
        Returns:
            Path: The path of the file where the CCM was saved.
        """
        # Retrieve and convert the CCM to float32
        ccm = ccm_model.getCCM()
        ccm.astype(np.float32)

        # Define the filename and save the CCM as a comma-separated text file
        filename = self.uploads_folder / f"{img_name}.txt"
        np.savetxt(filename, ccm, delimiter=',')
        return filename


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to process raw images and generate a color correction matrix.

    Locates the local batch folder, initializes a ColorChecker instance,
    processes raw images to detect a ColorChecker chart, and if found, computes and saves
    the color correction matrix.
    """
    log.info(f"Starting color correction matrix generation for batch {cfg.batch_id}")
    # Locate the LTS directory for the given batch
    lts_dir = find_lts_dir(cfg.batch_id, cfg.paths.lts_locations, local=True)

    # Build the path to the local uploads folder
    uploads_folder = Path(cfg.paths.data_dir) / lts_dir.name / "semifield-upload" / cfg.batch_id

    # Check that the uploads folder is valid (exists and contains files)
    if not uploads_folder:
        log.error("Local batch folder either not found or RAW files are incomplete. Exiting.")
        return

    # Create a ColorChecker instance
    colorchecker = ColorChecker(cfg, uploads_folder)
    # Retrieve the list of raw image paths
    image_paths = colorchecker.list_raw_images()
    log.info(f"{cfg.batch_id} - Found {len(image_paths)} raw images")
    image_height: int = cfg.colorchecker.height
    image_width: int = cfg.colorchecker.width

    # List to hold tuples of image name and processed BGR image that contains the ColorChecker
    ccm_images = []
    for image in image_paths:
        # Preprocess each raw image (read and reshape)
        raw_image = colorchecker.preprocess_raw_image(image, image_height, image_width)
        # Check for the presence of a ColorChecker chart in the image
        color_checker_exists, ccm_bgr_img = colorchecker.color_checker_exists(raw_image)

        if color_checker_exists:
            log.info(f"{image.stem} - found color checker")
            ccm_images.append((image.stem, ccm_bgr_img))
            break  # Process only the first image with a detected color checker
        else:
            log.info(f"{image.stem} - color checker not found")

    # Process the image(s) with detected ColorChecker to compute and save the CCM
    for image_name, ccm_bgr_img in ccm_images:
        log.info(f"Processing color correction matrix for {image_name}")
        current_ccm_model = colorchecker.process_ccm_image(cfg.colorchecker.ccm_gamma, ccm_bgr_img)
        filename = colorchecker.save_ccm(current_ccm_model, image_name)
        log.info(f"Saved color correction matrix for {filename}")
        break  # Only process the first valid image
    
    log.info("Color correction matrix generation complete.")

if __name__ == "__main__":
    main()
