import cv2
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import hydra
from typing import Tuple
log = logging.getLogger(__name__)
np.set_printoptions(suppress=True)


class ColorChecker:
    def __init__(self, cfg: DictConfig):
        """
        Class constructor
        """
        self.cfg = cfg
        self.batch_id = self.cfg.batch_id
        self.raw_files_mask = self.cfg.file_masks.raw_files
        self.task_cfg = self.cfg.colorchecker
        self.uploads_folder = None

    def list_raw_images(self):
        """
        Method to list all raw images available in the batch
        """
        if not self.uploads_folder:
            self.uploads_folder = (Path(self.cfg.paths.data_dir) /
                              'semifield-upload' /
                              self.batch_id / 'raw')
            # log.error(f"{self.batch_id} doesn't exist")
        raw_files = []
        for file_mask in self.raw_files_mask:
            raw_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        return raw_files

    @staticmethod
    def preprocess_raw_image(image_path: str, height: int,
                             width: int) -> np.ndarray:
        """
        Method to convert and reshape raw image
        """
        raw_image = np.fromfile(image_path, dtype=np.uint16).astype(np.uint16)
        raw_image = raw_image.reshape(height, width)
        return raw_image

    @staticmethod
    def color_checker_exists(raw_image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Expects numpy array of raw image (resized using height, width).
        Looks for color correction module in the image,
        returns True if module is found, else returns False.
        """
        try:
            bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerBG2BGR)
            bgr_image = (bgr_image / 256).astype(np.uint8)
            ccm_detector = cv2.mcc.CCheckerDetector.create()
            color_checker_exists = ccm_detector.process(bgr_image,
                                                        cv2.ccm.COLORCHECKER_Macbeth,
                                                        1)
            del ccm_detector
            return color_checker_exists, bgr_image
        except Exception as e:
            log.error(e)
            return False, None

    @staticmethod
    def process_ccm_image(gamma: int,
                          ccm_bgr_img: np.ndarray) -> cv2.ccm.ColorCorrectionModel:
        """
        Generate a color correction model for
        an image where a ColorChecker chart has been detected.
        Args:
            gamma (int): Gamma of the image
            ccm_bgr_img (np.ndarray): Input BGR image containing
            a ColorChecker chart.
        Returns:
            cv2.ccm.ColorCorrectionModel: The computed color correction model.
        Raises:
            ValueError: If an unexpected number of ColorCheckers is detected.
        """
        # Create a ColorChecker detector and process the image
        ccm_detector = cv2.mcc.CCheckerDetector.create()
        ccm_detector.process(ccm_bgr_img, cv2.ccm.COLORCHECKER_Macbeth, 1)

        # Get the list of detected ColorCheckers in the image
        color_checkers = ccm_detector.getListColorChecker()

        # Validate that exactly one ColorChecker is detected
        if len(color_checkers) != 1:
            raise ValueError(
                f"Unexpected checker count, detected ({len(color_checkers)}), expected (1)")

        # Extract the first detected ColorChecker
        log.info(f"Detected {len(color_checkers)} color checkers")
        log.info(type(color_checkers))
        log.info(type(color_checkers[0]))
        checker = color_checkers[0]
        # Get the sRGB color values from the ColorChecker chart
        chart_sRGB = checker.getChartsRGB()
        np.set_printoptions(suppress=True)
        
        # Reshape and normalize the color values
        src = chart_sRGB[:, 1].copy().reshape(24, 1, 3)
        src /= 255.0

        # Create and configure the color correction model
        ccm_model = cv2.ccm.ColorCorrectionModel(src,
                                                 cv2.ccm.COLORCHECKER_Macbeth)
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

    def save_ccm(self, ccm_model: cv2.ccm.ColorCorrectionModel, img_name: str):
        """
        Save a color correction model to a file.
        Args:
            ccm_model (cv2.ccm.ColorCorrectionModel): The color correction model.
            img_name (str): The image file name that has the color checker
            module.
        Returns:
            filename (str): The filename to save the model to.
        """
        ccm = ccm_model.getCCM()
        ccm.astype(np.float32)

        filename = self.uploads_folder / f"{img_name}.txt"
        
        np.savetxt(filename, ccm, delimiter=',')
        return filename

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    colorchecker = ColorChecker(cfg)
    image_paths = colorchecker.list_raw_images()
    log.info(f"{cfg.batch_id} - Found {len(image_paths)} raw images")
    image_height = cfg.colorchecker.height
    image_width = cfg.colorchecker.width
    ccm_images = []
    for image in image_paths:
        raw_image = colorchecker.preprocess_raw_image(image, image_height,
                                                      image_width)
        color_checker_exists, ccm_bgr_img = colorchecker.color_checker_exists(
            raw_image)
    
        if color_checker_exists:
            log.info(f"{image.stem} - found color checker")
            ccm_images.append((image.stem, ccm_bgr_img))
            break

    for image_name, ccm_bgr_img in ccm_images:
        log.info(f"Processing color correction matrix for {image_name}")
        current_ccm_model = colorchecker.process_ccm_image(
            cfg.colorchecker.ccm_gamma, ccm_bgr_img)
        filename = colorchecker.save_ccm(current_ccm_model, image_name)
        log.info(f"Saved color correction matrix for {filename}")
        break
    return

if __name__ == "__main__":
    main()