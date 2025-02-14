import cv2
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

log = logging.getLogger(__name__)


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
        # todo: @jinamshah
        #   lts vs local vs both
        # for path in self.cfg.paths.lts_locations:
        #     semif_uploads = os.path.join(path, "semifield-upload")
        #     batch_ids = [x.name for x in Path(semif_uploads).glob("*")]
        #     if self.batch_id in batch_ids:
        #         self.uploads_folder = Path(semif_uploads) / self.batch_id
        #         break
        if not self.uploads_folder:
            self.uploads_folder = (Path(self.cfg.paths.data_dir) /
                              'semifield-upload' /
                              self.batch_id)
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
        Args:
            image_path (str): Path to the image
            height (int): Height of the image
            width (int): Width of the image
        Returns:
            raw_image (np.ndarray): Preprocessed image
        """
        raw_image = np.fromfile(image_path, dtype=np.uint16).astype(np.uint16)
        raw_image = raw_image.reshape(height, width)
        return raw_image

    @staticmethod
    def color_checker_exists(raw_image: np.ndarray) -> (bool, np.ndarray):
        """
        Expects numpy array of raw image (resized using height, width).
        Looks for color correction module in the image,
        returns True if module is found, else returns False.
        Args:
            raw_image (np.ndarray): resized raw image
        Returns:
            color_checker_exists (bool): True if color correction module found
            bgr_image (np.ndarray): bgr image
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
        checker = color_checkers[0]
        # Get the sRGB color values from the ColorChecker chart
        chart_sRGB = checker.getChartsRGB()
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


def main(cfg: DictConfig) -> None:
    """
    Main function that calls ColorChecker.
    1. List raw images
    2. Preprocess raw images (using np)
    3. Detect color checkers
    """
    log.info(f"Detecting color checkers for {cfg.batch_id} and saving color "
             f"correction matrix/matrices")
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
        # todo: @jinamshah
        #   condition changes if multiple colorcheckers need to be used
        if color_checker_exists:
            log.info(f"{image.stem} - found color checker")
            ccm_images.append((image.stem, ccm_bgr_img))
            break

    # todo: @jinamshah
    #   changes expected here depending on if color checker is generated per
    #   species/per season/per batch
    #   multiprocessing?
    for image_name, ccm_bgr_img in ccm_images:
        log.info(f"Processing color correction matrix for {image_name}")
        current_ccm_model = colorchecker.process_ccm_image(
            cfg.colorchecker.ccm_gamma, ccm_bgr_img)
        filename = colorchecker.save_ccm(current_ccm_model, image_name)
        log.info(f"Saved color correction matrix for {filename}")
        break
    return
