import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from utils import find_lts_dir

log = logging.getLogger(__name__)


def log_image_stats(image: np.ndarray, label: str = "Image"):
    if image.size == 0:
        log.info(f"{label} is empty.")
    else:
        log.info(
            f"{label} - dtype: {image.dtype}, range: [{np.min(image)}, {np.max(image)}], shape: {image.shape}"
        )

class Demosaicer:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_raw_image(self, raw_file: Path) -> np.ndarray:
        """
        Loads a RAW image file using dimensions specified in the configuration.
        
        Parameters:
            raw_file (Path): Path to the RAW image file.
        
        Returns:
            np.ndarray: A 2D numpy array representing the RAW image.
        """
        log.info(f"Loading: {raw_file}")
        im_height, im_width = self.cfg.raw2png.height, self.cfg.raw2png.width
        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))
        return nparray

    def demosaic_image(self, nparray: np.ndarray) -> np.ndarray:
        """
        Demosaics a RAW image using the Menon2007 algorithm.
        
        Parameters:
            nparray (np.ndarray): A 2D numpy array representing the RAW image.
        
        Returns:
            np.ndarray: A normalized demosaiced image in the RGB color space.
        """
        # Alternative using OpenCV (commented out):
        demosaiced = cv2.cvtColor(nparray, cv2.COLOR_BayerBG2RGB_EA)
        # demosaiced = demosaicing_CFA_Bayer_Menon2007(nparray, pattern="RGGB")
        demosaiced = demosaiced.astype(np.float64) / 65535.0
        return demosaiced

    def save_image(self, image: np.ndarray, output_path: Path):
        """Saves an image as a PNG file with no compression."""
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        log.info(f"Saved: {output_path}")

    def process_image(self, raw_file: Path, output_dir: Path):
        """
        Processes a single RAW image file: loading, demosaicing, converting, and saving.
        
        Parameters:
            raw_file (Path): Path to the RAW image file.
            output_dir (Path): Directory where the processed image will be saved.
        """
        log.info(f"Processing: {raw_file}")

        # Load the RAW image.
        raw_array = self.load_raw_image(raw_file)

        # Demosaic the image.
        demosaiced = self.demosaic_image(raw_array)
        log_image_stats(demosaiced, "Demosaiced RGB")

        # Convert the demosaiced image back to 16-bit.
        demosaiced_16 = (demosaiced * 65535).astype(np.uint16)
        log_image_stats(demosaiced_16, "Demosaiced 16-bit")

        # Convert RGB to BGR for saving with OpenCV.
        corrected_image_bgr = cv2.cvtColor(demosaiced_16, cv2.COLOR_RGB2BGR)

        # Save the final image as PNG (filename is based on the RAW file stem).
        raw_file_name = f"{raw_file.stem}.png"
        self.save_image(corrected_image_bgr, output_dir / raw_file_name)

class BatchDemoasaicer:

    def __init__(self, cfg: DictConfig):
        """
        Initializes the batch processor with configuration settings.
        
        Args:
            cfg (DictConfig): Configuration object containing paths and processing parameters.
        """
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.src_dir, self.output_dir = self.setup_paths()
        self.processor = Demosaicer(cfg)

    def setup_paths(self):
        """
        Sets up and validates required directories for processing.
        
        Returns:
            tuple: A tuple containing the source and output directories.
        """
        self.lts_dir_name = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=True).name

        src_dir = Path(self.cfg.paths.data_dir, self.lts_dir_name, "semifield-upload", self.batch_id)
        output_dir = (
            Path(self.cfg.paths.data_dir)
            / self.lts_dir_name
            / "semifield-developed-images"
            / self.batch_id
            / "debayered"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
        return src_dir, output_dir

    def process_files(self):
        """
        Processes RAW image files by applying demosaicing.
        Filters files based on file size and specific file stems.
        """
        all_raw_files = list(self.src_dir.glob("*.RAW"))
        log.info(f"Found {len(all_raw_files)} RAW files.")

        # Determine the largest file size to filter out incomplete or corrupted files.
        max_file_size = max(f.stat().st_size for f in all_raw_files)
        raw_files = sorted([f for f in all_raw_files if f.stat().st_size == max_file_size])

        # Optionally filter files by specific names.
        # raw_files = [f for f in raw_files if f.stem in ["NC_1740166530"]]

        if not raw_files:
            log.info("No new files to process.")
            return

        log.info(f"Processing {len(raw_files)} RAW files.")
        for raw_file in raw_files:
            self.processor.process_image(raw_file, self.output_dir)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting debayering processing workflow.")
    batch_processor = BatchDemoasaicer(cfg)
    batch_processor.process_files()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
