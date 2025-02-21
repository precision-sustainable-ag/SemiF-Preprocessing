import logging
from pathlib import Path
from typing import Tuple, List, Union

import hydra
import numpy as np
from omegaconf import DictConfig
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import (CalibrationIlluminant, CFAPattern, DNGVersion,
                        Orientation, PhotometricInterpretation,
                        PreviewColorSpace)

from utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class RawToDNGConverter:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        self.task_cfg: DictConfig = cfg.raw2dng
        self.batch_id: str = cfg.batch_id
        self.raw_files_mask: List[str] = self.cfg.file_masks.raw_files
        self.ccm_files_mask: List[str] = self.cfg.file_masks.ccm_files

        self.height: int = self.task_cfg.height
        self.width: int = self.task_cfg.width
        self.num_pixels: int = self.height * self.width
        self.bpp: int = self.task_cfg.bpp

        self.color_gain_div: int = 100

        # Determine the LTS directory name for the batch using a local lookup
        self.lts_dir_name: str = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=True).name
        # Set the uploads folder path based on the data directory, LTS directory, and batch_id
        self.uploads_folder: Path = Path(self.cfg.paths.data_dir) / self.lts_dir_name / 'semifield-upload' / self.batch_id

    def list_files(self) -> Tuple[List[Path], List[Path]]:
        """
        List all raw image and CCM files available in the uploads folder.

        Returns:
            Tuple[List[Path], List[Path]]: A tuple containing a list of raw image files and a list of CCM files.

        Raises:
            ValueError: If no CCM files are found.
        """
        raw_files: List[Path] = []
        # Gather raw image files using provided file masks
        for file_mask in self.raw_files_mask:
            raw_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        
        ccm_files: List[Path] = []
        # Gather CCM files using provided file masks
        for file_mask in self.ccm_files_mask:
            ccm_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        
        if not ccm_files:
            raise ValueError("No CCM files available")
        
        return raw_files, ccm_files

    def load_raw_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load raw image data from a file into a 16-bit numpy array and reshape it.

        Args:
            file_path (Union[str, Path]): Path to the raw image file.

        Returns:
            np.ndarray: 2D numpy array representing the raw image.
        """
        # Read file as a flat array of 16-bit unsigned integers
        rawImage = np.fromfile(str(file_path), dtype=np.uint16).astype(np.uint16)
        # Reshape the array to the specified dimensions
        raw_image = np.reshape(rawImage, (self.height, self.width))

        # Log information about the loaded image
        log.info(f"Loaded raw image from {file_path}")
        log.info(f"Raw image shape: {raw_image.shape}")
        log.info(f"Raw image dtype: {raw_image.dtype}")
        log.info(f"Raw image min: {np.min(raw_image)}")
        log.info(f"Raw image max: {np.max(raw_image)}")

        return raw_image

    def configure_dng_tags(self, ccm_file: Union[str, Path]) -> DNGTags:
        """
        Set up and configure DNG tags for the conversion using the provided CCM file.

        Args:
            ccm_file (Union[str, Path]): Path to the CCM file.

        Returns:
            DNGTags: Configured DNG tags object.
        """
        t: DNGTags = DNGTags()
        # Set image dimensions and tiling information
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, Orientation.Horizontal)
        # Set photometric interpretation and pixel sample information
        t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.BitsPerSample, self.bpp)
        t.set(Tag.CFARepeatPatternDim, [2, 2])
        t.set(Tag.CFAPattern, CFAPattern.RGGB)  # RGGB best so far; other patterns yield poorer results
        # Set black and white level information
        t.set(Tag.BlackLevel, 0)
        t.set(Tag.WhiteLevel, 65535)
        # Set calibration illuminant
        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
        # Set camera properties and DNG version information
        t.set(Tag.Make, "SVS")
        t.set(Tag.Model, "Camera Model")
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
        t.set(Tag.BaselineExposure, [[1, 1]])  # Neutral exposure setting
        # Set color correction matrix tags using CCM file data
        t.set(Tag.AsShotNeutral, self.get_ashot_neutral())
        t.set(Tag.ColorMatrix1, self.set_color_correction(ccm_file))

        return t

    def get_ashot_neutral(self) -> List[List[int]]:
        """
        Generate balanced AsShotNeutral values based on predefined color gains.

        Returns:
            List[List[int]]: A list containing numerator and denominator pairs for red, green, and blue channels.
        """
        # [denominator, numerator] 
        # Define gains for each color channel
        color_gain_div = self.color_gain_div
        r_gain: float = 1.3
        g_gain: float = 1.0
        b_gain: float = 1.0

        as_shot_neutral: List[List[int]] = [
            [color_gain_div, round(r_gain * color_gain_div)],  # Red channel
            [color_gain_div, color_gain_div],                  # Green channel
            [color_gain_div, round(b_gain * color_gain_div)]     # Blue channel
        ]
        return as_shot_neutral

    def set_color_correction(self, ccm_file: Union[str, Path]) -> List[Tuple[int, int]]:
        """
        Load and normalize the color correction matrix from the given CCM file.

        Args:
            ccm_file (Union[str, Path]): Path to the CCM file.

        Returns:
            List[Tuple[int, int]]: Normalized color correction matrix as a list of (numerator, denominator) tuples.
        """
        # Load the CCM from file; assume comma-separated values
        ccm = np.loadtxt(ccm_file, delimiter=',')
        normalized_ccm: List[Tuple[int, int]] = []
        # Normalize each row so that the sum of elements equals color_gain_div
        for row in ccm:
            row_sum = sum(row)
            # Create a normalized tuple for each value in the row
            normalized_row = [(round((value / row_sum) * self.color_gain_div), self.color_gain_div) for value in row]
            normalized_ccm.extend(normalized_row)

        return normalized_ccm

    @staticmethod
    def convert_to_dng(raw_image: np.ndarray, dng_tags: DNGTags, output_filename: str) -> None:
        """
        Convert a raw image to DNG format using specified DNG tags and save it.

        Args:
            raw_image (np.ndarray): The raw image data.
            dng_tags (DNGTags): Configured DNG tags for the image.
            output_filename (str): Path where the converted DNG file will be saved.

        Raises:
            ValueError: If raw_image is None.
        """
        if raw_image is None:
            raise ValueError("Raw image data not loaded.")

        # Initialize the RAW to DNG converter
        converter = RAW2DNG()
        # Set conversion options with provided DNG tags; no compression applied here
        converter.options(dng_tags, path="", compress=False)
        # Convert the raw image and save to the specified output file
        converter.convert(raw_image, filename=output_filename)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to initialize the RawToDNGConverter and process raw images into DNG format.

    The function retrieves raw and CCM files from the uploads folder, then converts each raw image 
    to DNG format using the corresponding CCM data, saving the result to an output directory.
    """
    log.info(f"Starting raw to DNG conversion for batch {cfg.batch_id}")
    # Initialize the converter with the provided configuration
    raw2dng_conv = RawToDNGConverter(cfg)
    # Retrieve lists of raw image files and CCM files
    raw_files, ccm_files = raw2dng_conv.list_files()
    log.info(f"Found {len(raw_files)} raw images and {len(ccm_files)} ccm files")
    
    # Define the output directory for converted DNG files and ensure it exists
    output_dir: Path = Path(cfg.paths.data_dir) / raw2dng_conv.lts_dir_name / 'semifield-developed-images' / cfg.batch_id / 'dngs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each raw file with each available CCM file
    for ccm_file in ccm_files:
        for raw_file in raw_files:
            # Load raw image data
            raw_data = raw2dng_conv.load_raw_image(raw_file)
            # Configure DNG tags using the current CCM file
            dng_tags = raw2dng_conv.configure_dng_tags(ccm_file)
            log.info(f"DNG Tags: {dng_tags}")
            # Define the output filename for the converted DNG file
            output_filename: str = str(Path(output_dir, f"{raw_file.stem}.dng"))
            # Convert the raw image to DNG format
            raw2dng_conv.convert_to_dng(raw_data, dng_tags, output_filename)
            log.info(f"Converted {raw_file} to {output_filename}")
    log.info("Raw to DNG conversion complete.")


if __name__ == "__main__":
    main()
