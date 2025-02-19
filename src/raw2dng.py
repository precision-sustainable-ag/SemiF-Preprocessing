from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import Orientation, PhotometricInterpretation, CFAPattern, CalibrationIlluminant, DNGVersion, PreviewColorSpace
import numpy as np
import struct
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class RawToDNGConverter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_cfg = cfg.raw2dng
        self.batch_id = cfg.batch_id
        self.uploads_folder = None
        self.raw_files_mask = self.cfg.file_masks.raw_files
        self.ccm_files_mask = self.cfg.file_masks.ccm_files

        self.height = self.task_cfg.height
        self.width = self.task_cfg.width
        self.num_pixels = self.height * self.width
        self.bpp = self.task_cfg.bpp

        self.color_gain_div = 10000

    def list_files(self):
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
        ccm_files = []
        for file_mask in self.ccm_files_mask:
            ccm_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        if not ccm_files:
            raise ValueError("No CCM files available")

        return raw_files, ccm_files

    def load_raw_image(self, file_path):
        """Load raw data from file into a 16-bit numpy array."""
        rawImage = np.fromfile(file_path, dtype=np.uint16).astype(np.uint16)
        raw_image = np.reshape(rawImage, (self.height, self.width))

        log.info(f"Loaded raw image from {file_path}")
        log.info(f"Raw image shape: {raw_image.shape}")
        log.info(f"Raw image dtype: {raw_image.dtype}")
        log.info(f"Raw image min: {np.min(raw_image)}")
        log.info(f"Raw image max: {np.max(raw_image)}")

        return raw_image
    
    def configure_dng_tags(self, ccm_file):
        """Set DNG tags for the conversion."""
        t = DNGTags()
        # Image dimensions
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, Orientation.Horizontal)
        
        # Photometric interpretation
        t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.BitsPerSample, self.bpp)
        t.set(Tag.CFARepeatPatternDim, [2, 2])
        t.set(Tag.CFAPattern, CFAPattern.RGGB) # RGGB best so far. RGGB and BGGR similar. GBRG and GRBG very bad
        
        # Black and white levels
        # t.set(Tag.BlackLevel, (4096 >> (16 - self.bpp)))
        t.set(Tag.BlackLevel, 0)
        t.set(Tag.WhiteLevel, 65535)
        # t.set(Tag.BlackLevel, 2048)
        # t.set(Tag.WhiteLevel, 60000)
        # t.set(Tag.WhiteLevel, ((1 << self.bpp) - 1))
        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)

        # Camera properties
        t.set(Tag.Make, "SVS")
        t.set(Tag.Model, "Camera Model")
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
        t.set(Tag.BaselineExposure, [[0,1]])
        # t.set(Tag.BaselineExposure, [[4, 1]])  # Equivalent to +2 EV

        # Color correction matrix
        t.set(Tag.AsShotNeutral, self.get_ashot_neutral())
        # t.set(Tag.ColorMatrix1, self.set_color_correction(ccm_file))

        return t

    def get_ashot_neutral(self):
        """Get as shot neutral values from raw image."""
        color_gain_div = self.color_gain_div
        gain_r = 1
        gain_b = 3
        gain_r = int(gain_r * color_gain_div)
        gain_b = int(gain_b * color_gain_div)
        log.info(f"As Shot Neutral: {gain_r}, {color_gain_div}, {gain_b}")
        # as_shot_neutral = [
        #     [color_gain_div, gain_r], # red
        #     [color_gain_div, color_gain_div], # green
        #     [color_gain_div, gain_b]] # blue
        as_shot_neutral = [
        [color_gain_div, color_gain_div],  # red
        [color_gain_div, color_gain_div],  # green
        [color_gain_div, color_gain_div]   # blue
            ]
        return as_shot_neutral
    
    def set_color_correction(self, ccm_file):
        """Set color correction matrix from the loaded CCM file."""
        color_gain_div = self.color_gain_div
        ccm = np.loadtxt(ccm_file, delimiter=',')
        ccm1 = list()
        for color in ccm.flatten().tolist():
            ccm1.append((int(color * color_gain_div), color_gain_div))
        log.info(f"Loaded CCM from {ccm_file}")
        log.info(f"Color Correction Matrix: {ccm1}")
        return ccm1

    @staticmethod
    def convert_to_dng(raw_image, dng_tags, output_filename):
        """Convert the loaded raw image to DNG format and save to output file."""
        if raw_image is None:
            raise ValueError("Raw image data not loaded.")

        converter = RAW2DNG()
        converter.options(dng_tags, path="", compress=False)
        converter.convert(raw_image, filename=output_filename)

def main(cfg):
    """Main function to initialize converter and process all raw images in the directory."""

    raw2dng_conv = RawToDNGConverter(cfg)
    raw_files, ccm_files = raw2dng_conv.list_files()
    log.info(
        f"Found {len(raw_files)} raw images and {len(ccm_files)} ccm files")
    output_dir = Path(cfg.paths.data_dir) / 'semifield-upload' / cfg.batch_id / 'dngs'
    output_dir.mkdir(parents=True, exist_ok=True)
    for ccm_file in ccm_files:
        for raw_file in raw_files:
            raw_data = raw2dng_conv.load_raw_image(raw_file)
            dns_tags = raw2dng_conv.configure_dng_tags(ccm_file)
            log.info(f"DNG Tags: {dns_tags}")
            output_filename = str(Path(output_dir, f"{raw_file.stem}.dng"))
            raw2dng_conv.convert_to_dng(raw_data, dns_tags, output_filename)
            log.info(f"Converted {raw_file} to {output_filename}")