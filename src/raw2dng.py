from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
import numpy as np
import struct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import logging
import random
import cv2

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

    # def __init__(self, raw_file_path: str, output_filename: str, height: int, width: int, bpp: int = 12):
    #     self.raw_file_path = raw_file_path
    #     self.output_filename = output_filename
    #     self.height = height
    #     self.width = width
    #     self.bpp = bpp
    #     self.num_pixels = width * height
    #     self.raw_image = None
    #     self.dng_tags = DNGTags()

    # Set default color matrix for demo purposes
    # self.ccm1 = [
    #     [19549, 10000], [-7877, 10000], [-2582, 10000],
    #     [-5724, 10000], [10121, 10000], [1917, 10000],
    #     [-1267, 10000], [-110, 10000], [6621, 10000]
    # ]
    def list_files(self):
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
        ccm_files = []
        for file_mask in self.ccm_files_mask:
            ccm_files.extend(list(self.uploads_folder.glob(f"*{file_mask}")))
        if not ccm_files:
            raise ValueError("No CCM files available")

        return raw_files, ccm_files

    def load_raw_image(self, file_path):
        """Load raw data from file into a 16-bit numpy array."""
        with open(file_path, mode='rb') as rf:
            raw_data = struct.unpack("H" * self.num_pixels,
                                     rf.read(2 * self.num_pixels))

        raw_flat_image = np.zeros(self.num_pixels, dtype=np.uint16)
        raw_flat_image[:] = raw_data[:]
        raw_image = np.reshape(raw_flat_image, (self.height, self.width))
        # raw_image = raw_image >> (16 - self.bpp)

        # rawImage = np.fromfile(file_path, dtype=np.uint16).astype(np.uint16)
        # rawImage = rawImage.reshape((self.height, self.width))
        # rawImage_rgb = cv2.cvtColor(rawImage, cv2.COLOR_BayerBG2RGB)
        # rawImageRGBFloat = rawImage_rgb.astype(np.float64) / 65535.0
        #
        # # img_float = np.dot(rawImageRGBFloat, color_matrix) * 20000.0
        # img_float = rawImageRGBFloat
        # print(np.max(np.max(img_float)))
        # img_float[img_float < 0] = 0
        # img_float[img_float > 65535] = 65535.0
        # colour_image_gamma_adjusted = img_float.astype(np.uint16)
        #
        # inv_gamma = 1.0 / 2
        # # inv_gamma = 1.0 / gamma
        # table = ((np.arange(0, 65536.0) / 65535.0) ** inv_gamma) * 65535.0
        # # Ensure table is 16-bit
        # table = table.astype(np.uint16)
        # colour_image_gamma_adjusted = table[colour_image_gamma_adjusted]
        # raw_image = colour_image_gamma_adjusted

        log.info(f"Loaded raw image from {file_path}")
        log.info(f"Raw image shape: {raw_image.shape}")
        log.info(f"Raw image dtype: {raw_image.dtype}")
        log.info(f"Raw image min: {np.min(raw_image)}")
        log.info(f"Raw image max: {np.max(raw_image)}")

        return raw_image

    def configure_dng_tags(self, ccm_file):
        """Set DNG tags for the conversion."""
        t = DNGTags()
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, Orientation.Horizontal)

        t.set(Tag.PhotometricInterpretation,
              PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.BitsPerSample, self.bpp)
        t.set(Tag.CFARepeatPatternDim, [2, 2])
        t.set(Tag.CFAPattern, CFAPattern.RGGB)

        # t.set(Tag.BlackLevel, (4096 >> (16 - self.bpp)))
        t.set(Tag.BlackLevel, 0)
        t.set(Tag.WhiteLevel, 65535)
        # t.set(Tag.WhiteLevel, ((1 << self.bpp) - 1))

        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
        t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
        # t.set(Tag.BaselineExposure, [[-150, 3000]])
        t.set(Tag.BaselineExposure, [[-150, 100]])
        t.set(Tag.Make, "SVS")
        t.set(Tag.Model, "Camera Model")
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)

        # todo: @jinamshah
        #   needs verification (especially around color gain div)

        color_gain_div = 10000
        gain_r = 2
        gain_b = 1
        gain_r = int(gain_r * color_gain_div)
        gain_b = int(gain_b * color_gain_div)
        as_shot_neutral = [[color_gain_div, gain_r],
                           [color_gain_div, color_gain_div],
                           [color_gain_div, gain_b]]

        t.set(Tag.AsShotNeutral, as_shot_neutral)
        # t.set(Tag.BaselineExposure, [[1,1]])

        ccm = np.loadtxt(ccm_file, delimiter=',')
        ccm1 = list()
        for color in ccm.flatten().tolist():
            ccm1.append((int(color * color_gain_div), color_gain_div))
        t.set(Tag.ColorMatrix1, ccm1)
        return t

    @staticmethod
    def convert_to_dng(raw_image, dng_tags, output_filename):
        """Convert the loaded raw image to DNG format and save to output file."""
        if raw_image is None:
            raise ValueError("Raw image data not loaded.")

        converter = RAW2DNG()
        converter.options(dng_tags, path="", compress=False)
        converter.convert(raw_image, filename=output_filename)


# def process_file(raw_file_path, output_dir, height, width, bpp):
#     """Process a single raw file and convert it to DNG."""
#     output_filename = f"{raw_file_path.stem}.dng"
#     output_path = output_dir / output_filename
#
#     # Initialize and use the converter for each file
#     converter = RawToDNGConverter(
#         raw_file_path=str(raw_file_path),
#         output_filename=str(output_path),
#         height=height,
#         width=width,
#         bpp=bpp
#     )
#
#     converter.load_raw_image()
#     converter.configure_dng_tags()
#     converter.convert_to_dng()
#     print(f"Converted {raw_file_path} to {output_filename}")

def main(cfg):
    """Main function to initialize converter and process all raw images in the directory."""

    raw2dng_conv = RawToDNGConverter(cfg)
    raw_files, ccm_files = raw2dng_conv.list_files()
    log.info(
        f"Found {len(raw_files)} raw images and {len(ccm_files)} ccm files")

    # todo: @jinamshah
    #   multiple ccm per batch or not / per season/ per species
    for ccm_file in ccm_files:
        for raw_file in raw_files:
            raw_data = raw2dng_conv.load_raw_image(raw_file)
            dns_tags = raw2dng_conv.configure_dng_tags(ccm_file)
            output_filename = f"./data/test/{raw_file.stem}.dng"
            raw2dng_conv.convert_to_dng(raw_data, dns_tags, output_filename)
            log.info(f"Converted {raw_file} to {output_filename}")
            break
        break

    # input_dir = Path(cfg.paths.local_upload, cfg.batch_id)
    # output_dir = Path(cfg.paths.semif_developed, cfg.batch_id, "dng")
    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # height, width, bpp = 9528, 13376, 16
    #
    # # List all .RAW files in the input directory
    # raw_files = list(input_dir.glob("*.RAW"))
    # # seed = random.randint(0,1000)
    # seed = 914
    # print(seed)
    # random.seed(seed)
    #
    # raw_files = random.sample(raw_files, 50)
    # log.info(f"Processing {len(raw_files)} raw files in parallel.")
    #
    # # Prepare the partial function with fixed arguments for process_file
    # process_file_partial = partial(process_file, output_dir=output_dir, height=height, width=width, bpp=bpp)
    #
    # # Use ProcessPoolExecutor with map to process files concurrently
    # with ProcessPoolExecutor(max_workers=12) as executor:
    #     executor.map(process_file_partial, raw_files)
    # print(seed)
