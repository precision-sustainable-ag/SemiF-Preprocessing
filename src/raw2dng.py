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

log = logging.getLogger(__name__)

class RawToDNGConverter:
    def __init__(self, raw_file_path: str, output_filename: str, height: int, width: int, bpp: int = 12):
        self.raw_file_path = raw_file_path
        self.output_filename = output_filename
        self.height = height
        self.width = width
        self.bpp = bpp
        self.num_pixels = width * height
        self.raw_image = None
        self.dng_tags = DNGTags()
        
        # Set default color matrix for demo purposes
        # self.ccm1 = [
        #     [19549, 10000], [-7877, 10000], [-2582, 10000],
        #     [-5724, 10000], [10121, 10000], [1917, 10000],
        #     [-1267, 10000], [-110, 10000], [6621, 10000]
        # ]
        
    def load_raw_image(self):
        """Load raw data from file into a 16-bit numpy array."""
        with open(self.raw_file_path, mode='rb') as rf:
            raw_data = struct.unpack("H" * self.num_pixels, rf.read(2 * self.num_pixels))
        
        raw_flat_image = np.zeros(self.num_pixels, dtype=np.uint16)
        raw_flat_image[:] = raw_data[:]
        self.raw_image = np.reshape(raw_flat_image, (self.height, self.width))
        self.raw_image = self.raw_image >> (16 - self.bpp)
        log.info(f"Loaded raw image from {self.raw_file_path}")
        log.info(f"Raw image shape: {self.raw_image.shape}")
        log.info(f"Raw image dtype: {self.raw_image.dtype}")
        log.info(f"Raw image min: {np.min(self.raw_image)}")
        log.info(f"Raw image max: {np.max(self.raw_image)}")

    def configure_dng_tags(self):
        """Set DNG tags for the conversion."""
        t = self.dng_tags
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, Orientation.Horizontal)
        t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.BitsPerSample, self.bpp)
        t.set(Tag.CFARepeatPatternDim, [2, 2])
        t.set(Tag.CFAPattern, CFAPattern.RGGB)
        t.set(Tag.BlackLevel, (4096 >> (16 - self.bpp)))
        t.set(Tag.WhiteLevel, ((1 << self.bpp) - 1))
        # t.set(Tag.ColorMatrix1, self.ccm1)
        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
        t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
        t.set(Tag.BaselineExposure, [[-150, 3000]])
        t.set(Tag.Make, "Camera Brand")
        t.set(Tag.Model, "Camera Model")
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)

    def convert_to_dng(self):
        """Convert the loaded raw image to DNG format and save to output file."""
        if self.raw_image is None:
            raise ValueError("Raw image data not loaded.")
        
        converter = RAW2DNG()
        converter.options(self.dng_tags, path="", compress=False)
        converter.convert(self.raw_image, filename=self.output_filename)

def process_file(raw_file_path, output_dir, height, width, bpp):
    """Process a single raw file and convert it to DNG."""
    output_filename = f"{raw_file_path.stem}.dng"
    output_path = output_dir / output_filename

    # Initialize and use the converter for each file
    converter = RawToDNGConverter(
        raw_file_path=str(raw_file_path),
        output_filename=str(output_path),
        height=height,
        width=width,
        bpp=bpp
    )
    
    converter.load_raw_image()
    converter.configure_dng_tags()
    converter.convert_to_dng()
    print(f"Converted {raw_file_path} to {output_filename}")

def main(cfg):
    """Main function to initialize converter and process all raw images in the directory."""
    input_dir = Path(cfg.paths.local_upload, cfg.batch_id)
    output_dir = Path(cfg.paths.semif_developed, cfg.batch_id, "dng")
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width, bpp = 9528, 13376, 16
    
    # List all .RAW files in the input directory
    raw_files = list(input_dir.glob("*.RAW"))
    # seed = random.randint(0,1000)
    seed = 914
    print(seed)
    random.seed(seed)

    raw_files = random.sample(raw_files, 50)
    log.info(f"Processing {len(raw_files)} raw files in parallel.")

    # Prepare the partial function with fixed arguments for process_file
    process_file_partial = partial(process_file, output_dir=output_dir, height=height, width=width, bpp=bpp)

    # Use ProcessPoolExecutor with map to process files concurrently
    with ProcessPoolExecutor(max_workers=12) as executor:
        executor.map(process_file_partial, raw_files)
    print(seed)