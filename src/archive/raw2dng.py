import multiprocessing
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime
from omegaconf import DictConfig
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
from tqdm import tqdm
from typing import Dict

from utils import utils

log = logging.getLogger(__name__)


class RawToDNGConverter:
    def __init__(self, raw2dng_cfg: DictConfig, dng_tags_cfg: DictConfig,
                 batch_id: str, file_masks: DictConfig,
                 paths: Dict) -> None:
        """
        Class constructor.
        Separate parameters due to multiprocessing incompatibility of OmegaConf.
        Args:
            raw2dng_cfg (DictConfig): task specific configuration
            dng_tags_cfg (DictConfig): dng tags configuration
            batch_id (str): Batch id
            file_masks (DictConfig): File masks from config
            paths (Dict): paths configuration
        """
        self.task_cfg = raw2dng_cfg
        self.dng_tags = dng_tags_cfg
        self.batch_id = batch_id
        self.uploads_folder = None
        self.raw_files_mask = file_masks.raw_files
        self.ccm_files_mask = file_masks.ccm_files

        self.height = self.task_cfg.height
        self.width = self.task_cfg.width
        self.num_pixels = self.height * self.width
        self.bpp = self.task_cfg.bpp

        self.uploads_folder = utils.locate_lts_location(paths['lts_locations'],
                                                   self.batch_id,
                                                        "semifield-upload")
        if not self.uploads_folder:
            self.uploads_folder = (Path(paths['data_dir']) /
                                   'semifield-upload' /
                                   self.batch_id)
        if not self.uploads_folder:
            log.error(f"{self.batch_id} doesn't exist")

        self.developed_images_folder = utils.create_developed_images(
            self.uploads_folder)

    def list_files(self):
        """
        Method to list all raw images and ccms available in the batch
        Returns:
            raw_files (List[Path]): list of raw image paths
            ccm_files (List[Path]): saved CCM txt files
        """
        # todo: @jinamshah
        #   lts vs local vs both
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
        """
        Load raw data from file into a 16-bit numpy array.
        Args:
            file_path (Path): Path to the raw image
        """
        raw_image = np.fromfile(file_path, dtype=np.uint16).astype(np.uint16)
        raw_image = np.reshape(raw_image, (self.height, self.width))
        log.info(f"Loaded raw image from {file_path}")
        return raw_image

    def configure_dng_tags(self, ccm_file: Path) -> DNGTags:
        """
        Set DNG tags for the conversion.
        Args:
            ccm_file (Path): CCM file path
        """
        t = DNGTags()
        # dng metadata details
        t.set(Tag.Make, self.dng_tags.camera_make)
        t.set(Tag.Model, self.dng_tags.camera_model)
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
        # Basic image details
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, Orientation.Horizontal)
        # t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.SamplesPerPixel, self.dng_tags.samples_per_pixel)
        t.set(Tag.BitsPerSample, self.bpp)
        # Photometric interpretation
        t.set(Tag.PhotometricInterpretation,
              PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.CFARepeatPatternDim, self.dng_tags.cfa_repeat_pattern_dim)
        t.set(Tag.CFAPattern, CFAPattern.RGGB)
        # Image calibration
        t.set(Tag.BlackLevel, self.dng_tags.black_level)
        t.set(Tag.WhiteLevel, self.dng_tags.white_level)
        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
        t.set(Tag.BaselineExposure, self.dng_tags.baseline_exposure)
        # exposure configuration per color channel
        as_shot_neutral = [
            [int(self.dng_tags.gain_r * self.dng_tags.color_gain_div),
             self.dng_tags.color_gain_div],
            [int(self.dng_tags.gain_g * self.dng_tags.color_gain_div),
             self.dng_tags.color_gain_div],
            [int(self.dng_tags.gain_b * self.dng_tags.color_gain_div),
             self.dng_tags.color_gain_div]
        ]
        t.set(Tag.AsShotNeutral, as_shot_neutral)
        # apply normalized color correction matrix to image
        ccm = np.loadtxt(ccm_file, delimiter=',')
        ccm1 = []
        for row in ccm:
            row_sum = sum(row)
            normalized_row = [
                (int(value / row_sum * self.dng_tags.color_gain_div),
                 self.dng_tags.color_gain_div) for
                value in row]
            ccm1.extend(normalized_row)
        t.set(Tag.ColorMatrix1, ccm1)
        return t

    def convert_to_dng(self, raw_image: np.array, dng_tags: DNGTags,
                       output_filename: str) -> str:
        """
        Convert the loaded raw image to DNG format and save to output file.
        Args:
            raw_image (np.array): Raw image
            dng_tags (DNGTags): DNG tags
            output_filename (str): Output filename
        Returns:
            str: Output filename
        """
        if raw_image is None:
            raise ValueError("Raw image data not loaded.")
        os.makedirs(os.path.join(self.developed_images_folder, 'dngs'),
                    exist_ok=True)
        output_filename = os.path.join(self.developed_images_folder, 'dngs',
                                       output_filename)
        converter = RAW2DNG()
        converter.options(dng_tags, path="", compress=False)
        converter.convert(raw_image, filename=output_filename)
        return output_filename


def proc_wrapper(args):
    """
    Wrapper around process_image for async processing.
    """
    return process_image(*args)


def process_image(raw2dng_cfg, dng_tags_cfg, batch_id, file_masks, paths,
                  raw_file, ccm_file):
    """
    Multiprocessing function to convert raw image to DNG format in parallel.
    Args:
        raw2dng_cfg (DictConfig): Raw DNG config
        dng_tags_cfg (DictConfig): DNG tags config
        batch_id (str): Batch ID
        file_masks (DictConfig): File masks config
        paths (Dict): Paths
        raw_file (Path): Raw image to convert
        ccm_file (Path): Corresponding CCM file for raw image
    """
    raw2dng_conv = RawToDNGConverter(raw2dng_cfg, dng_tags_cfg, batch_id,
                                     file_masks, paths)
    raw_data = raw2dng_conv.load_raw_image(raw_file)
    dns_tags = raw2dng_conv.configure_dng_tags(ccm_file)
    output_filename = f"{raw_file.stem}.dng"
    raw2dng_conv.convert_to_dng(raw_data, dns_tags, output_filename)


def main(cfg: DictConfig) -> None:
    """
    Main function to initialize converter and process all raw images in the directory.
    Args:
        cfg (DictConfig): Configuration dictionary
    """

    raw2dng_cfg = cfg.raw2dng
    dng_tags_cfg = cfg.dng_tags
    batch_id = cfg.batch_id
    file_masks = cfg.file_masks
    paths = dict(cfg.paths)
    raw2dng_conv = RawToDNGConverter(raw2dng_cfg, dng_tags_cfg, batch_id,
                                     file_masks, paths)
    raw_files, ccm_files = raw2dng_conv.list_files()
    log.info(
        f"Found {len(raw_files)} raw images and {len(ccm_files)} ccm files")
    del raw2dng_conv

    # todo: @jinamshah
    #   multiple ccm per batch or not / per season/ per species
    #   for now, applying the first ccm to all images
    args = []
    for ccm_file in ccm_files:
        for raw_file in raw_files:
            args.append((raw2dng_cfg, dng_tags_cfg, batch_id,
                                 file_masks, paths, raw_file, ccm_file))
        break
    log.info(f"Generated arguments for processing raw images in parallel")

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(cfg.max_workers) as pool:
        total_tasks = len(args)
        results = list(tqdm(pool.imap_unordered(proc_wrapper, args),
                            total=total_tasks, desc="converted dngs"))
        tuple(results)
    log.info(f"{len(args)} raw images convert to dng format")
