import os
import shutil
import numpy as np
from pathlib import Path
import logging
from omegaconf import DictConfig
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import DNGVersion, PreviewColorSpace, Orientation, PhotometricInterpretation, CFAPattern, CalibrationIlluminant

from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.utils import find_lts_dir


log = logging.getLogger(__name__)


class RawToDNGConverter:
    def __init__(self, 
                 dng_tags_cfg: DictConfig,
                 batch_id: str,
                 lts_dir: Path,
                 developed_dng_dir: Path,
                 ccm9x9: np.array
                 ) -> None:
        """
        Class constructor.
        Separate parameters due to multiprocessing incompatibility of OmegaConf.
        Args:
            dng_tags_cfg (DictConfig): dng tags configuration
            batch_id (str): Batch id
            file_masks (DictConfig): File masks from config
            lts_dir (Path): LTS directory
        """
        self.dng_tags = dng_tags_cfg
        self.batch_id = batch_id
        
        self.lts_dir = lts_dir
        self.developed_dng_dir = developed_dng_dir
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)

        
        self.height = self.dng_tags.ImageLength
        self.width = self.dng_tags.ImageWidth

        self.ccm9x9 = ccm9x9


    def load_raw_image(self, file_path):
        """
        Load raw data from file into a 16-bit numpy array.
        Args:
            file_path (Path): Path to the raw image
        """
        raw_image = np.fromfile(file_path, dtype=np.uint16).astype(np.uint16)
        raw_image = np.reshape(raw_image, (self.height, self.width))
        log.info(f"Loaded raw image from {file_path.name}")
        return raw_image

    @staticmethod
    def get_3x3_linear(transformation_matrix: np.ndarray) -> np.ndarray:
        """
        Gets the first 3 columns of ccm to get linear rgb relationships.
        Returns: 3x3 array representing the ccm
        """
        return transformation_matrix[:3, :3]

    @staticmethod
    def compress_to_3x3(transformation_matrix):
        """
        Assigns weights to linear, quadratic and cubic rgb relationships.
        Returns: 3x3 array representing the ccm
        """
        # todo: needs calibartion
        linear_weights = [0.6, 0.3, 0.1]
        return np.sum([w * transformation_matrix[i * 3:(i + 1) * 3, :3]
                       for i, w in enumerate(linear_weights)], axis=0)


    def format_ccm4pidng(self,ccm_9x9):
        """Convert 9x9 CCM to PIDNG-compatible 3x3 format"""
        # either get the linear channels or compress all channels
        # ccm_3x3 = self.get_3x3_linear(ccm_9x9)
        ccm_3x3 = self.compress_to_3x3(ccm_9x9)
        normalized_ccm = []
        for row in ccm_3x3:
            row_sum = np.sum(row)
            normalized_row = [[int((value / row_sum) * 10000), 10000] for value in row]
            normalized_ccm.extend(normalized_row)
        return normalized_ccm

    def configure_dng_tags(self) -> DNGTags:
        """Set DNG tags for the conversion."""
        t = DNGTags()
        # DNG metadata details
        t.set(Tag.Make, self.dng_tags.Make)
        t.set(Tag.Model, self.dng_tags.Model)
        t.set(Tag.DNGVersion, getattr(DNGVersion, self.dng_tags.DNGVersion))
        t.set(Tag.DNGBackwardVersion, getattr(DNGVersion, self.dng_tags.DNGBackwardVersion))
        t.set(Tag.PreviewColorSpace, getattr(PreviewColorSpace, self.dng_tags.PreviewColorSpace))
        
        # Basic image details
        t.set(Tag.ImageWidth, self.width)
        t.set(Tag.ImageLength, self.height)
        t.set(Tag.TileWidth, self.width)
        t.set(Tag.TileLength, self.height)
        t.set(Tag.Orientation, getattr(Orientation, self.dng_tags.Orientation))
        t.set(Tag.FocalLength, [[self.dng_tags.FocalLength, 1]])
        t.set(Tag.FocalLengthIn35mmFilm, self.dng_tags.FocalLengthIn35mmFilm)

        # t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.SamplesPerPixel, self.dng_tags.SamplesPerPixel)
        t.set(Tag.BitsPerSample, self.dng_tags.BitsPerSample)
        
        # Photometric interpretation
        t.set(Tag.PhotometricInterpretation, getattr(PhotometricInterpretation, self.dng_tags.PhotometricInterpretation))
        t.set(Tag.CFARepeatPatternDim, self.dng_tags.CFARepeatPatternDim)
        t.set(Tag.CFAPattern, getattr(CFAPattern, self.dng_tags.CFAPattern))
        # Image calibration
        t.set(Tag.BlackLevel, self.dng_tags.BlackLevel)
        t.set(Tag.WhiteLevel, self.dng_tags.WhiteLevel)
        t.set(Tag.CalibrationIlluminant1, getattr(CalibrationIlluminant, self.dng_tags.CalibrationIlluminant1))
        t.set(Tag.BaselineExposure, [self.dng_tags.BaselineExposure])
        t.set(Tag.AsShotNeutral, self.dng_tags.AsShotNeutral)
        
        # TODO: Implement our own ccm instead of this standard one
        # uncalibrated color matrix, just for demo.
        ccm1 = [[19549, 10000], [-7877, 10000], [-2582, 10000],
           [-5724, 10000], [10121, 10000], [1917, 10000],
           [-1267, 10000], [ -110, 10000], [ 6621, 10000]]
        # ccm1 = self.format_ccm4pidng(self.ccm9x9)
        t.set(Tag.ColorMatrix1, ccm1)
        return t

    def convert_to_dng(self, raw_image: np.array, dng_tags: DNGTags,
                       raw_file: Path) -> Path:
        """
        Convert the loaded raw image to DNG format and save to output file.
        Args:
            raw_image (np.array): Raw image
            dng_tags (DNGTags): DNG tags
            raw_file (Path): raw file path
        Returns:
            Path: Output filename
        """
        if raw_image is None:
            raise ValueError("Raw image data not loaded.")
        
        converter = RAW2DNG()
        converter.options(dng_tags, path=str(self.developed_dng_dir), compress=False)
        converter.convert(raw_image, filename=raw_file.stem)
        return self.developed_dng_dir / f"{raw_file.stem}.dng"


class DNGConversionPipeline:
    """
    A pipeline to handle the conversion of raw images to DNG format.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dng_tags_cfg = cfg.dng_tags
        self.batch_id = cfg.batch_id
        self.file_masks = cfg.file_masks
        self.local_data_dir = Path(cfg.paths.data_dir)
        # get calculated 9x9 color correction matrix
        self.ccm9x9 = self._get_ccm9x9()

        self.lts_dir = self._find_lts_dir()
        if not self.lts_dir:
            raise ValueError(f"LTS directory not found for batch {self.batch_id}. Exiting.")

        self.uploads_folder = self.lts_dir / 'semifield-upload' / self.batch_id
        self.raw_files = self._list_files()
        log.info(f"Found {len(self.raw_files)} raw images.")

        self._set_developed_output_folder()

    def _get_ccm9x9(self) -> Path:
        ccm_path = Path(self.cfg.paths.image_development) / "color_matrix" / f"{self.cfg.ccm_name}.npz"
        os.makedirs(ccm_path.parent, exist_ok=True)
        if not ccm_path.exists():
            profiles_backup = self.cfg.paths.img_dev_lts_bkp
            log.warning(f"Color matrix file not found locally, copying from {profiles_backup}")
            color_matrix_backup_path = Path(profiles_backup) / "color_matrix" / f"{self.cfg.ccm_name}.npz"
            shutil.copy(color_matrix_backup_path, ccm_path)
        return np.load(ccm_path)['matrix']

    def _find_lts_dir(self) -> Path:
        """Locate the long-term storage directory."""
        lts_dir = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=False)
        if lts_dir is None:
            log.error(f"LTS directory not found for batch {self.batch_id}. Exiting.")
        return lts_dir
    
    def _set_developed_output_folder(self):
        # Create and set the DNG folder in the developed batch folder
        self.developed_dng_dir = self.local_data_dir / self.lts_dir.name / 'semifield-developed-images' / self.batch_id / 'dngs'
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)

    def _list_files(self) -> Tuple[List[Path], List[Path]]:
        """List all raw image files and CCM files in the batch."""
        raw_files = [file for mask in self.file_masks.raw_files for file in self.uploads_folder.glob(f"*{mask}")]
        return raw_files

    def _generate_processing_args(self) -> List[Tuple]:
        """
        Generate arguments for processing raw images in parallel.
        Currently applying the first CCM file to all images.
        """
        # todo: @jinamshah
        #   multiple ccm per batch or not / per season/ per species
        #   for now, applying the first ccm to all images
        args = []
        
        for raw_file in self.raw_files:
            args.append((self.dng_tags_cfg, self.batch_id, self.lts_dir, raw_file, self.developed_dng_dir, self.ccm9x9))
        return args

    def run(self, multiproc: bool = False) -> None:
        """Run the DNG conversion pipeline."""
        args = self._generate_processing_args()
        
        log.info(f"Processing {len(args)} raw images")

        if multiproc:
            self._process_in_parallel(args)
        
        else:
            self._process_sequentially(args)

        log.info("All raw images converted to DNG format.")

    def _process_in_parallel(self, args: List[Tuple]) -> None:
        """
        Process images using multiprocessing.
        """
        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            future_to_file = {executor.submit(process_image, *arg): arg[3] for arg in args}

            for future in as_completed(future_to_file):
                raw_file = future_to_file[future]
                try:
                    output_name = future.result()
                    log.info(f"Successfully converted {raw_file} to DNG format.")
                except Exception as e:
                    log.error(f"Error processing {raw_file}: {e}")

    def _process_sequentially(self, args: List[Tuple]) -> None:
        """
        Process images sequentially.
        """
        log.info("Processing images sequentially.")
        for arg in args:
            try:
                process_image(*arg)
                log.info(f"Successfully converted {arg[3]} to DNG format.")
            except Exception as e:
                log.exception(f"Error processing {arg[3]}")

def process_image(dng_tags_cfg, batch_id, lts_dir, raw_file, developed_dng_dir, ccm_9x9):
    """
    Multiprocessing function to convert raw image to DNG format in parallel.
    Args:
        dng_tags_cfg (DictConfig): DNG tags config
        batch_id (str): Batch ID
        file_masks (DictConfig): File masks config
        lts_dir (Dict): LTS directory
        raw_file (Path): Raw image to convert
        ccm_9x9 (np.array): 9x9 of the original color correction matrix
        has three components: linear, quadratic and cubic rgb relations
        linear: Basic color channel mixing (cols 1-3)
        quadratic: Non-linear intensity response (cols 4-6)
        cubic: High-order non-linearities (cols 7-9)
    """

    raw2dng_conv = RawToDNGConverter(dng_tags_cfg, batch_id, lts_dir, developed_dng_dir, ccm_9x9)
    log.debug("Initialized raw to DNG converter.")

    raw_data = raw2dng_conv.load_raw_image(raw_file)
    log.debug("Loaded raw image data.")
    try:
        dns_tags = raw2dng_conv.configure_dng_tags()
    except Exception as e:
        log.exception(f"Error configuring DNG tags: {e}")
        return
    dng_dst_path = raw2dng_conv.convert_to_dng(raw_data, dns_tags, raw_file)
    log.info(f"Converted {raw_file} to {dng_dst_path}.")
    return dng_dst_path

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and run the DNG conversion pipeline.
    """
    try:
        pipeline = DNGConversionPipeline(cfg)
        pipeline.run(multiproc=True)  # Set to True for multiprocessing
    except Exception as e:
        log.error(f"Pipeline failed: {e}")