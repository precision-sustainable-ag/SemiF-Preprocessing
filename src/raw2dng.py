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
                 ccm_file: Path = None):
        """
        Class constructor.
        Separate parameters due to multiprocessing incompatibility of OmegaConf.
        Args:
            dng_tags_cfg (DictConfig): dng tags configuration
            batch_id (str): Batch id
            file_masks (DictConfig): File masks from config
            lts_dir (Path): LTS directory
            ccm_file (Path, optional): Path to the CCM `.npy` file. Defaults to None.
        """
        self.dng_tags = dng_tags_cfg
        self.batch_id = batch_id
        
        self.lts_dir = lts_dir
        self.developed_dng_dir = developed_dng_dir
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)
        self.ccm_file = ccm_file  # Path to CCM file

        
        self.height = self.dng_tags.ImageLength
        self.width = self.dng_tags.ImageWidth

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

    def load_ccm(self):
        """Loads the CCM from a NumPy `.npy` file if provided."""
        if self.ccm_file and self.ccm_file.exists():
            log.info(f"Loading CCM from {self.ccm_file}")
            ccm = np.load(self.ccm_file)
            return self.format_ccm4pidng(ccm)
        else:
            log.warning("CCM file not found or not provided. Using default color matrix.")
            return [[19549, 10000], [-7877, 10000], [-2582, 10000],    
                    [-5724, 10000], [10121, 10000], [1917, 10000],
                    [-1267, 10000], [-110, 10000], [6621, 10000]]  # Default matrix


    def format_ccm4pidng(self, ccm):
        # Not implemented yet
        ccm1 = []
        for row in ccm:
            row_sum = sum(row)
            normalized_row = [
                (int((value / row_sum) * 10000),
                    10000) for
                value in row]
            ccm1.extend(normalized_row)
        return ccm1

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
        

        # **Load and set the ColorMatrix1**
        ccm1 = self.load_ccm()
        t.set(Tag.ColorMatrix1, ccm1)

        return t

    def convert_to_dng(self, raw_image: np.array, dng_tags: DNGTags,
                       raw_file: str) -> str:
        """
        Convert the loaded raw image to DNG format and save to output file.
        Args:
            raw_image (np.array): Raw image
            dng_tags (DNGTags): DNG tags
            raw_file (str): raw file path
        Returns:
            str: Output filename
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

        self.lts_dir = self._find_lts_dir()
        if not self.lts_dir:
            raise ValueError(f"LTS directory not found for batch {self.batch_id}. Exiting.")

        self.uploads_folder = self.lts_dir / 'semifield-upload' / self.batch_id
        self.raw_files = self._list_files()
        log.info(f"Found {len(self.raw_files)} raw images.")

        self._set_developed_output_folder()

        # CCM Path
        self.ccm_name = f"{self.cfg.ccm_name}.npy"
        self.local_ccm_path = Path(self.cfg.paths.image_development) / "color_matrices" / self.ccm_name


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
            args.append((self.dng_tags_cfg, self.batch_id, self.lts_dir, raw_file, self.developed_dng_dir, self.local_ccm_path))
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

def process_image(dng_tags_cfg, batch_id, lts_dir, raw_file, developed_dng_dir, local_ccm_path):
    """
    Multiprocessing function to convert raw image to DNG format in parallel.
    Args:
        dng_tags_cfg (DictConfig): DNG tags config
        batch_id (str): Batch ID
        file_masks (DictConfig): File masks config
        lts_dir (Dict): LTS directory
        raw_file (Path): Raw image to convert
    """
    raw2dng_conv = RawToDNGConverter(dng_tags_cfg, batch_id, lts_dir, developed_dng_dir, local_ccm_path)
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