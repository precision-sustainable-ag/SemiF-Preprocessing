from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import logging
from omegaconf import DictConfig
from pidng.core import RAW2DNG, DNGTags, Tag

from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.utils import find_lts_dir


log = logging.getLogger(__name__)


class RawToDNGConverter:
    def __init__(self, 
                 exif_cfg: DictConfig,
                 batch_id: str,
                 lts_dir: Path,
                 developed_dng_dir: Path,
                 ccm_file: Path = None):
        """
        Class constructor.
        Separate parameters due to multiprocessing incompatibility of OmegaConf.
        Args:
            exif_cfg (DictConfig): dng tags configuration
            batch_id (str): Batch id
            file_masks (DictConfig): File masks from config
            lts_dir (Path): LTS directory
            ccm_file (Path, optional): Path to the CCM `.npy` file. Defaults to None.
        """
        self.exif_cfgs = exif_cfg
        self.batch_id = batch_id
        
        self.lts_dir = lts_dir
        self.developed_dng_dir = developed_dng_dir
        self.developed_dng_dir.mkdir(parents=True, exist_ok=True)
        self.color_profile = ccm_file  # Path to CCM file
        self.matrix_den = 100000  # Denominator for rational representation
        
        self.height = self.exif_cfgs.DNG_ImageLength
        self.width = self.exif_cfgs.DNG_ImageWidth

        self._load_color_profile()
        
    def load_raw_image(self, file_path):
        """
        Load raw data from file into a 16-bit numpy array.
        Args:
            file_path (Path): Path to the raw image
        """
        raw_image = np.fromfile(file_path, dtype=np.uint16).astype(np.uint16)
        raw_image = np.reshape(raw_image, (self.height, self.width))
        log.debug(f"Loaded raw image from {file_path.name}")
        return raw_image

    def _load_color_profile(self):
        """Loads the CCM from a NumPy `.npy` file if provided."""
        if not self.color_profile or not self.color_profile.exists():
            # Raise error if CCM file is not found
            raise FileNotFoundError(f"CCM file not found: {self.color_profile}")
        log.debug(f"Loading Color profile from {self.color_profile}")
        # Load the color profile
        color_profile = np.load(self.color_profile, allow_pickle=True).item()
        # Extract matrices and gains
        ccm = color_profile["color_matrix"]
        fm = color_profile["forward_matrix"]
        wb_gains = color_profile["wb_gains"]
        
        # Transpose and reshape matrices
        t_ccm = ccm.T
        t_fm = fm.T
        r, g, b = wb_gains
        
        # Convert to rational representation
        self.ccm_rational = [
            [int(round(v * self.matrix_den)), self.matrix_den]
            for v in t_ccm.reshape(-1)
        ]
        self.fm_rational = [
            [int(round(v * self.matrix_den)), self.matrix_den]
            for v in t_fm.reshape(-1)
        ]

        self.as_shot_neutral = [
            [int(round(self.matrix_den / r)), self.matrix_den],
            [int(round(self.matrix_den / g)), self.matrix_den],
            [int(round(self.matrix_den / b)), self.matrix_den],
        ]


    @staticmethod
    def calculate_dt_from_epoch_gmt(file_stem: int) -> str:
        epoch_gmt = int(file_stem.split('_')[-1])
        dt = datetime.fromtimestamp(epoch_gmt, tz=timezone.utc)
        return dt.strftime("%Y:%m:%d %H:%M:%S")
    
    
    def configure_dng_tags(self) -> DNGTags:
        """Set DNG tags for the conversion."""
        t = DNGTags()
        # Imagespecific tags
        t.set(Tag.ImageWidth,  self.exif_cfgs.SVCamImageWidth)
        t.set(Tag.ImageLength, self.exif_cfgs.SVCamImageHeight)
        t.set(Tag.BitsPerSample, self.exif_cfgs.BitsPerSample)
        t.set(Tag.PhotometricInterpretation, self.exif_cfgs.PhotometricInterpretation)
        t.set(Tag.Orientation, self.exif_cfgs.Orientation)
        t.set(Tag.SamplesPerPixel, self.exif_cfgs.SamplesPerPixel)
        t.set(Tag.CFARepeatPatternDim, self.exif_cfgs.CFARepeatPatternDim)
        t.set(Tag.CFAPattern, self.exif_cfgs.CFAPattern)
        t.set(Tag.RowsPerStrip, self.exif_cfgs.RowsPerStrip)

        # Camera specific tags
        t.set(Tag.Make,  self.exif_cfgs.Make)
        t.set(Tag.Model, self.exif_cfgs.Model)
        t.set(Tag.EXIFPhotoBodySerialNumber, self.exif_cfgs.SerialNumber)
        t.set(Tag.EXIFPhotoLensModel, self.exif_cfgs.LensModel)
        t.set(Tag.FocalLength, [[int(self.exif_cfgs.FocalLength * self.matrix_den), self.matrix_den]])  # rational
        # t.set(Tag.FocalLengthIn35mmFormat, self.exif_cfgs.FocalLengthIn35mmFormat)
        t.set(Tag.FocalLengthIn35mmFilm, self.exif_cfgs.FocalLengthIn35mmFilm)  # rational
        t.set(Tag.FNumber, [[int(self.exif_cfgs.FNumber * self.matrix_den), self.matrix_den]])
        t.set(Tag.FocalPlaneXResolution, [[int(self.exif_cfgs.FocalPlaneXResolution * self.matrix_den), self.matrix_den]])
        t.set(Tag.FocalPlaneYResolution, [[int(self.exif_cfgs.FocalPlaneYResolution * self.matrix_den), self.matrix_den]])
        t.set(Tag.FocalPlaneResolutionUnit, [self.exif_cfgs.FocalPlaneResolutionUnit])
        # t.set(Tag.PixelSize, ccfg.PixelSize)

        # DNG Core tags
        t.set(Tag.DNGVersion, self.exif_cfgs.DNGVersion)
        t.set(Tag.DNGBackwardVersion, self.exif_cfgs.DNGBackwardVersion)
        # 16-bit black and white levels
        t.set(Tag.BlackLevel, self.exif_cfgs.BlackLevel)
        t.set(Tag.WhiteLevel, self.exif_cfgs.WhiteLevel)

        # Color
        t.set(Tag.ColorMatrix1, self.ccm_rational)
        t.set(Tag.ColorMatrix2, self.ccm_rational)
        # Forward matrix
        t.set(Tag.ForwardMatrix1, self.fm_rational)
        t.set(Tag.ForwardMatrix2, self.fm_rational)
        # WB settings
        t.set(Tag.AsShotNeutral, self.as_shot_neutral)
        # Other tags
        t.set(Tag.CalibrationIlluminant1, self.exif_cfgs.CalibrationIlluminant1)
        t.set(Tag.PreviewColorSpace, self.exif_cfgs.PreviewColorSpace)
        t.set(Tag.BaselineExposure, [self.exif_cfgs.BaselineExposure])
        
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

        # Final tags having to do with time and file time stamp
        dng_tags.set(Tag.DateTime, self.calculate_dt_from_epoch_gmt(raw_file.stem))
        dng_tags.set(Tag.DateTimeOriginal, self.calculate_dt_from_epoch_gmt(raw_file.stem))

        converter.options(dng_tags, path=str(self.developed_dng_dir), compress=False)
        converter.convert(raw_image, filename=raw_file.stem)
        return self.developed_dng_dir / f"{raw_file.stem}.dng"


class DNGConversionPipeline:
    """
    A pipeline to handle the conversion of raw images to DNG format.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exif_cfg = cfg.exif
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
            args.append((self.exif_cfg, self.batch_id, self.lts_dir, raw_file, self.developed_dng_dir, self.local_ccm_path))
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

def process_image(exif_cfg, batch_id, lts_dir, raw_file, developed_dng_dir, local_ccm_path):
    """
    Multiprocessing function to convert raw image to DNG format in parallel.
    Args:
        exif_cfg (DictConfig): DNG tags config
        batch_id (str): Batch ID
        file_masks (DictConfig): File masks config
        lts_dir (Dict): LTS directory
        raw_file (Path): Raw image to convert
    """
    raw2dng_conv = RawToDNGConverter(exif_cfg, batch_id, lts_dir, developed_dng_dir, local_ccm_path)
    log.debug("Initialized raw to DNG converter.")

    raw_data = raw2dng_conv.load_raw_image(raw_file)
    log.debug("Loaded raw image data.")
    try:
        dns_tags = raw2dng_conv.configure_dng_tags()
    except Exception as e:
        log.exception(f"Error configuring DNG tags: {e}")
        return
    dng_dst_path = raw2dng_conv.convert_to_dng(raw_data, dns_tags, raw_file)
    log.debug(f"Converted {raw_file} to {dng_dst_path}.")
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