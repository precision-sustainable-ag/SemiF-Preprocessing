import logging
import multiprocessing
import os
import shutil
import subprocess
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class PngToJpgConverter:
    def __init__(self, input_path: Path, output_path: Path,
                 pp3_file: str, val_rt_script: str) -> None:
        """
        Class constructor for each image
        """
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.pp3_file = str(Path(pp3_file).resolve())
        self.val_rt_script = str(Path(val_rt_script).resolve())

    def validate_rawtherapee(self) -> str | None:
        """
        Verify rawtherapee installation, install if not present
        Returns absolute path of rawtherapee-cli
        """
        try:
            result = subprocess.run(
                [self.val_rt_script],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split("'")[1]
        except subprocess.CalledProcessError as e:
            log.error(f"error validating rawtherapee: {e}")
            return None

    def convert(self, rt_cli: str) -> bool:
        """
        Convert png to jpg using rawtherapee and predefined profile
        Returns true if converted successfully
        """
        if not rt_cli:
            return False
        cmd = [
            rt_cli,
            "-O", self.output_path,
            "-p", self.pp3_file,
            "-j99", "-Y",
            "-c", self.input_path
        ]
        try:
            max_threads = os.cpu_count()  # Total available cores
            num_instances = 4  # Number of parallel conversions
            threads_per_instance = max(1, max_threads // num_instances)

            # Set environment per process
            env = {
                **os.environ,
                "LANG": "en_US.UTF-8",
                "OMP_NUM_THREADS": str(threads_per_instance),
                "OMP_DYNAMIC": "TRUE",  # Allows OpenMP to optimize thread count
                "OMP_NESTED": "FALSE"  # Disables nested parallelism
            }
            _ = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Error converting png to jpg: {e}")
            return False


def process_image(args: tuple) -> bool:
    """
    Multiprocessing wrapper to convert each image to jpg
    Args:
        args: tuple (png_file, output_path, rt_pp3)
    Returns:
        is_converted (bool): true if converted successfully
    """
    png_file, output_path, rt_pp3, val_rt_script = args
    png2jpg_conv = PngToJpgConverter(png_file, output_path, rt_pp3,
                                     val_rt_script)
    rt_cli = png2jpg_conv.validate_rawtherapee()
    is_converted = png2jpg_conv.convert(rt_cli)
    del png2jpg_conv
    return is_converted


def main(cfg: DictConfig) -> None:
    """
    Main function to convert pngs to jpgs. Accepts omegaconf dictConfig
    """

    log.info("Converting pngs to jpgs")
    # check local storage for converted png files
    lts_dir = find_lts_dir(cfg.batch_id,
                                cfg.paths.lts_locations,
                                local=True)
    pngs_folder = (cfg.paths.data_dir / lts_dir.name /
                   "semifield-developed-images" / cfg.batch_id / "pngs")
    if not pngs_folder.exists():
        log.error(f"{cfg.batch_id} doesn't have any png files")
    png_files = []
    for file_mask in cfg.file_masks.png_files:
        png_files.extend(list(pngs_folder.glob(f"*{file_mask}")))

    # identify output location and create args for multiprocessing
    output_dir = Path(lts_dir) / "semifield-developed-images" / cfg.batch_id
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Converting {len(png_files)} png files to jpgs")
    tasks = [
        (png_file, output_dir/f'{png_file.stem}.jpg', cfg.png2jpg.rt_pp3,
         cfg.png2jpg.validate_rt) for png_file in png_files
    ]
    with multiprocessing.Pool() as pool:
        results = []
        with tqdm(total=len(tasks), desc="pngs converted") as pbar:
            for result in pool.imap_unordered(process_image, tasks):
                results.append(result)
                pbar.update()
    if sum(results) == len(tasks):
        log.info(f"All png files converted successfully")
    else:
        log.warning(f"Failed to convert {len(results) - sum(results)} pngs")

    if cfg.png2jpg.remove_pngs:
        if "research-project" in str(png_files[0]) or "screberg" in str(
                png_files[0]):
            log.warning(
                "Refusing to remove file from LTS research-project directory.")
        else:
            shutil.rmtree(pngs_folder)
            log.info(f"Deleted local pngs: {pngs_folder}")