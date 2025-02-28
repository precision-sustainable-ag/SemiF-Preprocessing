import logging
import multiprocessing
import os
import subprocess
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from utils import utils

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
    png2jpg_conv = PngToJpgConverter(png_file, output_path, rt_pp3, val_rt_script)
    rt_cli = png2jpg_conv.validate_rawtherapee()
    is_converted = png2jpg_conv.convert(rt_cli)
    del png2jpg_conv
    return is_converted


def main(cfg: DictConfig) -> None:
    """
    Main function to convert pngs to jpgs. Accepts omegaconf dictConfig
    """

    log.info("Converting pngs to jpgs")
    # TODO @jinamshah:
    #  changes around developed images possible if storing pngs temporarily

    # locate batch's developed images folder in LTS
    # TODO: this will fail with error if files not present in LTS
    lts_dir_name = utils.find_lts_dir(cfg.batch_id,
                                                 cfg.paths.lts_locations,
                                                 local=False)
    pngs_folder = (Path(lts_dir_name) / "semifield-developed-images" /
                   cfg.batch_id / "pngs")

    # if not present in LTS locations, pngs must've been created locally
    if not pngs_folder.exists():
        lts_dir_name = utils.find_lts_dir(cfg.batch_id,
                                          cfg.paths.lts_locations,
                                          local=True)
        pngs_folder = (cfg.paths.data_dir / lts_dir_name /
                       "semifield-developed-images" / cfg.batch_id / "pngs")
    if not pngs_folder.exists():
        log.error(f"{cfg.batch_id} doesn't have any png files")
    png_files = []
    for file_mask in cfg.file_masks.png_files:
        png_files.extend(list(pngs_folder.glob(f"*{file_mask}")))
    tasks = []
    for png_file in png_files:
        output_path = pngs_folder.parent / f'{png_file.stem}.jpg'
        tasks.append((png_file, output_path, cfg.png2jpg.rt_pp3,
                      cfg.png2jpg.validate_rt))

    log.info(f"Converting {len(png_files)} png files using multiprocessing")
    start_time = datetime.now()
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
    log.info(f"conversion time: {datetime.now() - start_time}")