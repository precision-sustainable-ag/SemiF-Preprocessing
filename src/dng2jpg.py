import logging
import os
import subprocess
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class DNGToJpgConverter:
    """
    Converts a dng file to a jpg file (100% quality) using rawtherapee-cli
    """
    def __init__(self, input_path: Path, output_path: Path,
                 pp3_file: Path, val_rt_script: Path) -> None:
        """
        Class constructor for each image
        """
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.pp3_file = str(pp3_file.resolve())
        self.val_rt_script = str(val_rt_script.resolve())

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
        Convert dng to jpg using rawtherapee and predefined profile
        Args:
            rt_cli: path to rawtherapee-cli (verified installation)
        Returns true if converted successfully
        """

        if not rt_cli:
            return False
        cmd = [
            rt_cli,
            "-O", self.output_path,
            "-p", self.pp3_file,
            "-j100",
            "-js3",
            "-c", self.input_path
        ]
        try:
            max_threads = 50  # Total number of threads to use
            num_instances = 12  # Expected number of parallel threads
            threads_per_instance = max(1, max_threads // num_instances) #
            # prevents oversubscription of resources

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
            log.error(f"Error converting dng to jpg: {e}")
            return False


def process_image(args: tuple) -> bool:
    """
    Multiprocessing wrapper to convert each image to jpg
    Args:
        args: tuple (dng_file, output_path, rt_pp3)
    Returns:
        is_converted (bool): true if converted successfully
    """
    dng_file, output_path, rt_pp3, val_rt_script = args
    log.info(f"Converting {dng_file.name} to jpg")
    dng2jpg_conv = DNGToJpgConverter(dng_file, output_path, rt_pp3,
                                     val_rt_script)
    rt_cli = dng2jpg_conv.validate_rawtherapee()
    is_converted = dng2jpg_conv.convert(rt_cli)
    if is_converted:
        log.info(f"Successfully converted {dng_file} to jpg")
    else:
        log.warning(f"Failed to convert {dng_file} to jpg")
    del dng2jpg_conv    # explicit garbage collection
    return is_converted


def main(cfg: DictConfig) -> None:
    """
    Entry point for converting dng to jpg.
    This function is only used if dng2jpg is run as a separate task.
    """
    log.info("Converting dngs to jpgs")
    
    # check local storage for converted dng files
    lts_dir = find_lts_dir(cfg.batch_id,
                                cfg.paths.lts_locations,
                                local=False, developed=True, dngs=True)
    if not lts_dir:
        log.error(f"Couldn't find {cfg.batch_id} in any of the NFS locations")
        return
    
    # get and check if dngs folder exists
    dngs_folder = lts_dir / "semifield-developed-images" / cfg.batch_id / "dngs"
    
    # get all dng files
    dng_files = []
    for file_mask in cfg.file_masks.dng_files:
        dng_files.extend(list(dngs_folder.glob(f"*{file_mask}")))
    log.info(f"Converting {len(dng_files)} dng files to jpgs")

    # identify output location and create args for multiprocessing
    output_dir = Path(lts_dir) / "semifield-developed-images" / cfg.batch_id / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # create tasks for multiprocessing
    pp3_path = Path(cfg.paths.image_development) / "dev_profiles" / f"{cfg.dng2jpg.rt_pp3_name}.pp3"
    validate_rt_cli_script = Path(cfg.paths.scripts) / "validate_rawtherapee.sh"
    tasks = [(dng_file, output_dir / f'{dng_file.stem}.jpg', pp3_path, validate_rt_cli_script) for dng_file in dng_files]
    # convert dngs to jpgs
    results = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        future_to_task = {executor.submit(process_image, task): task for task in tasks}
        for future in as_completed(future_to_task):
            results.append(future.result())

    if sum(results) == len(tasks):
        log.info("All dng files converted successfully")
    else:
        log.warning(f"Failed to convert {len(results) - sum(results)} dngs")