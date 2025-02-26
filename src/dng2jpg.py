import logging
import subprocess
from pathlib import Path
from omegaconf import DictConfig
from utils import utils

log = logging.getLogger(__name__)

class DngToJpgConverter:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path

    def convert(self, input_path, output_path):
        cmd = [
            "rawtherapee-cli",
            "-O", output_path,
            "-j99",
            "-c", input_path
        ]
        try:
            results = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env={"LANG": "en_US.UTF-8", "OMP_NUM_THREADS": "1"} # limits
                # internal threading of process
            )
            log.info(results.stdout)
            log.info(results.stderr)
        except subprocess.CalledProcessError as e:
            log.error(e)
        return

def main(cfg: DictConfig) -> None:
    """
    Main function to convert dngs to jpgs. Accepts omegaconf dictConfig
    """
    log.info("Converting dngs to jpgs")

    developed_images_folder = utils.locate_lts_location(
        cfg.paths.lts_locations, cfg.batch_id, 'semifield-developed-images')

    dng_folder = developed_images_folder / 'dngs'
    dng_files = []
    for file_mask in cfg.file_masks.dng_files:
        dng_files.extend(list(dng_folder.glob(f"*{file_mask}")))
    log.info(f"Located {len(dng_files)} dng files in "
             f"{developed_images_folder.parent.parent}")

    for dng_file in dng_files:
        output_path = developed_images_folder / f'{dng_file.stem}.jpg'
        dng2jpg_conv = DngToJpgConverter(dng_file, output_path)
        break