import logging
import subprocess
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig
from utils import utils

log = logging.getLogger(__name__)


class PngToJpgConverter:
    def __init__(self, input_path: Path, output_path: Path,
                 pp3_file: str) -> None:
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.pp3_file = str(Path(pp3_file).resolve())

    def convert(self):
        rt_path = Path(
            "./scripts/squashfs-root/usr/bin/rawtherapee-cli").resolve()
        cmd = [
            # "rawtherapee-cli",
            str(rt_path),
            "-O", self.output_path,
            "-p", self.pp3_file,
            "-j99",
            "-c", self.input_path
        ]
        try:
            start_time = datetime.now()
            results = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env={"LANG": "en_US.UTF-8", "OMP_NUM_THREADS": "1"}  # limits
                # internal threading of process
            )
            log.info(f"out: {results.stdout}")
            log.info(f"err: {results.stderr}")
            log.info(f"image conversion time: {datetime.now() - start_time}")
        except subprocess.CalledProcessError as e:
            log.error(e)
        return


def main(cfg: DictConfig) -> None:
    """
    Main function to convert pngs to jpgs. Accepts omegaconf dictConfig
    """
    log.info("Converting pngs to jpgs")
    # rawtherapee profile for setting default exposure for images
    rt_pp3 = cfg.png2jpg.rt_pp3
    # TODO @jinamshah:
    #  changes around developed images possible if storing pngs temporarily

    # locate batch's developed images folder in LTS
    developed_images_folder = utils.locate_lts_location(
        cfg.paths.lts_locations, cfg.batch_id, 'semifield-developed-images')

    if not developed_images_folder:
        developed_images_folder = (Path(cfg.paths.data_dir) /
                                   'semifield-developed-images' / cfg.batch_id)
    if not developed_images_folder:
        log.error(f"{cfg.batch_id} doesn't exist")

    png_folder = developed_images_folder / 'pngs'
    png_files = []
    for file_mask in cfg.file_masks.png_files:
        png_files.extend(list(png_folder.glob(f"*{file_mask}")))

    log.info(f"Located {len(png_files)} png files in "
             f"{developed_images_folder.parent.parent}")

    for png_file in png_files:
        output_path = developed_images_folder / f'{png_file.stem}.jpg'
        log.info(f"{png_file}, {output_path}")
        png2jpg_conv = PngToJpgConverter(png_file, output_path, rt_pp3)
        png2jpg_conv.convert()
        break