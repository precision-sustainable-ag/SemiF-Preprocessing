import os
from pathlib import Path

import yaml
from omegaconf import DictConfig

from dataclasses import dataclass

@dataclass
class PipelineKeys:
    account_url: str
    down_dev: str
    up_dev: str
    down_cut: str
    up_cut: str
    down_upload: str
    up_upload: str

    ms_lic: str

def read_keys(keypath):
    with open(keypath, "r") as file:
        pipe_keys = yaml.safe_load(file)
        sas = pipe_keys["SAS"]
        account_url = sas["account_url"]
        # semif cutouts
        up_cut = sas["cutouts"]["upload"]
        down_cut = sas["cutouts"]["download"]
        # semif developed-images
        up_dev = sas["developed"]["upload"]
        down_dev = sas["developed"]["download"]
        # semif-upload blob
        down_upload = sas["upload"]["download"]
        up_upload = sas["upload"]["upload"]

        keys = PipelineKeys(
            account_url=account_url,
            down_dev=down_dev,
            up_dev=up_dev,
            down_cut=down_cut,
            up_cut=up_cut,
            down_upload=down_upload,
            up_upload=up_upload,
            ms_lic=pipe_keys["metashape"]["lic"],
        )
    return keys

def make_autosfm_dirs(cfg):
    Path(cfg.paths.autosfm).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.down_photos).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.down_masks).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.proj_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.refs).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.orthodir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.demdir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.inspection_dir).mkdir(parents=True, exist_ok=True)


def config_gcp_path(cfg):
    batch_id = cfg.batch_id
    season = cfg.season
    gcp_dir = Path(cfg.paths.marker_dir) / season
    state_id = batch_id.split("_")[0]

    gcp_reference_path = None

    season_csvs = [str(x) for x in Path(gcp_dir).glob("*.csv")]
    gcp_reference_path = [x for x in season_csvs if state_id in Path(x).stem][0]
    cfg.paths.marker_file = gcp_reference_path
    print(f"Using GCP reference file: {gcp_reference_path}")

    return cfg


def parse_yml(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_config(cfg):
    keys = read_keys(cfg.paths.pipeline_keys)
    cfg.paths.metashape_key = keys.ms_lic
    # Prep config
    cfg = config_gcp_path(cfg)
    make_autosfm_dirs(cfg)
    return cfg


def autosfm_present(cfg: DictConfig) -> None:
    """Checks batch autosfm directory for data. Checks for presences of directories and files.

    Args:
        cfg (DictConfig): _description_

    Returns:
        data (list): list of dictionaries
    """
    asfm_list = []
    # asfm dir
    asfm = Path(cfg.paths.autosfm)
    asfm_rel = "./" + os.path.relpath(cfg.paths.autosfm)
    asfm_ex = asfm.exists()
    asfm_dict = {
        "main_dir": "autosfm",
        "item": "asfm",
        "relative_path": asfm_rel,
        "present": asfm_ex,
    }
    asfm_list.append(asfm_dict)
    # PSX project directory
    proj_dir = Path(cfg.paths.proj_dir)
    proj_dir_rel = "./" + os.path.relpath(cfg.paths.proj_dir)
    proj_dir_ex = proj_dir.exists()
    proj_dir_dict = {
        "main_dir": "autosfm",
        "item": "proj_dir",
        "relative_path": proj_dir_rel,
        "present": proj_dir_ex,
    }
    asfm_list.append(proj_dir_dict)
    # Metashape project psx file
    proj_path = Path(cfg.paths.proj_path)
    proj_path_rel = "./" + os.path.relpath(cfg.paths.proj_path)
    proj_path_ex = proj_path.exists()
    proj_path_dict = {
        "main_dir": "autosfm",
        "item": "proj_path",
        "relative_path": proj_path_rel,
        "present": proj_path_ex,
    }
    asfm_list.append(proj_path_dict)
    # Downscaled photos
    down_photos = Path(cfg.paths.down_photos)
    down_photos_rel = "./" + os.path.relpath(cfg.paths.down_photos)
    down_photos_ex = down_photos.exists()
    down_photos_dict = {
        "main_dir": "autosfm",
        "item": "down_photos",
        "relative_path": down_photos_rel,
        "present": down_photos_ex,
    }
    asfm_list.append(down_photos_dict)
        # Check if all are present
    presence = [x["present"] for x in asfm_list]
    return True if all(presence) else False

