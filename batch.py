import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig

from main import run_single_batch
from src.utils.unprocessed_stats import BatchStatusChecker

log = logging.getLogger(__name__)


def is_missing(val: Any) -> bool:
    """Return True if the value is None, float nan, or string 'nan'."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() == "nan":
        return True
    return False


def cleanup_batch_cfg(batches: List[Dict]) -> List[Dict]:
    """Skip batch dicts missing required fields."""
    required = ("batch_id", "season", "bbot_version", "crs")
    cleaned = []
    for b in batches:
        if all(not is_missing(b.get(k)) for k in required):
            cleaned.append({k: b[k] for k in required})
    return cleaned


def sort_batches(batches: List[Dict]) -> List[Dict]:
    """Sort batch dicts descending by batch_id."""
    return sorted(batches, key=lambda x: x["batch_id"], reverse=True)


def exclude_states(batches: List[Dict], states: Optional[List[str]]) -> List[Dict]:
    """Exclude batches from given states (e.g., ['TX', 'NC'])."""
    if not states:
        return batches
    return [b for b in batches if b["batch_id"].split("_")[0] not in states]


def exclude_bbot_versions(batches: List[Dict], versions: Optional[List[str]]) -> List[Dict]:
    """Exclude batches with specific bbot_versions."""
    if not versions:
        return batches
    return [b for b in batches if b["bbot_version"] not in versions]


def assign_bbot_and_crs(batch_cfgs: List[Dict], season_config: Dict) -> List[Dict]:
    """
    Add bbot_version and crs to batch_cfgs based on site/date using season_config.
    """
    for batch in batch_cfgs:
        batch_id = batch.get("batch_id")
        batch["bbot_version"] = None
        batch["crs"] = None
        if not batch_id or "nan" in str(batch_id).lower():
            continue

        try:
            site, date_str = batch_id.split("_")
            batch_date = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue

        for season_info in season_config.get(site, {}).values():
            start = datetime.strptime(season_info["start"], "%Y-%m-%d")
            end = datetime.strptime(season_info["end"], "%Y-%m-%d")
            if start <= batch_date <= end:
                batch["bbot_version"] = season_info["bbot_version"]
                batch["crs"] = season_info["crs"]
                break
    return batch_cfgs


def gather_batches(cfg: DictConfig) -> List[Dict]:
    """
    Gather batch dictionaries from config, or from batch status checker.
    """
    if cfg.batch_ids.batch_ids:
        return [{
            "batch_id": bid,
            "season": cfg.season,
            "bbot_version": cfg.bbot_version,
        } for bid in cfg.batch_ids.batch_ids]

    checker = BatchStatusChecker(cfg)
    df = checker.check_batches()
    df = df[df["exists_in_uploads"] & ~df["exists_in_developed"]]
    return [
        {
            "batch_id": row.batch_id,
            "season": row.season,
            "bbot_version": row.bbot_version,
        }
        for _, row in df.iterrows()
    ]


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    batches = gather_batches(cfg)
    if not batches:
        log.info("No batches to process.")
        return
    
    batches = exclude_states(batches, getattr(cfg.batch_ids.exclude, "states", []))
    batches = exclude_bbot_versions(batches, getattr(cfg.batch_ids.exclude, "bbot_versions", []))
    batches = sort_batches(batches)
    batches = assign_bbot_and_crs(batches, cfg.date_ranges)
    batches = cleanup_batch_cfg(batches)

    if not batches:
        log.warning("No valid batches after filtering and cleanup.")
        return
    
    log.info(f"Processing {len(batches)} batches: {[b['batch_id'] for b in batches]}")
    exit()
    for batch_cfg in batches:
        try:
            run_single_batch(cfg, batch_cfg=batch_cfg)
        except Exception:
            log.exception(f"Failed processing batch {batch_cfg['batch_id']}")
            continue

    log.info("Finished processing all batches.")


if __name__ == "__main__":
    main()
