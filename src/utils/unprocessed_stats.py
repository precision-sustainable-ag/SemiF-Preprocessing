import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from tqdm import tqdm
log = logging.getLogger(__name__)

REQUIRED_DIRS = [
    "images", 
    "metadata", 
    # "meta_masks", 
    # "reference"
    ]

REQUIRED_COLUMNS = [
    'has_images', 
    'has_metadata', 
    # 'has_meta_masks', 
    # 'has_reference'
    ]
class BatchAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.standardize_columns()
        self.required_columns = REQUIRED_COLUMNS

    def standardize_columns(self):
        self.df.columns = self.df.columns.str.strip().str.lower()

    def get_unprocessed_batches(self) -> pd.DataFrame:
        unprocessed = self.df[self.df['processed'] == False].copy()
        return unprocessed


class BatchStatusChecker:
    VALID_BATCH_REGEX = re.compile(r"^(TX|NC|MD)_\d{4}-\d{2}-\d{2}$")

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.lts_locations: List[str] = cfg.paths.lts_locations
        self.required_dirs = REQUIRED_DIRS
        self.season_config = cfg.date_ranges
        self.cache_path = Path(cfg.paths.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_minutes = 1000 #cfg.batch_ids.max_age_minutes
        self.force_reload = True 

    def is_cache_valid(self) -> bool:
        if not self.cache_path.exists():
            return False
        age = (time.time() - self.cache_path.stat().st_mtime) / 60
        return age < self.max_age_minutes
    
    def match_season(self, site: str, date_str: str) -> Tuple[Optional[str], Optional[str]]:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        for season, data in self.season_config.get(site, {}).items():
            start = datetime.strptime(data["start"], "%Y-%m-%d")
            end = datetime.strptime(data["end"], "%Y-%m-%d")
            if start <= date_obj <= end:
                return season, data["bbot_version"]
        return None, None

    def get_all_batches(self, path: Path) -> set:
        if not path.exists():
            return set()
        return {d.name for d in path.iterdir() if d.is_dir() and self.VALID_BATCH_REGEX.match(d.name)}

    def check_batches(self) -> pd.DataFrame:
        if self.is_cache_valid() and not self.force_reload:
            log.info(f"Using cached batch status from {self.cache_path}")
            return pd.read_csv(self.cache_path)
        log.info("Scanning LTS locations for batch data...")
        batch_records = {}

        for lts in tqdm(self.lts_locations, desc="Scanning LTS locations", unit="location"):
            lts_path = Path(lts)
            developed_path = lts_path / "semifield-developed-images"
            uploads_path = lts_path / "semifield-upload"

            developed_batches = self.get_all_batches(developed_path)
            uploads_batches = self.get_all_batches(uploads_path)
            all_batches = set(developed_batches) | set(uploads_batches)
            log.info(f"Found {len(all_batches)} batches in {lts_path.name}")
            for batch_id in tqdm(all_batches, desc="Processing batches", leave=False):
                if batch_id not in batch_records:
                    match = self.VALID_BATCH_REGEX.match(batch_id)
                    site = match.group(1)
                    date_str = batch_id.split("_")[1]
                    season, bbot_version = self.match_season(site, date_str)

                    batch_records[batch_id] = {
                        "batch_id": batch_id,
                        "season": season,
                        "bbot_version": bbot_version,
                        "storage_location": lts_path.name,
                        "exists_in_uploads": False,
                        "exists_in_developed": False,
                        "processed": False,
                        "has_images": False,
                        "has_metadata": False,
                        # "has_meta_masks": False,
                        # "has_reference": False,
                    }

                record = batch_records[batch_id]

                # Check developed location
                dev_folder = developed_path / batch_id
                if dev_folder.exists():
                    record["exists_in_developed"] = True
                    status = {
                        f"has_{d}": (dev_folder / d).exists()
                        for d in self.required_dirs
                    }
                    record.update(status)
                    record["processed"] = all(status.values())

                    record["preprocessed"] = True if record["has_images"] else False

                # Check uploads location
                upload_folder = uploads_path / batch_id
                if upload_folder.exists():
                    record["exists_in_uploads"] = True

        df = pd.DataFrame(batch_records.values())
        return df.sort_values(by=["batch_id"])

def preprocess_dataframe(df: pd.DataFrame, season_mapping: Dict[str, list]) -> pd.DataFrame:
    # Create a date column from batch_id
    df['date'] = pd.to_datetime(df['batch_id'].str.split('_').str[1], format='%Y-%m-%d')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['state'] = df['batch_id'].str.split('_').str[0]

    # Clean weird suffixes
    df['season'] = df['season'].str.replace('_MDbbotv3.0', '', regex=False)
    return df.sort_values(by=["batch_id"], ascending=False)

def group_by_state_season_year(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['state','season','year']).agg(
        {
        'batch_id': 'count'
        }
        ).reset_index().sort_values(['state','year','season'])

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"DataFrame saved to {path}")

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    # Instantiate and use the checker
    checker = BatchStatusChecker(cfg)
    df = checker.check_batches()

    # Instantiate and use the analyzer
    analyzer = BatchAnalyzer(df)
    unprocessed_df = analyzer.get_unprocessed_batches()

    # Preprocess the DataFrame
    cleaned_unprocessed_df = preprocess_dataframe(unprocessed_df, cfg.date_ranges.season_mappings)
    
    # Group by state, season, and year
    summary_df = group_by_state_season_year(cleaned_unprocessed_df)

    # Save unprocessed batches and the grouped by summary
    unprocessed_stats_dir = Path(cfg.paths.unprocessed_stats_dir)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_df(cleaned_unprocessed_df, unprocessed_stats_dir / f"unprocessed_batches_{suffix}.csv")
    save_df(summary_df, unprocessed_stats_dir / f"unprocessed_batches_summary_{suffix}.csv")
    save_df(analyzer.df, unprocessed_stats_dir / f"all_batches_{suffix}.csv")

if __name__ == "__main__":
    main()
