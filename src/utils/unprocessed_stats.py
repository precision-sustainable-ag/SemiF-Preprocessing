from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import hydra
import re
from omegaconf import DictConfig
import logging
from datetime import datetime

log = logging.getLogger(__name__)

import pandas as pd

class BatchAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.standardize_columns()
        self.required_columns = ['has_images', 'has_metadata', 'has_meta_masks', 'has_reference']

    def standardize_columns(self):
        self.df.columns = self.df.columns.str.strip().str.lower()

    def get_unprocessed_batches(self) -> pd.DataFrame:
        unprocessed = self.df[~self.df['processed']].copy()
        unprocessed['missing_components'] = (~unprocessed[self.required_columns[1:]]).sum(axis=1)
        unprocessed['preprocessed'] = unprocessed['has_metadata'] & unprocessed['has_reference']
        return unprocessed[
            (unprocessed['has_images']) & 
            (unprocessed['missing_components'] != 0)
        ]


class BatchStatusChecker:
    VALID_BATCH_REGEX = re.compile(r"^(TX|NC|MD)_\d{4}-\d{2}-\d{2}$")

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.lts_locations: List[str] = cfg.paths.lts_locations
        self.required_dirs = ["images", "metadata", "meta_masks", "reference"]
        self.season_config = cfg.date_ranges

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
        batch_records = {}

        for lts in tqdm(self.lts_locations, desc="Scanning LTS locations", unit="location"):
            lts_path = Path(lts)
            developed_path = lts_path / "semifield-developed-images"
            uploads_path = lts_path / "semifield-upload"

            developed_batches = self.get_all_batches(developed_path)
            uploads_batches = self.get_all_batches(uploads_path)
            all_batches = developed_batches.union(uploads_batches)

            for batch_id in all_batches:
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
                        "has_meta_masks": False,
                        "has_reference": False,
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

                # Check uploads location
                upload_folder = uploads_path / batch_id
                if upload_folder.exists():
                    record["exists_in_uploads"] = True

        return pd.DataFrame(batch_records.values())

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Create a date column from batch_id
    df['date'] = pd.to_datetime(df['batch_id'].str.split('_').str[1], format='%Y-%m-%d')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['state'] = df['batch_id'].str.split('_').str[0]
    # Replace "cover crops 2024/2025MDbbotv3.0" with "cover crops 2024/2025" in season column that already exists
    df['season'] = df['season'].str.replace('_MDbbotv3.0', '', regex=False)
    # Create a general season column
    df['general_season'] = df['season'].apply(
        lambda x: 'cover' if 'cover' in str(x).lower() else 
                  'weeds' if 'weeds' in str(x).lower() else 
                  'cash' if 'cash' in str(x).lower() else None
    )
    return df.sort_values(by=["batch_id"])

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    # Instantiate and use the checker
    checker = BatchStatusChecker(cfg)
    df = checker.check_batches().sort_values(by=["batch_id"])

    # Instantiate and use the analyzer
    analyzer = BatchAnalyzer(df)
    unprocessed_df = analyzer.get_unprocessed_batches()

    # Preprocess the DataFrame
    cleaned_unprocessed_df = preprocess_dataframe(unprocessed_df)
    
    # Group by state, general_season, and year
    cleaned_unprocessed_df = cleaned_unprocessed_df[cleaned_unprocessed_df['preprocessed'] == False]
    summary_df = cleaned_unprocessed_df.groupby(['state','general_season','year']).agg(
        {
        'batch_id': 'count'
        }
        ).reset_index().sort_values(['state','year','general_season'])
    
    # Save unprocessed batches and the grouped by summary
    unprocessed_stats_dir = Path(cfg.paths.unprocessed_stats_dir)
    unprocessed_stats_dir.mkdir(parents=True, exist_ok=True)
    cleaned_unprocessed_df.to_csv(unprocessed_stats_dir / "unprocessed_batches.csv", index=False)
    summary_df.to_csv(unprocessed_stats_dir / "unprocessed_batches_summary.csv", index=False)
    analyzer.df.to_csv(unprocessed_stats_dir / "all_batches.csv", index=False)

if __name__ == "__main__":
    main()
