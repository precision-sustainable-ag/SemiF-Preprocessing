from pathlib import Path
import pandas as pd
from datetime import datetime

def simple_batch_info(batch_dir):
    preprocessing_csv = batch_dir / f"{batch_dir.name}_preprocessing_inspection_results.csv"
    
    if not preprocessing_csv.exists():
        return False
    
    return True

def batch_is_preprocessed(batch_dir):
    if batch_dir.exists():
        samples_images = list(Path(batch_dir, "preprocessing_samples").glob("*.jpg"))
        if len(samples_images) >= 50:
            return True
    return False
    


def collect_batch_data(upload_dir, dev_dir, start_date, end_date):
    batches = sorted(upload_dir.glob("*"))
    batch_data = []

    for batch in batches:
        batch_id = batch.name
        # Extract the date part from the batch name
        try:
            batch_date = datetime.strptime(batch_id.split("_")[1], "%Y-%m-%d")
        except (IndexError, ValueError):
            # Skip batches with invalid or missing date formats
            continue

        # Check if the batch date is within the specified range
        if start_date <= batch_date <= end_date:
            dev_batch_dir = dev_dir / batch_id
            inspection_csv_exists = simple_batch_info(dev_batch_dir)
            batch_data.append({
                "BatchID": batch_id,
                "IsPreprocessed": batch_is_preprocessed(dev_batch_dir),
                "Inspected": inspection_csv_exists
            })

    return batch_data


def save_batch_data_to_csv(batch_data, output_file):
    batch_df = pd.DataFrame(batch_data)
    batch_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    upload_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload")
    dev_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images")
    start_date = datetime.strptime("2024-12-02", "%Y-%m-%d")
    end_date = datetime.today()

    batch_data = collect_batch_data(upload_dir, dev_dir, start_date, end_date)
    save_batch_data_to_csv(batch_data, "batch_info.csv")
