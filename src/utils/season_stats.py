import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from omegaconf import DictConfig
import hydra

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    Image as PlatypusImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


class SeasonStatsCollector:
    VALID_BATCH_REGEX = re.compile(r"^(TX|NC|MD)_\d{4}-\d{2}-\d{2}$")

    def __init__(self, cfg: DictConfig, season: str, state_id: str):
        
        self.lts_locations: List[str] = cfg.paths.lts_locations
        
        self.output_dir = Path(cfg.paths.season_stats_dir) / state_id / season
        self.species_json_path = cfg.paths.species_info
        self.class_id_to_name = self.load_species_mapping()

    @staticmethod
    def parse_state_date_from_batch(batch_name: str):
        try:
            parts = batch_name.split("_")
            return parts[0], datetime.strptime(parts[-1], "%Y-%m-%d")
        except Exception:
            return None, None

    def load_species_mapping(self) -> dict:
        with open(self.species_json_path, "r") as f:
            species_info = json.load(f)
        class_id_to_name = {}
        for sp_data in species_info["species"].values():
            class_id = sp_data.get("class_id")
            name = sp_data.get("common_name", "unknown")
            if class_id is not None:
                class_id_to_name[class_id] = name
        return class_id_to_name

    def filter_by_date(self, batch_names, state_id, start_date=None, end_date=None):
        filtered_batch_names = []
        for batch_name in batch_names:
            state, batch_date = self.parse_state_date_from_batch(batch_name)
            if state != state_id:
                continue
            if not batch_date:
                continue
            if start_date and batch_date < start_date:
                continue
            if end_date and batch_date > end_date:
                continue
            filtered_batch_names.append(batch_name)
        return filtered_batch_names

    def get_season_from_metadata(self, metadata_dir: Path):
        metadata_files = sorted(metadata_dir.glob("*.json"))
        if not metadata_files:
            return "unknown"
        with open(metadata_files[0], "r") as f:
            data = json.load(f)
        return data.get("season", "unknown")

    @staticmethod
    def bin_area_by_species(area_by_species, area_bins):
        binned_area = []
        for species, areas in area_by_species.items():
            for area in areas:
                log_bin = pd.cut([area], bins=area_bins, labels=[
                    f"<{area_bins[1]}",
                    f"[{area_bins[1]}, {area_bins[2]})",
                    f"[{area_bins[2]}, {area_bins[3]})",
                    f"[{area_bins[3]}, {area_bins[4]})",
                    f"[{area_bins[4]}, {area_bins[5]})",
                    f"[{area_bins[5]}, {area_bins[6]})",
                    f"[{area_bins[6]}, {area_bins[7]})",
                    f">={area_bins[-2]}"
                ])[0]
                binned_area.append({"species": species, "area_sqcm": area, "area_bin": str(log_bin)})
        return binned_area

    def get_all_batches(self, path: Path) -> set:
        if not path.exists():
            return set()
        return {d.name for d in path.iterdir() if d.is_dir() and self.VALID_BATCH_REGEX.match(d.name)}
    
    def collect(self, state_id: str, start_date: datetime = None, end_date: datetime = None):

        batch_dates = []
        batch_summary = []
        image_counts = []
        image_sizes = []
        species_counts = Counter()
        primary_species_counts = Counter()
        area_by_species = defaultdict(list)
        area_by_primary_species = defaultdict(list)
        bbox_species_per_batch = defaultdict(Counter)

        lts_dirs = [Path(lts) for lts in self.lts_locations]
        for lts in tqdm(lts_dirs, desc="LTS Locations"):

            developed_dir = lts / "semifield-developed-images"
            batches = self.get_all_batches(developed_dir)
            filtered_batches = self.filter_by_date(batches, state_id, start_date, end_date)

            log.info(f"Processing {len(filtered_batches)} batches in {lts.name}")

            for batch_name in tqdm(filtered_batches, desc="Batches", leave=False):
                batch_dir = developed_dir / batch_name
                _, batch_date = self.parse_state_date_from_batch(batch_name)
                images_dir = batch_dir / "images"
                metadata_dir = batch_dir / "metadata"
                if not images_dir.exists() or not metadata_dir.exists():
                    log.warning(f"Missing images or metadata for batch {batch_name} in {lts.name}")
                    continue        

                jpgs = list(images_dir.glob("*.jpg"))
                num_images = len(jpgs)
                total_size = sum(f.stat().st_size for f in jpgs)

                image_counts.append(num_images)
                image_sizes.append(total_size)
                batch_dates.append(batch_date)

                batch_bboxes = 0
                metadata_files = sorted(metadata_dir.glob("*.json"))
                for meta_file in tqdm(metadata_files, desc="Metadata", leave=False):
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                    annotations = data.get("annotations", [])
                    batch_bboxes += len(annotations)
                    for ann in annotations:
                        class_id = ann.get("category_class_id", "unknown")
                        common_name = self.class_id_to_name.get(class_id, f"Unknown ({class_id})")
                        bbox_species_per_batch[batch_dir.name][common_name] += 1
                        area = ann.get("global_coordinates", {}).get("area_sqm", None)
                        if area is not None:
                            area *= 10000  # Convert sqm to sqcm
                        species_counts[common_name] += 1

                        is_primary = ann.get("is_primary", None)
                        if is_primary == True:
                            primary_species_counts[common_name + "_primary"] += 1
                            if area:
                                area_by_primary_species[common_name].append(area)
                        elif is_primary == False:
                            primary_species_counts[common_name + "_non_primary"] += 1

                        if area:
                            area_by_species[common_name].append(area)

                batch_summary.append({
                    "batch_id": batch_dir.name,
                    "num_images": num_images,
                    "total_image_size_GiB": total_size / 2**30,
                    "num_bboxes": batch_bboxes,
                    "lts_location": lts.name,
                })

        # Time-based stats
        df_dates = pd.Series(batch_dates)
        week_stats = df_dates.dt.to_period("W").value_counts().sort_index()
        month_stats = df_dates.dt.to_period("M").value_counts().sort_index()

        # Area binning
        area_bins = [-float("inf"), 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, float("inf")]
        binned_area = self.bin_area_by_species(area_by_species, area_bins)
        binned_primary_area = self.bin_area_by_species(area_by_primary_species, area_bins)

        # Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        batch_summary_df = pd.DataFrame(batch_summary)
        # Merge bbox species counts into batch summary
        bbox_species_per_batch_df = pd.DataFrame(bbox_species_per_batch).fillna(0).astype(int).T
        bbox_species_per_batch_df.index.name = "batch_id"
        batch_summary_df = batch_summary_df.merge(bbox_species_per_batch_df, on="batch_id", how="left").sort_values("batch_id")
        # Add an index column
        batch_summary_df.insert(0, "Index", range(1, len(batch_summary_df) + 1))
        batch_summary_df.to_csv(self.output_dir / "batch_summary.csv", index=False)
        
        species_counts_df = pd.DataFrame(species_counts.items(), columns=["species", "count"])
        species_counts_df.to_csv(self.output_dir / "species_counts.csv", index=False)

        binned_area_df = pd.DataFrame(binned_area)
        binned_area_df.to_csv(self.output_dir / "area_by_species_binned.csv", index=False)

        primary_species_counts_df = pd.DataFrame(primary_species_counts.items(), columns=["species", "count"])
        primary_species_counts_df.to_csv(self.output_dir / "primary_species_counts.csv", index=False)
        
        binned_primary_area_df = pd.DataFrame(binned_primary_area)        
        binned_primary_area_df.to_csv(self.output_dir / "primary_area_by_species_binned.csv", index=False)
        
        week_stats.reset_index().rename(columns={"index": "week", "count": "batch_count"}).to_csv(self.output_dir / "weekly_batches.csv", index=False)
        month_stats.reset_index().rename(columns={"index": "month", "count": "batch_count"}).to_csv(self.output_dir / "monthly_batches_present_only.csv", index=False)

        results = {
            "batch_summary": batch_summary_df,
            "species_counts": species_counts_df,
            "area_binned": binned_area_df,
            "primary_species_counts": primary_species_counts_df,
            "primary_area_binned": binned_primary_area_df,
            "weekly_batches": week_stats.reset_index().rename(columns={"index": "week", 0: "batch_count"}),
            "monthly_batches": month_stats.reset_index().rename(columns={"index": "month", 0: "batch_count"}),
        }
        return results

    @staticmethod
    def parse_bin_for_sorting(bin_label):
        if bin_label.startswith("<"):
            return -float('inf')
        elif bin_label.startswith(">="):
            return float('inf')
        match = re.match(r"\[([\d.eE+-]+), ([\d.eE+-]+)\)", bin_label)
        if match:
            return float(match.group(1))
        return float('inf')

    @staticmethod
    def plot_species_counts_labeled(species_df: pd.DataFrame, output_dir: Path):
        species_df_sorted = species_df.sort_values("count", ascending=False)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(species_df_sorted["species"], species_df_sorted["count"])
        ax.set_xticks(range(len(species_df_sorted["species"])))
        ax.set_xticklabels(species_df_sorted["species"], rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Species")
        ax.set_ylabel("Count")
        ax.set_title("Species Counts")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "species_counts_plot_labeled.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_species_counts_primary(df: pd.DataFrame, output_dir: Path):
        # Step 2: Split into 'base_species' and 'status'
        df["base_species"] = df["species"].apply(lambda x: x.replace("_non_primary", "").replace("_primary", ""))
        df["status"] = df["species"].apply(lambda x: "primary" if "primary" in x and "non" not in x else "non_primary")

        # Step 3: Pivot for total count calculation
        pivot_df = df.pivot(index="base_species", columns="status", values="count").fillna(0)
        pivot_df["total_count"] = pivot_df["primary"] + pivot_df["non_primary"]

        # Step 4: Sort by total count
        pivot_df = pivot_df.sort_values("total_count", ascending=False)

        # Step 5: Reset index for plotting
        pivot_df = pivot_df.reset_index()

        # Step 6: Melt back to long format for seaborn
        pivot_df_melted = pivot_df.melt(
            id_vars=["base_species"],
            value_vars=["primary", "non_primary"],
            var_name="status",
            value_name="count"
        )

        # Step 7: Plot
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 5))

        sns.barplot(
            data=pivot_df_melted,
            x="base_species",
            y="count",
            hue="status",
            palette="muted",
            ax=ax
        )

        # Beautify
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.title("Primary vs Non-Primary Counts by Species")
        plt.legend(title="Status")

        # Add labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

        plt.tight_layout()
        save_path = output_dir / "species_primary_vs_nonprimary_sorted.png"
        plt.savefig(save_path, dpi=300)
        plt.show()


    @staticmethod
    def plot_area_distribution_labeled(area_df: pd.DataFrame, output_dir: Path, primary: bool = False):
        area_df = area_df[area_df["species"] != "colorchecker"]
        unique_bins = area_df["area_bin"].unique()
        parsed_bins = sorted(unique_bins, key=SeasonStatsCollector.parse_bin_for_sorting)
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=area_df,
            kind="count",
            x="area_bin",
            col="species",
            col_wrap=3,
            sharey=True,
            sharex=False,
            height=5,
            aspect=1.2,
            order=parsed_bins,
        )
        g.set_xticklabels(rotation=45)
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Area Bin (cm²)", "Count")
        for ax in g.axes.flatten():
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)
        plt.tight_layout()
        save_path = output_dir / ("primary_area_distribution_plot_labeled.png" if primary else "area_distribution_plot_labeled.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        
class SeasonReportGenerator:
    def __init__(self, state: str, season: str, 
                 batch_summary_path: Path, monthly_batches_path: Path,
                 weekly_batches_path: Path, species_counts_path: Path,
                 primary_species_img_path: Path,
                 species_counts_img_path: Path, area_distribution_img_path: Path,
                primary_area_distribution_img_path: Path,
                 output_pdf_path: Path):
        self.state = state
        self.season = season
        self.batch_summary_path = batch_summary_path
        self.monthly_batches_path = monthly_batches_path
        self.weekly_batches_path = weekly_batches_path
        self.species_counts_path = species_counts_path
        self.primary_species_img_path = primary_species_img_path
        self.species_counts_img_path = species_counts_img_path
        self.area_distribution_img_path = area_distribution_img_path
        self.primary_area_distribution_img_path = primary_area_distribution_img_path
        self.output_pdf_path = output_pdf_path
        self.styles = getSampleStyleSheet()

        # Load data
        self.batch_summary = pd.read_csv(batch_summary_path)
        self.monthly_batches = pd.read_csv(monthly_batches_path)
        self.weekly_batches = pd.read_csv(weekly_batches_path)
        self.species_counts = pd.read_csv(species_counts_path)

    def create_table_from_dataframe(self, df, title: str, max_rows=30, max_cols=10):
        """
        Creates a Table from a pandas DataFrame.
        Limits rows and columns for readability if needed.
        """
        # Limit columns if too many
        if max_cols and df.shape[1] > max_cols:
            df = df.iloc[:, :max_cols]

        # Limit rows if too many
        if max_rows and df.shape[0] > max_rows:
            df = df.iloc[:max_rows]

        # Convert to list of lists (header + data)
        data = [df.columns.tolist()] + df.values.tolist()

        # Create the table
        table = Table(data, repeatRows=1)

        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        # Title + table
        title_paragraph = Paragraph(title, self.styles["Heading3"])
        spacer = Spacer(1, 12)
        return [title_paragraph, spacer, table, spacer]

    def generate_summary_text(self) -> List[str]:
        batch_summary = self.batch_summary
        monthly_batches = self.monthly_batches
        weekly_batches = self.weekly_batches
        species_counts = self.species_counts

        average_images_per_batch = batch_summary['num_images'].mean()
        total_image_size_gib = batch_summary['total_image_size_GiB'].sum()
        total_image_size_tib = total_image_size_gib / 1024

        summary_text = [
            f"Total batches: {batch_summary['batch_id'].nunique()}",
            f"Total images: {batch_summary['num_images'].sum():,}",
            f"Average number of images per batch: {average_images_per_batch:.2f}",
            f"Total plant instances: {species_counts[species_counts['species']!='colorchecker']['count'].sum():,}",
            f"Total image (jpg) size: {total_image_size_gib:.2f} GiB ({total_image_size_tib:.2f} TiB)"
        ]

        monthly_summary = "\n".join([f"{row['month']}: {row['batch_count']} batches" for _, row in monthly_batches.iterrows()])
        summary_text.append("Monthly Batches:\n" + monthly_summary)

        weekly_summary = "\n".join([f"{row['week']}: {row['batch_count']} batches" for _, row in weekly_batches.iterrows()])
        summary_text.append("Weekly Batches:\n" + weekly_summary)
        return summary_text

    def add_image_preserving_aspect(self, story, img_path: Path, max_width=8*inch):
        with PILImage.open(img_path) as img:
            width, height = img.size
            aspect = height / width
            scaled_width = max_width
            scaled_height = scaled_width * aspect
            story.append(PlatypusImage(str(img_path), width=scaled_width, height=scaled_height))

    def create_full_pdf(self):
        doc = SimpleDocTemplate(str(self.output_pdf_path), pagesize=letter)
        story = [Paragraph(f"Season Report:", self.styles["Title"]), Spacer(1, 1),
                 Paragraph(f"{self.state} {self.season.replace('_', '/')}", self.styles["Title"]), Spacer(1, 12)]

        # Add summary text
        summary_text = self.generate_summary_text()
        for paragraph in summary_text:
            story.append(Paragraph(paragraph.replace("\n", "<br/>"), self.styles["BodyText"]))
            # story.append(Spacer(1, 12))

        # Add page break
        story.append(PageBreak())
        
        # Add plots
        for img_path, title in [
            (self.species_counts_img_path, "Total Species Counts"),
            (self.area_distribution_img_path, "Area Distribution by Species and Bin"),
            (self.primary_species_img_path, "Primary vs Non-Primary Counts by Species"),
            (self.primary_area_distribution_img_path, "Primary Area Distribution by Species and Bin"),
        ]:
            # story.append(Spacer(1, 24))
            story.append(Paragraph(title, self.styles["Heading3"]))
            story.append(Spacer(1, 1))
            self.add_image_preserving_aspect(story, img_path)
            story.append(Spacer(1, 1))
        
        # Add batch summary table
        story.append(PageBreak())
        story.extend(self.create_table_from_dataframe(
            self.batch_summary.sort_values("batch_id"),
            title="Batch Summary Table",
            max_rows=30,
            max_cols=11
        ))

        doc.build(story)

    def generate_report(self):
        self.create_full_pdf()
        return self.output_pdf_path

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig):

    ############# Set these ##############
    season = "summer_weeds_2024"
    state = "TX"
    start = datetime(2024, 3, 26)
    end = datetime(2024, 9, 8)
    ############################################
    
    # Ensure the root directory exists
    output_dir = Path(cfg.paths.season_stats_dir, state, season)
    output_plot_dir = output_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    stats = SeasonStatsCollector(cfg,season, state)
    
    results = stats.collect(state_id=state, start_date=start, end_date=end)
    
    # # Load results (optional)
    # results = {
    #     "batch_summary": pd.read_csv(output_dir / "batch_summary.csv"),
    #     "species_counts": pd.read_csv(output_dir / "species_counts.csv"),
    #     "area_binned": pd.read_csv(output_dir / "area_by_species_binned.csv"),
    #     "weekly_batches": pd.read_csv(output_dir / "weekly_batches.csv"),
    #     "monthly_batches": pd.read_csv(output_dir / "monthly_batches_present_only.csv"),
    #     "primary_species_counts": pd.read_csv(output_dir / "primary_species_counts.csv"),
    #     "primary_area_binned": pd.read_csv(output_dir / "primary_area_by_species_binned.csv"),
    # }
    
    # Load results
    species_df = results["species_counts"]
    area_df = results["area_binned"]
    primary_species_df = results["primary_species_counts"]
    primary_area_df = results["primary_area_binned"]

    # Create plots
    stats.plot_species_counts_labeled(species_df, output_plot_dir)
    stats.plot_area_distribution_labeled(area_df, output_plot_dir)
    stats.plot_species_counts_primary(primary_species_df, output_plot_dir)
    stats.plot_area_distribution_labeled(primary_area_df, output_plot_dir, primary=True)

    # Create the report
    fixed_full_report = SeasonReportGenerator(
        state=state,
        season=season,
        batch_summary_path=output_dir / "batch_summary.csv",
        monthly_batches_path=output_dir / "monthly_batches_present_only.csv",
        weekly_batches_path=output_dir / "weekly_batches.csv",
        species_counts_path=output_dir / "species_counts.csv",
        primary_species_img_path=output_plot_dir / "species_primary_vs_nonprimary_sorted.png",
        species_counts_img_path=output_plot_dir / "species_counts_plot_labeled.png",
        area_distribution_img_path=output_plot_dir / "area_distribution_plot_labeled.png",
        primary_area_distribution_img_path=output_plot_dir / "primary_area_distribution_plot_labeled.png",
        output_pdf_path=output_dir / f"Season_Analysis_Report_{state}_{season}.pdf"
    )
    # Generate the report
    final_fixed_output = fixed_full_report.generate_report()

if __name__ == "__main__":
    main()    
    