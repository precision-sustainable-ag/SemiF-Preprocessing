import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PlatypusImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from tqdm import tqdm


class SeasonStatsCollector:
    def __init__(self, root_dir: Path, output_dir: Path, species_json_path: Path):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.species_json_path = species_json_path
        self.class_id_to_name = self.load_species_mapping()

    @staticmethod
    def parse_date_state_from_batch(batch_name: str):
        try:
            parts = batch_name.split("_")
            return datetime.strptime(parts[-1], "%Y-%m-%d"), parts[0]
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

    def filter_by_date(self, batch_dirs, start_date=None, end_date=None):
        filtered_dirs = []
        for batch_dir in batch_dirs:
            batch_date, _ = self.parse_date_state_from_batch(batch_dir.name)
            if not batch_date:
                continue
            if start_date and batch_date < start_date:
                continue
            if end_date and batch_date > end_date:
                continue
            filtered_dirs.append(batch_dir)
        return filtered_dirs

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

    def collect(self, start_date: datetime = None, end_date: datetime = None):
        batch_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        batch_dirs = self.filter_by_date(batch_dirs, start_date, end_date)

        batch_dates = []
        batch_summary = []
        image_counts = []
        image_sizes = []
        species_counts = Counter()
        primary_species_counts = Counter()
        area_by_species = defaultdict(list)
        area_by_primary_species = defaultdict(list)

        for batch_dir in tqdm(batch_dirs, desc="Processing batches"):
            batch_date, state = self.parse_date_state_from_batch(batch_dir.name)
            images_dir = batch_dir / "images"
            metadata_dir = batch_dir / "metadata"
            if not images_dir.exists() or not metadata_dir.exists():
                continue        

            jpgs = list(images_dir.glob("*.jpg"))
            num_images = len(jpgs)
            total_size = sum(f.stat().st_size for f in jpgs)

            image_counts.append(num_images)
            image_sizes.append(total_size)
            batch_dates.append(batch_date)

            batch_bboxes = 0
            metadata_files = sorted(metadata_dir.glob("*.json"))
            for meta_file in tqdm(metadata_files, desc="Processing metadata files", leave=False):
                with open(meta_file, "r") as f:
                    data = json.load(f)
                annotations = data.get("annotations", [])
                batch_bboxes += len(annotations)
                for ann in annotations:
                    class_id = ann.get("category_class_id", "unknown")
                    common_name = self.class_id_to_name.get(class_id, f"Unknown ({class_id})")
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
                "batch_date": batch_date.strftime("%Y-%m-%d"),
                "num_images": num_images,
                "total_image_size_GiB": total_size / 2**30,
                "num_bboxes": batch_bboxes
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
        batch_summary_df.to_csv(self.output_dir / "batch_summary.csv", index=False)
        
        species_counts_df = pd.DataFrame(species_counts.items(), columns=["species", "count"])
        species_counts_df.to_csv(self.output_dir / "species_counts.csv", index=False)

        binned_area_df = pd.DataFrame(binned_area)
        binned_area_df.to_csv(self.output_dir / "area_by_species_binned.csv", index=False)

        primary_species_counts_df = pd.DataFrame(primary_species_counts.items(), columns=["species", "count"])
        primary_species_counts_df.to_csv(self.output_dir / "primary_species_counts.csv", index=False)
        
        binned_primary_area_df = pd.DataFrame(binned_primary_area)        
        binned_primary_area_df.to_csv(self.output_dir / "primary_area_by_species_binned.csv", index=False)
        
        week_stats.reset_index().rename(columns={"index": "week", 0: "batch_count"}).to_csv(self.output_dir / "weekly_batches.csv", index=False)
        month_stats.reset_index().rename(columns={"index": "month", 0: "batch_count"}).to_csv(self.output_dir / "monthly_batches_present_only.csv", index=False)

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
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(species_df_sorted["species"], species_df_sorted["count"])
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
        fig, ax = plt.subplots(figsize=(12, 6))

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
        plt.title("Primary vs Non-Primary Counts by Species (Sorted by Total Count)")
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
            height=6,
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
            (self.primary_species_img_path, "Primary vs Non-Primary Counts by Species"),
            (self.species_counts_img_path, "Total Species Counts"),
            (self.area_distribution_img_path, "Area Distribution by Species and Bin"),
            (self.primary_area_distribution_img_path, "Primary Area Distribution by Species and Bin"),
        ]:
            # story.append(Spacer(1, 24))
            story.append(Paragraph(title, self.styles["Heading3"]))
            story.append(Spacer(1, 1))
            self.add_image_preserving_aspect(story, img_path)
            story.append(Spacer(1, 1))

        doc.build(story)

    def generate_report(self):
        self.create_full_pdf()
        return self.output_pdf_path


if __name__ == "__main__":
    
    ############# Set these ##############
    season = "cover crops 2024_2025"
    state = "NC"
    start = datetime(2024, 12, 2)
    end = datetime(2025, 4, 4)
    root_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images")
    output_dir = Path("data/season_stats")
    species_json_path = Path("data/semifield-utils/species_information/species_info.json")
    ############################################
    
    # Ensure the root directory exists
    output_dir = Path("data/season_stats", state, season)
    output_plot_dir = output_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    stats = SeasonStatsCollector(
        root_dir=root_dir, 
        output_dir=output_dir, 
        species_json_path=species_json_path
        )
    
    results = stats.collect(start_date=start, end_date=end)
    
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
        primary_species_img_path=output_plot_dir / "primary_species_counts_labeled.png",
        species_counts_img_path=output_plot_dir / "species_counts_plot_labeled.png",
        area_distribution_img_path=output_plot_dir / "area_distribution_plot_labeled.png",
        primary_area_distribution_img_path=output_plot_dir / "primary_area_distribution_plot_labeled.png",
        output_pdf_path=output_dir / f"Season_Analysis_Report_{state}_{season}.pdf"
    )
    # Generate the report
    final_fixed_output = fixed_full_report.generate_report()
