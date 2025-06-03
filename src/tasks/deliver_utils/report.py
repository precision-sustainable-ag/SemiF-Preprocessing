import logging
import random
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from src.utils.utils import find_lts_dir
from src.utils.artifact_utils import read_artifact

log = logging.getLogger(__name__)

class ImageReport:
    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.batch_id
        
        self.bbot_version = str(cfg.bbot_version)
        self.lts_dir = find_lts_dir(self.batch_id, cfg.paths.lts_locations, developed=True, jpgs=True)
        self.upload_directory = Path(self.lts_dir) / "semifield-upload" / self.batch_id
        self.developed_directory = Path(self.lts_dir) / "semifield-developed-images" / self.batch_id
        self.output_report_dir = Path(cfg.paths.inspection_dir)
        self.output_report_dir.mkdir(parents=True, exist_ok=True)

        self.plot_file_base = self.output_report_dir / "plots"
        self.plot_file_base.mkdir(parents=True, exist_ok=True)

        self.raw_image_files = self.find_raw_image_files()
        self.developed_image_files = list(Path(self.developed_directory, "images").glob("*.jpg"))
        self.image_data = []

        self.local_sample_dir = self.output_report_dir / "remapped_samples"

        self.log_parser = LogParser(cfg, self.output_report_dir)

        self.sample_size = cfg.report.sample_size

    def find_raw_image_files(self) -> List[Path]:
        if "3.1" in str(self.bbot_version):
            extension = "*.RAW"
        else:
            # Check if 'SONY' directory exists in the upload directory
            if (self.upload_directory / "SONY").exists():
                extension = "SONY/*.ARW"
            else:
                # If 'SONY' directory does not exist, use the default extension
                extension = "*.ARW"
        raw_image_files = sorted(self.upload_directory.glob(extension))
        log.debug(f"Found {len(raw_image_files)} raw image files in {self.upload_directory}")
        if not raw_image_files:
            extension = "*.jpg"
            jpg_image_files = sorted(Path(self.developed_directory, "images").glob(extension))
            if jpg_image_files:
                log.info(f"Using developed JPGs from {self.developed_directory} instead of raw images.")
                raw_image_files = jpg_image_files
        return raw_image_files
        
    def calculate_total_images(self) -> int:
        return len(self.raw_image_files)

    def calculate_total_size(self) -> int:
        return sum(image.stat().st_size for image in self.raw_image_files)

    def calculate_developed_total_size(self) -> int:
        return sum(image.stat().st_size for image in self.developed_image_files)
    
    def calculate_average_size(self) -> float:
        total_images = self.calculate_total_images()
        total_size = self.calculate_total_size()
        return total_size / total_images if total_images > 0 else 0

    def calculate_average_developed_size(self) -> float:
        total_images = len(self.developed_image_files)
        total_size = self.calculate_developed_total_size()
        return total_size / total_images if total_images > 0 else 0
    
    def find_max_image_size(self) -> int:
        return max((image.stat().st_size for image in self.raw_image_files), default=0)

    def count_partial_uploads(self) -> int:
        max_size = self.find_max_image_size()
        return sum(1 for image in self.raw_image_files if image.stat().st_size < max_size)

    def matches_stem_pattern(self, stem: str) -> bool:
        """
        Check if the file stem matches the expected pattern which is <state>_<epoch>. State should either be MD,NC, or TX.
        """
        pattern = re.compile(r"^(MD|NC|TX)_(\d+)$")
        return bool(pattern.match(stem))
    
    def extract_image_metadata(self) -> None:
        stem_match_flag = True
        for image in self.raw_image_files:
            try:
                stem = image.stem
                if not self.matches_stem_pattern(stem):
                    stem_match_flag = False
                    if not stem_match_flag:
                        log.warning(f"Filename {stem} does not match the expected regex pattern ('^(.*)_(\d+)$').")
                
                state, epoch = stem.split("_")[0], stem.split("_")[1]

                # check if epoch is a valid integer
                if not epoch.isdigit():
                    log.warning(f"Epoch {epoch} in filename {stem} is not a valid integer.")
                    continue

                file_size = image.stat().st_size
                file_mtime = datetime.fromtimestamp(image.stat().st_mtime)
                capture_datetime = datetime.fromtimestamp(int(epoch))
                upload_delay = file_mtime - capture_datetime

                self.image_data.append({
                    "batch_id": self.batch_id,
                    "filename": image.name,
                    "state": state,
                    "epoch": int(epoch),
                    "file_size_bytes": file_size,
                    "file_size_kib": file_size / 1024,
                    "file_datetime_est_modified": file_mtime,
                    "capture_datetime_epoch": capture_datetime,
                    "average_upload_time_seconds": upload_delay.total_seconds(),
                    "average_upload_time_minutes": upload_delay,
                })
            except Exception as e:
                log.error(f"Error extracting metadata from {image}: {e}")
                continue
            
        self.image_data = sorted(self.image_data, key=lambda x: x["epoch"])
        log.info(f"Extracted {len(self.image_data)} records")

    def get_first_and_last_upload(self):
        if not self.image_data:
            self.extract_image_metadata()
        if not self.image_data:
            return None, None
        first_upload = self.image_data[0]["file_datetime_est_modified"]
        last_upload = self.image_data[-1]["file_datetime_est_modified"]
        return first_upload, last_upload

    def generate_capture_line_plot(self) -> None:
        """
        Generates a line plot of the capture times of the images.
        """
        if not self.image_data:
            self.extract_image_metadata()
        timestamps = [datetime.fromtimestamp(data["epoch"]) for data in self.image_data]
        df = pd.DataFrame({'Date': timestamps})
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], range(len(df)), marker='o', linestyle='-', color='b')

        plt.xlabel('Capture Time (EDT)')
        plt.ylabel('Image Index')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        plt.title('Capture Time Plot (based on epoch in the filename)')

        file_path = self.plot_file_base / f"capture_time_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()
        log.info(f"Capture time plot saved to {file_path}")

    def generate_modified_line_plot(self) -> None:
        """
        Generates a line plot of the modified times of the images.
        """
        if not self.image_data:
            self.extract_image_metadata()
        timestamps = sorted([data["file_datetime_est_modified"] for data in self.image_data])

        df = pd.DataFrame({'Date': timestamps})
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], range(len(df)), linestyle='-', color='b')

        plt.xlabel('Upload Time (EDT)')
        plt.ylabel('Image Index')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        plt.title('Upload Time Plot (based on file modified time)')
        file_path = self.plot_file_base / f"upload_time_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()
        log.info(f"Upload time plot saved to {file_path}")
    
    def generate_average_upload_time_plot(self) -> None:
        """
        Generates a line plot of the average upload time differences between images.
        """
        if not self.image_data:
            self.extract_image_metadata()
        # sort self.image_data by epoch
        self.image_data = sorted(self.image_data, key=lambda x: x["epoch"])
        timestamps = [data["average_upload_time_minutes"] for data in self.image_data]
        # convert datetime.timedeltas into seconds
        timestamps = [time.total_seconds() for time in timestamps]
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(timestamps)), timestamps, linestyle='-', color='b')

        plt.xlabel('Image Index')
        plt.ylabel('Time Difference (s)')
        plt.grid(True)
        plt.title('Capture / Upload Time Difference Plot')
        file_path = self.plot_file_base / f"upload_time_difference_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()
        log.info(f"Upload time difference plot saved to {file_path}")

    def calculate_average_upload_time(self) -> float:
        """
        Calculates the average upload time between images.
        """
        if not self.image_data:
            self.extract_image_metadata()
        if len(self.image_data) <= 1:
            return 0
        timestamps = [data["epoch"] for data in self.image_data]
        time_differences = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return sum(time_differences) / len(time_differences) if time_differences else 0

    def _generate_summary_section(self, c: canvas.Canvas) -> None:
        """
        Generates the summary section of the PDF report.
        """
        batch_id = self.batch_id
        total_images = self.calculate_total_images()
        total_size = self.calculate_total_size()
        developed_total_size = self.calculate_developed_total_size()
        avg_size = self.calculate_average_size()
        avg_developed_size = self.calculate_average_developed_size()
        first_upload, last_upload = self.get_first_and_last_upload()
        partial_uploads = self.count_partial_uploads()

        c.setFont("Helvetica", 12)

        c.drawString(50, 750, f"SemiField Bbot V{self.bbot_version} Collection Report")
        c.drawString(50, 730, f"Batch ID: {batch_id}")
        c.drawString(50, 710, f"Total Raw Images: {total_images}")
        c.drawString(50, 690, f"Total Raw Size: {total_size / (1024 ** 3):.2f} GiB")
        c.drawString(50, 670, f"Average Raw Image Size: {avg_size / (1024 ** 2):.2f} MiB")
        c.drawString(50, 650, f"Total Developed Size: {developed_total_size / (1024 ** 3):.2f} GiB")
        c.drawString(50, 630, f"Average Developed Image Size: {avg_developed_size / (1024 ** 2):.2f} MiB")
        
        if first_upload and last_upload:
            c.drawString(50, 610, f"First Upload: {first_upload}")
            c.drawString(50, 590, f"Last Upload: {last_upload}")
        
        if partial_uploads:
            c.drawString(50, 570, f"Partial Uploads: {partial_uploads}")

        # Add the line plot to the PDF
        capture_plot_path = self.plot_file_base / f"capture_time_plot_{batch_id}.png"
        if capture_plot_path.exists():
            c.drawImage(capture_plot_path, 5, 400, width=500//1.6, height=300//1.5)
        
        # Add the line plot to the PDF
        upload_plot_path = self.plot_file_base / f"upload_time_plot_{batch_id}.png"
        if upload_plot_path.exists():
            c.drawImage(upload_plot_path, 300, 400, width=500//1.6, height=300//1.5)

        # Parse module timings
        module_timings_df = self.log_parser.extract_module_timings()
        if not module_timings_df.empty:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 370, "Module timing Information:")

            # Get the value of Duration for the row with ScriptModule == "Total"
            total_duration = module_timings_df.loc[module_timings_df['ScriptModule'] == 'Total', 'DurationSeconds'].values[0]
            total_duration_hours = int(total_duration // 3600)
            total_duration_minutes = int((total_duration % 3600) // 60)
            total_duration_seconds = int(total_duration % 60)
            total_duration_str = f"{total_duration_hours}h {total_duration_minutes}m {total_duration_seconds}s"
            c.setFont("Helvetica", 12)
            c.drawString(50, 350, f"Total Duration: {total_duration_str}")

            # Prepare table data
            table_data = [["Module", "Duration (h:m:s)"]]
            module_timings_df = module_timings_df[module_timings_df['ScriptModule'] != 'Total']
            for _, row in module_timings_df.iterrows():
                module = row['ScriptModule']
                duration = row['DurationSeconds']
                duration_hours = int(duration // 3600)
                duration_minutes = int((duration % 3600) // 60)
                duration_seconds = int(duration % 60)
                duration_str = f"{duration_hours}h {duration_minutes}m {duration_seconds}s"
                table_data.append([module, duration_str])

            # Create the table
            table = Table(table_data, colWidths=[160, 130, 130, 100])

            # Style the table
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ]))

            # Draw the table
            table.wrapOn(c, 50, 250)
            table.drawOn(c, 50, 110)  # Adjust Y value as needed
            
        else:
            c.drawString(50, 540, "No module timings found in logs.")

    def _add_metashape_page(self, c: canvas.Canvas, file_name: str, x: int, y: int, width: int, height: int) -> None:
        c.showPage()  # Start a new page
        # Metashape report page
        metashape_page_1 = self.output_report_dir / f"metashape_report_pages/{file_name}"
        if metashape_page_1.exists():
            c.drawImage(metashape_page_1, x, y, width=width, height=height, preserveAspectRatio=True)
    
    def _add_sample_images(self, c: canvas.Canvas) -> None:
        """
        Adds sample images to the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas object.
        """
        # Set heading for sample images
        if self.local_sample_dir.exists() and list(self.local_sample_dir.glob("*.jpg")):
            sample_images = list(self.local_sample_dir.glob("*.jpg"))
            num_samples = min(self.sample_size, len(sample_images))  # Or change to len(sample_images) for all
            selected_images = sorted(random.sample(sample_images, num_samples))

            images_per_page = 12
            images_per_row = 3

            spacing_x = 10
            spacing_y = 1
            page_width, page_height = letter
            left_margin = 25
            top_margin = 770
            available_width = page_width - 2 * left_margin
            image_w = (available_width - (images_per_row - 1) * spacing_x) / images_per_row
            image_h = image_w  # Square layout
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left_margin, top_margin, "Sample images")

            for i, image_path in enumerate(selected_images):
                if i % images_per_page == 0:
                    if i > 0:
                        c.showPage()
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(left_margin, top_margin, "Sample images")

                index_on_page = i % images_per_page
                row = index_on_page // images_per_row
                col = index_on_page % images_per_row

                x = left_margin + col * (image_w + spacing_x)
                y = top_margin - 10 - row * (image_h + spacing_y)
                if image_path.exists():
                    c.drawImage(str(image_path),
                                x, y - image_h,
                                width=image_w,
                                height=image_h,
                                preserveAspectRatio=True,
                                anchor='c')
        else:
            log.warning("Sample images not available")
            c.drawString(50, 350, "Sample images not available")
        
    def _add_plot(self, c: canvas.Canvas, plot_file_name: str, x: int, y: int, width: int= 250, height: int = 200, title: str = None, newpage: bool = True) -> None:
        if newpage:
            c.showPage()  # Start a new page for the three analytical plots
        if title:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 750, title)
        count_plot_path = self.plot_file_base / plot_file_name
        if Path(count_plot_path).exists():
            c.drawImage(count_plot_path, x, y, width=width*2.0, height=height*2.0, preserveAspectRatio=True)

    def _add_errors_and_warnings(self, c: canvas.Canvas) -> None:
        """
        Adds errors and warnings to the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas object.
        """
        # Parse errors and warnings
        errors_df = self.log_parser.extract_error_blocks()
        if not errors_df.empty:
            c.showPage()  # Start a new page for the errors and warnings
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 750, "Errors and Warnings:")
            c.setFont("Helvetica", 10)
            y_position = 730

            for _, row in errors_df.iterrows():
                module = row['ScriptModule']
                level = row['Level']
                message = row['LogSnippet']
                if y_position < 50:
                    c.showPage()
                    y_position = 750
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y_position, "Continued Errors & Warnings:")
                    y_position -= 20
                    c.setFont("Helvetica", 10)
                c.drawString(25, y_position, f"[{module}] - {level.upper()} - {message}")
                y_position -= 15
        else:
            c.showPage()  # Start a new page for the errors and warnings
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 750, "Errors and Warnings:")

    def generate_pdf_report(self):
        
        if not self.image_data:
            self.extract_image_metadata()

        pdf_output_path = self.output_report_dir / f"{self.batch_id}_report.pdf"
        c = canvas.Canvas(str(pdf_output_path), pagesize=letter)
        # ----------------------------------------------------
        # Section 1: Summary Information
        # ----------------------------------------------------
        self._generate_summary_section(c)
        # --------------------------------------------------------
        # Section 2: Metashape Report Image Page 1
        # --------------------------------------------------------
        self._add_metashape_page(c, "page_001.jpg", -150, -125, 850, 1000)
        self._add_metashape_page(c, "page_002.jpg", -115, -120, 850, 1000)
        self._add_metashape_page(c, "page_003.jpg", -115, -120, 850, 1000)
        self._add_metashape_page(c, "page_004.jpg", -115, -120, 850, 1000)
        # --------------------------------------------------------
        # Section 5: Sample Images
        # --------------------------------------------------------
        self._add_sample_images(c)
        # -----------------------------------------
        # Section 6: Add Area and Density
        # -----------------------------------------
        self._add_plot(c, "species_counts.png", 50, 350, 250, 200, title="Analysis Plots: Species Counts and Area Distribution", newpage=True)
        self._add_plot(c, "area_log_scaled_histograms.png", 50, 0, 250, 200, newpage=False)
        self._add_plot(c, "species_centroid_density.png", 50, 350, 250, 200, title="Analysis Plots: Spatial Density", newpage=True)
        
        #-------------------------------------------
        # Section 8: Log errors
        #-------------------------------------------
        self._add_errors_and_warnings(c)

        # Save the PDF
        c.save()
        log.info(f"PDF report saved to {pdf_output_path}")

    def generate_report(self) -> None:
        """
        Main execution function to generate all visual plots and the final PDF report.
        """
        log.info(f"Generating report for batch: {self.batch_id}")
        self.extract_image_metadata()
        self.generate_capture_line_plot()
        self.generate_modified_line_plot()
        self.generate_average_upload_time_plot()
        self.generate_pdf_report()
        log.info(f"Completed report for batch: {self.batch_id}")

class LogParser:
    def __init__(self, cfg: DictConfig, output_report_dir: Path = None):
        self.artifact_yaml_path = Path(cfg.paths.artifact_path)
        
        if not self.artifact_yaml_path.exists():
            raise FileNotFoundError(f"Artifact YAML file not found: {self.artifact_yaml_path}")
    
        self.artifact = read_artifact(self.artifact_yaml_path)
        
        self.output_report_dir = output_report_dir

    def extract_error_blocks(self) -> pd.DataFrame:
        messages = []
        modules = []
        levels = []

        warn_errors = self.artifact.get("warning_and_errors", {})
        for module, entries in warn_errors.items():
            if entries is None:
                continue
            for entry in entries:
                level = "ERROR" if "ERROR" in entry else "WARNING"
                messages.append(entry)
                modules.append(module)
                levels.append(level)

        return pd.DataFrame({
            "ErrorIndex": range(1, len(messages) + 1),
            "ScriptModule": modules,
            "Level": levels,
            "LogSnippet": messages,
        })

    def extract_module_timings(self) -> pd.DataFrame:
        task_durations = self.artifact.get("task_durations", {})
        records = []

        for module, duration_str in task_durations.items():
            if duration_str is None:
                continue
            h, m, s = map(int, duration_str.split(":"))
            duration_seconds = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
            records.append({
                "ScriptModule": module,
                "DurationSeconds": duration_seconds,
            })

        total_duration = sum(r["DurationSeconds"] for r in records)
        records.append({
            "ScriptModule": "Total",
            "DurationSeconds": total_duration,
        })

        return pd.DataFrame(records)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra-based entry point to generate an image batch report and optionally parse logs for warnings/errors.

    Args:
        cfg (DictConfig): Hydra config with paths and batch settings.
    """
    try:
        image_report = ImageReport(cfg)
        image_report.generate_report()
        log.info(f"Report generated for batch: {cfg.batch_id}")
    except Exception as e:
        log.error(f"Error generating report: {e}")
        raise
    
    if cfg.report.save2lts:
        try:
            # Copy report to LTS developed inspection directory
            report_dst = image_report.developed_directory / "inspection"
            report_src = str(image_report.output_report_dir / f"{image_report.batch_id}_report.pdf")
            shutil.copy(report_src, report_dst)
            log.info(f"Report copied to LTS directory: {report_dst}")
        except Exception as e:
            log.error(f"Error copying report to LTS directory: {e}")
            raise
        

if __name__ == "__main__":
    main()