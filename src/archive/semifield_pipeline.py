import subprocess
import sys
import time
import yaml
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import GPUtil

class ImagePipeline:
    def __init__(self, config_path, batch_name):
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.batch_name = batch_name
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        self.nfs_path = self.config['paths']['primary_nfs']
        self.im_input_dir = Path(self.config['paths']['local_upload']) / batch_name
        self.im_output_dir = Path(self.config['semif_developed']) / batch_name / "images"
        self.im_output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = Path(self.config['paths']['dev_profiles'], f"{batch_name}.pp3") 
        assert self.profile_path.exists(), f"Development profile ({self.profile_path}) does not exist"

        self.raw_extension = ".RAW" if self.config['bbot_version'] == 3.1 else ".ARW"

        # Initialize logger
        self.init_logger()

    def init_logger(self):
        """Initialize a logger with a log file based on batch name and timestamp."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        log_dir = Path(self.config['paths']['log_dir']) / self.batch_name / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{time_str}.txt"

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.log = logging.getLogger()
        self.log.info(f"Logger initialized for batch {self.batch_name}")

    def develop_images(self):
        # Implementation of the develop_images method
        self.log.info("Starting image development...")
        num_of_raw_images = len(list(self.im_input_dir.glob(f"*{self.raw_extension}")))
        
        if num_of_raw_images == 0:
            message = f"No RAW images found in the input directory ({self.im_input_dir})"
            self.log.warning(message)
            return            
        
        else:
            exe_command = f"./RawTherapee_5.9.AppImage --cli \
                -O {self.im_output_dir} \
                -p {self.profile_path} \
                -j99 \
                -c {self.im_input_dir}"
            exe_command2 = 'bash -c "OMP_NUM_THREADS=90; ' + exe_command + '"'
            self.log.info(f"Executing command: {exe_command2}")

            try:
                # Run the rawtherapee command
                subprocess.run(exe_command2, shell=True, check=True)
                self.log.info("Image development completed successfully.")

            except Exception as e:
                self.log.error(f"Error during image development: {e}")
                raise e

        print("Image development has finished for batch " + str(batch_name))


    def do_weed_detection_yolov8(self):
        # Implementation of the do_weed_detection_yolov8 method
        image_source = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name) + "/images"
        batch_name = str(batch_name)
        self.log("Finding available GPU for weed detection..")
        gpu_idx = GPUtil.getFirstAvailable(order='memory', maxMemory=0.6, attempts=90, interval=120)[0]
        self.log.info("Found available gpu.. Will use ID: " + str(gpu_idx))
        env_command = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
        #env_command = "export CUDA_VISIBLE_DEVICES=9"
        exe_command = f"/home/psa_images/anaconda3/envs/yolov8/bin/python predict_yolov8.py \
            --source {image_source} \
            --batch_name {batch_name}"

        try:
            # Run the rawtherapee command
            #process_id = subprocess.run(['/bin/bash', '-i', '-c', exe_command])
            self.log.info([env_command, '/bin/bash', '-c', exe_command])
            process_id = subprocess.run(['/bin/bash', '-c', env_command + ";" + exe_command])
        except Exception as e:
            raise e
        
        self.log.info("Weed detection has finished for batch " + str(batch_name))
 
    def move_local_raw_data_to_nsf(self):
        dest = Path(self.nfs_path, "semifield-upload")
        dest.mkdir(parents=True, exist_ok=True)  # Ensure destination exists

        # Perform rsync without removing source files
        subprocess.call(['rsync', '-avzh', '--progress', str(self.im_input_dir), str(dest)])

        # Verify completeness of transfer
        src_files = {file.name for file in self.im_input_dir.glob('*')}
        dest_files = {file.name for file in Path(dest, self.im_input_dir.name).glob('*')}

        if src_files == dest_files:
            # Delete source files if all are successfully transferred
            for file in self.im_input_dir.glob('*'):
                if file.is_file():
                    file.unlink()  # Safely delete file
            self.log.info(f"All files successfully moved from {self.im_input_dir} to {dest}. Source files deleted.")
        else:
            missing_files = src_files - dest_files
            print(f"Error: The following files were not moved: {missing_files}")

        
    def move_local_output_data_to_nsf(self):
        src = Path("/home/psa_images/temp_data/semifield-outputs", self.batch_name)
        dest = Path(self.nfs_path, "semifield-developed-images")
        dest.mkdir(parents=True, exist_ok=True)  # Ensure destination exists

        # Perform rsync without removing source files
        subprocess.call(['rsync', '-avzh', '--progress', str(src), str(dest)])

        # Verify completeness of transfer
        src_files = {file.name for file in src.glob('**/*') if file.is_file()}
        dest_files = {file.name for file in dest.glob('**/*') if file.is_file()}

        if src_files == dest_files:
            # Delete source files if all are successfully transferred
            for file in src.glob('**/*'):
                if file.is_file():
                    file.unlink()  # Safely delete file
            self.log.info(f"All files successfully moved from {src} to {dest}. Source files deleted.")
        else:
            missing_files = src_files - dest_files
            self.log.error(f"Error: The following files were not moved: {missing_files}")


    def update_access_rights(self):
        # Implementation of the update_access_rights method
        subprocess.call(['chmod', '-R', '777', self.im_output_dir])
        time.sleep(10)
        
    def run_pipeline(self):
        self.log.info("Pipeline execution started.")
        try:
            if self.config['develop_images']:
                self.develop_images()
            if self.config['weed_detection']:
                self.do_weed_detection_yolov8()
            if self.config['fix_access_rights']:
                self.update_access_rights()
            if self.config['backup_and_delete_local_raw_data']:
                self.move_local_raw_data_to_nsf()
            if self.config['backup_and_delete_local_output_data']:
                self.move_local_output_data_to_nsf()
        except Exception as e:
            self.log.error(f"Pipeline execution failed: {e}")
            raise
        self.log.info("Pipeline execution completed successfully.")



if __name__ == "__main__":
    # Command-line arguments for batch_name and config file
    assert len(sys.argv) > 2, "Usage: script.py <batch_name> <config.yaml>"
    batch_name = sys.argv[1]
    config_file = sys.argv[2]

    # Initialize and run the pipeline
    pipeline = ImagePipeline(config_file, batch_name)
    pipeline.run_pipeline()
