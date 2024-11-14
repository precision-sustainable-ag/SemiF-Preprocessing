import glob
import json
import ntpath
import os
import subprocess
import sys
import time
import yaml

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import GPUtil
import imageio
import numpy as np
from PIL import Image
from skimage import color
from skimage.measure import label


class ImagePipeline:
    def __init__(self, config_path, batch_name):
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.batch_name = batch_name
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        self.nfs_path = self.config['nfs_path']
        self.im_input_dir = Path(self.config['image_input_dir']) / batch_name
        self.im_output_dir = Path(self.config['image_output_dir']) / batch_name / "images"
        self.im_output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = Path(self.config['image_development_profile_path'], f"{batch_name}.pp3") 
        assert self.profile_path.exists(), f"Development profile ({self.profile_path}) does not exist"

        self.raw_extension = ".RAW" if self.config['bbot_version'] == 3.1 else ".ARW"

    def develop_images(self):
        # Implementation of the develop_images method
        num_of_raw_images = len(list(self.im_input_dir.glob(f"*{self.raw_extension}")))
        
        if num_of_raw_images == 0:
            print(f"No RAW images found in the input directory ({self.im_input_dir})")
            return            
        
        else:
            exe_command = f"./RawTherapee_5.9.AppImage --cli \
                -O {self.im_output_dir} \
                -p {self.profile_path} \
                -j99 \
                -c {self.im_input_dir}"
            exe_command2 = 'bash -c "OMP_NUM_THREADS=90; ' + exe_command + '"'
            
            print(exe_command2)
            try:
                # Run the rawtherapee command
                #subprocess.run(exe_command, shell=True, check=True)
                subprocess.run(exe_command2, shell=True, check=True)
                print("")
            except Exception as e:
                raise e

    def do_weed_detection_yolov8(self):
        # Implementation of the do_weed_detection_yolov8 method
        pass

    def generate_mask(self, img, output_path):
        # Implementation of the generate_mask method
        pass

    def get_largest_cc(self, segmentation):
        # Implementation of the get_largest_cc method
        pass

    def do_mask_generation(self):
        # Implementation of the do_mask_generation method
        pass

    def upload_to_azure(self):
        # Implementation of the upload_to_azure method
        pass

    def move_local_raw_data_to_nsf(self):
        # Implementation of the move_local_raw_data_to_NSF method
        dest = self.nfs_path + "semifield-upload"
        #subprocess.call('rsync --remove-source-files -h ' + src + ' ' + dest)
        #subprocess.Popen(['rsync', '-avzh', '--remove-source-files', '--progress', src, dest])
        subprocess.call(['rsync', '-avzh', '--remove-source-files', '--progress', self.im_input_dir, dest])
        
    
    def move_local_output_data_to_nsf(self):
        # Implementation of the move_local_output_data_to_NSF method
        pass

    def update_access_rights(self):
        # Implementation of the update_access_rights method
        #subprocess.Popen(['chmod', '-R', '777', src])
        subprocess.call(['chmod', '-R', '777', self.im_output_dir])
        time.sleep(10)
        
    def run_pipeline(self):
        # Execute the pipeline based on the configuration
        if self.config['develop_images']:
            self.develop_images()
        if self.config['weed_detection']:
            self.do_weed_detection_yolov8()
        if self.config['mask_generation']:
            self.do_mask_generation()
        if self.config['upload_when_completed']:
            self.upload_to_azure()
        if self.config['fix_access_rights']:
            self.update_access_rights()
        if self.config['backup_and_delete_local_raw_data']:
            self.move_local_raw_data_to_nsf()
        if self.config['backup_and_delete_local_output_data']:
            self.move_local_output_data_to_nsf()


if __name__ == "__main__":
    # Command-line arguments for batch_name and config file
    assert len(sys.argv) > 2, "Usage: script.py <batch_name> <config.yaml>"
    batch_name = sys.argv[1]
    config_file = sys.argv[2]

    # Initialize and run the pipeline
    pipeline = ImagePipeline(config_file, batch_name)
    pipeline.run_pipeline()
