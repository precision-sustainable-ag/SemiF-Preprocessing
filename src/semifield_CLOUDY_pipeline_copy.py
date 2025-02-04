import os
import glob
import json
import logging
import signal
import subprocess
import sys
import time
import psutil

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import GPUtil
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.measure import label

class ImagePipeline:
    def __init__(
            self, 
            batch_name, 
            nfs_path, 
            local_semif_uploads_paths, 
            local_semif_outputs_batch_dir, 
            dev_profiles_dir, 
            access_right_update_paths
            ):
        
        self.batch_name = batch_name
        
        self.nfs_path = nfs_path
        self.local_semif_uploads_paths = local_semif_uploads_paths
        self.local_semif_outputs_batch_dir = local_semif_outputs_batch_dir
        self.dev_profiles_dir = dev_profiles_dir
        self.access_right_update_paths = access_right_update_paths
        
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.DEVELOP_IMAGES = False
        self.WEED_DETECTION = True
        self.MASK_GENERATION = False
        # self.UPLOAD_WHEN_COMPLETED = False
        self.FIX_ACCESS_RIGHTS = True
        self.BACKUP_LOCAL_RAW_DATA = True
        self.BACKUP_LOCAL_OUTPUT_DATA = True
        self.VERIFY = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.log_file_dir = Path("./logs", self.batch_name)
        self.log_file_dir.mkdir(parents=True, exist_ok=True)
        log_filename = Path(self.log_file_dir, f"{self.batch_name}_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.log_initial_settings()

    def verify_all_dir_contents(self):
        logging.info(f"Verifying all directory contents for batch {self.batch_name}")
        # Check local uploads
        local_semif_uploads_dir = Path(f"/home/psa_images/temp_data/semifield-upload/{self.batch_name}", "SONY")
        if local_semif_uploads_dir.exists():
            local_upload_batches = [x.stem for x in local_semif_uploads_dir.glob("*.ARW")]
            logging.info(f"Number of Local ARWs in temp_data/semifield-upload/SONY: {len(local_upload_batches)}")

        elif Path(f"/home/psa_images/temp_data/semifield-upload/{self.batch_name}").exists():
            local_semif_uploads_dir = Path(f"/home/psa_images/temp_data/semifield-upload/{self.batch_name}")
            local_upload_batches = [x.stem for x in local_semif_uploads_dir.glob("*.ARW")]
            logging.info(f"Number of Local ARWs in temp_data/semifield-upload: {len(local_upload_batches)}")

        
        # Check local outputs images and pp3s
        local_semif_outputs_batch_dir = Path(f"/home/psa_images/temp_data/semifield-outputs/{self.batch_name}/images")
        if local_semif_outputs_batch_dir.exists():    
            local_output_batches_jpgs = [x.stem for x in local_semif_outputs_batch_dir.glob("*.jpg")]
            local_output_batches_pp3s = [x.stem for x in local_semif_outputs_batch_dir.glob("*.pp3")]
            logging.info(f"Number of Local JPGs in temp_data/semifield-outputs/images: {len(local_output_batches_jpgs)}")
            logging.info(f"Number of Local PP3s in temp_data/semifield-outputs/images: {len(local_output_batches_pp3s)}")

        
        nfs_developed_dir = Path(self.nfs_path, "semifield-developed-images", self.batch_name, "images")
        if nfs_developed_dir.exists():
            nfs_developed_batches = [x.stem for x in nfs_developed_dir.glob("*.jpg")]
            logging.info(f"Number of NFS JPGs in semifield-developed-images: {len(nfs_developed_batches)}")
        
        nfs_upload_dir = Path(self.nfs_path, "semifield-upload", self.batch_name)
        if nfs_upload_dir.exists():
            nfs_upload_batches = [x.stem for x in nfs_upload_dir.glob("*.ARW")]
            logging.info(f"Number of NFS ARWs in semifield-upload: {len(nfs_upload_batches)}")

        
        if local_semif_outputs_batch_dir.exists() and nfs_upload_dir.exists():
            if len(local_output_batches_jpgs) == len(nfs_upload_batches):
                logging.info(f"The number of preprocessed jpgs ({local_output_batches_jpgs}) and uploaded arw images ({nfs_developed_batches}) is the same.")
                logging.info(f"Batch {self.batch_name} has successfully been preprocessed.")

            else:
                logging.error(f"The number of preprocessed jpgs ({len(local_output_batches_jpgs)}) and uploaded arw images ({len(nfs_developed_batches)}) is NOT the same.")
                logging.error(f"Batch {self.batch_name} preprocessing has failed. Please check the color profiling was correctly performed in RawTherapee and rerun.")

        logging.info(f"Verification of all directory contents for batch {self.batch_name} completed.")
        

    def log_initial_settings(self):
        logging.info("Starting ImagePipeline for batch: %s", self.batch_name)
        logging.info("Pipeline settings:")
        logging.info("  NFS_PATH: %s", self.nfs_path)
        logging.info("  BATCH_NAME: %s", self.batch_name)
        logging.info("  DEVELOP_IMAGES: %s", self.DEVELOP_IMAGES)
        logging.info("  WEED_DETECTION: %s", self.WEED_DETECTION)
        logging.info("  MASK_GENERATION: %s", self.MASK_GENERATION)
        # logging.info("  UPLOAD_WHEN_COMPLETED: %s", self.UPLOAD_WHEN_COMPLETED)
        logging.info("  FIX_ACCESS_RIGHTS: %s", self.FIX_ACCESS_RIGHTS)
        logging.info("  BACKUP_LOCAL_RAW_DATA: %s", self.BACKUP_LOCAL_RAW_DATA)
        logging.info("  BACKUP_LOCAL_OUTPUT_DATA: %s", self.BACKUP_LOCAL_OUTPUT_DATA)

    def run(self):
        try:
            if self.DEVELOP_IMAGES:
                self.develop_images()
            if self.WEED_DETECTION:
                self.do_weed_detection_yolov8()
            if self.MASK_GENERATION:
                self.do_mask_generation()
            # if self.UPLOAD_WHEN_COMPLETED:
            #     self.upload_to_azure()
            if self.FIX_ACCESS_RIGHTS:
                self.update_access_rights()
            if self.BACKUP_LOCAL_RAW_DATA:
                self.backup_and_verify_local_raw_data()
            if self.BACKUP_LOCAL_OUTPUT_DATA:
                self.backup_and_verify_local_output_data()
            if self.VERIFY:
                self.verify_all_dir_contents()
        except Exception as e:
            logging.error(f"Pipeline failed for batch {self.batch_name} with error: {e}")
            self.terminate_all_processes()
            sys.exit(1)

    def terminate_all_processes(self):
        logging.info("Terminating all subprocesses.")
        try:
            current_pid = os.getpid()
            parent = psutil.Process(current_pid)
            children = parent.children(recursive=True)
            for child in children:
                logging.info("Terminating subprocess with PID: %d", child.pid)
                child.terminate()
        except Exception as e:
            logging.error("Failed to terminate subprocesses: %s", e)

    def develop_images(self):
        logging.info("Starting image development for batch %s", self.batch_name)
        
        
        for dev_im_input_path in self.local_semif_uploads_paths:
            for zippath in glob.iglob(str(dev_im_input_path / Path("*.JPG"))):
                os.remove(zippath)
                logging.debug("Removed JPG image %s", zippath)
            
            list_of_raw_images = glob.glob(str(dev_im_input_path / Path("*.ARW")))
            logging.info(f"Number of raw images to develop: {len(list_of_raw_images)}")
            if len(list_of_raw_images) > 0:
                
                if not self.dev_profiles_dir.exists():
                    logging.error("Development profile does not exist for batch %s", self.batch_name)
                    return
                semif_outputs_batch_image_dir = Path(self.local_semif_outputs_batch_dir, "images")
                os.makedirs(semif_outputs_batch_image_dir, exist_ok=True)
                
                exe_command = f"./RawTherapee_5.8.AppImage --cli \
                    -O {semif_outputs_batch_image_dir} \
                        -p {self.dev_profiles_dir} \
                            -j99 \
                                -c {dev_im_input_path}"
                exe_command2 = 'bash -c "OMP_NUM_THREADS=60; ' + exe_command + '"' # OMP_NUM_THREADS=90;

                try:
                    subprocess.run(exe_command2, shell=True, check=True)
                    logging.info("Image development completed for batch %s", self.batch_name)
                except subprocess.CalledProcessError as e:
                    logging.error("Image development failed for batch %s with error: %s", self.batch_name, e)

    def do_weed_detection_yolov8(self):
        logging.info("Starting YOLOv8 weed detection for batch %s", self.batch_name)
        image_source = Path(self.local_semif_outputs_batch_dir, "images")
        
        try:
            gpu_idx = GPUtil.getFirstAvailable(order='memory', maxMemory=0.6, attempts=90, interval=120)[0]
            logging.info("Using GPU ID: %d", gpu_idx)
        except Exception as e:
            logging.error("GPU not available: %s", e)
            return
        
        env_command = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
        exe_command = f"/home/mkutuga/miniconda3/envs/semif_py311/bin/python predict_yolov8.py --source {image_source} --batch_name {self.batch_name}"
        
        try:
            subprocess.run(f"{env_command} && {exe_command}", shell=True, check=True)
            # subprocess.run(['/bin/bash', '-c', env_command + ";" + exe_command])
            logging.info("Weed detection completed for batch %s", self.batch_name)
        except subprocess.CalledProcessError as e:
            logging.error(f"Weed detection failed for batch {self.batch_name} with error: {e}. Exiting.")
            sys.exit(1)
    
    def getLargestCC(self, segmentation):
        labels = label(segmentation)
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
        return largestCC
    
    def generate_mask(self, img, im_output_path_and_filename ):
        img_red = img[:,:,0]
        img_green = img[:,:,1]
        img_blue = img[:,:,2]
        img_exbrg = -4.0*img_green.astype(float) - 4.0*img_red.astype(float) + 4.0*img_blue.astype(float)

        median = cv2.medianBlur((img_exbrg>-50).astype('uint8'),19)
        median = cv2.medianBlur(median.astype('uint8'),121)
        #median = ndimage.percentile_filter((img_exbrg>-40).astype('uint8'), percentile=50, size=20)
        
        kernel = np.ones((2700, 2700), np.uint8)#np.ones((500, 900), np.uint8)
        img_dilated = cv2.dilate(median, kernel, iterations=1)
        
        img_connected_component = self.getLargestCC(img_dilated)
        
        #print(np.sum(img_connected_component))
        img_final = np.ones(img.shape[0:2])
        #if ((np.sum(img_connected_component)>2300000) and (np.sum(img_connected_component)<40000000)):
        if ((np.sum(img_connected_component)>1200000) and (np.sum(img_connected_component)<40000000)):
            img_final[img_connected_component==1] = 0
        
        #### ADDED weed frabric masking CODE ####
        img_hsv = color.rgb2hsv(img)
        median2 = cv2.medianBlur(((img_hsv[..., 1]*255)<55).astype('uint8'),21)
        img_final[median2==1] = 0
        
        imageio.imsave(im_output_path_and_filename, (img_final * 255).astype(np.uint8), compress_level=3)

    def do_mask_generation(self):
        logging.info("Starting mask generation for batch %s", self.batch_name)
        masking_im_input_path = Path(self.local_semif_outputs_batch_dir, "images")
        masking_im_output_path = Path(self.local_semif_outputs_batch_dir, "masks")

        filename_list = glob.glob(str(masking_im_input_path / Path("*.jpg")))
        filename_list.sort()
        
        for i, im_filename_long in enumerate(filename_list):
            logging.info("Processing image %d of %d: %s", i + 1, len(filename_list), im_filename_long)
            os.makedirs(masking_im_output_path, exist_ok=True)
            img = np.asarray(Image.open(im_filename_long))
            im_filename = Path(im_filename_long).stem
            img = np.asarray(img)
            
            im_output_path_and_filename = str(masking_im_output_path) + "/" + im_filename + "_mask.png"
            
            #generate_mask(img, im_output_path_and_filename)
            self.executor.submit(self.generate_mask, img, im_output_path_and_filename)
            time.sleep(0.1)
        
        self.executor.shutdown(wait=True)
        logging.info("Mask generation completed for batch %s", self.batch_name)

    def _backup_and_verify_data(self, src, dest, data_type, remove_only=False):
        if not remove_only:
            try:
                logging.info(f"Moving {data_type} data for batch {self.batch_name} to NFS")
                subprocess.run(['rsync', '-avzh', '--progress', src, dest], check=True)
                logging.info("%s data moved successfully for batch %s", data_type.capitalize(), self.batch_name)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to move %s data for batch %s: %s", data_type, self.batch_name, e)
                return

        src_files = set(os.listdir(src))
        dest_files = set(os.listdir(dest))
        
        if src_files.issubset(dest_files):
            logging.info("All %s files successfully moved to NFS for batch %s", data_type, self.batch_name)
            for file in src_files:
                os.remove(os.path.join(src, file))
            logging.info(f"Local {data_type} files deleted after successful transfer for batch {self.batch_name}")
        else:
            not_moved_files = src_files - dest_files
            logging.warning(f"Not all {data_type} files were moved to NFS for batch {self.batch_name}")
            logging.warning(f"Files not moved to NFS for batch {self.batch_name}: {not_moved_files}")

    def backup_and_verify_local_raw_data(self):
        logging.info("Starting backup of local raw data to longterm_images2/semifield-upload for batch %s", self.batch_name)
        for path in self.local_semif_uploads_paths:
            if path.exists():
                src = path
                break
        dest = os.path.join(self.nfs_path, "semifield-upload")
        batch_dest = Path(dest, self.batch_name)
        if batch_dest.exists():
            batch_images = [x.stem for x in batch_dest.glob("*.ARW")]
            src_images = [x.stem for x in src.glob("*.ARW")]
            if len(batch_images) == len(src_images):
                logging.info(f"Batch {self.batch_name} raw data already exists in NFS.")
                return
            
        self._backup_and_verify_data(src, dest, "raw")
        logging.info(f"Backup of local raw data completed for batch {self.batch_name}")

    def backup_and_verify_local_output_data(self):
        logging.info(f"Starting backup of local output data to longterm_images2/semifield-developed-images for batch {self.batch_name}")
        src = self.local_semif_outputs_batch_dir
        dest = os.path.join(self.nfs_path, "semifield-developed-images")
        self._backup_and_verify_data(src, dest, "output")
        logging.info(f"Backup of local output data completed for batch {self.batch_name}")

    def update_access_rights(self):
        for path in self.access_right_update_paths:
            try:
                logging.info("Updating access rights for path: %s", path)
                subprocess.run(['chmod', '-R', '777', path], check=True)
                logging.info("Access rights updated for path: %s", path)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to update access rights for path %s: %s", path, e)
            
            time.sleep(10)

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        logging.error("Batch name argument is required.")
        sys.exit(1)
    batch_name = sys.argv[1]
    
    nfs_path = "/mnt/research-projects/s/screberg/longterm_images2/"
    local_semif_uploads_paths = [
        Path(f"temp_data/semifield-upload/{batch_name}"),
        Path(f"temp_data/semifield-upload/{batch_name}/SONY")
    ]
    local_semif_outputs_batch_dir = Path(f"temp_data/semifield-outputs/{batch_name}")
    dev_profiles_dir = Path(f"data/persistent_data/semifield-utils/image_development/dev_profiles/{batch_name}.pp3")
    access_right_update_paths = ["temp_data", "persistent_data"]
    
    pipeline = ImagePipeline(
        batch_name, 
        nfs_path, 
        local_semif_uploads_paths, 
        local_semif_outputs_batch_dir, 
        dev_profiles_dir, 
        access_right_update_paths
        )
    
    pipeline.run()