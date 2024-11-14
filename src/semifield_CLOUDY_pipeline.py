#import cv2
import glob
import json
import ntpath
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import GPUtil
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

#from scipy import ndimage
from skimage.measure import label

executor = ThreadPoolExecutor(max_workers=16)
futures = []

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))


assert (len(sys.argv) > 1)
    
batch_name = sys.argv[1]

DEVELOP_IMAGES = True
WEED_DETECTION = True
MASK_GENERATION = False
UPLOAD_WHEN_COMPLETED = False
FIX_ACCESS_RIGHTS = True
BACKUP_AND_DELETE_LOCAL_RAW_DATA = True
BACKUP_AND_DELETE_LOCAL_OUTPUT_DATA = True


nfs_path = "/mnt/research-projects/s/screberg/longterm_images2/"
# nfs_path = "/mnt/research-projects/s/screberg/GROW_DATA/"


def develop_images(batch_name):
    dev_im_input_path1 = Path("/home/psa_images/temp_data/semifield-upload/") / str(batch_name)
    dev_im_input_path2 = Path("/home/psa_images/temp_data/semifield-upload/") / str(batch_name) / "SONY"
    dev_im_input_paths = [dev_im_input_path1, dev_im_input_path2]
    for dev_im_input_path in dev_im_input_paths:
        # remove jpg images from the raw folder
        for zippath in glob.iglob(str(dev_im_input_path / Path("*.JPG"))):
            os.remove(zippath)
        list_of_raw_images = glob.glob(str(dev_im_input_path / Path("*.ARW")))
        if(len(list_of_raw_images)>0):
            dev_im_output_path = Path("/home/psa_images/temp_data/semifield-outputs/") / str(batch_name) / "images/"
            dev_im_development_profile_path = Path("/home/psa_images/persistent_data/semifield-utils/image_development/dev_profiles/") / (str(batch_name)+".pp3")
            
            assert (os.path.exists(dev_im_development_profile_path))
            
            os.makedirs(dev_im_output_path, exist_ok = True)
            
            exe_command = f"./RawTherapee_5.8.AppImage --cli \
                -O {dev_im_output_path} \
                -p {dev_im_development_profile_path} \
                -j99 \
                -c {dev_im_input_path}"
            exe_command2 = 'bash -c "OMP_NUM_THREADS=90; ' + exe_command + '"'
            
            print(exe_command2)
            try:
                # Run the rawtherapee command
                #subprocess.run(exe_command, shell=True, check=True)
                subprocess.run(exe_command2, shell=True, check=True)
                print("")
            except Exception as e:
                raise e
                
                  
            
    print("Image development has finished for batch " + str(batch_name))
    
    print("Uploading rawtherapee development profiles to azure..")
    dev_im_development_profile_path = Path("/home/psa_images/persistent_data/semifield-utils/image_development/dev_profiles/") / (str(batch_name)+".pp3")
    exe_command = f"/home/psa_images/semifield_tools/azcopy copy \
        {dev_im_development_profile_path} \
        '# SAS key goes here' \
        --recursive \
        --overwrite=true"
        
    print(exe_command)
    try:
        process_id = subprocess.run(exe_command, shell=True, check=True)
    except Exception as e:
        raise e
    #os.killpg(os.getpgid(process_id.pid), signal.SIGKILL)
    #process_id.wait()
    print("Profile has been uploaded for batch " + str(batch_name))
    

def do_weed_detection_updated(batch_name):

    image_source = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name) + "/images/\*.jpg"
    export_dir = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name)
    
    exe_command = f"/home/psa_images/anaconda3/envs/yolo/bin/python detect.py \
        --source {image_source} \
        --weights 'runs/train/exp5/weights/best.pt' \
        --project 'run_weeds' \
        --name {export_dir} \
        --data 'wir.yaml' \
        --device '1' \
        --export-predictions \
        --exist-ok"

    print(exe_command)
    try:
        # Run the rawtherapee command
        process_id = subprocess.run(['/bin/bash', '-c', exe_command])
        print("")
    except Exception as e:
        raise e
    print(process_id)
    print("Weed detection has finished for batch " + str(batch_name))
    

def do_weed_detection_yolov8(batch_name):

    image_source = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name) + "/images"
    batch_name = str(batch_name)
    print("Finding available GPU for weed detection..")
    gpu_idx = GPUtil.getFirstAvailable(order='memory', maxMemory=0.6, attempts=90, interval=120)[0]
    print("Found available gpu.. Will use ID: " + str(gpu_idx))
    env_command = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    exe_command = f"/home/psa_images/anaconda3/envs/yolov8/bin/python predict_yolov8.py \
        --source {image_source} \
        --batch_name {batch_name}"

    print(exe_command)
    try:
        print([env_command, '/bin/bash', '-c', exe_command])
        process_id = subprocess.run(['/bin/bash', '-c', env_command + ";" + exe_command])
        print("")
    except Exception as e:
        raise e
    print(process_id)
    print("Weed detection has finished for batch " + str(batch_name))


def generate_mask(img, im_output_path_and_filename ):
    img_red = img[:,:,0]
    img_green = img[:,:,1]
    img_blue = img[:,:,2]
    img_exbrg = -4.0*img_green.astype(float) - 4.0*img_red.astype(float) + 4.0*img_blue.astype(float)

    median = cv2.medianBlur((img_exbrg>-50).astype('uint8'),19)
    median = cv2.medianBlur(median.astype('uint8'),121)
    #median = ndimage.percentile_filter((img_exbrg>-40).astype('uint8'), percentile=50, size=20)
    
    kernel = np.ones((2700, 2700), np.uint8)#np.ones((500, 900), np.uint8)
    img_dilated = cv2.dilate(median, kernel, iterations=1)
    
    img_connected_component = getLargestCC(img_dilated)
    
    #print(np.sum(img_connected_component))
    img_final = np.ones(img.shape[0:2])
    #if ((np.sum(img_connected_component)>2300000) and (np.sum(img_connected_component)<40000000)):
    if ((np.sum(img_connected_component)>1200000) and (np.sum(img_connected_component)<40000000)):
        img_final[img_connected_component==1] = 0
    
    #### ADDED weed frabric masking CODE ####
    img_hsv = color.rgb2hsv(img)
    median2 = cv2.medianBlur(((img_hsv[..., 1]*255)<55).astype('uint8'),21)
    img_final[median2==1] = 0
    
    imageio.imsave(im_output_path_and_filename, (img_final*255).astype(np.uint8), compress_level=3)

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def do_mask_generation(batch_name):
    masking_im_input_path = Path("/home/psa_images/temp_data/semifield-outputs/") / str(batch_name) / "images/"
    masking_im_output_path = Path("/home/psa_images/temp_data/semifield-outputs/") / str(batch_name) / "masks/"
    
    filenameList = glob.glob(str(masking_im_input_path / Path("*.jpg")))
    filenameList.sort()
    i = 0
    for im_filename_long in filenameList:
        print(str(i) + " of " + str(len(filenameList)))
        print(im_filename_long)
        os.makedirs(masking_im_output_path, exist_ok=True)
        img = Image.open(im_filename_long)
        width, height = img.size
        im_filename = ntpath.basename(im_filename_long)[0:-4]
        img = np.asarray(img)
        
        im_output_path_and_filename = str(masking_im_output_path) + "/" + im_filename + "_mask.png"
        
        #generate_mask(img, im_output_path_and_filename)
        a = executor.submit(generate_mask, img, im_output_path_and_filename)
        time.sleep(0.1)
        
        i=i+1
    time.sleep(300)
        
def upload_to_azure(batch_name):
    time.sleep(30)
    export_dir = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name)
    exe_command = f"/home/psa_images/semifield_tools/azcopy copy \
        {export_dir} \
        '# SAS key goes here' \
        --recursive \
        --overwrite=true"
        
    print(exe_command)
    try:
        # Run the rawtherapee command
        process_id = subprocess.run(exe_command, shell=True, check=True)
        print("")
    except Exception as e:
        #raise e
        print("")
        
    ### repeat to upload failed files ###
    exe_command = f"/home/psa_images/semifield_tools/azcopy copy \
        {export_dir} \
        '# SAS key goes here' \
        --recursive \
        --overwrite=false"
        
    print(exe_command)
    try:
        # Run the rawtherapee command
        process_id = subprocess.run(exe_command, shell=True, check=True)
        print("")
    except Exception as e:
        print("")
    
    #os.killpg(os.getpgid(process_id.pid), signal.SIGKILL)
    #process_id.wait()
    print("Weed detection has finished for batch " + str(batch_name))

def move_local_raw_data_to_NSF(batch_name):
    dev_im_input_path1 = "/home/psa_images/temp_data/semifield-upload/" + str(batch_name)
    src = dev_im_input_path1
    dest = nfs_path + "semifield-upload"
    #subprocess.call('rsync --remove-source-files -h ' + src + ' ' + dest)
    #subprocess.Popen(['rsync', '-avzh', '--remove-source-files', '--progress', src, dest])
    subprocess.call(['rsync', '-avzh', '--remove-source-files', '--progress', src, dest])
    
def move_local_output_data_to_NSF(batch_name):
    src = "/home/psa_images/temp_data/semifield-outputs/" + str(batch_name)
    dest = nfs_path + "semifield-developed-images"
    #subprocess.call('rsync -avzh --remove-source-files --progress ' + src + ' ' + dest)
    #subprocess.Popen(['rsync', '-avzh', '--remove-source-files', '--progress', src, dest])
    subprocess.call(['rsync', '-avzh', '--remove-source-files', '--progress', src, dest])
    
def download_rawtherapee():
    exe_command = "wget https://rawtherapee.com/shared/builds/linux/RawTherapee_5.9.AppImage"
    try:
        # Run the rawtherapee command
        process_id = subprocess.run(exe_command, shell=True, check=True)
        print("")
    except Exception as e:
        raise e
def update_access_rights():



    src = "/home/psa_images/temp_data"
    #subprocess.Popen(['chmod', '-R', '777', src])
    subprocess.call(['chmod', '-R', '777', src])
    
    src = "/home/psa_images/persistent_data"
    #subprocess.Popen(['chmod', '-R', '777', src])
    subprocess.call(['chmod', '-R', '777', src])
    
    time.sleep(10)


if(DEVELOP_IMAGES):
    develop_images(batch_name)
if(WEED_DETECTION):
    #do_weed_detection_updated(batch_name)
    do_weed_detection_yolov8(batch_name)
if(MASK_GENERATION):
    do_mask_generation(batch_name)
if(UPLOAD_WHEN_COMPLETED):
    upload_to_azure(batch_name)
if(FIX_ACCESS_RIGHTS):
    update_access_rights()
if(BACKUP_AND_DELETE_LOCAL_RAW_DATA):
    move_local_raw_data_to_NSF(batch_name)
if(BACKUP_AND_DELETE_LOCAL_OUTPUT_DATA):
    move_local_output_data_to_NSF(batch_name)
    
