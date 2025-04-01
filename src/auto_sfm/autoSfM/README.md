# autoSfM

## Metashape and Python
This project uses Metashape version 1.8.0 and Python SDK version 1.8.0. The code has been tested with Python version 3.8

## Local Setup:
Set the environment variable ```agisoft_LICENSE``` to the path to the license file. This file is generated after activation using the activation key, in the directory where the Metashape software is located. This can be done using the command 
```sh
export agisoft_LICENSE="<PATH_TO_LICENSE_FILE>"
```

## Installing Docker

Reference: https://docs.docker.com/engine/install/ubuntu/

```sh
sudo apt-get update

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Test the installation by running
```sh
sudo docker run hello-world
```

Post Installation: Make docker runnable as non-root
```sh
sudo groupadd docker
sudo usermod -aG docker $USER
# This command will make the changes effective
newgrp docker
```

Test using
```sh
docker run hello-world
```

Configure Docker to start on boot
```sh
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

## Docker with GPU (optional)

Follow this instructions to install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Make changes to the [DockerFile](Dockerfile)
```Docker
# Pull the offical base image
FROM ubuntu:18.04
# Place a comment above and uncomment below to use gpu
# FROM nvidia/cuda:11.4.0-base
```

## Docker execution structure
- Building the image: The Dockerfile will pull a base Ubuntu 18.04 image, will copy all the code and dependencies (Metashape installation tar and Python API whl) to the image, and create environment variables. ```execute_pipeline.sh``` is marked as the ```ENTRYPOINT```, i.e. when the container is run, it will act as an executable and execute ```execute_pipeline.sh```
- The execution shell script will activate the license, run the autoSfM pipeline an deactivate the license before exiting
- Activation key is passed at runtime, and not while building the image to prevet leakage
- config.yml passed at runtime by modifying ```volumes/config/cnfig.yml``` file

## Setting up the code and directories

- Clone the repository
  ```sh
  git clone https://github.com/precision-sustainable-ag/autoSfM.git
  ```
- Copy metashape tar.gz file to ```autoSfM/volumes/metashape``` and extract the contents
  ```sh
  tar -xzvf metashape-pro_1_8_0_amd64.tar.gz
  ```
- Set up the data directory
  ```sh
  mkdir <BASE_DIR>/storage
  ```
- Set up the export directory
  ```sh
  mkdir <BASE_DIR>/exports
  ```

Where ```<BASE_DIR>``` is the base directory where the data and exports will be saved. Note that ```<BASE_DIR>``` and all the subdirectories should be owned by the current user since Docker container runs as a non-root user. The directory and all the subdirectories can be owned using:
```sh
sudo chown -R <USER>:<GROUP> <BASE_DIR>
```

## Docker volumes
3 local directories are mounted on the final Docker container. This means that the data which the Docker container writes to the mounts will be available outside the container (in the directories which are used for the respective mounts) even when the container has finished execution.

Following describe the paths in the Docker container where local directories are mounted:
1. ```/home/psi_docker/autoSfM/volumes```: This mount contains the ```config``` and ```metashape``` directories. ```config``` will contain the ```config.yml``` file, and ```metashape``` will contain the ```metashape-pro``` directory, which contains the Metashape software. The ```volumes``` directory from the cloned repository on the host (local directory) should be mounted here.

2. ```/home/psi_docker/autoSfM/storage```: This mount will contain the input data to be processed. Each set of images should be contained in a single directory inside ```/home/psi_docker/autoSfM/storage```. Any directory which will contain the subdirectories can be used for the mount (eg. ```/mnt/data/auto_sfm/storage/circular_12bit_referenced_dataset```), Here, ```/mnt/data/auto_sfm/storage/``` on the host can be mounted at ```/home/psi_docker/autoSfM/storage``` in the container). The GCP file can be located anywhere in ```/home/psi_docker/autoSfM/storage```, as long as the file is unique per set of photos. This means that the data located in ```/mnt/data/auto_sfm/storage``` will be accessible to the Docker container at ```/home/psi_docker/autoSfM/storage```.

3. ```/home/psi_docker/autoSfM/exports```: This mount will contain the data exported by the code. The code will make separate directories for each project with the name of the project (defined in ```config.yml```). Each project directory will contain 3 directories upon complete execution: ```ortho```, ```projects```, and ```reference```.

   a. ```ortho```: This will contain the orthomisaic generated by the code.

   b. ```project```: This will contain the saved Metashape project.
   
   c. ```reference```: This will contain the exported reference files.

   For example, ```/mnt/data/auto_sfm/exports``` can be mounted at ```/home/psi_docker/autoSfM/exports```. This means any files which the Docker container writes to ```/home/psi_docker/autoSfM/exports``` will be accessible at ```/mnt/data/auto_sfm/exports``` even when the container stops.

   Another directory ```downsized_photos``` may be present if the option ```downsize -> enabled``` is set to ```True``` in the config.

## Directory Structure
The ```storage``` and ```exports``` directories have to follow the following directory structure for the pipeline to work correctly (by itself and with SemiF-Annotation). The directory structure for the ```exports``` is created automatically from the batch_id and the path to the ```storeage``` and ```exports``` specified in the ```config.yml``` file.

```
storage
├── NC_2022-03-11
│   ├── developed
│   │   ├── row1_1.jpg
│   │   └── ...
│   └── GroundControlPoints.csv
├── NC_2022-03-29
│   └── ...
└── ...
```
```
exports
├── NC_2022-03-11
│   ├── ortho
│   │   └── orthomosaic.tif
│   ├── reference
│   │   ├── camera_reference.csv
│   │   ├── error_statistics.csv
│   │   ├── fov.csv
│   │   └── gcp_reference.csv
│   ├── project
│   │   ├── NC_2022-03-11.psx
│   │   └── ...
│   ├── downscaled_photos
│   │   ├── row1_1.jpg
│   │   ├── row1_2.jpg
│   │   └── ...
│   └── logfile.log
├── NC_2022-03-29
└── ...
```

## Reference files
After the execution of the code, 4 data files are generated. The names of the files have been hardcoded for now, but can be made flexible through the config file. Following are the files:
1. ```camera_reference.csv```: Camera calibration parameters, estimated rotations, alignment status and estimated location in real-world coordinates.
2. ```error_statistics.csv```: Number and percentage of GCPs detected, and the number and percentage of cameras aligned.
3. ```fov.csv```: 4 camera field of view coordinates and image height and width in real-world coordinates.
4. ```gcp_reference.csv```: GCP location coordinates, estimation errors per coordinate, variance per coordinate, and detection status.

## Docker build and run instructions
- **Building the image**
  ```sh
  docker build -t sfm .
  ```
  The config file and the mounts does not afect the building of the image.

- **Running the container**

  The processing is carried out through arguments in a YAML config file located at ```volumes/config/config.yml```. The ```volumes``` directory is mounted at ```/home/psi_docker/autoSfM/volumes``` in the Docker container. This means that once the image is built, the processing for multiple projects can be done by modifying the config file without the need for rebuilding the image.
  ```sh
  docker run \
         -v <REPOSITORY_PATH>/autoSfM/volumes/:/home/psi_docker/autoSfM/volumes \
         -v <DATA_PATH>/storage/:/home/psi_docker/autoSfM/storage \
         -v <DATA_PATH>/exports/:/home/psi_docker/autoSfM/exports \
         sfm <LICENSE_KEY>
  ```
  where ```<REPOSITORY_PATH>``` is the absolute path to the parent directory where the local clone of the repository is located, and ```<DATA_PATH>``` if the absolute path to the directory where the data is located.

  For example:
  ```sh
  docker run \
         -v /home/azureuser/autoSfM/volumes/:/home/psi_docker/autoSfM/volumes \
         -v /mnt/data/auto_sfm/storage/:/home/psi_docker/autoSfM/storage \
         -v /mnt/data/auto_sfm/exports/:/home/psi_docker/autoSfM/exports \
         sfm <LICENSE_KEY>
  ```

- **Restrictions**

  A different Metashape project is created every time the container is run, and an error is thrown if a project already exists. If the execution is interrupted and a set of photos has to be processed again, change the project name, or delete the files generated for the project in the ```exports``` directory.

- **User in the Docker container**

  The Docker container runs as a non-root user ```psi_docker```. This has been defined in the Dockerfile. As a result, any files which the container writes to the mounts are owned by the current user on the host machine.


## Breakdown of the config.yml file

The config file is meant for the user to modify the batch to be processed. The directory structure is not changed through the config file, but is defined through the code. This ensures that the directory structure is followed for the pipeline to be compatible with other pipelines.

```yml
# Required config
# All the paths in the config file should be the paths of the Docker container
export_path: "/home/psi_docker/autoSfM/exports" # The path to the export mount
storage_path: "/home/psi_docker/autoSfM/storage" # Path in the Docker container
batch_id: "quickstart"

# Electives: The enabled key controls if the pipeline will generate the respective feature or not
downscale:
  enabled: True
  factor: 0.25 # between (0, 1]

align_photos:
  downscale: 1 # Accuracy parameter, lower means higher accuracy
  # The autosave key determines if the project is saved after computing the respective feature
  # It is recommended to keep this set to True
  autosave: True

depth_map:
  enabled: True
  downscale: 4
  autosave: True

dense_cloud:
  enabled: True
  autosave: True

dem: 
  enabled: True
  autosave: True

orthomosaic:
  enabled: True
  export: 
    enabled: True
  autosave: True

camera_fov:
  enabled: True

```
