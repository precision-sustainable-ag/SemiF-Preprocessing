import subprocess
import time
from pathlib import Path
from omegaconf import DictConfig
import logging
import hydra

from src.utils.utils import find_lts_dir

log = logging.getLogger(__name__)


class ImagePipeline:
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.batch_id = cfg.batch_id
        
        # Determine the LTS directory name using batch_id and configuration paths
        self.lts_dir_name = find_lts_dir(self.batch_id, self.cfg.paths.lts_locations, local=True).name
        log.info(f"LTS directory name determined: {self.lts_dir_name}")
        
        # Define the input directory for DNG images and the output directory for developed images
        self.im_input_dir = Path(self.cfg.paths.data_dir) / self.lts_dir_name / "semifield-developed-images" / self.batch_id / "pngs_iteration3"
        self.im_output_dir = Path(self.cfg.paths.data_dir) / self.lts_dir_name / "semifield-developed-images" / self.batch_id / "images_iteration3"
        self.im_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Input directory set to: {self.im_input_dir}")
        log.info(f"Output directory set to: {self.im_output_dir}")

        # Define the path to the RawTherapee profile to use for image development
        self.profile_path = Path(self.cfg.paths.image_development, "dev_profiles", cfg.png2jpg.profile_name + ".pp3")

        # Flags from the configuration to control pipeline behavior
        self.develop_images_flag = self.cfg.png2jpg.develop_images
        self.fix_access_rights_flag = self.cfg.png2jpg.fix_access_rights
        log.info(f"Develop images flag: {self.develop_images_flag}")
        log.info(f"Fix access rights flag: {self.fix_access_rights_flag}")

    def develop_images(self) -> None:
        """
        Develop images by converting RAW DNG files to JPEG using RawTherapee.
        It checks for the existence of RAW images in the input directory and, if found,
        constructs and executes a command to process these images using RawTherapee.
        """
        log.info("Starting image development process.")
        # Count the number of RAW images in the input directory
        num_of_raw_images = len(list(self.im_input_dir.glob("*.png")))
        log.info(f"Number of RAW images found: {num_of_raw_images}")
        
        if num_of_raw_images == 0:
            log.error(f"No RAW images found in the input directory ({self.im_input_dir})")
            return
        else:
            # Construct the command to run RawTherapee in CLI mode
            exe_command = (
                f"./squashfs-root/usr/bin/rawtherapee-cli "
                f"-Y "
                f"-j99 "
                f"-O {self.im_output_dir} "
                f"-p {self.profile_path} "
                f"-c {self.im_input_dir}"
            )
            # Wrap the command in a bash call to set the OMP_NUM_THREADS environment variable
            exe_command2 = f'bash -c "OMP_NUM_THREADS=90; {exe_command}"'
            
            try:
                # Run the RawTherapee command
                subprocess.run(exe_command2, shell=True, check=True)

            except subprocess.CalledProcessError as e:
                log.error("RawTherapee command failed.")
                log.error(e)
                raise e

    def update_access_rights(self) -> None:
        """
        Update file access rights for the developed images.
        """
        log.info("Updating access rights for developed images.")
        result = subprocess.call(['chmod', '-R', '777', str(self.im_output_dir)])
        log.info(f"chmod call returned: {result}")
        time.sleep(3)
        log.info("Access rights updated successfully.")

    def run_pipeline(self) -> None:
        """
        Execute the image processing pipeline based on configuration flags.
        """
        if self.develop_images_flag:
            log.info("Develop images flag is enabled. Starting image development.")
            self.develop_images()
        else:
            log.info("Develop images flag is disabled. Skipping image development.")

        if self.fix_access_rights_flag:
            log.info("Fix access rights flag is enabled. Updating access rights.")
            self.update_access_rights()
        else:
            log.info("Fix access rights flag is disabled. Skipping access rights update.")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and run the image processing pipeline.
    """
    log.info("Starting image pipeline execution.")
    pipeline = ImagePipeline(cfg)
    pipeline.run_pipeline()
    log.info("Image pipeline execution completed.")


if __name__ == "__main__":
    main()
