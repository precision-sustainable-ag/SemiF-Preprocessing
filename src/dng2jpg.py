import subprocess
import time
from pathlib import Path
from omegaconf import DictConfig
import hydra
# import GPUtil


class ImagePipeline:
    def __init__(self, cfg):
        # Load configuration from YAML file

        self.cfg = cfg
        self.batch_id = cfg.batch_id
        
        self.im_input_dir = Path(self.cfg.paths.data_dir, "semifield-upload") / self.batch_id / "dngs"
        self.im_output_dir = Path(self.cfg.paths.data_dir) / "semifield-upload" / self.batch_id / "images"
        self.im_output_dir.mkdir(parents=True, exist_ok=True)

        self.extension = ".dng"

        self.develop_images_flag = self.cfg.dng2jpg.develop_images
        self.fix_access_rights_flag = self.cfg.dng2jpg.fix_access_rights

    def develop_images(self):
        # Implementation of the develop_images method
        num_of_raw_images = len(list(self.im_input_dir.glob(f"*{self.extension}")))
        
        if num_of_raw_images == 0:
            print(f"No RAW images found in the input directory ({self.im_input_dir})")
            return            
        
        else:
            exe_command = f"./RawTherapee_5.8.AppImage --cli \
                -Y \
                -O {self.im_output_dir} \
                -j99 \
                -c {self.im_input_dir}"
            
            exe_command2 = 'bash -c "OMP_NUM_THREADS=90; ' + exe_command + '"'
            
            try:
                # Run the rawtherapee command
                subprocess.run(exe_command2, shell=True, check=True)
                print("")
            except Exception as e:
                raise e
            
    def update_access_rights(self):
        # Implementation of the update_access_rights method
        subprocess.call(['chmod', '-R', '777', self.im_output_dir])
        time.sleep(2)
        
    def run_pipeline(self):
        # Execute the pipeline based on the configuration
        if self.develop_images_flag:
            self.develop_images()
        if self.fix_access_rights_flag:
            self.update_access_rights()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize and run the pipeline
    pipeline = ImagePipeline(cfg)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
    