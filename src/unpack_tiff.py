import numpy as np
import cv2
from pathlib import Path
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_bilinear

class TIFFProcessor:
    def __init__(self, file_path, im_height, im_width):
        self.file_path = Path(file_path)
        self.im_height = im_height
        self.im_width = im_width
        self.packed_img = None
        self.unpacked_img = None
        self.demosaiced_img = None

    def load_image(self):
        """Loads the packed TIFF image."""
        self.packed_img = cv2.imread(str(self.file_path))
        if self.packed_img is None:
            raise FileNotFoundError(f"Could not load image from {self.file_path}")
        print(f"Loaded image with shape: {self.packed_img.shape}")

    def unpack_image(self):
        """Unpacks a 12-bit packed image to 16-bit format."""
        if self.packed_img is None:
            raise ValueError("Image not loaded. Please load the image first.")

        layer1 = self.packed_img[:, :, 0]
        nparray = layer1.reshape((int(self.im_height * self.im_width * 1.5))).astype(np.uint16)
        nparray_16bit = np.empty(self.im_width * self.im_height, dtype=np.uint16)

        # Unpack 12-bit packed data into 16-bit
        nparray_16bit[::2] = (nparray[::3] << 4) | (nparray[1::3] & 0b00001111)
        nparray_16bit[1::2] = nparray[2::3] | ((nparray[1::3] & 0b11110000) << 4)

        self.unpacked_img = nparray_16bit.reshape((self.im_height, self.im_width)) << 4
        print(f"Unpacked image to shape: {self.unpacked_img.shape}, dtype: {self.unpacked_img.dtype}")

    def demosaic_image(self):
        """Applies demosaicing to the unpacked 16-bit image."""
        if self.unpacked_img is None:
            raise ValueError("Image not unpacked. Please unpack the image first.")
        
        image_data = (self.unpacked_img).astype(np.float32) / 65535.0
        # colour_image = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
        colour_image = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
        # Clamp values to the [0, 1] range
        colour_image = np.clip(colour_image, 0, 1)
        
        gamma = 2.2
        colour_image = np.power(colour_image, 1.0 / gamma)
        # Convert to BGR format for OpenCV compatibility
        self.demosaiced_img = cv2.cvtColor(colour_image.astype(np.float32), cv2.COLOR_RGB2BGR)
        print(f"Demosaiced image shape: {self.demosaiced_img.shape}, dtype: {self.demosaiced_img.dtype}")

    def save_image(self, output_format="16bit"):
        """Saves the demosaiced image in either 8-bit or 16-bit format."""
        if self.demosaiced_img is None:
            raise ValueError("Image not demosaiced. Please demosaic the image first.")
        
        if output_format == "8bit":
            # Scale to 8-bit and save
            save_img = (self.demosaiced_img * 255).astype(np.uint8)
            output_file = self.file_path.with_name(f"{self.file_path.stem}_unpacked_8bit.png")
            cv2.imwrite(str(output_file), save_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif output_format == "16bit":
            # Scale to 16-bit and save
            save_img = (self.demosaiced_img * 65535).astype(np.uint16)
            output_file = self.file_path.with_name(f"{self.file_path.stem}_unpacked_16bit.png")
            cv2.imwrite(str(output_file), save_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            raise ValueError("Invalid output format. Choose '8bit' or '16bit'.")
        
        print(f"Image saved as {output_format} at {output_file}")

# Example usage:
file_path = "data/results/NC_2024-11-12/NC_1731423628.tiff"
processor = TIFFProcessor(file_path=file_path, im_height=9528, im_width=13376)
processor.load_image()
processor.unpack_image()
processor.demosaic_image()
processor.save_image(output_format="8bit")  # Change to "8bit" to save as 8-bit
