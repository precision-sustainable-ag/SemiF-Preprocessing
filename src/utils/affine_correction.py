import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from preprocess import Preprocessor
log = logging.getLogger(__name__)


def affine_color_correction(rgb_img, source_matrix, target_matrix):
    """Affine color correction of RGB image.

    Correct the color of the input image based on the target color matrix using an affine transformation
    in the RGB space. The vector containing the regression coefficients is calculated as the one that minimizes the
    Euclidean distance between the transformed source color values and the target color values.

    Inputs:
    rgb_img         = an RGB image with color chips visualized
    source_matrix   = array of RGB color values (intensity in the range [0-1]) from
                      the image to be corrected where each row is one
                      color reference and the columns are organized as index,R,G,B
    target_matrix   = array of target RGB color values (intensity in the range [0-1])
                      where each row is one color reference and the columns are
                      organized as index,R,G,B

    Outputs:
    corrected_img   = color corrected image


    :param rgb_img: numpy.ndarray
    :return source_matrix: numpy.ndarray
    :return target_matrix: numpy.ndarray
    :return corrected_img: numpy.ndarray
    """
    # matrices must have the same number of color references
    if source_matrix.shape != target_matrix.shape:
        raise ValueError('Source and target matrices must have the same number of color references')

    h, w, c = rgb_img.shape

    # number of references
    n = source_matrix.shape[0]

    # the column zero (index) of the matrices is not used in this model
    # augment matrix of source values with a column of 1s for the constant part of
    # the affine transformation
    S = np.concatenate((source_matrix[:, 1:].copy(), np.ones((n, 1))), axis=1)

    # make vectors of taget values for each color
    T = target_matrix[:, 1:].copy()
    tr = T[:, 0]
    tg = T[:, 1]
    tb = T[:, 2]

    # calculate regression vector for each color as the pseudoinverse of the source
    # values matrix multiplied by each color target vector
    ar = np.matmul(np.linalg.pinv(S), tr)
    ag = np.matmul(np.linalg.pinv(S), tg)
    ab = np.matmul(np.linalg.pinv(S), tb)

    # img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    img_rgb = rgb_img
    # reshape image as a 2D array where the rows are pixels and the colums are color channels
    # and augment the channels with a column of 1s for the affine transformation
    img_pix = np.concatenate((img_rgb.reshape(h*w, c).astype(np.float64)/255, np.ones((h*w, 1))), axis=1)

    # calculate the corrected colors, eliminate values outside the range [0-1] and
    # convert to [0-255] unit8
    img_r_cc = (255*np.clip(np.matmul(img_pix, ar), 0, 1)).astype(np.uint8)
    img_g_cc = (255*np.clip(np.matmul(img_pix, ag), 0, 1)).astype(np.uint8)
    img_b_cc = (255*np.clip(np.matmul(img_pix, ab), 0, 1)).astype(np.uint8)

    # reconstruct the RGB (actually BGR for openCV) image
    corrected_img = np.stack((img_b_cc, img_g_cc, img_r_cc), axis=1).reshape(h, w, c)
    return corrected_img


def get_matrices(cfg: DictConfig) -> np.ndarray:
    """
    Compute and save the transformation matrix based on configuration values.

    This function reads the reference and measured colors from the configuration,
    computes the transformation matrix, saves it to disk, and returns it.

    Parameters:
        cfg (DictConfig): Hydra configuration with color checker information.

    Returns:
        np.ndarray: The computed 9x9 transformation matrix.
    """
    reference_colors = np.array([
        [ref['number']] + [x / 255.0 for x in ref["rgb_target"]]
        for ref in cfg.ccm
    ])
    
    measured_colors = np.array([
        [meas['number']] + [x / 255.0 for x in meas["rgb_sample"]]
        for meas in cfg.ccm
    ])
    
    return reference_colors, measured_colors


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Compute or save the transformation matrix.
    reference_colors, measured_colors = get_matrices(cfg)
    img_path = "data/longterm_images2/semifield-upload/NC_2025-02-21/NC_1740166530.RAW"
    img = Preprocessor.load_raw_image(img_path, cfg)
    demosaiced = Preprocessor.demosaic_image(img)
    
    demosaiced = (demosaiced * 255.0).astype(np.uint8)
    
    corrected_img = affine_color_correction(demosaiced, measured_colors,reference_colors)

    corrected_img = (corrected_img / 255.0).astype(np.float32)
    
    gamma_corrected = Preprocessor.apply_gamma_correction(corrected_img, gamma=0.9)

    corrected_img = (gamma_corrected * 255.0).astype(np.uint8)
    
    corrected_img_path = Path(img_path).with_suffix(".corrected.png")
    
    
    cv2.imwrite(str(corrected_img_path), corrected_img)

if __name__ == "__main__":
    main()
