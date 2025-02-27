import numpy as np
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def get_transformation_components(target_matrix: np.ndarray, source_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate components required for generating a transformation matrix.

    Depending on the shape of the input matrices, the first column is assumed to be an identifier.

    Parameters:
        target_matrix (np.ndarray): Reference color matrix.
        source_matrix (np.ndarray): Measured color matrix.

    Returns:
        tuple: (matrix_a, matrix_m, matrix_b) to be used for calculating the transformation.
    """
    _, t_r, t_g, t_b = np.split(target_matrix, 4, axis=1)
    _, s_r, s_g, s_b = np.split(source_matrix, 4, axis=1)

    # Build the design matrix using powers of the source colors.
    matrix_a = np.hstack([s_r, s_g, s_b, s_r**2, s_g**2, s_b**2, s_r**3, s_g**3, s_b**3])
    # Compute the pseudo-inverse (least-squares solution) of the design matrix.
    matrix_m = np.linalg.solve(matrix_a.T @ matrix_a, matrix_a.T)
    # Build the target matrix with powers.
    # matrix_b = np.hstack([t_r, t_g, t_b, t_r**2, t_g**2, t_b**2, t_r**3, t_g**3, t_b**3])
    matrix_b = np.hstack([t_r, t_r**2, t_r**3, t_g, t_g**2, t_g**3, t_b, t_b**2, t_b**3])
    return matrix_a, matrix_m, matrix_b

def calc_transformation_matrix(matrix_m: np.ndarray, matrix_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate the transformation matrix and its deviance."""
    t_r, t_r2, t_r3, t_g, t_g2, t_g3, t_b, t_b2, t_b3 = np.split(matrix_b, 9, 1)

    # multiply each 22x1 matrix from target color space by matrix_m
    red = np.matmul(matrix_m, t_r)
    green = np.matmul(matrix_m, t_g)
    blue = np.matmul(matrix_m, t_b)

    red2 = np.matmul(matrix_m, t_r2)
    green2 = np.matmul(matrix_m, t_g2)
    blue2 = np.matmul(matrix_m, t_b2)

    red3 = np.matmul(matrix_m, t_r3)
    green3 = np.matmul(matrix_m, t_g3)
    blue3 = np.matmul(matrix_m, t_b3)

    # concatenate each product column into 9X9 transformation matrix
    transformation_matrix = np.concatenate((red, green, blue, red2, green2, blue2, red3, green3, blue3), 1)

    # find determinant of transformation matrix
    t_det = np.linalg.det(transformation_matrix)

    return 1-t_det, transformation_matrix

def compute_transformation_matrix(cfg: DictConfig) -> np.ndarray:
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
    
    _, matrix_m, matrix_b = get_transformation_components(reference_colors, measured_colors)
    deviance, transformation_matrix = calc_transformation_matrix(matrix_m, matrix_b)
    log.info(f"Transformation matrix deviance: {deviance:.6f}")
    return transformation_matrix

def save_matrix(cfg: DictConfig, matrix: np.ndarray) -> None:
    """Save a matrix to a file."""
    output_dir = Path(cfg.paths.data_dir) / "semifield-utils" / "image_development" / "color_matrix"
    ccm_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.ccm
    matrix_file = Path(output_dir, f"{ccm_name}.npz")
    np.savez(matrix_file, matrix=matrix)
    log.info(f"Transformation matrix saved to {matrix_file}")

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Compute or save the transformation matrix.
    transformation_matrix = compute_transformation_matrix(cfg)
    save_matrix(cfg, transformation_matrix)

if __name__ == "__main__":
    main()
