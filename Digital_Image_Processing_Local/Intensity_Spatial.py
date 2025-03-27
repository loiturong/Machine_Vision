import numpy as np
import cv2 as cv

__all__ = ["negative_transform", "negative_colors_transform",
           "log_transform", "gamma_transform"]

# region Constant
L = 255

# endregion

def negative_transform(image: np.ndarray) -> np.ndarray:
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    image_np = L - 1 - image_np

    return image_np

def negative_colors_transform(image: np.ndarray) -> np.ndarray:
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 3), "Must be an RGB Image."

    image_np = L - 1 - image_np

    return image_np

def log_transform(image: np.ndarray,
                  constant_factor: int = L) -> np.ndarray:
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    constant = (constant_factor - 1.0) / np.log(1.0 * constant_factor)
    image_np = constant * np.log(1.0 + image_np)

    # convert back to 8-bit integer to show in scale 0-255
    return np.array(image_np, dtype=np.uint8)

def gamma_transform(image: np.ndarray,
                    gamma: np.float16 = 2.5) -> np.ndarray:
    """Power-law transformations or Gamma Transformations"""
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    # compute constant value with respect to gamma value (and L)
    constant = np.power(L - 1.0 , 1.0 - gamma)

    image_np = constant * np.power(image_np, gamma)

    return np.array(image_np, dtype=np.uint8)