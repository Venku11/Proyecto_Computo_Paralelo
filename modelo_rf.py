import numpy as np
import cv2
import rasterio
import joblib

from sklearn.ensemble import RandomForestClassifier
from ground_truth import cargar_ground_truth


def leer_rgb(tif_path):

    with rasterio.open(tif_path) as src:
        img = src.read()

    rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8

