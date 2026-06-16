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

def extraer_features(tif_path):

    rgb = leer_rgb(tif_path)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 180)

    h, w = gray.shape

    x_coords = np.tile(np.arange(w), (h, 1)) / w
    y_coords = np.tile(np.arange(h).reshape(-1, 1), (1, w)) / h

    features = np.dstack([
        rgb[:, :, 0],
        rgb[:, :, 1],
        rgb[:, :, 2],
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2],
        gray,
        edges,
        x_coords * 255,
        y_coords * 255
    ])