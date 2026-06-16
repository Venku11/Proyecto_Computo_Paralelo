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


    return features.astype(np.float32)


def crear_muestras_entrenamiento(tif_path, max_por_clase=3000):

    features = extraer_features(tif_path)

    ruta_gt = tif_path.replace("images", "labels").replace(".tif", ".json")
    gt = cargar_ground_truth(ruta_gt, features.shape[:2])

    X = features.reshape(-1, features.shape[2])
    y = (gt.reshape(-1) > 0).astype(np.uint8)

    indices_estructura = np.where(y == 1)[0]
    indices_fondo = np.where(y == 0)[0]

    if len(indices_estructura) == 0:
        return None, None

    n_estructura = min(max_por_clase, len(indices_estructura))
    n_fondo = min(max_por_clase, len(indices_fondo))

    indices_estructura = np.random.choice(indices_estructura, n_estructura, replace=False)
    indices_fondo = np.random.choice(indices_fondo, n_fondo, replace=False)

    indices = np.concatenate([indices_estructura, indices_fondo])
    np.random.shuffle(indices)

    return X[indices], y[indices]

def entrenar_random_forest(lista_imagenes, modelo_salida="modelo_rf.joblib", max_imagenes=20):

    X_total = []
    y_total = []

    imagenes_entrenamiento = lista_imagenes[:max_imagenes]

    for ruta in imagenes_entrenamiento:

        try:
            X, y = crear_muestras_entrenamiento(ruta)

            if X is not None:
                X_total.append(X)
                y_total.append(y)
                print("Imagen agregada al entrenamiento:", ruta)

        except Exception as e:
            print("Error entrenando con", ruta, ":", e)

    X_total = np.vstack(X_total)
    y_total = np.concatenate(y_total)

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    modelo.fit(X_total, y_total)

    joblib.dump(modelo, modelo_salida)

    print("Modelo Random Forest guardado en:", modelo_salida)

