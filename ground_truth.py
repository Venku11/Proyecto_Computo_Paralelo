import json
import numpy as np
import cv2

from shapely import wkt


def cargar_ground_truth(json_path, shape_img):

    # Creamos Máscara Vacia
    mask = np.zeros(shape_img[:2], dtype=np.uint8)

    # Abrimos el JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Por posición de la imagen obtenemos los edificios
    features = data["features"]["xy"]

    for feature in features:

        #Obtener la seccion del edificio

        polygon = wkt.loads(feature["wkt"])

        #Obteenmos las coordenadas
        coords = np.array(polygon.exterior.coords, dtype=np.int32)

        #Dibujamos el edificio
        cv2.fillPoly(mask, [coords], 255)

    return mask