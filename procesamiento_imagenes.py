import rasterio
import numpy as np
import cv2

def procesarImagen(tif_path):
    #Hacemos lectura de la imagen con la libreria raterio
    with rasterio.open(tif_path) as src:
        img = src.read()

    #Debido a que las imagenes satelitales viene en bandas, convertimos a RGB y normalizamos los valores 
    rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    #Transformamos la imagen a formato HSV para poder hacer segmentacion de diferentes elementos
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

    #Aplicamos diferentes mascaras para filtrar: NUbes, áreas verdes y entornos urbanos (baja satiración: caracteristicas similares)
    mask_verde = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    mask_nubes = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    mask_baja_sat = cv2.inRange(hsv, (0, 0, 50), (180, 80, 255))

    #Construinos las mascaras compuestas, invertimos la mascara o se conservan las areas utiles
    mask_valid = cv2.bitwise_and(mask_baja_sat, cv2.bitwise_not(mask_verde))
    mask_valid = cv2.bitwise_and(mask_valid, cv2.bitwise_not(mask_nubes))

    #Buscamos separar dos entornos comunes en el dataset, imagenes que contienen mucha vegetación vs areas totalmente urbanas)
    porcentaje_verde = np.sum(mask_verde > 0) / mask_verde.size
    densidad = np.sum(mask_valid > 0) / mask_valid.size

    if porcentaje_verde > 0.4 and densidad < 0.3:
        tipo = "Mayor Área de Vegetación"
        area_min, area_max = 80, 5000
        ratio_min, ratio_max = 0.3, 3
    else:
        tipo = "Mayor Área Urbana"
        area_min, area_max = 10, 50000
        ratio_min, ratio_max = 0.1, 10

    #Creación del kernel
    kernel = np.ones((3,3), np.uint8)

    #Aqui tenemos que usar prcesamiento extras para las zonas urbanas 
    if tipo == "Mayor Área Urbana":
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, 80, 180)
        edges = cv2.dilate(edges, kernel, 1)
        edges = cv2.erode(edges, kernel, 1)

        mask_valid = cv2.bitwise_or(mask_valid, edges)

        edge_density = cv2.blur(edges.astype(np.float32), (15,15))
        mask_valid[edge_density < 5] = 0

    #Hacemos limpieza de las máscaras para eliminar ruido excesivo
    mask_clean = cv2.medianBlur(mask_valid, 5)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, 2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, 1)

    #Devolvemos los atributos de los componenetes y los validamos
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, 8)

    #La máscara final ya con los elementos válidos
    mask_final = np.zeros_like(mask_clean)

    #Validamos dichos componentes
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ratio = w / (h + 1e-8)

        if area_min < area < area_max and ratio_min < ratio < ratio_max:
            mask_final[labels == i] = 255

    #Retornamos la máscara final
    return mask_final    


    