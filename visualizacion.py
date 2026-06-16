# Librerias
import matplotlib.pyplot as plt
import numpy as np
import random
import rasterio

from procesamiento_imagenes import procesarImagen
from ground_truth import cargar_ground_truth


#Seccion de Graficas

def graficar_rendimiento(procesos, tiempos, speedups, eficiencias):

    #Tiempo
    plt.figure(figsize=(8,5))
    plt.plot(
        procesos,
        tiempos,
        marker='o',
        linewidth=2,
        color='royalblue'
    )
    plt.title("Tiempo de Ejecución vs Número de Procesos", fontsize=14)
    plt.xlabel("Número de Procesos", fontsize=12)
    plt.ylabel("Tiempo de Ejecución (s)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # SpeedUp
    plt.figure(figsize=(8,5))
    plt.plot(
        procesos,
        speedups,
        marker='o',
        linewidth=2,
        color='seagreen',
        label="Speedup Real"
    )
    plt.plot(
        procesos,
        procesos,
        linestyle='--',
        linewidth=2,
        color='darkorange',
        label="Speedup Ideal"
    )
    plt.title("Speedup del Sistema Paralelo", fontsize=14)
    plt.xlabel("Número de Procesos", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    #Eificiencia
    plt.figure(figsize=(8,5))
    plt.plot(
        procesos,
        eficiencias,
        marker='o',
        linewidth=2,
        color='crimson'
    )
    plt.title("Eficiencia Paralela", fontsize=14)
    plt.xlabel("Número de Procesos", fontsize=12)
    plt.ylabel("Eficiencia", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Metricas

def graficar_metricas(acc, prec, rec, f1, iou):
    metricas_nombres = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "IoU"
    ]
    metricas_valores = [acc, prec, rec, f1, iou]
    colores = [
        "royalblue",
        "seagreen",
        "darkorange",
        "crimson",
        "purple"
    ]
    plt.figure(figsize=(8,5))
    plt.bar(
        metricas_nombres,
        metricas_valores,
        color=colores
    )
    plt.title("Métricas de Evaluación", fontsize=14)
    plt.ylabel("Valor", fontsize=12)
    plt.ylim(0,1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()


# Unos ejemplos para reporte je

def mostrar_ejemplos(lista_imagenes, cantidad=5, usar_rf=False, modelo_rf="modelo_rf.pkl"):

    # Protección: si pides más imágenes de las que hay, ajusta la cantidad
    if len(lista_imagenes) < cantidad:
        cantidad = len(lista_imagenes)

    ejemplos = random.sample(lista_imagenes, cantidad)

    for i, ruta in enumerate(ejemplos):

        # Otra vez a procesar pero es con fines para el reporte x.x
        with rasterio.open(ruta) as src:
            img = src.read()

        rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(np.float32)
        # El + 1e-8 evita que el código explote por división entre cero
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Selección inteligente de modelo
        if usar_rf:
            from modelo_rf import predecir_random_forest
            pred = predecir_random_forest(ruta, modelo_rf)
        else:
            pred = procesarImagen(ruta)

        # Creamos la mascara
        ruta_gt = ruta.replace("images", "labels").replace(".tif", ".json")
        gt = cargar_ground_truth(ruta_gt, pred.shape)

        # Visualizar
        plt.figure(figsize=(16,5))
        plt.subplot(1,4,1)
        plt.imshow(rgb_uint8)
        plt.title("Imagen Original")
        plt.axis("off")
        
        plt.subplot(1,4,2)
        plt.imshow(pred, cmap="gray")
        plt.title("Predicción")
        plt.axis("off")
        
        plt.subplot(1,4,3)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(1,4,4)
        plt.imshow(rgb_uint8)
        plt.imshow(pred, cmap="jet", alpha=0.4)
        plt.title("Overlay")
        plt.axis("off")
        
        plt.tight_layout()
        
    plt.show()