#Librerias
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt

from paralelizar import worker, nivelacion_cargas
from loader import cargar_imagenes
from modelo_rf import entrenar_random_forest
from metricas_paralelas import calcular_metricas_paralelas
from configuracion import RUTA_DATASET, PROCESOS_LISTA, TIPO
from configuracion import USAR_RANDOM_FOREST, ENTRENAR_MODELO, MODELO_RF, MAX_IMAGENES_ENTRENAMIENTO, MAX_MUESTRAS_POR_CLASE
from visualizacion import graficar_rendimiento, graficar_metricas, mostrar_ejemplos


if __name__ == "__main__":

    lista_imagenes = cargar_imagenes(RUTA_DATASET, TIPO)

    print("Imágenes encontradas:", len(lista_imagenes))

    if len(lista_imagenes) == 0:
        print("No se encontraron imágenes.")
        print("Revisa RUTA_DATASET y TIPO en configuracion.py")
        exit()


    # ---------------- ENTRENAMIENTO DEL MODELO ----------------

    if USAR_RANDOM_FOREST and ENTRENAR_MODELO:
        print("\nEntrenando modelo Random Forest...")
        entrenar_random_forest(
            lista_imagenes,
            modelo_salida=MODELO_RF,
            max_imagenes=MAX_IMAGENES_ENTRENAMIENTO
        )

    tiempos = []
    metricas_globales = []

    for n_procesos in PROCESOS_LISTA:
        print(f"\nEjecutando con {n_procesos} procesos:")
        inicio = time.perf_counter()
        cargas = nivelacion_cargas(lista_imagenes, n_procesos)
        queue = mp.Queue()
        procesos = []

        for i in range(len(cargas)):
            p = mp.Process(
                target=worker,
                args=(cargas[i], i, queue, USAR_RANDOM_FOREST, MODELO_RF)
            )
            procesos.append(p)
            p.start()

        resultados = []

        for _ in procesos:
            resultados.extend(queue.get())

        for p in procesos:
            p.join()
        
        fin = time.perf_counter()
        tiempo = fin - inicio
        tiempos.append(tiempo)
        print(f"Tiempo con {n_procesos} procesos: {tiempo:.2f} segundos")
        metricas_globales.append(resultados)

    # ---------------- SPEEDUP Y EFICIENCIA ----------------

    resultados_paralelos = calcular_metricas_paralelas(PROCESOS_LISTA, tiempos)
    print("\nResultados de Rendimiento:")
    speedups = []
    eficiencias = []

    for r in resultados_paralelos:
        speedups.append(r["speedup"])
        eficiencias.append(r["eficiencia"])

        print(f"""
        Procesos:        {r["procesos"]}
        Tiempo:          {r["tiempo"]:.2f} s
        Speedup:         {r["speedup"]:.2f}
        Eficiencia:      {r["eficiencia"]:.2f}
        Amdahl:          {r["amdahl"]:.2f}
        Gustafson:       {r["gustafson"]:.2f}
        Karp-Flatt:      {r["karp_flatt"]:.4f}
        """)

    # Promedio de las metricas

    metricas_array = np.array(metricas_globales[-1])
    acc = metricas_array[:, 0].mean()
    prec = metricas_array[:, 1].mean()
    rec = metricas_array[:, 2].mean()
    f1 = metricas_array[:, 3].mean()
    iou = metricas_array[:, 4].mean()

    print("\nMétricas finales:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"IoU:       {iou:.3f}")

    # VISUALIZACIÓN
    graficar_rendimiento(PROCESOS_LISTA, tiempos, speedups, eficiencias)
    graficar_metricas(acc, prec, rec, f1, iou)
    mostrar_ejemplos(lista_imagenes, cantidad=5, usar_rf=USAR_RANDOM_FOREST, modelo_rf=MODELO_RF)