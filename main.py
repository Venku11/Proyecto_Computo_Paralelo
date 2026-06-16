#Librerias
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt

from paralelizar import worker, nivelacion_cargas
from loader import cargar_imagenes
from configuracion import RUTA_DATASET, PROCESOS_LISTA, TIPO
from configuracion import USAR_RANDOM_FOREST, ENTRENAR_MODELO, MODELO_RF, MAX_IMAGENES_ENTRENAMIENTO
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
    speedups = []
    eficiencias = []
    amdahl_lista = []
    gustafson_lista = []
    karp_flatt_lista = []
    metricas_globales = []

    for n_procesos in PROCESOS_LISTA:
        print(f"\nEjecutando con {n_procesos} procesos:")
        inicio = time.time()
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
        
        fin = time.time()
        tiempo = fin - inicio
        tiempos.append(tiempo)
        print(f"Tiempo con {n_procesos} procesos: {tiempo:.2f} segundos")
        metricas_globales.append(resultados)

    # ---------------- SPEEDUP Y EFICIENCIA ----------------

    for i in range(len(PROCESOS_LISTA)):
        S = tiempos[0] / tiempos[i]
        E = S / PROCESOS_LISTA[i]
        speedups.append(S)
        eficiencias.append(E)

    print("\nResultados de Rendimiento:")

    for i in range(len(PROCESOS_LISTA)):

        print(f"""
        Procesos:   {PROCESOS_LISTA[i]}
        Tiempo:     {tiempos[i]:.2f} s
        Speedup:    {speedups[i]:.2f}
        Eficiencia: {eficiencias[i]:.2f}
        """)

    # Promedio de las metricas

    metricas_array = np.array(metricas_globales[-1])
    acc = metricas_array[:, 0].mean()
    prec = metricas_array[:, 1].mean()
    rec = metricas_array[:, 2].mean()
    f1 = metricas_array[:, 3].mean()

    print("\nMétricas finales:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")

    # VISUALIZACIÓN
    graficar_rendimiento(PROCESOS_LISTA, tiempos, speedups, eficiencias)
    graficar_metricas(acc, prec, rec, f1)
    mostrar_ejemplos(lista_imagenes, cantidad=5)
    