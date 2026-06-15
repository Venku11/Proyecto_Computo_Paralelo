from procesamiento_imagenes import procesarImagen
from metricas import calcular_metricas
from ground_truth import cargar_ground_truth
import numpy as np
import time
def worker(lista_imagenes, id_proceso, queue):

    inicio = time.time()

    resultados = []

    total_imagenes = len(lista_imagenes)

    try:

        for ruta in lista_imagenes:

            #gemerar mascara
            mask_pred = procesarImagen(ruta)

            # Ajustamos ruta del JSON GT
            ruta_gt = ruta.replace("images", "labels").replace(".tif", ".json")

            try:

                # Cargar máscara real desde JSON
                gt = cargar_ground_truth(ruta_gt, mask_pred.shape)

                # Calcular métricas
                acc, prec, rec, f1 = calcular_metricas(mask_pred, gt)

            except Exception as e:

                print(f"Error en {ruta_gt}: {e}")

                acc = prec = rec = f1 = 0

            # guarda las metricas en la lista local del proceso
            resultados.append((acc, prec, rec, f1))

    except Exception as e:

        print(f"Error crítico en proceso {id_proceso}: {e}")

    finally:

        # envia SIEMPRE los datos
        queue.put(resultados)

        print(f"Proceso {id_proceso} terminado")



# la poderosa nivelacion de cargas

def nivelacion_cargas(D, n_p):

    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)

    out = []
    temp = []

    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []

    for i in range(len(n_D)):
        out[i].append(n_D[i])

    return out