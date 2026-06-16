from procesamiento_imagenes import procesarImagen
from modelo_rf import predecir_random_forest
from metricas import calcular_metricas
from ground_truth import cargar_ground_truth

import time


def worker(lista_imagenes, id_proceso, queue, usar_rf=False, modelo_rf="modelo_rf.joblib"):

    inicio = time.time()

    resultados = []

    try:

        for ruta in lista_imagenes:

            if usar_rf:
                mask_pred = predecir_random_forest(ruta, modelo_rf)
            else:
                mask_pred = procesarImagen(ruta)

            ruta_gt = ruta.replace("images", "labels").replace(".tif", ".json")

            try:
                gt = cargar_ground_truth(ruta_gt, mask_pred.shape)

                acc, prec, rec, f1, iou = calcular_metricas(mask_pred, gt)

            except Exception as e:

                print(f"Error en {ruta_gt}: {e}")

                acc = prec = rec = f1 = iou = 0

            resultados.append((acc, prec, rec, f1, iou))

    except Exception as e:

        print(f"Error crítico en proceso {id_proceso}: {e}")

    finally:

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