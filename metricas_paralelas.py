def calcular_metricas_paralelas(procesos, tiempos):

    resultados = []
    tiempo_base = tiempos[0]

    for i in range(len(procesos)):

        p = procesos[i]
        tiempo_p = tiempos[i]

        speedup = tiempo_base / (tiempo_p + 1e-8)
        eficiencia = speedup / p

        if p == 1:
            karp_flatt = 0
            amdahl = 1
            gustafson = 1
        else:
            karp_flatt = ((1 / speedup) - (1 / p)) / (1 - (1 / p))
            amdahl = 1 / (karp_flatt + ((1 - karp_flatt) / p))
            gustafson = p - karp_flatt * (p - 1)

        resultados.append({
            "procesos": p,
            "tiempo": tiempo_p,
            "speedup": speedup,
            "eficiencia": eficiencia,
            "amdahl": amdahl,
            "gustafson": gustafson,
            "karp_flatt": karp_flatt
        })

    return resultados