import glob

def cargar_imagenes(ruta_base, tipo="pre"):

    # Define el patron de búsqueda segun el parametro 'tipo'
    if tipo == "pre":

        # Busca archivos que terminen en '_pre_disaster.tif'
        patron = "*_pre_disaster.tif"
    elif tipo == "post":

        # Busca archivos que terminen en '_post_disaster.tif'
        patron = "*_post_disaster.tif"
    else:

        # Si el tipo no es 'pre' ni 'post', busca cualquier archivo .tif
        patron = "*.tif"

    # Refresa una lista con las rutas completas de todas las imagenes encontradas
    return glob.glob(f"{ruta_base}/{patron}")