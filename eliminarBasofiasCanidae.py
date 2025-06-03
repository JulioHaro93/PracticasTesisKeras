# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import os
import gc

directorio = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/preprocesadas_numpy/Canidae"

# Parte 1: Intervalos a eliminar
intervalos = [
    (0, 1), (9, 11), (60, 207), (210, 250), (271, 394),
    (401, 439), (455, 509), (636, 759), (925, 971), (1247, 1287),
    (2001, 2010), (2012, 2021), (2023, 2032), (2034, 2043), (2045, 2086),
    (3640, 3682), (3684, 3908), (4205,4220), (4238,4282), (5677, 5744),
    (5796,5805), (5807,5812), (6181, 6191),(6744,6810), (7765,7806),
    (8093,8096),(8723,8786),(8788,8956),(9021,9224), (9840,9891), (10381,10441),
    (10672,10716), (10784,10862), (11224,11311), (11425,11533), (12598,12836), 
    (13278,13603), (13991,14294), (14988,15111), (15438,15825), (16433,16485), (16488,16519),
    ( 16650,16771),(17049 ,17234), (17236,17269),(17580,18136), (19063,19143), (19242,19470),
    (19527,19548), (19524,19736), (19800,19970),(20012,20100),(200545,20609), (20996,21036),
    (21180,21261), (21550,21558), (23668,23845), (24006,24107), (24883,24929), (25175,25189),
    (25254,25407), (26651,26740), (27121,27178), (27240,24300), (27359,27570),  (27594,27638),
    (27841,27923), (28548,28599), (28623,28662),(28751,28757), (28845,29068),(307026,30799),
    (31076,31105), (31761,31974),(33882,33928), (35345,35423), (35522,35557), (35675,35688),
    (36506,36509), (36511,36519)
]
"""

Puedes encontrar el Canido bonito en la 16861

"""
# Parte 2: Casos aislados a eliminar
aislados = [
    6, 9, 14, 55, 2368, 3639, 5762, 5773, 5784, 5795, 8194, 9227,9238, 9249, 27593,
]

# 🧹 Eliminar por intervalos
for inicio, fin in intervalos:
    for i in range(inicio, fin + 1):
        archivo = os.path.join(directorio, f"Canidae_{i}.npy")
        if os.path.exists(archivo):
            os.remove(archivo)
    gc.collect()  # liberar memoria

# 🧹 Eliminar casos aislados
for i in aislados:
    archivo = os.path.join(directorio, f"Canidae_{i}.npy")
    if os.path.exists(archivo):
        os.remove(archivo)

print("Eliminación completada.")
