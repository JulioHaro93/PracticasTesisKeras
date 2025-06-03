#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 19:00:57 2025

@author: julio
"""

import os
import gc

directorio = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/preprocesadas_numpy/Leptophlebiidae"

# Parte 1: Intervalos a eliminar
intervalos = [
    (6,9),(125,140), (213,240), (293,295), (300,314), (334,345),
    (347,382), (397,407), (413,428), (452,500), (501,527), (535,552),
    (560,567),(733,765), (1261,1293), (1472,1633), (1665,1709), (1750,1799),
    (2094,2130), (2183,2222), (2347,2443), (3156,3217), (3624,3697), (4542,4565),
    (4598,4612), (4719,4750)
]
"""

Puedes encontrar el Canido bonito en la 16861

"""
# Parte 2: Casos aislados a eliminar
aislados = [6]

# 🧹 Eliminar por intervalos
for inicio, fin in intervalos:
    for i in range(inicio, fin + 1):
        archivo = os.path.join(directorio, f"Leptoplebiidae_{i}.npy")
        if os.path.exists(archivo):
            os.remove(archivo)
    gc.collect()  # liberar memoria

# 🧹 Eliminar casos aislados
for i in aislados:
    archivo = os.path.join(directorio, f"Leptophlebiidae_{i}.npy")
    if os.path.exists(archivo):
        os.remove(archivo)

print("Eliminación completada.")