#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:31:02 2025

@author: julio
"""


import os
import gc

directorio = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/preprocesadas_numpy/Heptageniidae"

# Parte 1: Intervalos a eliminar
intervalos = [
    (11, 14), (18,21), (55,76), (92,93),(97,98),(105,106), (1259,1297),
    (1611,1660), (2133,2195), (2321,2356),(2434,2452),(2543,2778),(2848,2853),
    (2958,2985), (3234,3271), (3282,3304), (3320,3694),(3813,3851),(3862,4074),
    (5122,2309),(6279,2391)
]
"""

Puedes encontrar el Canido bonito en la 16861

"""
# Parte 2: Casos aislados a eliminar
aislados = [
    0, 1,38,83,90,100,103
]

# 🧹 Eliminar por intervalos
for inicio, fin in intervalos:
    for i in range(inicio, fin + 1):
        archivo = os.path.join(directorio, f"Heptageniidae_{i}.npy")
        if os.path.exists(archivo):
            os.remove(archivo)
    gc.collect()  # liberar memoria

# 🧹 Eliminar casos aislados
for i in aislados:
    archivo = os.path.join(directorio, f"Heptageniidae_{i}.npy")
    if os.path.exists(archivo):
        os.remove(archivo)

print("Eliminación completada.")
