#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:19:33 2025

@author: julio
"""

import os
import gc

directorio = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/preprocesadas_numpy/Leptohyphidae"

# Parte 1: Intervalos a eliminar
intervalos = [
    (17, 19), (76,79), (265,300),(372,465),(507,547), 
    (671,684),(750,770),(2273,2309), (3210,2402), (2411,2474),
    (4528,4700), (4781,4884), (7019,7057), (7311,7379), (7503,7632),
    (8152, 8159), (8199,8200), (8285, 8326)
]
"""

Puedes encontrar el Canido bonito en la 16861

"""
# Parte 2: Casos aislados a eliminar
aislados = [
   3, 7,8162, 8192, 8195, 8207,8225, 8231, 8234,8236,
]

# 🧹 Eliminar por intervalos
for inicio, fin in intervalos:
    for i in range(inicio, fin + 1):
        archivo = os.path.join(directorio, f"Leptohyphidae_{i}.npy")
        if os.path.exists(archivo):
            os.remove(archivo)
    gc.collect()  # liberar memoria

# 🧹 Eliminar casos aislados
for i in aislados:
    archivo = os.path.join(directorio, f"Leptohyphidae_{i}.npy")
    if os.path.exists(archivo):
        os.remove(archivo)

print("Eliminación completada.")