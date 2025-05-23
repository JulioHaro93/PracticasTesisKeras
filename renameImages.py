#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:46:17 2025

@author: julio
"""

import os

carpeta = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/homogenas"  # Cambia esto al nombre de tu carpeta
extensiones = ('.jpg', '.png', '.jpeg')  # Tipos de archivos a renombrar

archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(extensiones)])
print("Hola mundo, esto sí está arrancando")
dato = ""
for i, nombre_original in enumerate(archivos):
    extension = os.path.splitext(nombre_original)[1]  # .jpg, .png, etc.
    #nuevo_nombre = f"{i:06d}{extension}"  # frame_julio_00001.jpg, etc.
    nuevo_nombre = f"{i}{extension}" # 1,2,3,4,5,..,10,11,100,1223,etc
    ruta_original = os.path.join(carpeta, nombre_original)
    ruta_nueva = os.path.join(carpeta, nuevo_nombre)

    os.rename(ruta_original, ruta_nueva)

print("Renombrado completo.")

carpeta = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/homogenea"  # Cambia esto al nombre de tu carpeta
extensiones = ('.jpg', '.png', '.jpeg')  # Tipos de archivos a renombrar

archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(extensiones)])
print("Hola mundo, esto sí está arrancando")
dato = ""
for i, nombre_original in enumerate(archivos):
    extension = os.path.splitext(nombre_original)[1]  # .jpg, .png, etc.
    #nuevo_nombre = f"{i:06d}{extension}"  # frame_julio_00001.jpg, etc.
    nuevo_nombre = f"{i}{extension}" # 1,2,3,4,5,..,10,11,100,1223,etc
    ruta_original = os.path.join(carpeta, nombre_original)
    ruta_nueva = os.path.join(carpeta, nuevo_nombre)

    os.rename(ruta_original, ruta_nueva)

print("Renombrado completo.")

carpeta = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptohyphidae/homogenea"  # Cambia esto al nombre de tu carpeta
extensiones = ('.jpg', '.png', '.jpeg')  # Tipos de archivos a renombrar

archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(extensiones)])
print("Hola mundo, esto sí está arrancando")
dato = ""
for i, nombre_original in enumerate(archivos):
    extension = os.path.splitext(nombre_original)[1]  # .jpg, .png, etc.
    #nuevo_nombre = f"{i:06d}{extension}"  # frame_julio_00001.jpg, etc.
    nuevo_nombre = f"{i}{extension}" # 1,2,3,4,5,..,10,11,100,1223,etc
    ruta_original = os.path.join(carpeta, nombre_original)
    ruta_nueva = os.path.join(carpeta, nuevo_nombre)

    os.rename(ruta_original, ruta_nueva)

print("Renombrado completo.")