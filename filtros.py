#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:20:04 2025

@author: julio
"""

import cv2
import numpy as np
import os

print("Inicia el bordeado de leptophlebiidae:")
ruta = '/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/homogenas'
extensiones = ('.jpg','.png','.jpeg')

archivos = sorted([f for f in os.listdir(ruta) if f.endswith(extensiones)])


for i, nombre_original in enumerate(archivos):

    stringBichito = '{}/{}.jpg'.format(ruta,str(i))
    bichito = cv2.imread(stringBichito)
    #cv2.imshow("Bichito",bichito)
    
    if bichito is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        continue

    bichito_grey = cv2.cvtColor(bichito, cv2.COLOR_BGR2GRAY)

    def angulizador(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy)
        return gx, gy, mag, ang

    gx, gy, mag, ang = angulizador(bichito_grey)

    # Normalizar magnitud para visualizarla
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = mag_vis.astype(np.uint8)

    #cv2.imshow('Magnitud del Gradiente', mag_vis)
    try:
        
        cv2.imwrite('/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/bordeadas/{}.jpg'.format(i), mag_vis)

    except:
        print("No se pudo guardar la nueva imagen")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print("Ha finalizado el filtrado de las imágenes con el filtro del borde para Leptophlebiidae")

print("##################################")
print("inicia el bordaedo para Heptageniidae")

ruta = '/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/homogenea'
extensiones = ('.jpg','.png','.jpeg')

archivos = sorted([f for f in os.listdir(ruta) if f.endswith(extensiones)])


for i, nombre_original in enumerate(archivos):

    stringBichito = '{}/{}.jpg'.format(ruta,str(i))
    bichito = cv2.imread(stringBichito)
    #cv2.imshow("Bichito",bichito)
    
    if bichito is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        continue

    bichito_grey = cv2.cvtColor(bichito, cv2.COLOR_BGR2GRAY)

    def angulizador(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy)
        return gx, gy, mag, ang

    gx, gy, mag, ang = angulizador(bichito_grey)

    # Normalizar magnitud para visualizarla
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = mag_vis.astype(np.uint8)

    #cv2.imshow('Magnitud del Gradiente', mag_vis)
    try:
        
        cv2.imwrite('/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/bordeadas/{}.jpg'.format(i), mag_vis)

    except:
        print("No se pudo guardar la nueva imagen")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print("Ha finalizado el filtrado de las imágenes con el filtro del borde para Heptageniidae")

print("#######################################################################")

print("Inicia el bordeado para Leptohyphidae")

ruta = '/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptohyphidae/homogenea'
extensiones = ('.jpg','.png','.jpeg')

archivos = sorted([f for f in os.listdir(ruta) if f.endswith(extensiones)])


for i, nombre_original in enumerate(archivos):

    stringBichito = '{}/{}.jpg'.format(ruta,str(i))
    bichito = cv2.imread(stringBichito)
    #cv2.imshow("Bichito",bichito)
    
    if bichito is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        continue

    bichito_grey = cv2.cvtColor(bichito, cv2.COLOR_BGR2GRAY)

    def angulizador(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy)
        return gx, gy, mag, ang

    gx, gy, mag, ang = angulizador(bichito_grey)

    # Normalizar magnitud para visualizarla
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = mag_vis.astype(np.uint8)

    #cv2.imshow('Magnitud del Gradiente', mag_vis)
    try:
        
        cv2.imwrite('/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptohyphidae/bordeadas/{}.jpg'.format(i), mag_vis)

    except:
        print("No se pudo guardar la nueva imagen")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
