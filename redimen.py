#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:30:13 2025

@author: julio
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import sklearn as skl
from sklearn.model_selection import train_test_split
import cv2
import os

# %matplotlib inline
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

print(np.__version__)

paths ={
        #'Betidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Beateidae/bordeadas/*.jpg',
        #'Canidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Caenidae/bordeadas/*.jpg',
        'Heptageniidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/bordeadas/*.jpg'
        #'Leptohyphidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptohyphidae/bordeadas/*.jpg',
        #'Leptophlebiidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/bordeadas/*.jpg'
        }




train_files =[]
train_labels =[]
test_files=[]
test_labels =[]

for label, pattern in paths.items():
  files = glob.glob(pattern)

  train, test = train_test_split(files, test_size=0.2, random_state=53)
  train_files.extend(train)

  train_labels.extend([label]*len(train))

  test_labels.extend([label]*len(test))

img= cv2.imread(train_files[550])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

#Redimensionar los tamaños con cv2.resize

# Lista de rutas a tus carpetas
carpetas = [
    #'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Beateidae/bordeadas/',
    #'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Caenidae/bordeadas/',
    #'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Heptageniidae/bordeadas/'
    '/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptohyphidae/bordeadas',
    #'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/Ephemeroptera/Leptophlebiidae/bordeadas/'
]

anchos = []
altos = []
"""
for carpeta in carpetas:

    rutas = glob.glob(os.path.join(carpeta, '*.jpg'))

    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is not None:
            altos.append(img.shape[0])
            anchos.append(img.shape[1])
"""
avg_altura =1111  #int(np.mean(altos))
avg_ancho =1933 #int(np.mean(anchos))

print(f"Tamaño promedio: {avg_ancho}x{avg_altura}")

extensiones = ['*.jpg', '*.jpeg', '*.png']

for carpeta in carpetas:
    carpeta_redimen = os.path.join(carpeta, 'redimen')
    os.makedirs(carpeta_redimen, exist_ok=True)

    for ext in extensiones:
        rutas = glob.glob(os.path.join(carpeta, ext))

        for ruta in rutas:
            img = cv2.imread(ruta)
            if img is not None:
                img_resized = cv2.resize(img, (avg_ancho, avg_altura))
                nombre_archivo = os.path.basename(ruta)
                ruta_guardado = os.path.join(carpeta_redimen, nombre_archivo)
                cv2.imwrite(ruta_guardado, img_resized)