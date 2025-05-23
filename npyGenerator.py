#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:46:17 2025

@author: julio
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import glob


img_size = (224,224)

X = []
y = []
paths ={
        'Beatidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Beatidae/*.jpg',
        'Canidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Canidae/*.jpg',
        'Heptageniidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Heptageniidae/*.jpg',
        'Leptohyphidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptohyphidae/*.jpg',
        'Leptophlebiidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptophlebiidae/*.jpg'
        }

output_dir = "/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/preprocesadas_numpy"
os.makedirs(output_dir, exist_ok=True)

for label, path_pattern in paths.items():
    print(label)
    files = glob.glob(path_pattern)
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    for i, file in enumerate(files):
        img = load_img(file, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        out_path = os.path.join(label_dir, f"{label}_{i}.npy")
        np.save(out_path, img_array)


"""

BASOFIA
ESTO ES BASOFIA, MUCHA BASOFIA


# Listas para guardar imágenes y etiquetas
X = []
y = []
paths ={
        'Betidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Beatidae/*.jpg',
        'Canidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Canidae/*.jpg',
        'Heptageniidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Heptageniidae/*.jpg',
        'Leptohyphidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptohyphidae/*.jpg',
        'Leptophlebiidae':'//home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptophlebiidae*.jpg'
        }

# Cargar imágenes y asignar etiquetas
for label, path_pattern in paths.items():
    files = glob.glob(path_pattern)
    for file in files:
        img = load_img(file, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalizar
        X.append(img_array)
        y.append(label)

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Codificar etiquetas a enteros
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # clases: 0, 1, 2, ...

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    directory='/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/',
    target_size=(224, 224),
    batch_size=10,
    class_mode='sparse',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    directory='/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas',
    target_size=(224, 224),
    batch_size=10,
    class_mode='sparse',
    subset='validation'
)


"""