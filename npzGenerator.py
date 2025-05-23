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
        'Betidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Beatidae/*.jpg',
        'Canidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Canidae/*.jpg',
        'Heptageniidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Heptageniidae/*.jpg',
        'Leptohyphidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptohyphidae/*.jpg',
        'Leptophlebiidae':'/home/julio/Documentos/TEST_TESIS/Tesis/db/procesadas/redimensionadas/Leptophlebiidae/*.jpg'
        }

output_dir = "/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy"
os.makedirs(output_dir, exist_ok=True)

for label, path_pattern in paths.items():
    files = glob.glob(path_pattern)
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    for i, file in enumerate(files):
        img = load_img(file, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        out_path = os.path.join(label_dir, f"{label}_{i}.npy")
        np.save(out_path, img_array)
