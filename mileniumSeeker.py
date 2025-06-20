# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:43:38 2025

@author: Julio
"""

import os
import random
import shutil
from pathlib import Path
import glob

# Ruta base de las familias
base_dir = Path(r"C:\Users\Julio\Documents\tesis\Tesis-BD\Ephemeroptera")
familias = ["Beateidae", "Caenidae", "Heptageniidae", "Leptohyphidae", "Leptophlebiidae"]

# Salidas
output_dir = Path(r"C:\Users\Julio\Documents\tesis\Tesis-BD\efemeropteros_yolo")
train_ratio = 0.8

# Crear carpetas destino
(train_img_dir := output_dir / "images/train").mkdir(parents=True, exist_ok=True)
(val_img_dir := output_dir / "images/val").mkdir(parents=True, exist_ok=True)
(train_lbl_dir := output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
(val_lbl_dir := output_dir / "labels/val").mkdir(parents=True, exist_ok=True)

pares = []

from pathlib import Path

base_dir = Path(r"C:\Users\Julio\Documents\tesis\Tesis-BD\Ephemeroptera")

# Detectar todas las familias automáticamente
familias = [d for d in base_dir.iterdir() if d.is_dir()]
print("Familias detectadas:")
for f in familias:
    print(" -", f.name)



for familia_dir in familias:
    txt_paths = list(familia_dir.rglob("*.txt"))
    #txt_paths = list(familia_dir.rglob("*.txt"))
    encontrados = 0
    print(txt_paths)
    # Preindexar todas las imágenes posibles por nombre base
    imagenes = {}
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        for img_path in familia_dir.rglob(f"*{ext}"):
            imagenes[img_path.stem] = img_path

    for lbl_path in txt_paths:
        nombre = lbl_path.stem
        if nombre in imagenes:
            img_path = imagenes[nombre]
            pares.append((img_path, lbl_path))
            encontrados += 1
        else:
            print(f"⚠️ Imagen no encontrada para {nombre} en {familia_dir}")

    if encontrados == 0:
        print(f"⚠️ No se encontraron pares válidos en {familia_dir}")
    else:
        print(f"✅ {encontrados} pares válidos encontrados en {familia_dir}")

# Mezclar y dividir
random.shuffle(pares)
split_index = int(len(pares) * train_ratio)
train_pares = pares[:split_index]
val_pares = pares[split_index:]

def copiar_pares(pares, dest_img_dir, dest_lbl_dir):
    for img_path, lbl_path in pares:
        shutil.copy(img_path, dest_img_dir / img_path.name)
        shutil.copy(lbl_path, dest_lbl_dir / lbl_path.name)

copiar_pares(train_pares, train_img_dir, train_lbl_dir)
copiar_pares(val_pares, val_img_dir, val_lbl_dir)

print(f"✅ Proceso completado: {len(train_pares)} imágenes para entrenamiento y {len(val_pares)} para validación.")
