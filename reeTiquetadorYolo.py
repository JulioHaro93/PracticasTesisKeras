# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:33:11 2025

@author: Julio
"""

import os
from pathlib import Path

# Ruta raíz de tus carpetas de familias
base_dir = Path(r"C:\Users\Julio\Documents\tesis\Tesis-BD\Ephemeroptera")

# Subcarpetas (familias) a procesar
familias = ["Beateidae", "Caenidae", "Heptageniidae", "Leptohyphidae", "Leptophlebiidae"]

# Carpetas a revisar dentro de cada familia
subcarpetas = ["labels/train", "labels/val"]

for familia in familias:
    for sub in subcarpetas:
        label_dir = base_dir / familia / sub
        if not label_dir.exists():
            print(f"[!] Carpeta no encontrada: {label_dir}")
            continue

        for txt_file in label_dir.glob("*.txt"):
            with open(txt_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = "0"  # Reemplazar clase por 0
                    new_line = " ".join(parts)
                    new_lines.append(new_line)

            # Sobrescribir el archivo
            with open(txt_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")

        print(f"[✓] Etiquetas modificadas en: {label_dir}")

