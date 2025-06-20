# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:23:46 2025

@author: Julio
"""

import os
import xml.etree.ElementTree as ET

# Cambia estas rutas
input_dir = "C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beateidae/labels/train"
output_dir ="C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beateidae/labels/train"
class_list = ["bug"]  # Asegúrate que "bug" sea tu única clase

os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    image_w = int(root.find("size/width").text)
    image_h = int(root.find("size/height").text)

    output_lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in class_list:
            continue  # skip unknown classes

        cls_id = class_list.index(cls)

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        xmax = int(bbox.find("xmax").text)
        ymin = int(bbox.find("ymin").text)
        ymax = int(bbox.find("ymax").text)

        # YOLO format: center_x center_y width height (normalized)
        x_center = ((xmin + xmax) / 2) / image_w
        y_center = ((ymin + ymax) / 2) / image_h
        width = (xmax - xmin) / image_w
        height = (ymax - ymin) / image_h

        output_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Guardar el archivo .txt
    base_filename = os.path.splitext(xml_file)[0]
    with open(os.path.join(output_dir, base_filename + ".txt"), "w") as f:
        f.write("\n".join(output_lines))
