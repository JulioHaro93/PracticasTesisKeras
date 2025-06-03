#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:42:59 2025

@author: julio
"""


import os
import cv2
import numpy as np
import tensorflow as tf

import matplotlib as plt
from ultralytics import YOLO


paths ={
        'Beatidae':'/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy/Beatidae/*.npy',
        'Canidae':'/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy/Canidae/*.npy',
        'Heptageniidae':'/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy/Heptageniidae/*.npy',
        'Leptohyphidae':'/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy/Leptohyphidae/*.npy',
        'Leptophlebiidae':'/content/drive/MyDrive/BD-Ephemeroptera/preprocesadas_numpy/Leptophlebiidae/*.npy'
        }

def getColours(cls_num):
    base_colors = [(255,0,0), (0,255,0), (0,0,255)]
    color_index = cls_num % len(base_colors)
    increments = [(1,-2,1), (-2,1,-1), (1,-1,2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * (cls_num//len(base_colors))%256 for i in range(3)]
    return tuple(color)

yolo = YOLO('yolov8s.pt')

#imageCapt = cv2.imread()