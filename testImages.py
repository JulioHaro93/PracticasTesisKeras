import os
import glob
import cv2
import skimage.measure
import numpy as np
import re

path = 'C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/recpadding/*.jpg'
diccionario = {}
for image_path in glob.glob(path):
    
    name = os.path.basename(image_path)
    
    image = cv2.imread(image_path)
    img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_convertida = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)
    entropy = skimage.measure.shannon_entropy(img_convertida)
    
    diccionario[name] = entropy

diccionario_ordenado = dict(
    sorted(diccionario.items(), key=lambda x: int(re.findall(r'\d+', x[0])[0]))
)

print(diccionario_ordenado)

valores = np.array(list(diccionario_ordenado.values()), dtype=float)

matriz_diferencias = np.abs(valores[:, None] - valores[None, :])

print(matriz_diferencias.shape)
print(matriz_diferencias)

