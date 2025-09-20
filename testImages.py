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

print("Iniciando prueba de student para diferencias significativas")

n = len(valores)
alpha = 0.05
confianza = 1 - alpha

grados_de_libertad = n - 1
t_critico = stats.t.ppf((1 + confianza) / 2, grados_de_libertad)
print(f"Valor crítico de t para {confianza*100}% de confianza y {grados_de_libertad} grados de libertad: {t_critico}")
media = np.mean(valores)
desviacion_estandar = np.std(valores, ddof=1)
error_estandar = desviacion_estandar / np.sqrt(n)
margen_de_error = t_critico * error_estandar
intervalo_confianza = (media - margen_de_error, media + margen_de_error)
print(f"Intervalo de confianza del {confianza*100}% para la media: {intervalo_confianza}")
print(f"Media: {media}, Desviación estándar: {desviacion_estandar}, Error estándar: {error_estandar}")
print("////////////////////////////////////////////////")

entropiaAlta= 'C:/Users/Usuario/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/BDWEB/Baetidae-0011-2.jpg'
print("Analizando imagen con alta entropía")

name = os.path.basename(entropiaAlta)
image = cv2.imread(entropiaAlta)
img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_convertida = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)
entropy = skimage.measure.shannon_entropy(img_convertida)
print(f"Entropía de la imagen {name}: {entropy}")

entropiaBaja= 'C:/Users/Usuario/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/BDWEB/frame_didier_06137-5.jpg'
print("Analizando imagen con alta entropía")

name = os.path.basename(entropiaBaja)
image = cv2.imread(entropiaBaja)
img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_convertida = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)
entropy = skimage.measure.shannon_entropy(img_convertida)
print(f"Entropía de la imagen {name}: {entropy}")

print("Fin de la prueba de entropía")
