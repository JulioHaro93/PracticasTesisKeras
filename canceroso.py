#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:28:24 2025

@author: julio


Ejemplo de clasificador con lunares de cáncer
"""


"""

En cada imagen se puede ver un lunar, en este caso el color permite diferenciarlos


Sacamos unas características de color para el lunar

Por lo tanto usamos el método de otsu para segmentarlo y sacar un color para cada imagen del lunar



"""

import cv2
import numpy as np
import glob #Ayuda a la lectura de los archivos de imágenes

paths = [
    '/home/julio/Documentos/Cancerosos/datasetLunares/datasetLunares/dysplasticNevi/train',
    '/home/julio/Documentos/Cancerosos/datasetLunares/datasetLunares/spitzNevus/train'
    ]

#img = cv2.readOpticalFlow(("/home/julio/Documentos/Cancerosos/datasetLunares/datasetLunares/dysplasticNevi/train/dysplasticNevi5.jpg"))



#Aquí determina de forma dinámica dependiendo de los valores de las imágenes para cada una de ellas
#Por lo que la máscara se ve ya sea por encima o por debajo, por lo que lo que nos interesa son todos los
#valores que están por debajo del umbral


def getFeatures(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #el threshold nos permite separar entre fondo y lunar.
    
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = np.uint8(1*(gray<threshold))
    
    #Sacamos primero el canal, una imágen con 3 canales
    #Todas las filas y todas las columnas nos dan el azul
    B=(1/255)* np.sum(img[:,:,0] * mask)/ np.sum(mask) #Así nos quita el fondo y deja sólo el lunar
    G=(1/255)* np.sum(img[:,:,1] * mask)/ np.sum(mask)
    R=(1/255)* np.sum(img[:,:,2] * mask)/ np.sum(mask)
    
    return [B,G,R]

labels =[]
features = []

#Para cada filename, usamos glob para leer el archivo a partir del path, si se concatena
#un *.jpg, lee todos los archivos que sean jpg
for label, path in enumerate(paths):
    for filename in glob.glob(path+'*.jpg'):      
        img = cv2. imread(filename)
        features.append(getFeatures(img))
        labels.append(label)

#Recuerda pasarlo a arreglo de numpy para que lo puedas trabajar

features = np.array(features)
labels = np.array(labels)

#Para que nos quede entre -1 y 1 las etiquetas

labels = 2*labels-1 

#Visualización del dataset en el espacio de características

import matplotlib.pyplot as plt
fig = plt.figure()

ax= fig.add_subplot(111, projection = '3d')



for i, feature_row in enumerate(features):
    if labels[i] == -1:
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker ='*', c='k')
        
    else:
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker ='*', c='r')
ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')

#Error en función de las constantes del hiper-plano
subFeatures = features[:, 1:,:]
loss= []

for w1 in np.linspace(-6,6,100):
    for w2 in np.linspace(-6,6,100):
        totalError = 0
        for i, in feature_row in enumerate(subFeatures):
            sample_error = (w1*feature_row[0]+w2*feature_row[1]-labels[i])**2
            
            totalError += sample_error
            loss.append([w1,w2,totalError])

loss = np.array(loss)

from matplotlib import cm
figure=plt.figure()

ax1 = fig. add_subplot(111, projection ='3d')

ax1.plot_trisurf(loss[:,0], loss[:,1], loss[:,2], cmap=cm.jet, linewidth =0)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('loss')

