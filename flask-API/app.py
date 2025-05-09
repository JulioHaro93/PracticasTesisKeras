# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:30:32 2025

@author: Julio

dependencias para abrir una API
"""


#Se utilizan las siguientes dependencias

import os
import requests
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify

print(tf.__version__)

#cargar el modelo pree entrenado
with open('fashion_model.json', 'r') as f:
    model_json = f.read()
    

# cargar los pesos en el modelo
model = tf.keras.models.model_from_json(model_json)

#Paso 3: Crear la API con flask
# Crear una ap´licación de Flask

model.load_weights("fashion_model.weights.h5")

app = Flask(__name__)

## Definir la función de clasificación de imágenes
@app.route("/api/v1/<string:img_name>", methods =["POST"])

def classify_image(img_name):
    #Donde se encuentran las imágenes
    
    upload_dir ="uploads/"
    
    #Cargar una de las imágenes de la carpeta
    
    image = Image.open(upload_dir + img_name)
    image= image.resize((28,28))
    
    
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28*28)  # Convertir en vector plano
    image_array = image_array / 255.0  # Normalizar (opcional pero común)
    #Definir la lista de posibles clases de la imagen
    
    
    classes = ["Camiseta", "Pantalón", "Sudadera", "Vestido", "Abrigo", "Sandalia", "Jersey", "Zapatilla", "Bolsa", "Botas"]
    
    
    #Te devuelve la mayor predicción del modelo pre entrenado
    prediction =model.predict([image_array])
    
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

app.run(port=5000, debug=False)


