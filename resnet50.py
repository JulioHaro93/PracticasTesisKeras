#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import sklearn as skl
from sklearn.model_selection import train_test_split
import cv2
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

X = []
y = []

def bdRecolector_5c():
    img_size = (224, 224)
    paths = {
    'Beatidae':Path(r'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/recdirect/*.jpg'), #c:\Users\Julio\Documents\tesis\Tesis-BD\Ephemeroptera\Beatidae\recdirect
    'Caenidae':Path(r'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Canidae/recdirect/*.jpg'),
    'Heptageniidae':Path(r'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Heptageniidae/recdirect/*.jpg'),
    'Leptohyphidae':Path(r'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptohyphidae/recdirect/*.jpg'),
    'Leptophlebiidae':Path(r'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptophlebiidae/recdirect/*.jpg')
    }


    label_map = {}
    label_counter = 0
    max_per_class = 500
    target_size = (224, 224)

    for label, path in paths.items():
        archivos = glob.glob(path)
        print(f"{label}: {len(archivos)} archivos encontrados")

        if len(archivos) > max_per_class:
            archivos = random.sample(archivos, max_per_class)

        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1

        for archivo in archivos:
            img = cv2.imread(archivo)  # Lee la imagen como BGR
            if img is None:
                print(f"⚠️ No se pudo leer {archivo}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(label_map[label])

        X = np.array(X, dtype=np.uint8)
        y = np.array(y)

    print(f"✅ Forma final de X: {X.shape}")
    print(f"✅ Forma final de y: {y.shape}")
    print(f"🧭 Mapeo de etiquetas: {label_map}")
    print(X[0])
    print(y[0])
    return X, y

def bdrecolector_2by2(labelcillo):
    paths = {
    'Baetidae':'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/recdirect/*.jpg', #c:\Users\Julio\Documents\tesis\Tesis-BD\Ephemeroptera\Beatidae\recdirect
    'Caenidae':'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Caenidae/recdirect/*.jpg',
    'Heptageniidae':'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Heptageniidae/recdirect/*.jpg',
    'Leptohyphidae':'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptohyphidae/recdirect/*.jpg',
    'Leptophlebiidae':'c:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Leptophlebiidae/recdirect/*.jpg'
    }

    X, y = [], []
    max_per_class = 500
    target_size = (224, 224)

    # 1. Recolectar imágenes de la clase objetivo
    if labelcillo not in paths:
        raise ValueError(f"'{labelcillo}' no es una etiqueta válida.")

    archivos_pos = glob.glob(paths[labelcillo])
    print(f"{labelcillo}: {len(archivos_pos)} archivos encontrados")

    if len(archivos_pos) > max_per_class:
        archivos_pos = random.sample(archivos_pos, max_per_class)

    for archivo in archivos_pos:
        img = cv2.imread(archivo)
        if img is None:
            print(f"⚠️ No se pudo leer {archivo}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        X.append(img)
        y.append(1)  # Etiqueta para labelcillo

    # 2. Recolectar imágenes del resto de clases
    otros_archivos = []
    for label, path in paths.items():
        if label == labelcillo:
            continue
        archivos = glob.glob(path)
        otros_archivos.extend(archivos)

    print(f"Otras clases: {len(otros_archivos)} archivos encontrados (sin {labelcillo})")
    if len(otros_archivos) > max_per_class:
        otros_archivos = random.sample(otros_archivos, max_per_class)

    for archivo in otros_archivos:
        img = cv2.imread(archivo)
        if img is None:
            print(f"⚠️ No se pudo leer {archivo}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        X.append(img)
        y.append(0)  # Etiqueta para otras clases

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)

    print(f"✅ Forma final de X: {X.shape}")
    print(f"✅ Forma final de y: {y.shape}")
    print(f"🧭 Clase positiva: '{labelcillo}' (etiqueta = 1), otras = 0")
    return X, y


def model_basic(X, y, nombrecillo): 
    print("Corriendo entrenamiento de red neuronal desde 0 para la familia {} y el resto de familias".format(nombrecillo))
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=4,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
    base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_shape=(224,224,3))
    #base_model.summary()
    model = Sequential()
    regularizer = tf.keras.regularizers.l2(0.001)
    model.add(base_model)
    model.add(Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=60, activation='relu', kernel_regularizer= regularizer))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
    #model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs= 20, batch_size=32, callbacks =[callbacks])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test accuracy: {}".format(test_accuracy))
    print("Test loss: {}".format(test_loss))
    
    model_name = 'ResNet50_{}_0.h5'.format(nombrecillo)
    tf.keras.models.save_model(model, model_name)


    converter = tf.lite.TFLiteConverter.from_keras_model(model)


    tflite_model = converter.convert()
    with open("tf_model2.tflite_basic", "wb") as f:
        f.write(tflite_model)
    
    df = pd.DataFrame(history.history)

    folder_path = f"./history/{nombrecillo}"
    
    df.to_csv(f"{folder_path}/hist_{nombrecillo}_0.csv", index=False)


def model_preentrenado(X,y, namecito, iteration):
    epocas = iteration*20
    print("Corriendo el entrenamiento de las épocas {} a {} para la familia {}".format(epocas,(epocas*2), namecito))
    #name = namecito
    #name = name + datetime.now()

    model = tf.keras.models.load_model('./ResNet50_{}_{}.h5'.format(namecito, iteration-1))
    model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
    history = model.fit(X_train, y_train, epochs= 20, batch_size=32)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test accuracy: {}".format(test_accuracy))
    print("Test loss: {}".format(test_loss))
    model_name = 'ResNet50_{}_{}.h5'.format(namecito, iteration)
    tf.keras.models.save_model(model, model_name)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    with open("tf_model2.tflite_{}".format(namecito), "wb") as f:
        f.write(tflite_model)
    
    folder_path = f"./history/{namecito}"
    os.makedirs(folder_path, exist_ok=True)
    df = pd.DataFrame(history.history)
    df.to_csv(f"{folder_path}/hist_{namecito}_{iteration}.csv", index=False)

def ResNet50_Sparce(X,y,iteration, clases):
    print("Corriendo entrenamiento de red neuronal desde 0 para el clasificador multiclase")
    print("familias: {}, {}, {}, {} y {}".format(clases[0], clases[1], clases[2], clases[3], clases[4]))
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=4,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )
    model = Sequential()
    if iteration ==0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
        base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_shape=(224,224,3))
        #base_model.summary()
        model = Sequential()
        regularizer = tf.keras.regularizers.l2(0.001)
        model.add(base_model)
        model.add(Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(units=60, activation='relu', kernel_regularizer= regularizer))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units = 1, activation='softmax'))
        #model.summary()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
        history = model.fit(X_train, y_train, epochs= 20, batch_size=32, callbacks =[callbacks])

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print("Test accuracy: {}".format(test_accuracy))
        print("Test loss: {}".format(test_loss))
        
        model_name = 'ResNet50_{}_0.h5'
        tf.keras.models.save_model(model, model_name)


        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()
        with open("tf_model_5c_0.tflite_basic", "wb") as f:
            f.write(tflite_model)
        
        df = pd.DataFrame(history.history)

        folder_path = f"./history/5clases"
        
        df.to_csv(f"{folder_path}/hist_5c_0.csv", index=False)
    elif int(iteration) == 2:
        epocas = iteration*20

        model = tf.keras.models.load_model('./ResNet50_{}_{}.h5'.format(epocas,iteration-1))
        model.summary()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_crossentropy'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)
        history = model.fit(X_train, y_train, epochs= 20, batch_size=32)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print("Test accuracy: {}".format(test_accuracy))
        print("Test loss: {}".format(test_loss))
        model_name = 'ResNet50_{}_{}.h5'.format(epocas,iteration)
        tf.keras.models.save_model(model, model_name)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()
        with open("tf_model2.tflite_{}_{}".format(epocas,iteration), "wb") as f:
            f.write(tflite_model)
        
        folder_path = f"./history/5clases"
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame(history.history)
        df.to_csv(f"{folder_path}/hist_5c_{iteration}.csv", index=False)
    else:
        print("No hay opción disponible")