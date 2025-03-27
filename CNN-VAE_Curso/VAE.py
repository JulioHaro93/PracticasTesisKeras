# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:22:48 2025

@author: Julio Cesar Haro Capetillo
"""

# Construcción del modelo Variational Auto Encoder

## Importar librerías que se utilizan

import numpy as np
import tensorflow as tf

## construcción del model VAE dentro de una clase

class ConvVAE(object):
    #Inicialización de los parámetros y variables de la clase convVAE
    '''
    Datos:
        * batch_size se utiliza así porque se hace por bloques, con valor a 1
        entrena de uno en uno
        * kl_tolerance hace alusión a la pérdida del error medio
        * is_training me permite validar que esté evaluando el aprendizaje
        * reuse es de tensorflow, es una variable que nos hace ver si utilziamos
        el ámbito de visibilidad de las variables (esto puede llegar a cambiar en la ejecución)
        * 
    
    '''
    def __init__(self, z_size=32, batch_size = 1, learning_rate= 0.0001, 
                 kl_tolerance= 0.5, is_training= False, reuse = False, 
                 gpu_mode = False):
        
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('conv_vae', resuse = self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'): #Función de la librería de ténsorwlof
                    tf.logging.info('Modelo entrenando por CPU')
                self._build_graph()
            else:
                tf.logging.info('Modelo entrenando con GPU')
                self._build_graph() ## crea el grafo computacional
        
        self._init_session() #Primer método del init
            
    '''
    Creación de un método para el modelo VAE
    Es para la arquitectura del modelo
    '''
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default(): #Este es de tensorflow que genera un grafo dinámico
            #el place holder se utiliza para recibir los datos antes de entrenar
            #None representa el número de muestras, pero se deja como None porque es desconocida
            
            self.x = tf.placeholder(tf.float32, shape = [None, 64,64,3])#Esto es un tensor de 32 bits
            
            
            #Se hacen 4 convoluciones, se le conoce como codificación, después viene la capa variacional
            #al final hace el autoencoder
            #En el VAE se hace la convolución, se usa relu de 32x4, la siguiente relu 64x4, y al final relu conv 128x4
            
            
            #Construir el Encoder del VAE
            #Esto es una capa de convolución, y los parámetros que recibe son
            #inputs: Son las imágenes, en este caso el placeholder de imágenes que se hayan cargado
            #filters es el número de filtros o de mapas de características que va a utilizar la red neuronal
            #En este caso son 32 las operaciones de convolución en la primer capa
            #el kernel_zise te dice cuál es el alto y el ancho de la matriz de convolución, en este caso es 4x4
            #Conv2D(inputs, filtters, kernel_size, strides, activation)
            #strides son el número de posición en convolución a convolución
            #activation, sino se especifica la función de activación toma la lineal de manera automática
            #Revisar la documentación: 
            #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
            #name es el nombre de la capa
            
            h = tf.layers.Conv2D(self.x, 32, 4,strides = 2, activation = tf.nn.relu, name ='encoder_conv1')
            h = tf.layers.Conv2D(h, 64, 4,strides = 2, activation = tf.nn.relu, name ='encoder_conv2')
            h = tf.layers.Conv2D(h, 128, 4,strides = 2, activation = tf.nn.relu, name ='encoder_conv3')
            h = tf.layers.Conv2D(h, 256, 4,strides = 2, activation = tf.nn.relu, name ='encoder_conv4')
            
            #Para la capa variacional se necesita un vector aplanado de tamaño 2x2x256
            #éste es un vector aplanado, un vector de resultados.
            #reshape() me redimensiona la salida de la última capa
            #En este caso el -1 significa que no importa la dimensión original
            h = tf.reshape(h,shape = [-1, 2*2*256]) #Y así queda unidimensional
            
            #Aquí termina la codificación del encoder, ahora fal ta la parte variacional.
            #Esta es la capa densa para el vector aplanado, esta parte tiene una forma estocástica
            #normalizada con mu =0, y sigma =1
            
            self.mu= tf.layers.dense(h, )
            self.sigma= tf.
            
    
    pass
        
    pass