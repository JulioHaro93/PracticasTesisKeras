#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun 4 May 2025
@author: julioHaro
"""

import numpy as np
import tensorflow as tf

## construcción del model MDN-RNN MODELO DE DENSIDAD Míxta y CAPA RECURSIVA

class MDNRNN(object):
    #Inicialización de los parámetros y variables de la clase MDNRNN
    '''
    Datos:
        * hps: Son los hiper parámetros
        * 
        * is_training me permite validar que esté evaluando el aprendizaje
        * reuse es de tensorflow, es una variable que nos hace ver si utilziamos
        el ámbito de visibilidad de las variables (esto puede llegar a cambiar en la ejecución)
        * 
    
    '''
    def __init__(self, hps, reuse = False, gpu_mode = False):
        self.hps = hps
        self.reuse = reuse
        with tf.variable_scope('mdn_rnn', resuse = self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'): #Función de la librería de ténsorwlof
                    tf.logging.info('Modelo entrenando por CPU')
                    self.g = tf.Graph()
                    if self.g.as_default():
                        self.build_model(hps)
            else:
                tf.logging.info('Modelo entrenando con GPU')
                self.g = tf.Graph() ## crea el grafo computacional
                if self.g.as_default():
                    self.build_model()
        self._init_session() #Primer método del init
            
    '''
    Creación de un método para el modelo MDN_RNN
    Es para la arquitectura del modelo
    '''
    def build_model(self,hps):
        # Construcción de la RNN
        """
        Parmámetros que se encargan de la memoria, recordar los modelos LSTM vistos en el CIC
        Al hablar de densidades mixtas se consideran al menos 5 distribuciones Gaussianas
        Las mayúsculas sólo se utilizan localmente en el método, por eso no se usa el objeto self
        """
        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture
        #Parámetros de la dimensión de las entradas y salidas
        #Una entrada es del VAE, y la otra es del mismo controlador (Recursión)
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.output_seq_width
        LENGTH = hps.max_seq_width
        if self.is_training:
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        
        pass