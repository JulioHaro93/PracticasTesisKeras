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
        LENGTH = hps.max_seq_len
        if self.is_training:
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        #Capa LSTM, no siempre se añade el dropout de olvido en todos los casos
        #Por eso se utilizan 3 variables booleanas que nos ayudan a activar la dropout
        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
        #Las variables booleanas siguientes, son las mismas que gobiernan las decisiones
        #a considerar en cada vuelta de la recurrencia
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout ==0 else True
        use_inpout_dropout = False if self.hps.use_input_dropout ==0 else True
        use_outpout_dropout = False if self.hps.use_output_fropout == 0 else True
        use_layer_norm = False if self.hps.use_layern_norm ==0 else True
        #Usa el dropout
        if use_recurrent_dropout:
            cell = cell_fn(hps.rnn_size, layer_norm = use_layer_norm, dropout_keep_prob = self.hps.recurrent_dropout_prob)
        #No usa el dropout
        else:
            cell = cell_fn(hps.rnn_size, layer_norm= use_layer_norm)
        if use_inpout_dropout:
            #imput_keep_prob
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = self.hps.input_dropout_prob)
        if use_outpout_dropout:
            cell = cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.hps.output_dropout_prob)
        self.cell = cell
        #Clase de placeholder
        """
        Es un submódulo de tensorflow, recurrente o no, espera un formato de salida y entrada
        pero en este caso se utiliza como referencia de un dato que va cambiando en cada
        parte de la recurrencia, un placeholder para las entradas y otro para las salidas
        """
        self.sequence_lengths = LENGTH
        self.input_x = tf.placeholder(dtype= tf.float32, shape = [self.hps.batch_size, self.sequence_lengths, INWIDTH])
        self.output_x = tf.placeholder(dtype = tf.float32, shape = [self.hps.batch_size, self.sequence_lengths, OUTWIDTH])
        actual_input_x = self.input_x

        self.initial_state = cell.zero_state(batch_size = self.hps.batch_size, dtype = tf.float32)
        #Para matrices de peos y bías obtenemos dos tensores
        NOUT = OUTWIDTH * KMIX * 3
        with tf.variable_scope("RNN"):
            output_w = tf.get_variable("output_w", shape = [self.hps.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", shape =[NOUT])
        output, last_state = tf.nn.dynamic_rnn(cell, inputs = actual_input_x, sequence_length = None,
                                               initial_state = self.initial_state,
                                               dtype = tf.float32,
                                               parallel_iterations=None,
                                               swap_memory = True, #para propagar errores de CPU a GPU
                                               time_major = False,
                                               scope = "RNN")
        """
        Sigue la parte de la capa de densidad mixta, en donde se considera la parte de la probabilidad
        con respecto a la normalidad de los datos, hasta aquí se trata de una red neuronal recurrente
        convencional cuya salida es determinista, esto puede limitar una red neuronal, 
        pero aquí se genera una nueva capa que va por encima a través de las densidades gaussianas una
        predicción estocástica (contrario al determinismo de la RNN)

        Aquí consideramos que KMIX parte del número de densidades
        """
        #Construcción de la capa de Densidad Mixta MDN

        #Comenzamos aplanando el vector de salida para que pueda multiplicarse
        #rnn_size te especifíca los 256 elementos para poder aplanar de 3 dimensiones a un vector columna de 256
        output = tf.reshape(output, [-1, hps.rnn_size])
        #Por fin hacemos la operación matricial para obtener la matriz resultante de la capa
        #Las dimensiones son dimensión1 por dimensión2
        output = tf.nn.xw_plus_b(output, output_w, output_b) #Este es el siguiente vector latente para la siguiente capa
        output = tf.reshape(output, [-1, KMIX*3]) #Quince dimensiones, ésto va a la capa del controlador

        self.final_state = last_state #último vector para poder retroalimentar para la RNN

        def get_mdn_coef(output):
            #Me parte la salida para poder generar las salidas
            """para el eje axis = 1 si horizontal, 0 si en vertical"""
            logmix, mean, logstd = tf.split(output, 3,1 )
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims = True)
            return logmix, mean, logstd
        out_logmix, out_mean, out_logstd = get_mdn_coef(output)
    
        self.output_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd
        