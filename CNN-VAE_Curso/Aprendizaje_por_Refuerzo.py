# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:36:28 2025

@author: Julio
"""

# Aprendizaje por refuerzo
# Pseudocódigo


obs = env.reset()

h = mdnrnn.initial_state

done = False

cumulative_reward =0

#En cada paso, dependiendo de loq ue devuelva el entorno
#en caso de que el entorno exprese que fue eficiente el aprendizaje, se incrementa el cumulative_reward después de cada ciclo

while not done:
    
    z = cnnvae(obs)
    #Por el modelo, z debe propagarse a la parte de la red recurrente
    #de las capas recurrentes sale una z y una h que son parámetros de entrada de la última capa llamada "controlador"
    
    #a_t = W_c * [z_t h_t] + b_controlador
    # =>
    a = controller([z,h]) #Esto es una lista de listas o vector plano de z y h, el controlador me genera la acción que regresa al "entorno"
    
    #El Entorno recibe una acció o instrucción, la cuál debe tomar una decisión, si la decisión es correcta, entonces cumulative_reward +=1
    #De lo contrario
    #obs vuelve a cambiar su valor por la nueva observación
    obs, reward, done = env.step(a)
    cummulative_reward+=reward
    #