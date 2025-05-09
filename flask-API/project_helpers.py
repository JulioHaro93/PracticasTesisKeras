# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:20:10 2025

@author: Julio
"""


from tensorflow.keras.datasets import fashion_mnist
#from scipy.misc import imsave

from PIL import Image
import numpy as np
import os




#Guardar una imagen con python

#img = Image.fromarray(image_array)
#img.save('output.png')

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

os.makedirs("uploads", exist_ok=True)

for i in range(5):
    
    img = Image.fromarray(X_test[i])
    img.save("uploads/{}.png".format(i))

