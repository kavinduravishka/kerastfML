import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Dropout
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import VGG16

import itertools
import random


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model.cnn.basic_nn import basic_nn


class convolutional_cnn(basic_nn):
    def __init__(self,labelcount=8,input_shape=(256,256,3)):
        basic_nn.__init__(self,labelcount=labelcount,input_shape=input_shape)
        
        conv_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
        
        for layer in conv_model.layers[:-4]:
            layer.trainable = False
    
        for layer in conv_model.layers:
            print(layer, layer.trainable)
        self.model = Sequential([
#            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=self.input_shape),
#            MaxPool2D(pool_size=(2, 2), strides=2),
#            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
#            MaxPool2D(pool_size=(2, 2), strides=2),
            conv_model,
            Flatten(),
            Dense(units=1024,activation='relu'),
            Dropout(0.5),
            Dense(units=self.labelcount, activation='softmax')
        ])
    
    