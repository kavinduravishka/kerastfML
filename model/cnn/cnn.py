import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Dropout
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import VGG16



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model.cnn.basic_nn import basic_nn


class convolutional_cnn(basic_nn):
    def __init__(self,labelcount=8,input_shape=(256,256,3)):
        basic_nn.__init__(self,labelcount=labelcount,input_shape=input_shape)
        
        conv_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
        
        for layer in conv_model.layers[:-4]:
            layer.trainable = False
    
        self.model = Sequential([
            conv_model,
            Flatten(),
            Dense(units=1024,activation='relu'),
            Dropout(0.5),
            Dense(units=self.labelcount, activation='softmax')
        ])
    
    