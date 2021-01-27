import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import itertools
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model.cnn.basic_cnn import basic_cnn


class deeplizard_basic(basic_cnn):
    def __init__(self,labelcount=2,input_shape=(256,256,3)):
        basic_cnn.__init__(self,labelcount=labelcount,input_shape=input_shape)
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=self.input_shape),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=self.labelcount, activation='softmax')
        ])
    
    