from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from numpy import argmax

class batch:
    def __init__(self,ppfunc):
        self.preprofunc={'vgg16':tf.keras.applications.vgg16.preprocess_input}
        
        if not (ppfunc==None or ppfunc==''):
            self.generator=ImageDataGenerator(preprocessing_function=self.preprofunc[ppfunc])
        else:
            self.generator=ImageDataGenerator()
        
        self.length=0
        self=iter(self)
        
    def __iter__(self):
        return self
            
    def getFromDir(self,directory,shape=None,batchsize=10,shuffle=False):
        self.labels=os.listdir(directory)
        self.labels.sort()
        self.iterset = self.generator.flow_from_directory(directory, target_size=shape,classes=self.labels, batch_size=batchsize)
        self.length=len(self.iterset)
        return self
    
    def __next__(self):
        outset=list(next(self.iterset))
        return outset
    
    def onehot2(self,label_type,onehot_array):
        if label_type=='index':
            return argmax(onehot_array,axis=1)
        elif label_type=='text':
            return [self.labels[i] for i in argmax(onehot_array,axis=1)]