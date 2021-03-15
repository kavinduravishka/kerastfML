from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from numpy import argmax

class batch:
    def __init__(self,shape=(256,256)):
        #preprocessing function
#        self.preprofunc={'vgg16':tf.keras.applications.vgg16.preprocess_input}
#        
#        if not (ppfunc==None or ppfunc==''):
#            self.generator=ImageDataGenerator(preprocessing_function=self.preprofunc[ppfunc],fill_mode='nearest')
#        else:
#            self.generator=ImageDataGenerator(fill_mode='nearest')
        
        self.generator=ImageDataGenerator()
        
        self.shape=shape
        self.length=0
        self=iter(self)
        self.outset=None
        
        
    def __iter__(self):
        return self
            
    def getFromDir(self,directory,batchsize=10,shuffle=True):
        self.labels=os.listdir(directory)
        self.labels.sort()
        self.iterset = self.generator.flow_from_directory(directory, target_size=self.shape,classes=self.labels, batch_size=batchsize)
        self.length=len(self.iterset)
        return self
    
    def __next__(self):
        self.outset=list(next(self.iterset))
        self.out_onehot=self.outset[1]
        #returns [images,labels]
        return self.outset
    
    def onehot2(self,label_type):
        if label_type=='index':
            return argmax(self.out_onehot,axis=1)
        elif label_type=='text':
            return [self.labels[i] for i in argmax(self.out_onehot,axis=1)]