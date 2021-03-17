from model.cnn.cnn import convolutional_cnn
from model.cnn.basic_nn import basic_nn
from data.batch import batch
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import itertools
import random

from tensorflow.keras.optimizers import Adam,RMSprop

class learner():
    def __init__(self,batchdirectory):
        self.traindir = batchdirectory+'train/'
        self.valdir = batchdirectory+'valid/'

        self.testdir = batchdirectory+'test/'
        self.modeldirectory = "model/saved/models/"
        
        self.cnnTrainable = True
        #self.cnnAvailable = False
        
    def get_trainbatch(self):
        trainbatch = batch()
        trainbatch.getFromDir(self.traindir)
        return trainbatch
    
    def get_validbatch(self):
        trainbatch = batch()
        trainbatch.getFromDir(self.valdir)
        return trainbatch
    
    def get_testbatch(self):
        trainbatch = batch()
        trainbatch.getFromDir(self.testdir,shuffle=False)
        return trainbatch
    
    def newModal(self):
        self.cnn = convolutional_cnn()
        self.cnnTrainable = True
        print("New CNN model initiated")
        return self
    
    def trainModel(self):
        try:
            if self.cnnTrainable == True:
                self.cnn.compilemodel(RMSprop,1e-4,'categorical_crossentropy',['accuracy'])
                self.history = self.cnn.fit(self.get_trainbatch(),self.get_validbatch())
                
                self.plot_train_history(self.history)
            else:
                print("CNN is not trainable")
        except AttributeError:
            print("No CNN is available for training. Create a CNN or load from file")
    
    def loadModel(self,name):
        try:
            self.cnn=basic_nn(None,None)
            self.cnn.deserialize(self.modeldirectory+ name)
            for layer in self.cnn.model.layers:
                layer.trainable = False
            self.cnnTrainable = False
            return self
        except OSError:
            print("A model with name '"+name+"' is not available")
        
    
    def saveModel(self,name):
        try:
            self.cnn.serialize(self.modeldirectory + name)
        except AttributeError:
            print("No CNN is available to save. Create a CNN or load from file")
    
    def listModel(self):
        modellist = os.listdir(self.modeldirectory)
        print("Available Models >>>")
        for modelname in modellist:
            print('\t'+modelname)
            
    def testModel(self):
        try:
            tstbtch = self.get_testbatch()
            true_classes = tstbtch.iterset.classes
            
            predictions = self.cnn.predict(tstbtch).getresult()
            predicted_classes = np.argmax(np.round(predictions),axis=-1)
            
            cm = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
            
            self.plot_confusion_matrix(cm,{k:i for i,k in enumerate(tstbtch.labels)})
        except AttributeError:
            print("No CNN is available to save. Create a CNN or load from file")
            
            
            
            
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    def plot_train_history(self,fit_history):
        acc = fit_history.history['accuracy']
        val_acc = fit_history.history['val_accuracy']
        
        loss = fit_history.history['loss']
        val_loss = fit_history.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
