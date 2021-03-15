from model.cnn.cnn import convolutional_cnn
from data.batch import batch
import os

from tensorflow.keras.optimizers import Adam

class learner():
    def __init__(self,batchdirectory):
        self.traindir = batchdirectory+'train/'
        self.valdir = batchdirectory+'valid/'

        self.testdir = batchdirectory+'test/'
        self.modeldirectory = "../model/saved/models/"
        
        
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
        trainbatch.getFromDir(self.testdir)
        return trainbatch
    
    def NewModal(self):
        self.cnn = convolutional_cnn()
        return self
    
    def trainModal(self):
        self.cnn.compilemodel(Adam,0.01,'categorical_crossentropy',['accuracy'])
        self.cnn.fit(self.get_trainbatch(),self.get_validbatch())
    
    def loadModel(self):
        pass
    
    def saveModel(self):
        pass
    
    def listModel(self):
        modellist = os.listdir(self.modeldirectory)
        return modellist