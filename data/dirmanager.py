import os
import shutil

class dirmanager:
    
    def __init__(self,datadir,batchdir):
        self.datadir=datadir
        self.batchdir = batchdir
        
        self.subdirectories = os.listdir(self.datadir)
        
    def createdirs(self):
        self.subdirectories = os.listdir(self.datadir)
        
        if not os.path.isdir(self.batchdir+'train/'):
            for subdir in self.subdirectories:
                os.makedirs(self.batchdir+'train/'+subdir)
            
        if not os.path.isdir(self.batchdir+'valid/'):
            for subdir in self.subdirectories:
                os.makedirs(self.batchdir+'valid/'+subdir)
                
        if not os.path.isdir(self.batchdir+'test/'):
            for subdir in self.subdirectories:
                os.makedirs(self.batchdir+'test/'+subdir)
                
    
    #def movedata(self,)