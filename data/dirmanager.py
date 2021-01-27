import os
import shutil
import random

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
                
        return self
                
    
    def movedata(self,trainset,validset,testset):
        total=trainset+validset+testset
        for subdir in self.subdirectories:
            
            print('Moving data from >>> '+subdir)
            imageset = os.listdir(self.datadir + subdir)
            if len(imageset)<total:
                print('\tNo enough images in the directory')
                continue
            else:
                random.shuffle(imageset)
                
                for image in imageset[0:trainset]:
                    shutil.move(self.datadir + subdir + '/' + image , self.batchdir+ 'train/' + subdir)
                    print("\tMoving : "+image,end='\r')
                
                for image in imageset[trainset:trainset+validset]:
                    shutil.move(self.datadir + subdir + '/' + image , self.batchdir+ 'valid/' + subdir)
                    print("\tMoving : "+image,end='\r')
                    
                for image in imageset[total-testset:total]:
                    shutil.move(self.datadir + subdir + '/' + image , self.batchdir+ 'test/' + subdir)
                    print("\tMoving : "+image,end='\r')
                    
        return self
    
    
    #def moveback(self):
        