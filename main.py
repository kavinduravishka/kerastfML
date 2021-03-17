import os

modpath = os.getcwd() + '/'
datadirectory = modpath+'kvasir-dataset/'
batchdirectory = modpath+'databatch/'

from learn.learn import learner
from data.dirmanager import dirmanager
from cmd import Cmd

dmngr = dirmanager(datadirectory,batchdirectory)
learner = learner(batchdirectory)

class kerasMLcli(Cmd):
    prompt = 'Keras ML > '
    intro = "Welcome! Type ? to list commands"
 
    def do_exit(self, inp):
        print("Bye")
        return True
    
    def help_exit(self):
        print('exit the application. Shorthand: x q Ctrl-D.')
 
    
    def do_organize(self,inp):
        inp = list(eval(inp))
        dmngr.createdirs()
        dmngr.movedata(inp[0],inp[1],inp[2])
 
    def help_organize(self):
        print("organize kwasir dataset\n\tusage :  organize (<train batch size>,<validation batch size>,<test batch size>")



    def do_newmodel(self,inp):
        learner.newModal()
        
    def do_loadmodel(self,inp):
        learner.loadModel(inp)
        
    def do_savemodel(self,inp):
        learner.saveModel(inp)
        
    def do_trainmodel(self,inp):
        learner.trainModel()
        
    def do_testmodel(self,inp):
        learner.testmodel()
        
    def do_listmodel(self,inp):
        learner.listModel()
        
    def do_summary(self,inp):
        learner.cnn.summary()
        
    
    def help_newmodel(self):
        print('init new CNN model')
        
    def help_loadmodel(self):
        print('Load from saved models\n\tusage: loadmodel <model name>')
        
    def help_savemodel(self):
        print('Save trained model with given name\n\tusage: savemodel <model name>')
        
    def help_trainmodel(self):
        print('Train CNN')
        
    def help_testmodel(self):
        print('Test final outcome of trained CNN')
        
    def help_listmodel(self):
        print('List saved models')
        
    def help_summary(self,inp):
        print('Print summary of currently loaded CNN')
        
    
    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)
 
        print("No command like: {}".format(inp))
 
    #do_EOF = do_exit
    #help_EOF = help_exit
 
shell = kerasMLcli()
shell.cmdloop()
