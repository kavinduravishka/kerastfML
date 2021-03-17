import os

modpath = os.getcwd() + '/'
datadirectory = modpath+'kvasir-dataset/'
batchdirectory = modpath+'databatch/'

from learn.learn import learner
from data.dirmanager import dirmanager
from cmd import Cmd

dmngr = dirmanager(datadirectory,batchdirectory)
mlearner = learner(batchdirectory)

