import os
import sys
from pathlib import Path as gtpath

modpath = str(gtpath(os.getcwd()).parent) +'/'

if modpath not in sys.path:
    sys.path.append(modpath)

datadirectory=modpath+'kvasir-dataset/'