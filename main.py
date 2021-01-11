import os
import sys
from pathlib import Path as gtpath

modpath = os.getcwd() + '/'

if modpath not in sys.path:
    sys.path.append(modpath)

datadirectory=modpath+'kvasir-dataset/'