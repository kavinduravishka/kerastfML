import os
import sys
from pathlib import Path as gtpath

modpath = gtpath(os.getcwd()).parent

if modpath not in sys.path:
    sys.path.append(modpath)
