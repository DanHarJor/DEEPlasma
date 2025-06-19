
# add current directory to PYTHONPATH such that pysgpp_swig can be imported
import os
import sys
sys.path.append(os.path.dirname(__file__))

# import pysgpp_swig and extensions
from .pysgpp_swig import *
from . import extensions
