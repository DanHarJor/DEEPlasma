#imports
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'enchanted-surrogates','src'))
sys.path.append(os.path.join(os.getcwd(),'GENE_UQ'))
from parsers import GENEparser
from GENE_UQ.Samplers import uniform

if __name__ == '__main__':
    print('hello')
    
    
    parser = GENEparser

