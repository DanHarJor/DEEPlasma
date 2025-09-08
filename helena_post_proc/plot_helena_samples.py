import sys
sys.path.append('/users/danieljordan/enchanted-surrogates/src')

# sys.path.append('/users/danieljordan/DEEPlasma')
# from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
from parsers.HELENAparser import HELENAparser


import os
import uuid
import matplotlib.pyplot as plt
import numpy as np
# assumes uuid hel dires

def is_valid_uuid(val):
    try:
        uuid_obj = uuid.UUID(val, version=4)
    except ValueError:
        return False      
    return str(uuid_obj) == val

def get_hel_dirs(base_run_dir):
    all_dirs = os.listdir(base_run_dir)
    
    hel_dirs = []
    for dir_ in all_dirs:
        if is_valid_uuid(dir_):
            hel_dirs.append(os.path.join(base_run_dir, dir_))
    return hel_dirs

def plot_helena_samples(base_run_dir, comparison_hel_run):
    hel_dirs = get_hel_dirs(base_run_dir)
    hel_dirs.append(comparison_hel_run)
    parser = HELENAparser()
    
    fig = plt.figure()
    
    
    for i, hel_dir in enumerate(hel_dirs):
        print('looking at hel dir', hel_dir)
        print(i,'out of', len(hel_dirs))
        
        psi, ne, Te = parser.get_europed_profiles(hel_dir, include_psi=True)
        pressure = np.array(ne) * np.array(Te)
        
        if i == len(hel_dirs)-1:
            plt.plot(psi,pressure, color='green', linewidth=3)
        else:
            plt.plot(psi,pressure, color='grey', alpha=0.7)
                    
        plt.ylabel('ne * Te')
        plt.xlabel('psi')
        
    fig.savefig(os.path.join(base_run_dir, 'helena_pressure_profiles2.png'))

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    comparison_hel_run = "/scratch/project_462000954/jet_97781_data/helena_run_97781_aaro"
    plot_helena_samples(base_run_dir, comparison_hel_run)