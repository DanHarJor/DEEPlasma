print('a')
import os
print('b')
import pandas as pd
print('c')
import argparse
print('d')
import re
print('e')
import yaml
print('f')
import pickle
print('g')
print('h')
import numpy as np
print('i')
import sys
print('j')
import importlib
print('k')
sys.path.append('/users/danieljordan/enchanted-surrogates/src')
# sys.path.append('/users/danieljordan/DEEPlasma')
print('l')
print('m')
import samplers

def sort_strings_by_number(strings):
    return sorted(strings, key=lambda s: int(re.search(r'\d+', s).group()))

def get_cycle_dirs(base_run_dir):
    listdir = os.listdir(base_run_dir)
    cycle_dirs = [d for d in listdir if 'active_cycle_' in d]
    cycle_dirs = sort_strings_by_number(cycle_dirs)
    cycle_dirs = [os.path.join(base_run_dir, c) for c in cycle_dirs]
    return cycle_dirs

def get_points(cycle_dir):
    df = pd.read_csv(os.path.join(cycle_dir,'runner_return.txt'))
    x = df.iloc[:,:-1].to_numpy()
    y = df[df.columns[-1]].to_numpy()
    return x, y

def get_points_upto(cycle_dir):
    cycle_dirs = get_cycle_dirs(os.path.dirname(cycle_dir))
    x_all, y_all = [],[]
    for cd in cycle_dirs:
        x, y = get_points(cd)
        x_all.append(x)
        y_all.append(y)
        if cd == cycle_dir:
            break
    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)
    
def get_parameters_bounds(base_run_dir):
    listdir = os.listdir(base_run_dir)
    config_files = [f for f in listdir if '.yaml' in f]
    if len(config_files)>1:
        raise ValueError('more than one config found in:', base_run_dir)
    if len(config_files) == 0:
        raise ValueError('no .yaml config files were found in:',base_run_dir)
    config_path = os.path.join(base_run_dir, config_files[0])
    config = load_configuration(config_path)
    return config.sampler['parameters'], config.sampler['bounds']

def get_config(base_run_dir):
    listdir = os.listdir(base_run_dir)
    config_files = [f for f in listdir if '.yaml' in f]
    if len(config_files)>1:
        raise ValueError('more than one config found in:', base_run_dir)
    if len(config_files) == 0:
        raise ValueError('no .yaml config files were found in:',base_run_dir)
    config_path = os.path.join(base_run_dir, config_files[0])
    config = load_configuration(config_path)
    return config

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    print('LOADING CONFIGURATION FILE')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.executor["config_filepath"] = config_path
    return config

def get_sasg(cycle_dir):
    from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
    import pysgpp

    # print('GETTING sasg FROM:', cycle_dir)
    base_run_dir = os.path.dirname(cycle_dir)
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters']
    
    sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    
    grid_file_path = os.path.join(cycle_dir, 'pysgpp_grid.txt')
    surpluses_file_path = os.path.join(cycle_dir, 'surpluses.mat')
    train_points_file = os.path.join(cycle_dir, 'train_points.pkl')
    with open(grid_file_path, 'r') as file:
        serialized_grid = file.read()
        sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
        sasg.gridStorage = sasg.grid.getStorage()
        sasg.gridGen = sasg.grid.getGenerator()
        surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
        sasg.alpha = surpluses
    with open(train_points_file, 'rb') as file:
        sasg.train = pickle.load(file)
    return sasg

def get_sampler(base_run_dir):
    # print('GETTING sasg FROM:', cycle_dir)
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    
    sampler = getattr(importlib.import_module(f"samplers.{config.sampler['type']}"),config.sampler['type'])(**config.sampler)
    return sampler

if __name__ == '__main__':
    x,y = get_points('/users/danieljordan/enchanted-surrogates/data_store/MMMG_sobolseq_testset/active_cycle_0')
    print(x,y)
    
