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
sys.path.append('/users/danieljordan/enchanted_plugins/enchanted-surrogates/src/')
# sys.path.append('/users/danieljordan/DEEPlasma')
print('l')
print('m')

import re

def sort_strings_by_number(strings):
    def extract_number(s):
        match = re.search(r'\d+', s)
        if match:
            num = int(match.group())
            return num if num != 0 else None  # Exclude '00' or '0'
        return None  # Exclude strings with no number

    # Filter out invalid entries
    filtered = [s for s in strings if extract_number(s) is not None]

    # Sort by extracted number
    return sorted(filtered, key=lambda s: extract_number(s))


def get_cycle_dirs(base_run_dir, cycle_num=None):
    listdir = os.listdir(base_run_dir)
    cycle_dirs = [d for d in listdir if 'active_cycle_' in d or 'batch_' in d]
    cycle_dirs = sort_strings_by_number(cycle_dirs)
    cycle_dirs = [os.path.join(base_run_dir, c) for c in cycle_dirs]
    
    if cycle_num:
        if cycle_num=='latest':
            cycle_dirs = [cycle_dirs[-1]]
        elif type(cycle_num)==type(1):
            # cycle_dirs = [f'active_cycle_{cycle_num}']
            cycle_dirs = cycle_dirs[-cycle_num:]
        elif 'every' in cycle_num:
            every_int = int(cycle_num.split('-')[-1])
            cycle_dirs = [cd for i,cd in enumerate(cycle_dirs) if i%every_int==0]
        if type(cycle_num)==type([]):
            cycle_dirs = cycle_dirs[cycle_num]
    
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

def get_test_points(test_dir):
    cycle_dirs = get_cycle_dirs(test_dir)
    x, y = get_points_upto(cycle_dirs[-1])
    return x,y

def get_test_points_brd(base_run_dir):
    config = get_config(base_run_dir)
    test_dir = config.sampler['test_dir']
    cycle_dirs = get_cycle_dirs(test_dir)
    print('debug get test points cycle_dirs:', cycle_dirs)
    x, y = get_points_upto(cycle_dirs[-1])
    return x,y
    
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
    # config.executor["config_filepath"] = config_path
    return config

def get_MMMGrunner(base_run_dir):
    listdir = os.listdir(base_run_dir)
    from runners.MMMGrunner import MMMGrunner
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    runner_args = config.executor['static_executor']['runner']
    runner = MMMGrunner(**runner_args)
    return runner

def get_sasg(cycle_dir, name=''):
    # import samplers
    from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
    from enchanted_surrogates.samplers.sgpp_sampler import SgppSampler
    import pysgpp

    # print('GETTING sasg FROM:', cycle_dir)
    base_run_dir = os.path.dirname(cycle_dir)
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    # bounds=np.array(config.sampler['bounds'])
    # parameters = config.sampler['parameters']
    
    try:
        sasg = SpatiallyAdaptiveSparseGrids(**config.sampler)
    except Exception as e:
        print('got exception:',e)
        sasg = SgppSampler(**config.executor_kwargs['sampler_kwargs'])
    
    grid_file_path = os.path.join(cycle_dir, name+'pysgpp_grid.txt')
    surpluses_file_path = os.path.join(cycle_dir, name+'surpluses.mat')
    train_points_file = os.path.join(cycle_dir, name+'train_points.pkl')
    virtual_boundary_points_file = os.path.join(cycle_dir, name+'virtual_boundary_points.pkl')
    anchor_boundary_points_file = os.path.join(cycle_dir, name+'anchor_boundary_points.pkl')
    
    with open(grid_file_path, 'r') as file:
        serialized_grid = file.read()
        sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
        sasg.gridStorage = sasg.grid.getStorage()
        sasg.gridGen = sasg.grid.getGenerator()
        surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
        sasg.alpha = surpluses
    with open(train_points_file, 'rb') as file:
        sasg.train = pickle.load(file)
    try:
        with open(virtual_boundary_points_file, 'rb') as file:
            sasg.virtual_boundary_points = pickle.load(file)
    except Exception as e:
        print('error found\n',e)
    try:
        with open(anchor_boundary_points_file, 'rb') as file:
            sasg.anchor_boundary_points = pickle.load(file)
    except Exception as e:
        print('error found\n',e)
    print('debug get sasg, train size:', len(sasg.train))
    return sasg

def get_sasg_zero_bounds(cycle_dir):
    from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
    import pysgpp

    sasg = get_sasg(cycle_dir) 
    new_grid = pysgpp.Grid.createLinearGrid(sasg.dim)
    new_grid_storage = new_grid.getStorage()
    # Transfer points from old grid to new grid
    new_alpha = pysgpp.DataVector(sasg.gridStorage.getSize())
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        new_grid_storage.insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        box_point = sasg.point_transform_unit2box(unit_point) 
        new_alpha[i] = sasg.train[box_point]
    pysgpp.createOperationHierarchisation(new_grid).doHierarchisation(new_alpha)
    sasg.grid = new_grid
    sasg.alpha = new_alpha
    print('sasg degree',sasg.grid.getDegree())
    return sasg

def get_sasg_zero_poly_grid(cycle_dir, degree=4):
    from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
    import pysgpp

    sasg = get_sasg(cycle_dir) 
    new_grid = pysgpp.Grid.createPolyGrid(sasg.dim, degree)
    new_grid_storage = new_grid.getStorage()
    # Transfer points from old grid to new grid
    new_alpha = pysgpp.DataVector(sasg.gridStorage.getSize())
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        new_grid_storage.insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        box_point = sasg.point_transform_unit2box(unit_point) 
        new_alpha[i] = sasg.train[box_point]
    pysgpp.createOperationHierarchisation(new_grid).doHierarchisation(new_alpha)
    sasg.grid = new_grid
    sasg.alpha = new_alpha
    print('sasg degree',sasg.grid.getDegree())
    return sasg

def get_weighted_mean(cycle_dir):
    from sklearn.neighbors import KernelDensity

    samples_x, samples_fx = get_points_upto(cycle_dir)
    # sasg = eval(f"{sasg_type}(cycle_dir)")
    # samples_x = np.array(list(sasg.train.keys()))
    # samples_fx = np.array(list(sasg.train.values()))
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(samples_x)
    samples_qx = np.exp(kde.score_samples(samples_x))
    px=1 # for uniform distribution
    weighted_mean = ( np.sum(samples_fx * (px/samples_qx)) / sum((px/samples_qx)) )
    return weighted_mean

def get_sasg_anchor(cycle_dir):
    level=1
    anchor=None
    sasg = get_sasg(cycle_dir)
    base_run_dir = os.path.dirname(cycle_dir)
    MMMGrunner = get_MMMGrunner(base_run_dir)
    function=MMMGrunner.mmg.evaluate
    sasg.add_boundary_anchors(level, anchor, function)
    return sasg

def get_sasg_mod_poly_grid(cycle_dir, degree=4):
    from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
    import pysgpp

    sasg = get_sasg(cycle_dir) 
    new_grid = pysgpp.Grid.createModPolyGrid(sasg.dim, degree)
    new_grid_storage = new_grid.getStorage()
    # Transfer points from old grid to new grid
    new_alpha = pysgpp.DataVector(sasg.gridStorage.getSize())
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        new_grid_storage.insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        box_point = sasg.point_transform_unit2box(unit_point) 
        new_alpha[i] = sasg.train[box_point]
    pysgpp.createOperationHierarchisation(new_grid).doHierarchisation(new_alpha)
    sasg.grid = new_grid
    sasg.alpha = new_alpha
    print('sasg degree',sasg.grid.getDegree())
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

def get_base_run_dir(config_file):
    config = load_configuration(config_file)
    base_run_dir = config.executor['base_run_dir']
    return base_run_dir

if __name__ == '__main__':
    x,y = get_points('/users/danieljordan/enchanted-surrogates/data_store/MMMG_sobolseq_testset/active_cycle_0')
    print(x,y)
    
