import pysgpp 
import pickle
import numpy as np
import pysgpp
import sys, os
sys.path.append('/users/danieljordan/enchanted-surrogates2/src')
# sys.path.append('/users/danieljordan/DEEPlasma')
from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
from highD_visualise import plot_matrix_contour, plot_slices
import argparse
import os
import yaml
import matplotlib.pyplot as plt

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

def vis_enchanted_sasg(base_run_dir, cycle_num='all'):
    '''
    cycle_num, int, str: The active cycle to perform plot for, can also be a string, 'latest' or 'all'
    '''
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters']

    sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    cycle_dirs = np.sort([d for d in listdir if 'active_cycle_' in d])
    if cycle_num=='latest':
        cycle_dir = cycle_dirs[-1]
    elif type(cycle_num)==type(1):
        cycle_dir = f'active_cycle_{cycle_num}'

    if cycle_num != 'all':
        cycle_dirs = [cycle_dir]

    for cycle_dir in cycle_dirs:
        grid_file_path = os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')
        surpluses_file_path = os.path.join(base_run_dir,cycle_dir, 'surpluses.mat')
        train_points_file = os.path.join(base_run_dir,cycle_dir, 'train_points.pkl')
        with open(grid_file_path, 'r') as file:
            serialized_grid = file.read()
            sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
            surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
            sasg.alpha = surpluses
        
        eval = lambda point: sasg.surrogate_predict([tuple(point)])[0]
        eval_many = lambda points: sasg.surrogate_predict(points)
        # eval = create_eval(grid_file_path, surpluses_file_path)
        # eval_many = create_eval_many(grid_file_path, surpluses_file_path)
        
        with open(train_points_file, 'rb') as file:
            train_points_dict = pickle.load(file)
        train_points = np.array(list(train_points_dict.keys()))
        
        # train_unit_points = sasg.points_transform_box2unit(train_points)
        fig_contours = plot_matrix_contour(function=eval, bounds=bounds, dimension_labels=parameters, points=train_points)
        fig_slices = plot_slices(function=eval_many, bounds=bounds, dimension_labels=parameters)
        fig_contours.savefig(os.path.join(base_run_dir, cycle_dir, 'contour_plots.png'))
        fig_slices.savefig(os.path.join(base_run_dir, cycle_dir, 'slice_plots.png'))
        plt.close(fig_contours)
        plt.close(fig_slices)

# def vis_enchanted_sasg_unit(base_run_dir, cycle_num='latest'):
#     listdir = os.listdir(base_run_dir)
#     config_file_name = [name for name in listdir if '.yaml' in name]
#     if len(config_file_name) > 1:
#         raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
#     config_file_name = config_file_name[0]
#     config = load_configuration(os.path.join(base_run_dir, config_file_name))
#     bounds=np.array(config.sampler['bounds'])
#     # print(bounds.shape, bounds)
#     unit_bounds=np.array([(0,1)]*len(bounds))
#     # print(bounds.shape, bounds)
#     parameters = config.sampler['parameters']

#     sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    
#     if cycle_num=='latest':
#         cycle_dirs = np.sort([d for d in listdir if 'active_cycle_' in d])
#         cycle_dir = cycle_dirs[-1]
#     else:
#         cycle_dir = f'active_cycle_{cycle_num}'
#     grid_file_path = os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')
#     surpluses_file_path = os.path.join(base_run_dir,cycle_dir, 'surpluses.mat')
#     train_points_file = os.path.join(base_run_dir,cycle_dir, 'train_points.pkl')
#     with open(grid_file_path, 'r') as file:
#       serialized_grid = file.read()
#     sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
#     surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
#     sasg.alpha = surpluses
    
#     # def eval(point):
#     #     sasg.surrogate_predict([])
    
#     # eval = lambda point: sasg.surrogate_predict([tuple(point)])[0]
#     # eval_many = lambda points: sasg.surrogate_predict(points)
#     eval = create_eval(grid_file_path, surpluses_file_path)
#     eval_many = create_eval_many(grid_file_path, surpluses_file_path)
    
#     with open(train_points_file, 'rb') as file:
#         train_points_dict = pickle.load(file)
#     train_points = np.array(list(train_points_dict.keys()))
#     train_points = sasg.points_transform_box2unit(train_points)
    
#     # train_unit_points = sasg.points_transform_box2unit(train_points)
#     fig_contours = plot_matrix_contour(function=eval, bounds=unit_bounds, dimension_labels=parameters, points=train_points)
#     fig_slices = plot_slices(function=eval_many, bounds=bounds, dimension_labels=parameters)
#     fig_contours.savefig(os.path.join(base_run_dir, 'contour_plots_unit.png'))
#     # fig_slices.savefig(os.path.join(base_run_dir, 'slice_plots.png'))

# def create_eval(grid_file, surpluses_file_path):
#     with open(grid_file, 'r') as file:
#       serialized_grid = file.read()
#     grid = pysgpp.Grid.unserialize(serialized_grid)
#     surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
#     def eval(pos):
#         opEval = pysgpp.createOperationEval(grid)
#         pos_v = pysgpp.DataVector(len(pos))
#         for i, p in enumerate(pos):
#             pos_v[i] = float(np.round(p,3))
#         return opEval.eval(surpluses, pos_v)
#     return eval

# def create_eval_many(grid_file, surpluses_file_path):
#     with open(grid_file, 'r') as file:
#       serialized_grid = file.read()
#     grid = pysgpp.Grid.unserialize(serialized_grid)
#     surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
#     def eval_many(positions):
#         positions = np.array(positions)
#         positions_dm = pysgpp.DataMatrix(positions)
#         opEval = pysgpp.createOperationMultipleEval(grid, positions_dm)
#         results = pysgpp.DataVector(len(positions))
#         opEval.eval(surpluses, results)
#         ans = np.array(results.array())
#         # ans = np.array([results.get(i) for i in range(len(positions))])
#         return ans
#     return eval_many


if __name__ == '__main__':
    _, base_run_dir = sys.argv
    vis_enchanted_sasg(base_run_dir)
    
    
# bounds=[[4,6.7], [2.1,3.5], [0.16,2.9]]
# parameters=['omt1','omt2','omn']

# sasg = SpatiallyAdaptiveSparseGrids(bounds=bounds, #[[0.488e-3,0.587e-3], [0.641e-2,0.867e-2], [1.156,1.927], [0.610, 0.670], [4.040, 6.733], [1.280,1.920], [2.170,2.399], [1.992, 2.435], [0.710, 0.870]], 
#                                     parameters=parameters)# ['beta','coll','omn', 'temp1', 'omt2', 'zeff', 'q0', 'shat', 'delta']) 


# dirs = os.listdir('./')
# cycle_dirs = np.sort([d for d in dirs if 'active_cycle_' in d])
# grid_file_path = os.path.join(cycle_dirs[-1], 'pysgpp_grid.txt')
# surpluses_file_path = os.path.join(cycle_dirs[-1], 'surpluses.mat')
# train_points_file = os.path.join(cycle_dirs[-1], 'train_points.pkl')

# eval = create_eval(grid_file_path, surpluses_file_path)
# eval_many = create_eval_many(grid_file_path, surpluses_file_path)
# # cycle_numbers = [d[13:] for d in cycle_dirs]
# train_points_file = os.path.join(np.sort(cycle_dirs)[-1], 'train_points.pkl')
# with open(train_points_file, 'rb') as file:
#     train_points_dict = pickle.load(file)

# train_points = np.array(list(train_points_dict.keys()))
# train_unit_points = sasg.points_transform_box2unit(train_points)

# mmg.plot_matrix_contour(function=eval, points=train_unit_points, dimension_labels = parameters)
# mmg.plot_slices(function=eval_many)