import os
import pandas as pd 
import numpy as np
import yaml
import argparse
import warnings
import pysgpp
import matplotlib.pyplot as plt
import sys

sys.path.append('/users/danieljordan/enchanted-surrogates2/src')
# sys.path.append('/users/danieljordan/DEEPlasma')
from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids

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

def sasg_test(base_run_dir, cycle_num='all'):
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters_labels']
    
    value_of_interest = config.general.get('value_of_interest', 'function value')
    parent_model = config.general.get('simulation_name', 'Parent Model')
    
    sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    cycle_dirs = np.sort([d for d in listdir if 'active_cycle_' in d])
    if cycle_num=='latest':
        cycle_dirs = [cycle_dirs[-1]]
    elif type(cycle_num)==type(1):
        cycle_dirs = [f'active_cycle_{cycle_num}']

    # Get test points from parent function if they exist
    test_dir = None
    for di in listdir:
        if 'sobolseq' in di or 'testset' in di:
            test_dir = os.path.join(base_run_dir,di)
            break
        
    if test_dir == None:
        warnings.warn('NO FOLDERS WITH sosbol_seq OR testset IN NAME, THAT COULD CONTAIN A TEST SET WERE FOUND')
        return None
    
    if os.path.exists(os.path.join(test_dir,'runner_return.csv')):        
        df_test = pd.read_csv(os.path.join(test_dir,'runner_return.csv'))
    elif os.path.exists(os.path.join(test_dir, 'runner_return.txt')):
        df_test = pd.read_csv(os.path.join(test_dir, 'runner_return.txt'))
    else:
        raise FileNotFoundError('NO RUNNER RETURN PATH WAS FOUND IN:',test_dir)

    test_x = np.array(df_test.iloc[:,0:-1].astype('float'))
    test_y = np.array(df_test.iloc[:,-1].astype('float'))
    
    MAPE = []
    for cycle_dir in cycle_dirs:
        if not os.path.exists(os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')):
            pass
        grid_file_path = os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')
        surpluses_file_path = os.path.join(base_run_dir,cycle_dir, 'surpluses.mat')
        train_points_file = os.path.join(base_run_dir,cycle_dir, 'train_points.pkl')
        with open(grid_file_path, 'r') as file:
            serialized_grid = file.read()
            sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
            surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
            sasg.alpha = surpluses
        
        # eval = lambda point: sasg.surrogate_predict([tuple(point)])[0]
        eval_many = lambda points: sasg.surrogate_predict(points)
        
        test_pred = eval_many(test_x)
        residuals = test_y - test_pred
        
        fig = plt.figure()
        plt.hexbin(test_y, residuals, gridsize=50, cmap='plasma', bins='log')
        plt.ylabel(f'Residuals, {value_of_interest}')
        plt.xlabel(f'{parent_model}, {value_of_interest}')
        fig.savefig(os.path.join(base_run_dir, cycle_dir, 'residual_plot.png'))
        plt.close(fig)
        mape = np.mean(np.abs(residuals/test_y)*100)
        MAPE.append(mape)
    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'], MAPE, '-o')
    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Mean Average Percentage Error, {value_of_interest}')
    fig.savefig(os.path.join(base_run_dir, 'MAPE.png'))
    plt.close(fig)
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    sasg_test(base_run_dir)