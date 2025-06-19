import os
import pandas as pd
import argparse
import re
import yaml

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

if __name__ == '__main__':
    x,y = get_points('/users/danieljordan/enchanted-surrogates/data_store/MMMG_sobolseq_testset/active_cycle_0')
    print(x,y)
    
