import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np

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

def plot_expectation(base_run_dir):
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'],df_cycle_info['quad_expectation'])
    plt.xlabel('number of samples')
    plt.ylabel('Expectation via Sparse Grid Quadrature')
    fig.savefig(os.path.join(base_run_dir, 'only_quad_expectation.png'))
    plt.close(fig)

def plot_sigma(base_run_dir):
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'],df_cycle_info['quad_double_sigma'])
    plt.xlabel('number of samples')
    plt.ylabel('Double Sigma via Sparse Grid Quadrature')
    fig.savefig(os.path.join(base_run_dir, 'only_quad_double_sigma.png'))
    plt.close(fig)

def plot_sobols(base_run_dir, xlabel='Number of Function Evaluations'):
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    parameters = config.sampler['parameters']
    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    brute_sobol_columns = [col for col in df_cycle_info.columns if 'brute_sobol_total_order_' in col]
    fig = plt.figure()
    if len(brute_sobol_columns) > len(parameters):
        raise ValueError(print('There are more parameters specified in the config file than there are dimensions that the sobol indicies were calculated for.'))
    order_args = np.argsort([df_cycle_info[col].iloc[-1] for col in brute_sobol_columns])
    # order = np.arange(len(parameters))[order_args]
    brute_sobol_columns = np.array(brute_sobol_columns)[order_args]
    parameters = np.array(parameters)[order_args]
    for col, param, ord in zip(brute_sobol_columns, parameters, np.arange(len(parameters))):
        plt.plot(df_cycle_info['num_samples'], df_cycle_info[col], label=f'{ord} '+param)
    plt.ylabel('Total Sobol Indicies (brute force)')
    plt.xlabel(xlabel)
    plt.legend()
    fig.savefig(os.path.join(base_run_dir, 'brute_total_sobol.png'))
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    plot_expectation(base_run_dir)
    plot_sigma(base_run_dir)
    plot_sobols(base_run_dir)