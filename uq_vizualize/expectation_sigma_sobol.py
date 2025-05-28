import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
import warnings
# Set font size globally
plt.rcParams.update({'font.size': 18})  # Adjust number as needed

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

def plot_expectation(base_run_dir, xlabel=None, ylabel=None):
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    if config.sampler.get('parameters_labels') != None:
        parameters = config.sampler['parameters_labels']
        parameters = [fr"{p}" for p in parameters]
        print(parameters)
    else:
        parameters = config.sampler['parameters']
    
    simulation_name=config.general.get('simulation_name', 'FUNCTION')
    value_of_interest=config.general.get('value_of_interest', 'VALUE')
    if xlabel==None:
        xlabel = f'Number of {simulation_name} Evaluations'
    if ylabel==None:
        ylabel = f'Expectation of {value_of_interest}'
    
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'],df_cycle_info['quad_expectation'], '-o')
    plt.fill_between(df_cycle_info['num_samples'], df_cycle_info['quad_expectation'].to_numpy().astype('float')-df_cycle_info['quad_double_sigma'].to_numpy().astype('float'),df_cycle_info['quad_expectation'].to_numpy().astype('float')+df_cycle_info['quad_double_sigma'].to_numpy().astype('float'), color='grey', label=r'2 Standard Deviations') 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'only_quad_expectation.png'))
    plt.close(fig)

# def plot_sigma(base_run_dir):
#     df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
#     fig = plt.figure()
#     plt.plot(df_cycle_info['num_samples'],df_cycle_info['quad_double_sigma'])
#     plt.xlabel('number of samples')
#     plt.ylabel('Double Sigma via Sparse Grid Quadrature')
#     fig.savefig(os.path.join(base_run_dir, 'only_quad_double_sigma.png'))
#     plt.close(fig)

def plot_sobols(base_run_dir, xlabel=None):
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    if config.sampler.get('parameters_labels') != None:
        parameters = config.sampler['parameters_labels']
        parameters = [fr"{p}" for p in parameters]
        print(parameters)
    else:
        parameters = config.sampler['parameters']
    
    simulation_name=config.general.get('simulation_name', 'FUNCTION')
    value_of_interest=config.general.get('value_of_interest', 'VALUE')
    if xlabel==None:
        xlabel = f'Number of {simulation_name} Evaluations'
    # if ylabel==None:
    #     ylabel = f'Expectation of {value_of_interest}'    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    
    # FIRST ORDER SOBOL INDICIES
    brute_sobol_columns = [col for col in df_cycle_info.columns if 'brute_sobol_first_order_' in col]
    if len(brute_sobol_columns) == 0:
        warnings.warn('THERE IS NO brute_sobol_first_order_ COLUMNS IN all_cycle_info.csv, IF YOU WANT THIS PLOT THEN ADD do_brute_force_sobol_indicies: True TO SpatiallyAdaptiveSparseGrids SAMPLER IN CONFIG FILE')

    fig = plt.figure()
    if len(brute_sobol_columns) > len(parameters):
        raise ValueError(print('There are more parameters specified in the config file than there are dimensions that the sobol indicies were calculated for.'))
    order_args = np.flip(np.argsort([df_cycle_info[col].iloc[-1] for col in brute_sobol_columns]))
    # order = np.arange(len(parameters))[order_args]
    brute_sobol_columns = np.array(brute_sobol_columns)[order_args]
    parameters = np.array(parameters)[order_args]
    
    # Define a new color cycle with a specific number of colors
    # num_colors = len(brute_sobol_columns)  # Set the number of colors you need
    # colors = [plt.cm.plasma(i / (num_colors - 1)) for i in range(num_colors)]
    colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#f781bf", "#a65628"]
    
    param_to_color = {}
    for col, param, ord, color in zip(brute_sobol_columns, parameters, np.flip(np.arange(len(parameters))),colors):
        plt.plot(df_cycle_info['num_samples'], df_cycle_info[col], '-o', label=f'{ord}: '+param, color=color)
        param_to_color[param] = color
    plt.ylabel('First Order Sobol Indicies')
    plt.yscale('log')
    plt.xlabel(xlabel)
    # plt.legend(facecolor='white')
    # plt.legend(loc="right", ncol=1, handletextpad=0.2, borderpad=0.1, bbox_to_anchor=(1, 0))
    # Move legend fully outside the plot to the right
    # plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=1)

    # Ensure enough space for the legend
    # fig.subplots_adjust(right=0.8)  # Adjust to allow space for the legend

    fig.tight_layout()
    plt.legend(fontsize=13, loc='center left')
    fig.savefig(os.path.join(base_run_dir, 'brute_first_sobol.png'))
    plt.legend(facecolor='white', ncol=2, handletextpad=0.1, borderpad=2, framealpha=1, loc='center')
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'brute_first_sobol_with_legend.png'))
    plt.close(fig)
    
    #TOTAL ORDER SOBOL INDICIES
    brute_sobol_columns = [col for col in df_cycle_info.columns if 'brute_sobol_total_order_' in col]
    if len(brute_sobol_columns) == 0:
        warnings.warn('THERE IS NO brute_sobol_total_order_ COLUMNS IN all_cycle_info.csv, IF YOU WANT THIS PLOT THEN ADD do_brute_force_sobol_indicies: True TO SpatiallyAdaptiveSparseGrids SAMPLER IN CONFIG FILE')

    fig = plt.figure()
    if len(brute_sobol_columns) > len(parameters):
        raise ValueError(print('There are more parameters specified in the config file than there are dimensions that the sobol indicies were calculated for.'))
    order_args = np.flip(np.argsort([df_cycle_info[col].iloc[-1] for col in brute_sobol_columns]))
    # order = np.arange(len(parameters))[order_args]
    brute_sobol_columns = np.array(brute_sobol_columns)[order_args]
    parameters = np.array(parameters)[order_args]
    for col, param, ord in zip(brute_sobol_columns, parameters, np.flip(np.arange(len(parameters)))):
        color = param_to_color[param]
        plt.plot(df_cycle_info['num_samples'], df_cycle_info[col], '-o', label=f'{ord}: '+param, color=color)
    plt.ylabel('Total Order Sobol Indicies')
    plt.yscale('log')
    plt.xlabel(xlabel)
    # plt.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.15), handletextpad=0.5, borderpad=0.2)
    plt.legend(fontsize=13, loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'brute_total_sobol.png'))
    plt.close(fig)
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    plot_expectation(base_run_dir)
    # plot_sigma(base_run_dir)
    plot_sobols(base_run_dir)