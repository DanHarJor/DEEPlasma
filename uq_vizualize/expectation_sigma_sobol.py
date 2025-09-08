print('PERFORMING IMPORTS')
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
import warnings
# Set font size globally
plt.rcParams.update({'font.size': 24})  # Adjust number as needed
print('IMPORTS FINISHED')

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
    random_means, random_vars, random_num_samples = get_random_comparison(base_run_dir)
    random_means = np.array(random_means)
    random_sigma = np.sqrt(np.array(random_vars))
    truest_mean = random_means[-1]    
    # test_x, test_y = get_test_set(base_run_dir)
    # truest_mean = np.nanmean(test_y)
    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    if 'do_surplus_based' in df_cycle_info.columns:
        do_surplus_based = df_cycle_info['do_surplus_based'].to_numpy()
        colors = np.where(do_surplus_based, 'orange', 'black')
    else:
        colors = np.repeat('black', len(df_cycle_info))
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    if config.sampler.get('parameters_labels') != None:
        parameters = config.sampler['parameters_labels']
        parameters = [fr"{p}" for p in parameters]
        print('PARAMETERS:',parameters)
    else:
        parameters = config.sampler['parameters']
    
    simulation_name=config.general.get('simulation_name', 'FUNCTION')
    value_of_interest=config.general.get('value_of_interest', 'VALUE')
    if xlabel==None:
        xlabel = f'Number of {simulation_name} Evaluations'
    if ylabel==None:
        ylabel = f'Expectated {value_of_interest}'
    
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'],df_cycle_info['quad_expectation'], '-o')
    plt.plot(random_num_samples,random_means, '-o')    
    # plt.scatter(df_cycle_info['num_samples'], df_cycle_info['quad_expectation'], c=colors)
    plt.hlines(truest_mean, 0, df_cycle_info['num_samples'].iloc[-1], color='black', label='Converged Growthrate')
    plt.fill_between(df_cycle_info['num_samples'], df_cycle_info['quad_expectation'].to_numpy().astype('float')-0.5*df_cycle_info['quad_double_sigma'].to_numpy().astype('float'),df_cycle_info['quad_expectation'].to_numpy().astype('float')+0.5*df_cycle_info['quad_double_sigma'].to_numpy().astype('float'), color='grey') 
    plt.fill_between(random_num_samples, random_means-random_sigma, random_means+random_sigma, color='orange', alpha=0.3) 
    plt.xlim(0, np.max(df_cycle_info['num_samples'].to_numpy()))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'only_quad_expectation.png'))
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'],np.abs(df_cycle_info['quad_expectation']-truest_mean), '-o')
    plt.plot(random_num_samples,np.abs(np.array(random_means)-truest_mean), '-o', label='Comparison, Random')    
    # plt.scatter(df_cycle_info['num_samples'], df_cycle_info['quad_expectation'], c=colors)
    # plt.hlines(truest_mean, 0, df_cycle_info['num_samples'].iloc[-1])
    # plt.fill_between(df_cycle_info['num_samples'], df_cycle_info['quad_expectation'].to_numpy().astype('float')-0.5*df_cycle_info['quad_double_sigma'].to_numpy().astype('float'),df_cycle_info['quad_expectation'].to_numpy().astype('float')+0.5*df_cycle_info['quad_double_sigma'].to_numpy().astype('float'), color='grey', label=r'Standard Deviation') 
    plt.xlabel(xlabel)
    plt.ylabel(f'Mean Error, {value_of_interest}')
    plt.legend()
    plt.xlim(0, np.max(df_cycle_info['num_samples'].to_numpy()))
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'only_quad_expectation_error.png'))
    plt.close(fig)

    
def plot_mse(base_run_dir):
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    if 'do_surplus_based' in df_cycle_info.columns:
        do_surplus_based = df_cycle_info['do_surplus_based'].to_numpy()
        colors = np.where(do_surplus_based, 'orange', 'black')
    else:
        colors = np.repeat('black', len(df_cycle_info))

    if not 'test_set_mse' in df_cycle_info.columns:
        return None
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'], df_cycle_info['test_set_mse'])
    plt.scatter(df_cycle_info['num_samples'], df_cycle_info['test_set_mse'], c=colors)
    fig.savefig(os.path.join(base_run_dir, 'MSE_cycle_info.png'))
    plt.close(fig)

def plot_surplus(base_run_dir):
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    if 'do_surplus_based' in df_cycle_info.columns:
        do_surplus_based = df_cycle_info['do_surplus_based'].to_numpy()
        colors = np.where(do_surplus_based, 'orange', 'black')
    else:
        colors = np.repeat('black', len(df_cycle_info))
    fig = plt.figure()
    if not 'mean_surplus' in df_cycle_info.columns:
        return None
    plt.plot(df_cycle_info['num_samples'], df_cycle_info['mean_surplus'])
    plt.scatter(df_cycle_info['num_samples'], df_cycle_info['mean_surplus'], c=colors)
    fig.tight_layout()
    plt.ylabel('Mean Surplus')
    # plt.yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'mean_surplus_cycle_info.png'))
    plt.close(fig)
    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'], df_cycle_info['max_surplus'])
    plt.scatter(df_cycle_info['num_samples'], df_cycle_info['max_surplus'],c=colors)
    plt.ylabel('Max Surplus')
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'max_surplus_cycle_info.png'))
    plt.close(fig)
    
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    plt.plot(df_cycle_info['num_samples'], df_cycle_info['mean_recent_surplus'])
    plt.scatter(df_cycle_info['num_samples'], df_cycle_info['mean_recent_surplus'],c=colors)
    plt.ylabel('Mean Cycle Surplus')
    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'mean_recent_surplus_cycle_info.png'))
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
    
    # SOBOL Bar CHART First Order
    print('PLOTTING SOBOL BAR CHART')
    if config.sampler.get('parameters_labels') != None:
        parameters = config.sampler['parameters_labels']
        parameters = [fr"{p}" for p in parameters]
        print(parameters)
    else:
        parameters = config.sampler['parameters']
    brute_sobol_columns = [col for col in df_cycle_info.columns if 'brute_sobol_first_order_' in col]
    order_args = np.flip(np.argsort([df_cycle_info[col].iloc[-1] for col in brute_sobol_columns]))
    # order = np.arange(len(parameters))[order_args]
    brute_sobol_columns = np.array(brute_sobol_columns)[order_args]
    parameters = np.array(parameters)[order_args]
    
    fig = plt.figure(figsize=(6.4,6))
    plt.title(f"N. {simulation_name} Eval: {df_cycle_info['num_samples'].to_numpy()[-1]}")
    plt.barh(parameters, df_cycle_info[brute_sobol_columns].iloc[-1,:].to_numpy(), color=colors, log=True)
    
    # for col, param, ord, color in zip(brute_sobol_columns, parameters, np.flip(np.arange(len(parameters))),colors):
    #     plt.plot(df_cycle_info['num_samples'], df_cycle_info[col], '-o', label=f'{ord}: '+param, color=color)
    #     param_to_color[param] = color

    fig.tight_layout()
    fig.savefig(os.path.join(base_run_dir, 'sobol_bar_last.png'))
    plt.close(fig)
def plot_cycle_info(base_run_dir):
    plot_expectation(base_run_dir)
    # plot_sigma(base_run_dir)
    # plot_sobols(base_run_dir)
    plot_mse(base_run_dir)
    plot_surplus(base_run_dir) 

def get_cycle_dirs(base_run_dir):
    listdir = os.listdir(base_run_dir)
    cycle_dirs = [d for d in listdir if 'active_cycle_' in d]
    
    ordinal = [int(d.split('_')[-1]) for d in cycle_dirs]
    # print('='*100,'debug ordinal', ordinal, np.argsort(ordinal))
    
    cycle_dirs = np.array(cycle_dirs)[np.argsort(ordinal)]
    cycle_dirs = [os.path.join(base_run_dir,cycle_dir) for cycle_dir in cycle_dirs]
    # print('debug cycle dirs',cycle_dirs)
    return cycle_dirs

def get_random_comparison(base_run_dir):
    print('RETRIVING RANDOM COMPARISON FROM', base_run_dir)
    listdir = os.listdir(base_run_dir)
    test_dir = None
    for di in listdir:
        if 'sobolseq' in di or 'comparison' in di:
            test_dir = os.path.join(base_run_dir,di)
            break
    if test_dir == None:
        raise FileNotFoundError('NO FOLDERS WITH sosbol_seq OR comparison IN NAME, THAT COULD CONTAIN A TEST SET WERE FOUND')
        
    cycle_dirs = get_cycle_dirs(test_dir)
    means = []
    num_samples = []
    output_values = []
    var = []
    for cycle_dir in cycle_dirs:
        df = pd.read_csv(os.path.join(cycle_dir,'runner_return.txt'))
        output_values = output_values + list(df[df.columns[-1]].to_numpy())
        mean = np.nanmean(output_values)
        var.append(np.nanvar(output_values))
        means.append(mean)
        num_samples.append(len(output_values))
    return means, var, num_samples 

def get_test_set(base_run_dir):
    print('RETRIVING TEST SET FROM', base_run_dir)
    listdir = os.listdir(base_run_dir)
    test_dir = None
    for di in listdir:
        if 'sobolseq' in di or 'testset' in di:
            test_dir = os.path.join(base_run_dir,di)
            break
    if test_dir == None:
        raise FileNotFoundError('NO FOLDERS WITH sosbol_seq OR testset IN NAME, THAT COULD CONTAIN A TEST SET WERE FOUND')
    print('FOUND TEST DIR', test_dir)
    
    file_path = os.path.join(test_dir,'merged_runner_return.csv')
    file_path_txt = os.path.join(test_dir, 'merged_runner_return.txt')
    if os.path.exists(file_path):        
        df_test = pd.read_csv(file_path)
        print('got merged_runner_return.csv', file_path)
    elif os.path.exists(file_path_txt):
        df_test = pd.read_csv(file_path_txt)
        print('got merged_runner_return.txt', file_path_txt)
    # elif os.path.exists(os.path.join(test_dir, 'runner_return.txt')):
    #     df_test = pd.read_csv(os.path.join(test_dir, 'runner_return.txt'))
    #     print('got runner_return.txt')    
    else:
        print('NO RUNNER RETURN FOUND, BEGINNIGN PARSING')
        finished_result = find_files(test_dir, 'GENE.finished')
        stopped_result = find_files(test_dir, 'stopped_by_monitor')
        result = finished_result + stopped_result
        run_dirs = [os.path.dirname(path) for path in result]
        if len(result) == 0:        
            raise FileNotFoundError(f'NO RUNNER RETURN PATH WAS FOUND IN:{test_dir},\n ALSO THERE SEEM TO BE NO FINNISHED OR EARLY STOPPED GENE RUNS IN: {test_dir}\n{file_path}\n{file_path_txt}')
        else:
            outputs = []
            for i, run_dir in enumerate(run_dirs):
                if i % 10 == 0:
                    print('NUMBER OF RUN_DIR PARSED:',i)
                outputs.append(parse_run_dir(run_dir, parameters))
            with open(os.path.join(test_dir, 'merged_runner_return.txt'), 'w') as file:
                lines = [runner_return_headder] + outputs
                lines = [line+'\n' for line in lines]
                file.writelines(lines)
            df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
            
    test_x = np.array(df_test.iloc[:,0:-1].astype('float'))
    # print('debug l tx', len(test_x))
    test_y = np.array(df_test.iloc[:,-1].astype('float'))
    
    # sasg = get_sasg(os.path.join(base_run_dir,'active_cycle_0'))
    # unit_test_x = sasg.points_transform_box2unit(test_x)
    # parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    # closest_boarder_distance = []
    # for tx in unit_test_x:
    #     lower = np.abs(tx-bounds.T[0])
    #     upper = np.abs(tx-bounds.T[1])
    #     closest_boarder_distance.append(np.min([lower,upper]))
    # closest_boarder_distance = np.array(closest_boarder_distance)
    test_x = test_x#[closest_boarder_distance>0.1]
    test_y = test_y#[closest_boarder_distance>0.1]
    if test_x.shape[1] <= 2:
        warnings.warn('test_x is 2D or less, this is probably not the correct dimensionality and can be caused by no runner_return_headder being at the top of the runner_return.txt file or in the configs file')
    return test_x, test_y


if __name__ == '__main__':
    _, base_run_dir = sys.argv
    # plot_expectation(base_run_dir)
    # plot_sigma(base_run_dir)
    plot_sobols(base_run_dir)
    # plot_mse(base_run_dir)
    # plot_surplus(base_run_dir)
    
    plot_cycle_info(base_run_dir)