import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_config_info(base_run_dir):
    print('STARTING SASG TEST')
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters']
    runner_return_headder = config.executor['runner_return_headder']
    
    value_of_interest = config.general.get('value_of_interest', 'function value')
    simulation_name = config.general.get('simulation_name', 'Parent Model')
    return parameters, bounds, simulation_name, value_of_interest

def sobol_batch_post_proc(base_run_dir):
    
    parameters, bounds, simulation_name, value_of_interest = get_config_info()
    
    df = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    plt.plot(df['num_samples'], df['mean'], marker='o')
    plt.fill_between(df['num_samples'], df['mean'].to_numpy()-df['std'].to_numpy(), df['mean'].to_numpy()+df['std'].to_numpy(), color='grey',label='Standard Deviation')
    plt.legend()
    xlabel = f'Number of {simulation_name} Evaluations'
    ylabel = f'Expectation of {value_of_interest}'
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Mean')
    figure.tight_layout()
    figure.savefig(os.path.join(base_dir, 'mean_std.png'), dpi=200)
    
if __name__ == '__main__':
    import sys
    _, base_run_dir = sys.argv
    sobol_batch_post_proc(base_run_dir)