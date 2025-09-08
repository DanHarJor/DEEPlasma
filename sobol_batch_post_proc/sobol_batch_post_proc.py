import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/users/danieljordan/DEEPlasma/sasg_post_proc')
from tools import get_cycle_dirs, get_points, get_parameters_bounds
import os, sys
from scipy.stats import sobol_indices, uniform
import numpy as np
pfd = os.path.dirname(__file__)
#base_run_dir should be for a random dataset like batch sobol sequence
def post_cycle_info(base_run_dir, do_sensitivity=True):
    print('STARTING RANDOM COMPARISON')
    cycle_dirs = get_cycle_dirs(base_run_dir)
    
    parameters, bounds = get_parameters_bounds(base_run_dir)
    num_samples = 0
    y_all = np.array([])
    x_all_list = []
    dfs = []
    for cycle_dir in cycle_dirs:
        print('LOOKING AT CYCLE DIR:',cycle_dir,'OUT OF:', len(cycle_dirs))
        x,y = get_points(cycle_dir)
        nan_mask = ~np.isnan(y)
        x,y = x[nan_mask], y[nan_mask]
        x_all_list.append(x)
        x_all = np.vstack(x_all_list)
        y_all = np.append(y_all, y)
        model = HistGradientBoostingRegressor()
        print('FITTING MODEL')
        model.fit(x_all,y_all)
        dists = []
        for b in bounds:
            assert b[1] > b[0]
            dists.append(uniform(loc=b[0], scale=b[1]-b[0]))
        func = lambda x: model.predict(x.T)
        mean=np.nanmean(y_all)
        std = np.sqrt(np.nanvar(y_all))
        num_samples = len(y_all)
        df = pd.DataFrame({'num_samples':[num_samples],'mean':[mean],'std':[std]})
        if do_sensitivity:
            print('COMPUTING SOBOL INDICIES')
            sobol = sobol_indices(func=func,  n=2**15, dists=dists)
            sobol_first_order, sobol_total_order = sobol.first_order, sobol.total_order
            for d, sfo in enumerate(sobol_first_order):
                df[f'brute_sobol_first_order_{d}'] = [sfo]
            for d, sto in enumerate(sobol_total_order):
                df[f'brute_sobol_total_order_{d}']= [sto]
        dfs.append(df)
        df.to_csv(os.path.join(cycle_dir,'post_cycle_info.csv'))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(base_run_dir, 'all_post_cycle_info.csv'))
    
    
def plot_post_cycle_info(base_run_dir):
    print('DOING PLOT COMPARISON')
    parameters, bounds = get_parameters_bounds(base_run_dir)

    cycle_dirs = get_cycle_dirs(base_run_dir)        
    dfs = []
    for cycle_dir in cycle_dirs:
        print('GETTING CYCLE INFO FOR',cycle_dir,' OUT OF:',len(cycle_dirs))
        df_ = pd.read_csv(os.path.join(cycle_dir, 'post_cycle_info.csv'))
        dfs.append(df_)
    df = pd.concat(dfs)
    print('debug columns', df.columns)
    # print('debug df', df)
    dim = len(parameters)
    # mean plot
    figure = plt.figure()
    plt.plot(df['num_samples'], df['mean'])
    plt.fill_between(df['num_samples'], df['mean'].to_numpy()-df['std'].to_numpy(), df['mean'].to_numpy()+df['std'].to_numpy(), alpha=0.2)
    plt.xlabel('N. Samples')
    plt.ylabel('Expected Value')
    figure.savefig(os.path.join(base_run_dir,'comparison_random_mean_std.png'))

    figure = plt.figure()
    plt.plot(df['num_samples'], df['mean'])
    plt.xlabel('N. Samples')
    plt.ylabel('Expected Value')
    figure.savefig(os.path.join(base_run_dir,'comparison_random_mean.png'))

    figure = plt.figure()
    plt.plot(df['num_samples'], df['std'])
    plt.xlabel('N. Samples')
    plt.ylabel('Std')
    figure.savefig(os.path.join(base_run_dir,'comparison_random_std.png'))

    # sobol plot
    fig = plt.figure()
    # Create main plot axes
    ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Main plot area
    ax_main.set_title("Main Plot")    
    for i in range(dim):
        ax_main.plot(df['num_samples'], df[f'brute_sobol_first_order_{i}'], label=f'x{i}')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('First order sobol indicie')
    ax_main.set_xlabel('Number of Parent Function Evaluations')
    # Create separate axes for legend
    # Create separate axes for legend
    ax_legend = fig.add_axes([0.75, 0.1, 0.2, 0.8])  # Legend area
    ax_legend.axis("off")  # Hide axes
    ax_legend.legend(*ax_main.get_legend_handles_labels(), loc="center")
    fig.savefig(os.path.join(base_run_dir, 'approx_sobol_first_order.png'),dpi=300)


    # sobol plot
    fig = plt.figure()
    # Create main plot axes
    ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Main plot area
    ax_main.set_title("Main Plot")    
    for i in range(dim):
        ax_main.plot(df['num_samples'], df[f'brute_sobol_total_order_{i}'], label=f'x{i}')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Total order sobol indicie')
    ax_main.set_xlabel('Number of Parent Function Evaluations')
    # Create separate axes for legend
    # Create separate axes for legend
    ax_legend = fig.add_axes([0.75, 0.1, 0.2, 0.8])  # Legend area
    ax_legend.axis("off")  # Hide axes
    ax_legend.legend(*ax_main.get_legend_handles_labels(), loc="center")
    fig.savefig(os.path.join(base_run_dir, 'approx_sobol_total_order.png'),dpi=300)
    
    
    # fig = plt.figure()
    # for i in range(dim):
    #     plt.plot(df['num_samples'], df[f'brute_sobol_total_order_{i}'], label=f'x{i}')
    #     plt.ylabel('Total order sobol indicie')
    #     plt.xlabel('Number of Parent Function Evaluations')
    # fig.savefig(os.path.join(base_run_dir, 'approx_sobol_total_order.png'),dpi=300)
        

# def plot_all_cycle_info(base_run_dir):
#     parameters, bounds, simulation_name, value_of_interest = get_config_info()
#     df = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
#     plt.plot(df['num_samples'], df['mean'], marker='o')
#     plt.fill_between(df['num_samples'], df['mean'].to_numpy()-df['std'].to_numpy(), df['mean'].to_numpy()+df['std'].to_numpy(), color='grey',label='Standard Deviation')
#     plt.legend()
#     xlabel = f'Number of {simulation_name} Evaluations'
#     ylabel = f'Expectation of {value_of_interest}'
#     plt.xlabel('Number of Evaluations')
#     plt.ylabel('Mean')
#     figure.tight_layout()
#     figure.savefig(os.path.join(base_dir, 'mean_std.png'), dpi=200)

    
if __name__ == '__main__':
    import sys
    _, base_run_dir = sys.argv
    post_cycle_info(base_run_dir, do_sensitivity=False)
