
print('PERFORMING IMPORTS')
# from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd
print('1')
import matplotlib.pyplot as plt
print('2')
from tools import get_cycle_dirs, get_points, get_parameters_bounds, get_config
print('3')
import os, sys
# from scipy.stats import sobol_indices, uniform
print('4')
import numpy as np
print('5')
import shutil
print('6')
pfd = os.path.dirname(__file__)
#base_run_dir should be for a random dataset like batch sobol sequence
print('IMPORTS DONE')

def random_comparison(base_run_dir, compare_run_dirs=None, do_sensitivity=True, do_tree=False, xlim=None, name=''):
    colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
            u'#bcbd22', u'#1f77b4', u'#8c564b', u'#e377c2']
        
    print('STARTING RANDOM COMPARISON')
    config = get_config(base_run_dir)
    test_dir = config.sampler['test_dir']
    # test_dir = os.path.join('/users/danieljordan/enchanted-surrogates/', test_dir)
    
    df_test = pd.read_csv(os.path.join(test_dir,'all_post_cycle_info.csv'))
    print('TEST DIR IS:',test_dir, len(df_test))
    print('debug', df_test['num_samples'])
    dfs_compare = []
    compare_labels = []
    if type(compare_run_dirs) != type(None):
        save_dir = os.path.join(base_run_dir, name+'comparison_many')
        print('SAVE DIR IS', save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for compare_run_dir in compare_run_dirs:
            compare_config = get_config(compare_run_dir)
            compare_label = compare_config.sampler['type']
            if compare_label == 'SpatiallyAdaptiveSparseGrids':
                compare_label = compare_label+'_'+compare_config.sampler['adaptive_strategy']['refinement_type']
            compare_labels.append(compare_label)
            config_file = [file for file in os.listdir(compare_run_dir) if '.yaml' in file]
            config_file = config_file[0]
            shutil.copy(os.path.join(compare_run_dir, config_file), os.path.join(base_run_dir, save_dir))
            if os.path.exists(os.path.join(compare_run_dir, 'all_post_cycle_info.csv')):
                df_compare = pd.read_csv(os.path.join(compare_run_dir, 'all_post_cycle_info.csv'))
            if os.path.exists(os.path.join(compare_run_dir, 'all_cycle_info.csv')):
                df_compare = pd.read_csv(os.path.join(compare_run_dir, 'all_cycle_info.csv'))
            dfs_compare.append(df_compare)
    else:
        save_dir = os.path.join(base_run_dir,name+'sobol_seq_comparison')
        print('SAVE DIR IS', save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    if do_tree:
        df_tree = pd.read_csv(os.path.join(base_run_dir,'all_tree_post_cycle_info.csv'))
    
    df = pd.read_csv(os.path.join(base_run_dir,'all_post_cycle_info.csv'))        
    # expectation comparison
    fig = plt.figure()
    plt.plot(df['num_samples'],df['mean'], label='sasg', marker='o')
    plt.plot(df_test['num_samples'], df_test['mean'], '--', color='green', label='Sobol sequence', marker='o')
    if do_tree: plt.plot(df_tree['num_samples'], df_tree['mean'], ':', color='cyan', label='sasg tree', marker='x')
    if type(compare_run_dirs) != type(None):
        for i, df_compare, compare_label in zip(range(len(dfs_compare)),dfs_compare, compare_labels):
            plt.plot(df_compare['num_samples'], df_compare['mean'], ':', color=colors[i], label=compare_label, marker='o')

    plt.ylabel('Expectation')
    plt.xlabel('N. Evaluations')
    fig.tight_layout()
    plt.legend()
    print('debug xlim', xlim)
    ax = fig.gca()
    ax.set_xlim(0, xlim)
    # plt.xlim(0,xlim)
    fig.savefig(os.path.join(save_dir, 'expectation_comparison.png'))
    plt.close(fig)
    
    # std comparison
    fig = plt.figure()
    plt.plot(df['num_samples'],df['std'], label='sasg', marker='o')
    plt.plot(df_test['num_samples'], df_test['std'], '--', color='green', label='Sobol sequence', marker='o')
    if do_tree: plt.plot(df_tree['num_samples'], df_tree['std'], ':', color='cyan', label='sasg tree', marker='x')
    if type(compare_run_dirs) != type(None):
        for i, df_compare, compare_label in zip(range(len(dfs_compare)),dfs_compare, compare_labels):
            plt.plot(df_compare['num_samples'], df_compare['std'], ':', color=colors[i], label=compare_label, marker='o')
    
    plt.ylabel('Standart Deviation, sqrt(var)')
    plt.xlabel('N. Evaluations')
    fig.tight_layout()
    plt.legend()
    plt.xlim(0,xlim)
    fig.savefig(os.path.join(save_dir, 'std_comparison.png'))
    plt.close(fig)
    
    if do_sensitivity:
        colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
            u'#bcbd22', u'#1f77b4', u'#8c564b', u'#e377c2']
        
        columns = df.columns
        sfo_col = [col for col in columns if 'brute_sobol_first_order' in col]
        dim = len(sfo_col)
        # mean first order sobol error
        # sobol plot
        fig = plt.figure()
        # Create main plot axes
        ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Main plot area
        ax_main.set_title("Main Plot")    
        for i in range(dim):
            # ax_main.plot(df['num_samples'], np.abs(df[f'brute_sobol_first_order_{i}'].to_numpy()-df_test[f'brute_sobol_first_order_{i}'].to_numpy()[-1]), label=f'x{i}')
            ax_main.plot(df['num_samples'], np.abs(df[f'brute_sobol_first_order_{i}']), label=f'x{i}', color=colors[i], marker='o')
            ax_main.plot(df_test['num_samples'], np.abs(df_test[f'brute_sobol_first_order_{i}']), '--', color=colors[i], marker='o')
            # ax_main.plot(df_tree['num_samples'], np.abs(df_tree[f'brute_sobol_first_order_{i}']), ':', label=f'x{i}', color=colors[i])
            if compare_run_dir != None:
                ax_main.plot(df_compare['num_samples'], np.abs(df_compare[f'brute_sobol_first_order_{i}']), ':', color=colors[i], marker='o')
            # ax_main.hlines(df_test[f'brute_sobol_first_order_{i}'].to_numpy()[-1], np.min(df['num_samples']), np.max(df['num_samples']), linestyles='--', color=colors[i])
        # ax_main.set_yscale('log')
        ax_main.set_ylabel('First order sobol indices Error, |sasg_sobol-true_sobol|')
        ax_main.set_xlabel('Number of Parent Function Evaluations')
        # Create separate axes for legend
        # Create separate axes for legend
        ax_legend = fig.add_axes([0.75, 0.1, 0.2, 0.8])  # Legend area
        ax_legend.axis("off")  # Hide axes
        ax_legend.legend(*ax_main.get_legend_handles_labels(), loc="center")
        ax_main.set_xlim(0,xlim)
        fig.savefig(os.path.join(save_dir, 'approx_sobol_first_order_comparison.png'),dpi=300)

        # mean total order sobol error
        # sobol plot
        fig = plt.figure()
        # Create main plot axes
        ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Main plot area
        ax_main.set_title("Main Plot")    
        for i in range(dim):
            ax_main.plot(df['num_samples'], df[f'brute_sobol_total_order_{i}'], label=f'x{i}', color=colors[i], marker='o')
            ax_main.plot(df_test['num_samples'], np.abs(df_test[f'brute_sobol_total_order_{i}']), '--', label=f'x{i}', color=colors[i], marker='o')
            # ax_main.plot(df_tree['num_samples'], np.abs(df_tree[f'brute_sobol_total_order_{i}']), ':', label=f'x{i}', color=colors[i])
            if compare_run_dir != None:
                ax_main.plot(df_compare['num_samples'], np.abs(df_compare[f'brute_sobol_total_order_{i}']), ':', color=colors[i], marker='o')

            # ax_main.plot(df['num_samples'], np.abs(df[f'brute_sobol_total_order_{i}'].to_numpy()-df_test[f'brute_sobol_total_order_{i}'].to_numpy()[-1]), label=f'x{i}')
            # ax_main.hlines(df_test[f'brute_sobol_total_order_{i}'].to_numpy()[-1], np.min(df['num_samples']), np.max(df['num_samples']), linestyles='--', color=colors[i])

        # ax_main.set_yscale('log')
        ax_main.set_ylabel('Total order sobol indices')
        ax_main.set_xlabel('Number of Parent Function Evaluations')
        # Create separate axes for legend
        # Create separate axes for legend
        ax_legend = fig.add_axes([0.75, 0.1, 0.2, 0.8])  # Legend area
        ax_legend.axis("off")  # Hide axes
        ax_legend.legend(*ax_main.get_legend_handles_labels(), loc="center")
        ax_main.set_xlim(0,xlim)
        fig.savefig(os.path.join(save_dir, 'approx_sobol_total_order_comparison.png'),dpi=300)

if __name__ == '__main__':
    # _, base_run_dir = sys.argv
    # # if compare_dir == 'None':
    # #     compare_dir = None
    # num_pro = os.environ.get('num_pro')
    # compare_dirs = ['/scratch/'+num_pro+'/DANIEL/data_store/full_12D_linearGrid/sasg_threshold',
    #                 '/scratch/'+num_pro+'/DANIEL/data_store/full_12D_linearGrid/sasg_volume',
    #                 '/scratch/'+num_pro+'/DANIEL/data_store/full_12D_linearGrid/MMMG_static_grid']
    #                 # '/scratch/'+num_pro+'/DANIEL/data_store/full_12D_linearGrid/active_GPyOPT_12D_MMMG']
    # # print('debug', compare_dir, type(compare_dir))
    # random_comparison(base_run_dir, compare_run_dirs=compare_dirs, do_sensitivity=False, do_tree=False, xlim=None, name='')
    

    # # compare_dirs=None
    # # random_comparison(base_run_dir, compare_run_dirs=compare_dirs, do_sensitivity=False, do_tree=True)
    
    _, base_run_dir = sys.argv
    # if compare_dir == 'None':
    #     compare_dir = None
    compare_dirs = ['/scratch/project_2007848/DANIEL/data_store/full_12D/sasg_threshold',
                    '/scratch/project_2007848/DANIEL/data_store/full_12D/sasg_volume',
                    '/scratch/project_2007848/DANIEL/data_store/full_12D/MMMG_static_grid',
                    '/scratch/project_2007848/DANIEL/data_store/full_12D/active_GPyOPT_12D_MMMG']
    # print('debug', compare_dir, type(compare_dir))
    random_comparison(base_run_dir, compare_run_dirs=compare_dirs, do_sensitivity=False, do_tree=False, xlim=700, name='xlim700_')
    

    # compare_dirs=None
    # random_comparison(base_run_dir, compare_run_dirs=compare_dirs, do_sensitivity=False, do_tree=True)