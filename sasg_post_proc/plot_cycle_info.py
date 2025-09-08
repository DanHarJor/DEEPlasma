print('PERFORMING IMPORTS')
from tools import get_cycle_dirs, get_points, get_parameters_bounds
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import numpy as np
def plot_cycle_info(base_run_dir):
    print('plotting cycle infor for:', base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    for cycle_dir in cycle_dirs:
        print('CYCLE DIR:',cycle_dir)
        dfs.append(pd.read_csv(os.path.join(cycle_dir, 'cycle_info.csv')))
    
    df = pd.concat(dfs)
    

    num_samples = df.pop('num_samples')
    columns = df.columns
    for col in columns:
        fig = plt.figure()
        plt.plot(num_samples,df[col])
        plt.xlabel('N. Parent Eval')
        plt.ylabel(col)
        fig.savefig(os.path.join(base_run_dir,col+'_cycle_info.png'))
        plt.close(fig)
def plot_post_cycle_info(base_run_dir, name='post_cycle_info_plots', fname='post_cycle_info.csv'):
    print('plotting cycle infor for:', base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    for cycle_dir in cycle_dirs:
        print('CYCLE DIR:',cycle_dir)
        if os.path.exists(os.path.join(cycle_dir, fname)):
            dfs.append(pd.read_csv(os.path.join(cycle_dir, fname)))
    
    df = pd.concat(dfs)
    
    num_samples = df.pop('num_samples').to_numpy()
    columns = df.columns
    columns = [col for col in columns if not 'Unn' in col]
    if not os.path.exists(os.path.join(base_run_dir, name)):
        os.makedirs(os.path.join(base_run_dir, name))
    for col in columns:
        fig = plt.figure()
        plt.plot(num_samples,df[col])
        plt.xlabel('N. Parent Eval')
        plt.ylabel(col)
        fig.tight_layout()
        if 'surplus' in col:
            plt.vlines(num_samples[18], np.min(df[col]), np.max(df[col]), color='red')
            # plt.vlines(num_samples[18], np.min(df[col]), np.max(df[col]), color='red')

            plt.yscale('log')
            # plt.ylim(1e-4,0.1)
            # plt.xlim(0,3000)
        fig.savefig(os.path.join(base_run_dir,name,col+'_post_cycle_info.png'))
        plt.close(fig)

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    plot_post_cycle_info(base_run_dir)
    # plot_post_cycle_info(base_run_dir, fname='boundary_anchors_cycle_info.csv', name='boundary_anchors_cycle_info_plots')