print('PERFORMING IMPORTS')
from tools import get_cycle_dirs, get_points, get_parameters_bounds
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
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

def plot_post_cycle_info(base_run_dir):
    print('plotting cycle infor for:', base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    for cycle_dir in cycle_dirs:
        print('CYCLE DIR:',cycle_dir)
        if os.path.exists(os.path.join(cycle_dir, 'post_cycle_info.csv')):
            dfs.append(pd.read_csv(os.path.join(cycle_dir, 'post_cycle_info.csv')))
    
    df = pd.concat(dfs)
    
    num_samples = df.pop('num_samples')
    columns = df.columns
    columns = [col for col in columns if not 'Unn' in col]
    if not os.path.exists(os.path.join(base_run_dir, 'post_cycle_info_plots')):
        os.makedirs(os.path.join(base_run_dir, 'post_cycle_info_plots'))
    for col in columns:
        fig = plt.figure()
        plt.plot(num_samples,df[col])
        plt.xlabel('N. Parent Eval')
        plt.ylabel(col)
        fig.savefig(os.path.join(base_run_dir,'post_cycle_info_plots',col+'_post_cycle_info.png'))
        plt.close(fig)

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    plot_post_cycle_info(base_run_dir)