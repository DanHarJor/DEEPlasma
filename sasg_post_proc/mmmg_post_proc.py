from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from tools import load_configuration, get_points, get_config, get_cycle_dirs
import sys, os
pfd = os.path.dirname(__file__)


sys.path.append('/users/danieljordan/enchanted-surrogates/src')

from runners.MMMGrunner import MMMGrunner

def mmmg_plot(base_run_dir):
    config = get_config(base_run_dir)
    runner_args = config.executor['static_executor']['runner']
    runner = MMMGrunner(**runner_args)
    mmg = runner.mmg
    fig = mmg.plot_matrix_contour()
    fig.savefig(os.path.join(base_run_dir,'parent_contours.png'))
    plt.close(fig)
    fig = mmg.plot_slices()
    fig.savefig(os.path.join(base_run_dir,'parent_slices.png'))
    plt.close(fig)
def mmmg_plot_random_forest_compare(base_run_dir):    
    print('RANDOM FOREST COMPARE:',base_run_dir)
    config = get_config(base_run_dir)
    runner_args = config.executor['static_executor']['runner']
    runner = MMMGrunner(**runner_args)
    mmg = runner.mmg
    
    cycle_dirs = get_cycle_dirs(base_run_dir)
    y_all = np.array([])
    x_all_list = []
    
    y_all = np.array([])
    x_all_list = []
    for cycle_dir in cycle_dirs:
        print('LOOKING AT CYCLE DIR:',cycle_dir,'OUT OF:', len(cycle_dirs))
        x,y = get_points(cycle_dir)
        x_all_list.append(x)
        x_all = np.vstack(x_all_list)
        y_all = np.append(y_all, y)
    model = HistGradientBoostingRegressor()
    print('FITTING MODEL')
    model.fit(x_all,y_all)
    forest = lambda x: model.predict(x)
    fig = mmg.plot_slices(compare_function=forest)
    fig.savefig(os.path.join(cycle_dir,'forest_slices.png'))
    plt.close(fig)
        

if __name__ == '__main__':
    _, base_run_dir =  sys.argv
    mmmg_plot(base_run_dir)
    # mmmg_plot_random_forest_compare(base_run_dir)