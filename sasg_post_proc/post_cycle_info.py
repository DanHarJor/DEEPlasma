print('PERFORMING IMPORTS')
from tools import get_cycle_dirs, get_config, get_sasg, get_parameters_bounds
import pandas as pd
import numpy as np
import os, sys
from tools import get_points_upto
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import sobol_indices, uniform
from scipy.stats.qmc import Sobol
print('IMPORTS DONE')

def post_cycle_info(base_run_dir):
    print('DOING POST CYCLE INFO')
    cycle_dirs = get_cycle_dirs(base_run_dir)
    grid_increase=None
    old_grid_size = 0
    for cycle_dir in cycle_dirs:
        print('FOR CYCLE DIR:',cycle_dir,'OUT OF:',len(cycle_dirs))
        if os.path.exists(os.path.join(cycle_dir,'pysgpp_grid.txt')):
            sasg = get_sasg(cycle_dir)
            print('debug 1')
            grid_size = sasg.gridStorage.getSize()
            print('debug 2')
            grid_increase = grid_size - old_grid_size
            print('debug 3')
            old_grid_size = grid_size
            print('debug 4')
            sasg.grid_increase = grid_increase
            print('debug 5')
            sasg.do_brute_force_sobol_indicies = False
            # sasg.brute_force_sobol_indicies_num_samples = float(2**15)
            print('debug 6')
            sasg.write_cycle_info(cycle_dir, fname='post_cycle_info.csv', save_grid=False)
    print('DOING JOIN')
    join_post_cycle_info(base_run_dir)
    
def tree_post_cycle_info(base_run_dir, do_sensitivity=True):
    parameters, bounds = get_parameters_bounds(base_run_dir)
    bounds = np.array(bounds)
    model = HistGradientBoostingRegressor()
    dists = []
    for b in bounds:
        assert b[1] > b[0]
        dists.append(uniform(loc=b[0], scale=b[1]-b[0]))
        
    print('DOING POST CYCLE INFO')
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    i = 0
    for cycle_dir in cycle_dirs:
        i+=1
        print('debug',i,i%20)
        if i%20 == 0:
            print('FOR CYCLE DIR:',cycle_dir,'OUT OF:',len(cycle_dirs))
            x,y = get_points_upto(cycle_dir)
            print('debug len y', len(y))
            model.fit(x,y)
            
            print('BRUTE FORCING MEAN AND STD')
            # Create a Sobol sequence generator
            sobol_seq = Sobol(d=len(bounds), scramble=False)
            power = 15
            # Generate points in the unit hypercube [0, 1]^d
            points = sobol_seq.random_base2(m=power)  # Generates 2^power points
            # Scale the points to the desired bounds
            lower_bounds = bounds.T[0]
            upper_bounds = bounds.T[1]
            scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)
            y_uniform = model.predict(scaled_points)
            mean = np.mean(y_uniform)
            std = np.sqrt(np.var(y_uniform))
            df = pd.DataFrame({'num_samples':[len(y)],'mean':[mean],'std':[std]})
            
            if do_sensitivity:
                print('COMPUTING SOBOL INDICIES')
                func = lambda x: model.predict(x.T)
                
                sobol = sobol_indices(func=func,  n=2**15, dists=dists)
                sobol_first_order, sobol_total_order = sobol.first_order, sobol.total_order
                
                for d, sfo in enumerate(sobol_first_order):
                    df[f'brute_sobol_first_order_{d}'] = [sfo]
                for d, sto in enumerate(sobol_total_order):
                    df[f'brute_sobol_total_order_{d}']= [sto]
            dfs.append(df)
    all_df = pd.concat(dfs)
    all_df.to_csv(os.path.join(base_run_dir,'all_tree_post_cycle_info.csv'))
        

def join_post_cycle_info(base_run_dir):
    print('JOINING POST CYCLE INFO INTO all_post_cycle_info.csv')
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    for cycle_dir in cycle_dirs:
        if os.path.exists(os.path.join(cycle_dir, 'post_cycle_info.csv')):    
            dfs.append(pd.read_csv(os.path.join(cycle_dir, 'post_cycle_info.csv')))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(base_run_dir,'all_post_cycle_info.csv'))

def join_cycle_info(base_run_dir):
    print('JOINING POST CYCLE INFO INTO all_cycle_info.csv')
    cycle_dirs = get_cycle_dirs(base_run_dir)
    dfs = []
    for cycle_dir in cycle_dirs:
        if os.path.exists(os.path.join(cycle_dir, 'cycle_info.csv')):
            dfs.append(pd.read_csv(os.path.join(cycle_dir, 'cycle_info.csv')))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(base_run_dir,'all_cycle_info.csv'))
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    join_cycle_info(base_run_dir)
    # tree_post_cycle_info(base_run_dir, do_sensitivity=False)
    # post_cycle_info(base_run_dir)