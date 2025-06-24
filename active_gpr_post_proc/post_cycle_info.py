import sys,os
sys.path.append('sasg_post_proc')
from tools import get_points_upto, get_cycle_dirs, get_sampler, get_parameters_bounds
from scipy.stats import sobol_indices, uniform
from scipy.stats.qmc import Sobol
import numpy as np
import pandas as pd

def post_cycle_info(base_run_dir):
    cycle_dirs = get_cycle_dirs(base_run_dir)
    parameters, bounds = get_parameters_bounds(base_run_dir)
    bounds = np.array(bounds)
    gpr_sampler = get_sampler(base_run_dir)        
    dists = []
    
    for b in bounds:
        assert b[1] > b[0]
        dists.append(uniform(loc=b[0], scale=b[1]-b[0]))

    dfs = []
    for cycle_dir in cycle_dirs:
        print('FOR CYCLE DIR:',cycle_dir,'OUT OF:',len(cycle_dirs))
        x, y = get_points_upto(cycle_dir)
        regressor = gpr_sampler.model_fit(x,y)
        # get mean and var
        sobol_seq = Sobol(d=len(bounds), scramble=False)
        power = 15
        # Generate points in the unit hypercube [0, 1]^d
        points = sobol_seq.random_base2(m=power)  # Generates 2^power points
        # Scale the points to the desired bounds
        lower_bounds = bounds.T[0]
        upper_bounds = bounds.T[1]
        scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)
        y_uniform = gpr_sampler.model_predict(scaled_points, regressor)
        mean = np.mean(y_uniform)
        std = np.sqrt(np.var(y_uniform))

        func = lambda x: gpr_sampler.model_predict(x.T, regressor)
        sobol = sobol_indices(func=func,  n=2**15, dists=dists)
        sobol_first_order, sobol_total_order = sobol.first_order, sobol.total_order

        df = pd.DataFrame({'num_samples':[len(y)],'mean':[mean],'std':[std]})
        for d, sfo in enumerate(sobol_first_order):
            df[f'brute_sobol_first_order_{d}'] = [sfo]
        for d, sto in enumerate(sobol_total_order):
            df[f'brute_sobol_total_order_{d}']= [sto]
        dfs.append(df)
    all_df = pd.concat(dfs)
    all_df.to_csv(os.path.join(base_run_dir,'all_post_cycle_info.csv'))
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    post_cycle_info(base_run_dir)