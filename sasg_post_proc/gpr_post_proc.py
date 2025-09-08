from tools import get_weighted_mean, get_cycle_dirs, get_points_upto
import pandas as pd
import sys, os
def post_cycle_info(base_run_dir, cycle_num='every-50'):
    cycle_dirs = get_cycle_dirs(base_run_dir, cycle_num)
    
    df = pd.DataFrame()
    
    weighted_means = []
    i = 0
    num_samples = []
    for cycle_dir in cycle_dirs:
        print('cycle dir:',cycle_dir, 'num', i, 'out of:', len(cycle_dirs))
        weighted_means.append(get_weighted_mean(cycle_dir))
        x, y = get_points_upto(cycle_dir)
        num_samples.append(len(x))
        i+=1
    
    df['mean']=weighted_means
    df['num_samples']=num_samples
    
    print('writing to:', os.path.join(base_run_dir, 'all_post_cycle_info.csv'))
    df.to_csv(os.path.join(base_run_dir, 'all_post_cycle_info.csv'))
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    
    post_cycle_info(base_run_dir) 