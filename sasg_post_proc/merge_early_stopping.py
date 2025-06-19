import os
import numpy as np
import pandas as pd
import warnings
def active_merge_early_stopping(base_run_dir):
    print('MERGING EARLY STOPPING FILES FROM run_dirs TO base_run_dir')
    listdir = os.listdir(base_run_dir)
    cycle_dirs = np.sort([d for d in listdir if 'active_cycle_' in d])
    dfs = []
    for cycle_dir in cycle_dirs:
        # assume dirs in the cycle directory is only run directories
        
        run_dirs = [os.path.join(base_run_dir,cycle_dir,dir) for dir in os.listdir(os.path.join(base_run_dir,cycle_dir)) if os.path.isdir(os.path.join(base_run_dir,cycle_dir,dir))]
        for run_dir in run_dirs:
            df = pd.read_csv(os.path.join(run_dir,'early_stopping_report.csv'), index_col=None)
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    # df["stable"] = df["stable"].astype(bool)
    # Explicitly drop "Unnamed: 0" if it still exists
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df.to_csv(os.path.join(base_run_dir, 'early_stopping_report.csv'), index=False)
    print('FINISHED MERGING THE EARLY STOPPING FILES FROM run_dirs TO base_run_dir')
    
def merge_early_stopping(base_run_dir):    
    def find_files(start_path, target_filename):
        matches = []
        for root, _, files in os.walk(start_path):
            if target_filename in files:
                matches.append(os.path.join(root, target_filename))
        return matches
    
    result = find_files(base_run_dir, 'early_stopping_report.csv')
    if len(result)==0:
        warnings.warn('THERE WERE NO early_stopping_report.csv FILES')
    else:
        dfs=[]
        for file in result:
            df = pd.read_csv(file, index_col=None)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        # df["stable"] = df["stable"].astype(bool)
        # Explicitly drop "Unnamed: 0" if it still exists
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]
        df.to_csv(os.path.join(base_run_dir, 'merged_early_stopping_report.csv'), index=False)

    
    
if __name__ == '__main__':
    import sys
    _, base_run_dir = sys.argv
    merge_early_stopping(base_run_dir)