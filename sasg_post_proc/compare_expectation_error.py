import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import get_cycle_dirs, get_points, get_parameters_bounds, get_config, get_MMMGrunner, get_test_points, get_test_points_brd

dir_to_compare = [
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/MMMG_sobolseq_testset_3D',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_linear_boundary_volume',
    # '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_volume',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_surplus',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_volume_1refpoint',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_variance_refinement',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_expectation',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_var_then_surplus_one_switch',
    '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_gpr'
    # '/scratch/project_2007848/DANIEL/data_store/dim_scan/dim_3_mod_linear_var_then_surplus',
]

labels = ['sobol seq', 'bound vol', 'surplus', 'vol', 'var', 'exp', 'var surp', 'gpr']

fig = plt.figure()

runner = get_MMMGrunner(dir_to_compare[0])
true_mean = runner.mmg.get_expectation(do_gaussian=False)
    

for base_run_dir, label in zip(dir_to_compare, labels):
        
    
    sasg_type='get_sasg'
    fname=sasg_type+'_post_all_cycle_info.csv'
    print(' debug df filename', os.path.join(base_run_dir,fname))
    try:
        df = pd.read_csv(os.path.join(base_run_dir,fname))        
    except FileNotFoundError:
        df = pd.read_csv(os.path.join(base_run_dir,'all_post_cycle_info.csv')        )
    plt.plot(df['num_samples'].to_numpy(),np.abs(df['mean']-true_mean), label=label, marker='o')
    # plt.vlines(df['num_samples'], np.min(np.abs(df['mean']-true_mean)), np.max(np.abs(df['mean']-true_mean)))
    plt.ylabel('Expectation Error')
    plt.xlabel('N. Evaluations')
    plt.yscale('log')
    fig.tight_layout()
    plt.legend()
    # print('debug xlim', xlim)
    # ax = fig.gca()
    

fig.savefig(os.path.join('/users/danieljordan/DEEPlasma/sasg_post_proc/figs', 'expectation_error_comparison_many.png'))
plt.close(fig)
    