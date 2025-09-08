from vis_enchanted_sasg import vis_enchanted_sasg
import sys
from merge_early_stopping import merge_early_stopping
from sasg_test import sasg_test, test_boundary_hypothesis, inspect_outliers, weighted_mean_test, MCvsQUAD

from random_comparison import random_comparison

from post_cycle_info import post_cycle_info
from plot_cycle_info import plot_post_cycle_info

sys.path.append('/users/danieljordan/DEEPlasma/')
from uq_vizualize.expectation_sigma_sobol import plot_expectation, plot_sobols, plot_cycle_info
import uuid
_, base_run_dir = sys.argv
import pandas as pd

# # for GENE
# print('MANAGING EARLY STOPPING FILES')
# merge_early_stopping(base_run_dir)
# print('VISUALISING SASG')
# vis_enchanted_sasg(base_run_dir, cycle_num='all')
# print('PLOTTING EXPECTATION')
# plot_expectation(base_run_dir)
# # print('PLOTTING SIGMA')
# # plot_sigma(base_run_dir)
# print('PLOTTING SOBOL')
# plot_sobols(base_run_dir)
# print('TESTING SURROGATE')
# sasg_test(base_run_dir)
# test_boundary_hypothesis(base_run_dir)
# inspect_outliers(base_run_dir)

# For MMMG
print('performing post proc')
sasg_type = 'get_sasg'
every = 'every-1'
post_cycle_info(base_run_dir, sasg_type=sasg_type, cycle_num=every)
plot_post_cycle_info(base_run_dir, name=sasg_type+'_post_cycle_info_plots', fname=sasg_type+'_post_cycle_info.csv')
# print('performing sasg test')
# sasg_test(base_run_dir, name=sasg_type+'_sasg_test', sasg_type=sasg_type, isMMMG=True, cycle_num=every)
# print('performing test boundary hypothesis')
# test_boundary_hypothesis(base_run_dir, sasg_type=sasg_type)
# print('performing weighted mean test')
# weighted_mean_test(base_run_dir, sasg_type=sasg_type)
# print('performing random comparison')
# random_comparison(base_run_dir, name=sasg_type+'_', fname=sasg_type+'_post_all_cycle_info.csv', isMMMG=True, gaussian_input_uncertanties=False)
# print('visualising')
# vis_enchanted_sasg(base_run_dir, cycle_num=every 'latest', isMMMG=True, sasg_type=sasg_type)
# print('MC vs QUAD')
# MCvsQUAD(base_run_dir, sasg_type=sasg_type)

# for MMG gaussian
# sasg_type='get_sasg'
# print('performing sasg test')
# sasg_test(base_run_dir, name=sasg_type+'_sasg_test_gaussian', sasg_type=sasg_type, isMMMG=True, cycle_num='every-7')
