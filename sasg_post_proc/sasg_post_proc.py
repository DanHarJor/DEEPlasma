from vis_enchanted_sasg import vis_enchanted_sasg
import sys
from merge_early_stopping import merge_early_stopping
from sasg_test import sasg_test

sys.path.append('/users/danieljordan/DEEPlasma/')
from uq_vizualize.expectation_sigma_sobol import plot_expectation, plot_sigma, plot_sobols
import uuid
_, base_run_dir = sys.argv
import pandas as pd

print('MANAGING EARLY STOPPING FILES')
merge_early_stopping(base_run_dir)
print('VISUALISING SASG')
vis_enchanted_sasg(base_run_dir, cycle_num='all')
print('PLOTTING EXPECTATION')
plot_expectation(base_run_dir)
print('PLOTTING SIGMA')
plot_sigma(base_run_dir)
print('PLOTTING SOBOL')
plot_sobols(base_run_dir)
print('TESTING SURROGATE')
sasg_test(base_run_dir)
