from vis_enchanted_sasg import vis_enchanted_sasg
import sys
sys.path.append('/users/danieljordan/DEEPlasma/')
from uq_vizualize.expectation_sigma_sobol import plot_expectation, plot_sigma, plot_sobols

_, base_run_dir = sys.argv

vis_enchanted_sasg(base_run_dir, cycle_num='all')
plot_expectation(base_run_dir)
plot_sigma(base_run_dir)
plot_sobols(base_run_dir, xlabel='Number of Local Linear GENE Simulations')