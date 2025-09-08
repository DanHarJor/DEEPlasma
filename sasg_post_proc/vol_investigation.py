import sys, os

from tools import get_base_run_dir

from sasg_test import weighted_mean_test, sasg_test
from post_cycle_info import post_cycle_info
from plot_cycle_info import plot_post_cycle_info
from vis_enchanted_sasg import vis_enchanted_sasg
if __name__ == '__main__':
    _, config_file = sys.argv
    base_run_dir = get_base_run_dir(config_file)
    post_cycle_info(base_run_dir)
    plot_post_cycle_info(base_run_dir)
    weighted_mean_test(base_run_dir)
    sasg_test(base_run_dir)
    vis_enchanted_sasg(base_run_dir, cycle_num='latest')
    vis_enchanted_sasg(base_run_dir, cycle_num=0)
    
    
    