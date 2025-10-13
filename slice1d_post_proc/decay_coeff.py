import sys, os
import pandas as pd
import f90nml
sys.path.append('/users/danieljordan/enchanted-surrogates/src/')
from runners.gene_single_monitor import calculate_decay_coefficient
from parsers.GENEparser import GENEparser
import numpy as np
from slice1d_post_proc import slice1d_from_df

def plot_decay_coeff(base_run_dir):
    gp = GENEparser()
    run_dirs = os.listdir(base_run_dir)
    dfs = []
    print('LOOKING AT RUN DIRS')
    decays = []
    for i, run_dir in enumerate(run_dirs):
        # if i == 10: break
        run_dir = os.path.join(base_run_dir, run_dir)
        if not os.path.exists(os.path.join(run_dir, 'GENE.finished')):
            continue
        print('LOOKING AT RUN DIR:', run_dir)
        decay = calculate_decay_coefficient(run_dir)
        decays.append(decay)
        nml = f90nml.read(os.path.join(run_dir, 'parameters'))
        omti = nml['_grp_species_0']['omt']
        omte = nml['_grp_species_1']['omt']
        omn = nml['_grp_species_1']['omn']
        ky, gr, freq = gp.read_omega(run_dir)
        
        dfi = pd.DataFrame({'omn':[omn], 'omte':[omte], 'omti':[omti], 'gr':[gr]})
        dfs.append(dfi)
        # $\omega_n$ ,$\omega_{T_e}$ ,$\omega_{T_i}$ 
    
    decay_norm = (decays - np.nanmin(decays)) / (np.nanmax(decays) - np.nanmin(decays))
    # print('debug nan', decays.min())
    # print('debug decays norm', decay_norm)
    df = pd.concat(dfs, ignore_index=True)
    df['decay'] = decay_norm
    print('SAVING DECAY DF')
    df.to_csv(os.path.join(base_run_dir, 'decay_coeff.csv'))

    slice1d_from_df(base_run_dir, df, name='slice1d_decay_coeff.png', num_out=2)

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    plot_decay_coeff(base_run_dir)