# field_file = '/scratch/project_462000954/daniel/enchanted_test/gene_beta_scan_test_write_when_stop_comparison_mode/e0fafdb4-a55c-4527-b2ff-d2baeee3bc63/field.dat'
# with open(field_file,'rb') as file:
#     field_data = file.readlines()
#     print(len(field_data))
#     for line in field_data:
#         print(line.decode('utf-8').strip())

import os,sys
# sys.path.append('/users/danieljordan/DEEPlasma/dependancies/enchanted-surrogates/src')

# from parsers.GENEparser import GENEparser

# sys.path.append('/scratch/project_462000954/reproduce_eped/dependancies/enchanted-surrogates/src')
sys.path.append('/users/danieljordan/enchanted-surrogates/src')
from parsers.GENEparser import GENEparser

parser = GENEparser()

# run_dir = '/scratch/project_462000954/daniel/enchanted_test/gene_files_when_stopped/9c4b8218-7a60-46ee-bc18-79c298d184fa/'
run_dir = '/scratch/project_462000954/daniel/enchanted_test/gene_files_when_stopped_bystopfile/74179a4e-4662-44fb-a7a5-34177ead4c0f/'
metrics = parser.micro_mode_classification_metrics(run_dir=run_dir, suffix='.dat')

# for met, v in metrics.items():
#     print('metric:',met, 'val:',v)
    
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(metrics['phi_z'][1],metrics['phi_z'][0])

fig.savefig(os.path.join('/users/danieljordan/DEEPlasma/early_convergence_post_proc/plots', 'mode_structure3.png'))

omega = parser.calculate_latest_omega(diagdir=run_dir)

print('omega:',omega)

# sys.path.append('/users/danieljordan/DEEPlasma/TPED/projects/GENE_sim_reader/src')

# from TPED.projects.GENE_sim_analysis.src.field_analysis.field_resolution import plot_mode_structure

# field_file = '/scratch/project_462000954/daniel/enchanted_test/gene_beta_scan_test_write_when_stop_comparison_mode/e0fafdb4-a55c-4527-b2ff-d2baeee3bc63/field.dat'
# plot_mode_structure(field_file)

# # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # import numpy as np
# # import os
# # import pickle

# # from IPython.display import clear_output
# # import matplotlib.pyplot as plt
# # import time


# # # from TPED.projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
# # # from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC


# # from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField as GF
# # from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder as GFF

# # def plot_mode_structure(filepath_list:str|list):
    
    
# #     if isinstance(filepath_list, str):
# #         if os.path.basename(filepath_list).startswith('field'):
# #             filepath_list = [filepath_list]
# #         elif os.path.isdir(filepath_list):
# #             filepath_list = GFF(filepath_list).find_files('field')
# #     elif isinstance(filepath_list, list):
# #         if all([os.path.basename(filepath).startswith('field') for filepath in filepath_list]):
# #             pass
# #         else:
# #             raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')   
# #     else:
# #         raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')



# #     for field_filepath in filepath_list:
# #         field = GF(field_filepath)
# #         field_dict = field.field_filepath_to_dict(time_criteria='last')

# #         zgrid = field_dict['zgrid']
# #         phi = field_dict['field_phi'][-1]

# #         fig = plt.figure()
# #         plt.title(r'$\phi$')
# #         plt.plot(zgrid,np.real(phi),color='red',label=r'$Re[\phi]$')
# #         plt.plot(zgrid,np.imag(phi),color='blue',label=r'$Im[\phi]$')
# #         plt.plot(zgrid,np.abs(phi),color='black',label=r'$|\phi|$')
# #         plt.xlabel(r'$z/\pi$',size=18)
# #         plt.legend()
# #         plt.show()
# #         fig.savefig(os.path.join('early_convergence_post_proc','plots', 'mode_structure.png'))