print('PERFORMING IMPORTS')
import os
import pandas as pd 
import numpy as np
import yaml
import argparse
import warnings
import pysgpp
import matplotlib.pyplot as plt
import sys
import f90nml
sys.path.append('/users/danieljordan/DEEPlasma')
from highD_visualise.highD_visualise import plot_single_slice, plot_slices, plot_matrix_contour
from slice1d_post_proc.slice1d_post_proc import slice1d_post_proc

from tools import get_config, get_parameters_bounds, get_MMMGrunner
from tools import get_sasg, get_MMMGrunner
from tools import get_sasg_zero_bounds, get_sasg_mod_poly_grid, get_sasg_zero_poly_grid, get_sasg_anchor
# get_sasg = get_sasg_zero_bounds


from sklearn.neighbors import KernelDensity

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 24})  # Adjust number as needed

sys.path.append('/users/danieljordan/enchanted-surrogates2/src')
sys.path.append('/users/danieljordan/enchanted-surrogates/src')

# sys.path.append('/users/danieljordan/DEEPlasma')
from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
# from parsers.GENEparser import GENEparser
print('IMPORTS COMPLETE ')
def find_files(start_path, target_filename):
    matches = []
    for root, _, files in os.walk(start_path):
        if target_filename in files:
            matches.append(os.path.join(root, target_filename))
    return matches

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    # print('LOADING CONFIGURATION FILE')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.executor["config_filepath"] = config_path
    return config

def get_parameters_labels(base_run_dir):
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    parameters_labels = config.sampler.get('parameters_labels', None)
    if parameters_labels == None:
        parameters_labels = config.sampler['parameters']
    return parameters_labels
    

def get_config_info(base_run_dir):
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters']
    runner_return_headder = config.executor['runner_return_headder']
    
    value_of_interest = config.general.get('value_of_interest', 'function value')
    parent_model = config.general.get('simulation_name', 'Parent Model')
    return parameters, bounds, parent_model, value_of_interest

# def quad_exp_test(base_run_dir):

def sample_shell_between_cubes(dim, dist, index, n_samples):
    center = 0.5
    # half_inner = s_inner / 2
    # half_outer = s_outer / 2

    lower_outer = center - (index+1)*dist
    upper_outer = center + (index+1)*dist

    lower_inner = center - index*dist
    upper_inner = center + index*dist

    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(lower_outer, upper_outer, size=(dim,))
        if np.any(x < lower_inner) or np.any(x > upper_inner):
            samples.append(x)

    return np.array(samples)

def MCvsQUAD(base_run_dir, sasg_type='get_sasg'):
    # Define a simple function
    cycle_dirs = get_cycle_dirs(base_run_dir)
    cycle_dir = cycle_dirs[-1]
    
    sasg = eval(f"{sasg_type}(cycle_dir)")
    runner = get_MMMGrunner(base_run_dir)
    def f(x): 
        # return sasg.unit_truncnorm_pdf(x)
        return runner.mmg.evaluate(x)
    dim = 12
    level=5
    # Build grid and interpolant
    grid = sasg.grid #
    # grid = pysgpp.Grid.createLinearGrid(dim)
    # grid.getGenerator().regular(level)
    storage = grid.getStorage()
    alpha = pysgpp.DataVector(storage.getSize())
    print('level:',level, '| size:', storage.getSize())
    for i in range(storage.getSize()):
        p = storage.getPoint(i)
        x = [p.getStandardCoordinate(j) for j in range(dim)]
        alpha[i] = float(f(x))
    pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

    # SG++ quadrature
    op_quad = pysgpp.createOperationQuadrature(grid)
    sgpp_integral = op_quad.doQuadrature(alpha)

    # Monte Carlo
    # samples = np.random.rand(100000, dim)
    samples = np.random.uniform(size=(1000,12))

    op_eval = pysgpp.createOperationEval(grid)
    mc_values = [op_eval.eval(alpha, pysgpp.DataVector(x)) for x in samples]
    mc_integral = np.mean(mc_values)
    
    mc_actual_integral = np.mean([f(x) for x in samples])

    print("SG++:", sgpp_integral)
    print("Monte Carlo Interpolant:", mc_integral)
    print("Monte Carlo Actual:", mc_actual_integral)
    
    print("Gradual increase of hypercube from center at 0.5 to edge at 0 and 1, to see where the true deviates from the interpolant integral")
    dist_ = []
    mc_integral_ = []
    mc_actual_integral_ = []
    mean_error_ = []
    mean_values_ = []
    mean_actual_values_ = []
    integral_ratio_ = []
    # for dist in np.linspace(0.01, 0.5, 20):
    num_shells = 20
    thickness=0.5/num_shells
    for i in range(num_shells):
        # samples = sample_shell_between_cubes(dim, dist=0.5/num_shells, index=i, n_samples=1000)
        samples = np.random.uniform(0.5+i*thickness, 0.5+(i+1)*thickness,size=(10000,12))
        mc_values = [op_eval.eval(alpha, pysgpp.DataVector(x)) for x in samples]
        mean_values_.append(np.mean(mc_values)) 
        mc_integral = np.mean(mc_values)
        mc_actual_values = [f(x) for x in samples]
        mean_actual_values_.append(np.mean(mc_actual_integral))
        mc_actual_integral = np.mean(mc_actual_values)
        mean_error = np.mean(np.abs(np.array(mc_values)-np.array(mc_actual_values)))
        dist = thickness*i
        print(f'dist: {dist}, interp integral:{mc_integral}, truest integral:{mc_actual_integral}')        
        dist_.append(dist)
        mc_actual_integral_.append(mc_actual_integral)
        mc_integral_.append(mc_integral)
        mean_error_.append(mean_error)
        if mc_integral < mc_actual_integral:
            integral_ratio_.append(mc_integral/mc_actual_integral)
        elif mc_actual_integral < mc_integral:
            integral_ratio_.append(mc_integral/mc_actual_integral)
        else:
            integral_ratio_.append(1)

    fig = plt.figure()
    plt.plot(dist_, mc_integral_, label='MC Interpolation Integral')
    plt.plot(dist_, mc_actual_integral_, label='MC Truest Integral')
    plt.yscale('symlog')
    plt.xlabel('hypercube with distance from center, 0.5')
    plt.ylabel('integral')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(cycle_dir, sasg_type+'_integral_different_cube_size.png'), dpi=300)

    fig = plt.figure()
    plt.plot(dist_, mean_error_, label='mean error')
    plt.plot(dist_, mean_values_, label='mean values')
    plt.plot(dist_, mean_actual_values_, label='mean actual values')
    
    plt.yscale('symlog')
    plt.xlabel('hypercube with distance from center, 0.5')
    plt.ylabel('mean error')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(cycle_dir, sasg_type+'_mean_error_different_cube_size.png'), dpi=300)

    fig = plt.figure()
    plt.plot(dist_, integral_ratio_)
    plt.ylabel('intepolant integral / true integral')
    plt.xlabel('hypercube with distance from center, 0.5')
    plt.tight_layout()
    fig.savefig(os.path.join(cycle_dir, sasg_type+'_integral_ratio_different_cube_size.png'), dpi=300)

def montecarlo_UQ_test(base_run_dir, sasg_type='get_sasg'):
    # When adding the gaussian Px to the expectation very large expectations were gained, x10⁹ when only order 1 should found.
    # to test the expectations of Px and fx I will use monte carlo techniques.
    # I will check the values of the functions used to get the integral to see if there are any large ones, I do this with a histogram
    
    cycle_dirs = get_cycle_dirs(base_run_dir)
    cycle_dir = cycle_dirs[-1]
    
    sasg = eval(f"{sasg_type}(cycle_dir)")
    
    fxu = lambda unit_point: sasg.lookup_function(unit_point, unit_or_box='unit')
    fxu_ans = []
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        fxu_ans.append(float(fxu(unit_point)))
        
    
    fxp = lambda unit_points: sasg.surrogate_predict(sasg.points_transform_unit2box(unit_points))
    
    unit_points = np.random.uniform(size=(100000,12))
    # from scipy.stats import norm, truncnorm
    # trunc = truncnorm(a=-2, b=2, loc=0.5, scale=0.25)
    # unit_points = trunc.rvs(size=(100000,12))
    print('debug unit points shape', unit_points.shape)
    box_points = sasg.points_transform_unit2box(unit_points)
    print('debug box_points shape', box_points.shape)
    
    Px = sasg.unit_truncnorm_pdf
    Px_ans=[Px(x) for x in unit_points]
    
    nc, nr, w, h = 4, 1, 5, 5
    fig, AX = plt.subplots(nr,nc, figsize=(w*nc,h*nr))
    integral_approx = np.mean(Px_ans)
    quad_integral_approx = sasg.quadrature_function_integral(Px, acted_on='unit_point')
    AX[0].hist(Px_ans, bins=20, density=True)
    AX[0].set_yscale('log')
    AX[0].set_title(f'Px: mean:{np.round(integral_approx,2)}, \nquad_int:{np.round(quad_integral_approx,2)}')
    
    fxp_ans = sasg.surrogate_predict(box_points)
    AX[1].hist(fxp_ans, bins=20, density=True)
    AX[1].set_yscale('log')
    AX[1].set_title(f'fxp, mean:{np.round(np.mean(fxp_ans),2)}, \nquad_int:{np.round(sasg.quadrature_function_integral(sasg.lookup_function))}')

    # MMMGrunner = get_MMMGrunner(base_run_dir)
    # true_mean = MMMGrunner.get_expectation(do_gaussian=True)
    AX[2].hist(Px_ans*fxp_ans, bins=20, density=True)
    def pxfx(unit_point):
        box_point = sasg.point_transform_unit2box(unit_point)
        return Px(unit_point)*sasg.lookup_function(box_point)
    AX[2].set_title(f"Px*fxp, mean:{np.round(np.mean(Px_ans*fxp_ans),2)}, \nquad int: {np.round(sasg.quadrature_function_integral(pxfx, acted_on='unit_point'))}")
    AX[2].set_yscale('log')
    
    AX[3].hist(fxu_ans, bins=20, density=True)
    AX[3].set_title(f'fxu, mean:{np.round(np.mean(fxu_ans),2)}')
    fig.tight_layout()
    fig.savefig(os.path.join(cycle_dir, sasg_type+'_montecarlo_UQ_test.png'))
    
    # look at the trunc norm as I am getting pdf values of most of the time
    nominal = np.repeat(0.5,12)
    # print(nominal, nominal.shape)
    x = np.stack([nominal for i in range(1000)])
    xi = np.linspace(0,1,1000)
    x.T[0] = xi
    # print(x.T[0])
    fig = plt.figure()
    # print(x, x.shape)
    
    plt.plot(x.T[0],[Px(xi) for xi in x])
    plt.title('truncnorm')
    plt.tight_layout()
    fig.savefig(os.path.join(cycle_dir,sasg_type+'_truncnorm.png'), dpi=300)

def trunc1d(base_run_dir):
    
    cycle_dirs = get_cycle_dirs(base_run_dir)
    cycle_dir = cycle_dirs[-1]
    
    # Trunc norm looks weird, mostly 0 then goes up to 10**8 for a half gaussian truncated at 0.5 and 0.75
    # trunc_gaussian = truncnorm(a=np.zeros(self.dim), b=np.ones(self.dim), loc=np.repeat(0.5,self.dim), scale=np.repeat(0.25,self.dim))
    # np.prod(trunc_gaussian.pdf(unit_point))
    from scipy.stats import norm, truncnorm
    trunc = truncnorm(a=-3, b=3, loc=0.5, scale=0.5/3)
    x1 = np.linspace(0,1,100)
    pdf = trunc.pdf(x1)
    fig = plt.figure()
    plt.plot(x1,pdf)
    fig.savefig(os.path.join(cycle_dir, 'trunc1d.png'))

def weighted_mean_test(base_run_dir, sasg_type='get_sasg', gaussian_input_uncertanties=False):
    print('PERFROMING WEIGHTED MEAN TEST')
    # there was an issue with the surrogate giving wild predictions and this was throwing off the expectation via quadrature
    # when uniform sampling and taking the mean it converged much faster, this assumes a block type surrogate model
    # to do this for the non uniform sasg sampling method that concentrates at discontinuities we need to weight the mean to account for the non uniform sampling
    # E(fx) = [ sum(fx * px / qx) / sum(px/qx) ] if px is uniform dist then E(fx) = [ sum(fx * 1 / qx) / sum(1/qx) ]. 
    # The /sum(px/qx) is to account for the fact that px and qx usually come from continutous distributions where the integral equals one and for out discrete samples we need the sum of the discrete evaluations to equal one.
    # Var = E(fx**2) + E(fx)**2 
    
    if gaussian_input_uncertanties:
        trunc_gaussian = truncnorm(a=0, b=1, loc=0.5, scale=0.25)
        unit_points = sasg.points_transform_box2unit(list(sasg.train.values()))
        px = trunc_gaussian.pdf(unit_points)
    else:
        px=1
    
    # doing random comparison
    random_means, random_num_samples = get_random_comparison(base_run_dir)
    truest_mean = random_means[-1]    

    # x_test, y_test = get_test_set(base_run_dir)
    # truest_mean = np.nanmean(y_test)
    
    save_dir = os.path.join(base_run_dir,sasg_type+'_weighted_mean_test')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    
    cycle_dirs = get_cycle_dirs(base_run_dir)
    weighted_means = []
    num_samples = []
    for cycle_dir in cycle_dirs:
        sasg = eval(f"{sasg_type}(cycle_dir)")
        samples_x = np.array(list(sasg.train.keys()))
        samples_fx = np.array(list(sasg.train.values()))
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(samples_x)
        samples_qx = np.exp(kde.score_samples(samples_x))
        weighted_mean = ( np.sum(samples_fx * (px/samples_qx)) / sum((px/samples_qx)) )
        # weighted_var = ( np.sum(samples_fx**2 * samples_qx**-1) / sum(samples_qx**-1) ) + weighted_mean**2
        
        # non_weighted_mean = (1/len(samples)) * np.sum(samples_fx)
        weighted_means.append(weighted_mean)
        num_samples.append(len(sasg.train))
    figure = plt.figure()
    plt.plot(num_samples, weighted_means,'-o')
    plt.plot(random_num_samples, random_means, '-o', label='Comparison, Random')
    plt.hlines(truest_mean, np.min(num_samples), np.max(num_samples), linewidths=6)
    plt.xlim(0,np.max(num_samples))
    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Weighted Mean, {value_of_interest}')
    plt.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(save_dir,'weighted_mean.png'))
    plt.close(figure)
    
    figure = plt.figure()
    plt.plot(num_samples, np.abs(np.array(weighted_means)-truest_mean), '-o')
    plt.plot(random_num_samples, np.abs(np.array(random_means)-truest_mean), '-o', label='Comparison, Random')
    # plt.hlines(truest_mean, np.min(num_samples), np.max(num_samples), linewidths=6)
    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Weighted Mean Error, {value_of_interest}')
    plt.xlim(0,np.max(num_samples))
    plt.yscale('log')
    plt.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(save_dir,'weighted_mean_error.png'))
    plt.close(figure)

    figure = plt.figure()
    plt.plot(num_samples[1:], np.abs(np.diff(np.array(weighted_means))), '-o')
    plt.plot(random_num_samples[1:], np.abs(np.diff(np.array(random_means))), '-o', label='Comparison, Random')
    # plt.hlines(truest_mean, np.min(num_samples), np.max(num_samples), linewidths=6)
    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Weighted Mean Diff, {value_of_interest}')
    plt.xlim(0,np.max(num_samples))
    plt.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(save_dir,'weighted_mean_diff.png'))
    plt.close(figure)
    print('FINISHED WEIGHTED MEAN TEST')
        
    
def sasg_test(base_run_dir, cycle_num='all', isMMMG=False, name='', sasg_type='get_sasg'):
    print('STARTING SASG TEST')
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    parameters_labels = get_parameters_labels(base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    num_cycle_dirs = len(cycle_dirs)
    # try:
    #     if name=='boundary_tree_':
    #         df_cycle_info = pd.read_csv(os.path.join(base_run_dir, name+'all_cycle_info.csv'))
    #     else:
    #         df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    # except:
    #     try:
    #         if name=='boundary_tree_':
    #             df_cycle_info = pd.read_csv(os.path.join(base_run_dir, name+'all_post_cycle_info.csv'))
    #         else:
    #             df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_post_cycle_info.csv'))
    #     except:
    #         raise FileNotFoundError('neither',os.path.join(base_run_dir, 'all_cycle_info.csv'), '\n or this was found', os.path.join(base_run_dir, 'all_post_cycle_info.csv'))
            
    save_dir = os.path.join(base_run_dir, name+'sasg_test')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if cycle_num=='latest':
        cycle_dirs = [cycle_dirs[-1]]
    elif type(cycle_num)==type(1):
        # cycle_dirs = [f'active_cycle_{cycle_num}']
        cycle_dirs = cycle_dirs[-cycle_num:]
    elif 'every' in cycle_num:
        every_int = int(cycle_num.split('-')[-1])
        cycle_dirs = [cd for i,cd in enumerate(cycle_dirs) if i%every_int==0]
    if type(cycle_num)==type([]):
        cycle_dirs = cycle_dirs[cycle_num]

    # Get test points from parent function if they exist
    test_x, test_y = get_test_set(base_run_dir)
        
    MAPE=[]
    ME=[]
    RMSE=[]
    SURP = []
    PEARS_COEFF=[]
    nr=1 
    nc=6
    w=5
    h=4.8
    compare_index = np.linspace(0,len(cycle_dirs), nc*nr).astype('int')
    compare_index[-1] -= 1
    subfig, AX = plt.subplots(nr,nc, figsize=(nc*w,nr*h), sharey=True, sharex=True)
    compare_index2 = np.array([0])
    key_outliers = np.empty((3,len(parameters)))
    num_samples = []
    for i, cycle_dir in enumerate(cycle_dirs):
        print('='*100, '\n SPARSE GRID TEST','\n CYCLE DIR:',cycle_dir, '||', i, 'OUT OF:', len(cycle_dirs), '='*100)
        # sasg = get_sasg(os.path.join(base_run_dir,cycle_dir), name=name)
        sasg=eval(f"{sasg_type}('{os.path.join(base_run_dir,cycle_dir)}')")
        num_samples.append(len(sasg.train))
        if not os.path.exists(os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')):
            pass
        
        SURP.append(np.mean(np.abs(np.array(sasg.alpha.array()))))
        # eval = lambda point: sasg.surrogate_predict([tuple(point)])[0]
        eval_many = lambda points: sasg.surrogate_predict(points)
        
        test_pred = eval_many(test_x)
        residuals = test_y - test_pred
        squared_error = residuals**2
        percentage_error = np.abs(residuals/test_y)*100
        
        def plot_error(error, name='error'):
            out = []
            hb_comp=None
            x_test, y_test = get_test_set(base_run_dir)
            fig = plt.figure()
            hb = plt.hexbin(test_y, error, gridsize=50, cmap='plasma', bins=None, mincnt=1)
            error = np.abs(error)
            error_std = np.sqrt(np.var(error))
                        
            if name=='Residuals':
                indexes=np.arange(len(test_y))
                x_outlier = np.array(x_test[error>2*error_std])#error_std]) 
                out.append(x_outlier)
                fig_ph, AX_ph = plt.subplots(1,len(parameters), figsize=(5*len(parameters),5))
                for k, p in enumerate(parameters_labels):
                    AX_ph[k].hist(x_outlier.T[k])
                    AX_ph[k].set_xlabel(p)
                    AX_ph[k].set_xlabel(p)
                    AX_ph[k].set_title(f'N. outliers: {len(x_outlier)}')
                fig_ph.tight_layout()
                fig_ph.savefig(os.path.join(base_run_dir,cycle_dir, sasg_type+'_'+name+'_outliers_hist.png'),dpi=200)
                plt.close(fig_ph)
                if i in compare_index:
                    print('COMPARING CYCLE DIR:',cycle_dir)
                    hb_comp = AX.flat[compare_index2[0]].hexbin(test_y, error, gridsize=50, cmap='plasma', bins=None, mincnt=1)
                    # AX.flat[compare_index2[0]].scatter(test_y[error>2*error_std], error[error>2*error_std], color = 'black')
                    if compare_index2[0]==0:
                        AX.flat[compare_index2[0]].set_ylabel(f'{name}, {value_of_interest}')
                    AX.flat[compare_index2[0]].set_xlabel(f'{parent_model}, {value_of_interest}')
                    AX.flat[compare_index2[0]].set_title(f"N. GENE Eval: {len(sasg.train)}")
                    if i == compare_index[-1]:
                        y_outlier = np.array(y_test[error>2*error_std])
                        figy = plt.figure()
                        plt.hist(y_outlier)
                        plt.xlabel(value_of_interest)
                        plt.ylabel('frequency of outliers')
                        figy.tight_layout()
                        figy.savefig(os.path.join(save_dir, f'{value_of_interest}_outlier_hist.png'), dpi=300)
                        plt.close(figy)
                        
                        key_outliers_indexes = [np.argmax(error[error>2*error_std]), np.argsort(error[error>2*error_std])[len(error[error>2*error_std])//2], np.argmin(error[error>2*error_std])]
                        key_outliers = x_outlier[key_outliers_indexes]
                        out.append(key_outliers)
                        AX.flat[compare_index2[0]].scatter(test_y[error>2*error_std][key_outliers_indexes], error[error>2*error_std][key_outliers_indexes], marker='x', color='black', label='key outliers')
                        AX.flat[compare_index2[0]].hlines(2*error_std, np.min(test_y), np.max(test_y), linewidths=6)
                        # Create a divider for the existing axes instance
                        divider = make_axes_locatable(AX.flat[compare_index2[0]])
                        # Append a new axes to the right of the current axes, with 5% width of the current axes
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        # Create the colorbar in the new axes
                        plt.colorbar(hb_comp, cax=cax)
                    compare_index2[0] += 1
            plt.ylabel(f'{name}, {value_of_interest}')
            plt.xlabel(f'{parent_model}, {value_of_interest}')
            # Add colorbar
            cb = plt.colorbar(hb)
            cb.set_label('Test Point Density')
            
            fig.tight_layout()
            fig.savefig(os.path.join(base_run_dir, cycle_dir, f'{sasg_type}_{name}_plot_hexbin.png'), dpi=200)
            plt.close(fig)

            fig = plt.figure()
            plt.scatter(test_y, error)
            plt.ylabel(f'{name}, {value_of_interest}')
            plt.hlines(error_std, np.min(test_y), np.max(test_y))
            plt.xlabel(f'{parent_model}, {value_of_interest}')
            fig.tight_layout()
            fig.savefig(os.path.join(base_run_dir, cycle_dir, f'{sasg_type}_{name}_plot_scatter.png'), dpi=200)
            plt.close(fig)
            return out
        
        plot_error(squared_error, name='Squared_Error')
        plot_error(percentage_error, name='Percentage_Error')
        out = plot_error(residuals, name='Residuals')
        if len(out)>1:
            x_outlier, key_outliers = out[0], out[1]
        else:
            x_outlier = out[0]
            
        mape = np.mean(np.abs(residuals/test_y)*100)
        me = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        pears_coeff = pearson_r(test_y, test_pred)
        MAPE.append(mape)
        ME.append(me)
        RMSE.append(rmse)
        PEARS_COEFF.append(pears_coeff)
    subfig.tight_layout()
    
    print('saving subfig:', os.path.join(base_run_dir,'hex_residuals_all.png'))
    subfig.savefig(os.path.join(save_dir,'hex_residuals_all.png'), dpi=200)
    plt.close(subfig)
    
    def plot_errorVsamples(error, name):
        fig = plt.figure()
        plt.plot(num_samples, error, '-o')
        plt.xlabel(f'Number of {parent_model} Evaluations')
        plt.ylabel(f'{value_of_interest}, {name}')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{name}.png'),dpi=200)
        plt.close(fig)
    
    
    plot_errorVsamples(MAPE, 'Mean_Average_Percentage_Error')
    plot_errorVsamples(ME, 'Mean Err')
    plot_errorVsamples(RMSE, 'Root_Mean_Squared_Error')
    plot_errorVsamples(SURP, 'Mean Surplus')
    plot_errorVsamples(PEARS_COEFF, 'Pearson Correlation Coefficient')
    print('FINISHED WEIGHTED MEAN TEST')
    
    if isMMMG:
        plot_outlier_slices_MMMG(base_run_dir, key_outliers, x_outliers=x_outlier, cycle_dirs=np.array(cycle_dirs)[compare_index], base_save_dir=save_dir, sasg_type=sasg_type)
    
    if len(sasg.anchor_boundary_points) > 0:
        dif = []
        for box_point, value in sasg.anchor_boundary_points:
            dif.append(sasg.predict(box_point))
        print('BOUNDARY INFERENCE TEST, should be 0',dif[0], np.sum(dif))
    
    print('FINISHED SASG TEST')

def pearson_r(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]    
    
def parse_run_dir(run_dir, parameters):
    parser = GENEparser()
    parameter_nml_map={
        'kymin':('box','kymin'),
        'omn1':('_grp_species_0','omn'),
        'omn':('_grp_species_0','omn'), #!!!! careful omn could be any species
        'omn2':('_grp_species_1','omn'),
        'omt1':('_grp_species_0','omt'),
        'omt2':('_grp_species_1','omt'),
        'coll':('general','coll'),
        'beta':('general','beta'),
        'q0':('geometry','q0'),
        'zeff':('general','zeff'),
        'shat':('geometry','shat'),
        'temp1':('_grp_species_0','temp'),
        'kappa':('geometry','kappa'),
        's_kappa':('geometry','s_kappa'),
        'delta':('geometry','delta'),
        'x0':('box','x0')
    }
    # read relevant output values
    report_path=os.path.join(run_dir,'early_stopping_report.csv')    
    sure = False
    df = pd.read_csv(report_path)
    while not sure: 
        if os.path.exists(os.path.join(run_dir, 'omega.dat')) and not df['stable'].loc[0]:
            ky, growthrate, frequency = parser.read_omega(run_dir, '.dat')
            output = [str(growthrate)]
            # print('setting output 1', output)
            sure=True
        elif os.path.exists(os.path.join(run_dir,f'early_converged_omega')):
            with open(os.path.join(run_dir,f'early_converged_omega'),'r') as file:
                line = file.read()
                growthrate, frequency = [float(value) for value in line.split(',')]
            output = [str(growthrate)]
            # print('setting output 2', output)
            sure=True
        elif df.iloc[0, 'timeout_reached']:
            output = ['timeout_reached']
            # print('setting output 3', output)
            sure = True
        elif df.iloc[0, 'stable']:
            output = ['stable']
            # print('setting output 4', output)
            sure=True
    
    namelist = f90nml.read(os.path.join(run_dir, 'parameters'))
    params = {}

    for pa in parameters:
        grp, var = parameter_nml_map[pa]
        params[pa] = namelist[grp][var]
    
    params_list = [str(v) for k,v in params.items()]
    return_list = params_list + output
    # print('runner returning', ','.join(return_list))
    # print('params', params)
    # print('output', output)
    return ','.join(return_list)

def get_predict_from_cycle_dir(cycle_dir, sasg_type='get_sasg'):
    sasg = eval(f"{sasg_type}(cycle_dir)")
    grid_file_path = os.path.join(base_run_dir,cycle_dir, 'pysgpp_grid.txt')
    surpluses_file_path = os.path.join(base_run_dir,cycle_dir, 'surpluses.mat')
    train_points_file = os.path.join(base_run_dir,cycle_dir, 'train_points.pkl')
    with open(grid_file_path, 'r') as file:
        serialized_grid = file.read()
        sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
        surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
        sasg.alpha = surpluses
    # eval = lambda point: sasg.surrogate_predict([tuple(point)])[0]
    return sasg.surrogate_predict

# def get_sasg(cycle_dir):
#     # print('GETTING sasg FROM:', cycle_dir)
    
#     base_run_dir = os.path.dirname(cycle_dir)
#     listdir = os.listdir(base_run_dir)
#     config_file_name = [name for name in listdir if '.yaml' in name]
#     if len(config_file_name) > 1:
#         raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
#     config_file_name = config_file_name[0]
#     config = load_configuration(os.path.join(base_run_dir, config_file_name))
#     bounds=np.array(config.sampler['bounds'])
#     parameters = config.sampler['parameters']
    
#     sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    
#     grid_file_path = os.path.join(cycle_dir, 'pysgpp_grid.txt')
#     surpluses_file_path = os.path.join(cycle_dir, 'surpluses.mat')
#     train_points_file = os.path.join(cycle_dir, 'train_points.pkl')
#     with open(grid_file_path, 'r') as file:
#         serialized_grid = file.read()
#         sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
#         sasg.gridStorage = sasg.grid.getStorage()
#         sasg.gridGen = sasg.grid.getGenerator()
#         surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
#         sasg.alpha = surpluses
#     with open(train_points_file, 'rb') as file:
#         sasg.train = pickle.load(file)
#     return sasg

import pickle
def get_predict_new_poly_degree(cycle_dir, new_poly_degree):
    sasg = get_sasg(cycle_dir) 
    new_grid = pysgpp.Grid.createModPolyGrid(sasg.dim, new_poly_degree)
    new_grid_storage = new_grid.getStorage()
    # Transfer points from old grid to new grid
    new_alpha = pysgpp.DataVector(sasg.gridStorage.getSize())
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        new_grid_storage.insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        box_point = sasg.point_transform_unit2box(unit_point) 
        new_alpha[i] = sasg.train[box_point]
    pysgpp.createOperationHierarchisation(new_grid).doHierarchisation(new_alpha)
    sasg.grid = new_grid
    sasg.alpha = new_alpha
    print('sasg degree',sasg.grid.getDegree())
    return sasg.surrogate_predict


def get_predict_with_bounds(cycle_dir):
    sasg = get_sasg(cycle_dir) 
    # new_grid = pysgpp.Grid.createModPolyGrid(sasg.dim, 3)
    # new_grid_storage = new_grid.getStorage()
    # Transfer points from old grid to new grid
    
    # first get a python version of alpha to be appended to as we add a grid point
    
    def add_boundary_points(grid, add_level='all'):
        print('add_level',add_level, 'num points before:', grid.getStorage().getSize())        
        num_bound_points=0
        storage = grid.getStorage()
        generator = grid.getGenerator()
        dim = storage.getDimension()
        for i in range(storage.getSize()):
            gp = storage.getPoint(i)
            for d in range(dim):
                level = gp.getLevel(d)
                if gp.isLeaf:
                    if level==add_level or add_level=='all':
                        # Left boundary
                        left_gp = pysgpp.HashGridPoint(gp)
                        left_gp.set(d, level, 0)
                        if not storage.isContaining(left_gp):
                            storage.insert(left_gp)
                            num_bound_points += 1

                        # Right boundary
                        right_gp = pysgpp.HashGridPoint(gp)
                        right_gp.set(d, level, 2 ** level)
                        if not storage.isContaining(right_gp):
                            storage.insert(right_gp)
                            num_bound_points += 1
        print('num points after', grid.getStorage().getSize())

    maxLevel = sasg.grid.getStorage().getMaxLevel()
    print('maxLevel', maxLevel)    
    add_boundary_points(sasg.grid, add_level='all')
    # fill a new alpha
    new_alpha = pysgpp.DataVector(sasg.gridStorage.getSize())
    for i in range(sasg.gridStorage.getSize()):
        gp = sasg.gridStorage.getPoint(i)
        unit_point = ()
        for j in range(sasg.dim):
            unit_point = unit_point + (gp.getStandardCoordinate(j),)
        box_point = sasg.point_transform_unit2box(unit_point) 
        if box_point in sasg.train:
            new_alpha[i] = sasg.train[box_point]
        else:
            new_alpha[i] = np.mean(list(sasg.train.values()))
    pysgpp.createOperationHierarchisation(sasg.grid).doHierarchisation(new_alpha)   
    # sasg.grid = new_grid
    sasg.alpha = new_alpha
    return sasg.surrogate_predict

def outlier_investigation(base_run_dir, sasg_type='get_sasg'):
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    parameters_labels = get_parameters_labels(base_run_dir)
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    cycle_dirs = get_cycle_dirs(base_run_dir)
    
    test_x, test_y = get_test_set(base_run_dir)      
    
    compare_index = np.linspace(0,len(cycle_dirs), 3).astype('int')
    compare_index[-1]-=1
    cycle_dirs = np.array(cycle_dirs)[compare_index]

    figure, AX = plt.subplots(1,5, figsize=(5*5,4.8), )
    
    for i, cycle_dir in enumerate(cycle_dirs):
        sasg = eval(f"{sasg_type}(cycle_dir)")
        pred_y = sasg.surrogate_predict(test_x)
        error = np.abs(pred_y-test_y)
        error_std = np.sqrt(np.nanvar(error))
        
        hb_comp = AX.flat[i].hexbin(test_y, error, gridsize=50, cmap='plasma', bins=None, mincnt=1)
        if i==0:
            AX.flat[i].set_ylabel(f'{value_of_interest}, Pred Err')
        AX.flat[i].set_xlabel(f'{parent_model}, {value_of_interest}')
        AX.flat[i].set_title(f"N. GENE Eval: {df_cycle_info['num_samples'].loc[compare_index[i]]}")
        if i > 0:
            AX.flat[i].sharey(AX.flat[i-1])
        if i == 0:
            # Create a divider for the existing axes instance
            divider = make_axes_locatable(AX.flat[i])
            # Append a new axes to the right of the current axes, with 5% width of the current axes
            cax = divider.append_axes("right", size="5%", pad=0.05)
            # Create the colorbar in the new axes
            plt.colorbar(hb_comp, cax=cax, label='Point Density')
        # if i ==2:
    AX.flat[i].hlines(2*error_std, np.nanmin(test_y), np.nanmax(test_y), linestyles='dashed', linewidths=4, color='red', zorder=10, label='Outlier Thres')
    AX.flat[i].legend(fontsize=20)
    x_outlier = np.array(test_x[error>2*error_std])
    parameter_index = 4
    omte = parameters_labels[parameter_index]
    AX.flat[i+1].hist(x_outlier.T[parameter_index])
    AX[i+1].set_xlabel(omte)
    AX[i+1].set_ylabel('N. outliers')
    # AX[i+1].set_title(f'N. outliers: {len(x_outlier)}')
    
    sasg = get_sasg(cycle_dirs[-1])
    listdir = os.listdir(base_run_dir)
    slice1d_dir = None
    for di in listdir:
        if 'slice1d' in di:
            slice1d_dir = os.path.join(base_run_dir,di)
            break
    slices = None
    if slice1d_dir != None:
        print('slice1d_dir:',slice1d_dir)
        slices = slice1d_post_proc(slice1d_dir)
    slices2 = slice1d_post_proc('/scratch/project_462000451/enchanted_test_out/gene_slice1d_grad_EPS')
    slices[4] = slices2[1]
    
    ax_slice = plot_single_slice(which=parameter_index,ax=AX[i+2],function=sasg.surrogate_predict, bounds=bounds, dimension_labels=parameters_labels, ylabel=value_of_interest, parent_model=parent_model, slices=slices)    
    
    figure.tight_layout()
    figure.savefig(os.path.join(base_run_dir, 'outlier_investigation.png'), dpi = 200)
    plt.close(figure)
    
    
def get_cycle_dirs(base_run_dir):
    listdir = os.listdir(base_run_dir)
    cycle_dirs = [d for d in listdir if 'active_cycle_' in d]
    
    ordinal = [int(d.split('_')[-1]) for d in cycle_dirs]
    
    cycle_dirs = np.array(cycle_dirs)[np.argsort(ordinal)]
    cycle_dirs = [os.path.join(base_run_dir,cycle_dir) for cycle_dir in cycle_dirs]
    return cycle_dirs

def get_random_comparison(base_run_dir):
    print('RETRIVING RANDOM COMPARISON FROM', base_run_dir)
    config = get_config(base_run_dir)
    test_dir = config.sampler['test_dir']    
    # listdir = os.listdir(base_run_dir)
    # test_dir = None
    # for di in listdir:
    #     if 'sobolseq' in di or 'comparison' in di:
    #         test_dir = os.path.join(base_run_dir,di)
    #         break
    # if test_dir == None:
    #     raise FileNotFoundError('NO FOLDERS WITH sosbol_seq OR comparison IN NAME, THAT COULD CONTAIN A TEST SET WERE FOUND')
    cycle_dirs = get_cycle_dirs(test_dir)
    means = []
    num_samples = []
    output_values = []
    for cycle_dir in cycle_dirs:
        df = pd.read_csv(os.path.join(cycle_dir,'runner_return.txt'))
        output_values = output_values + list(df[df.columns[-1]].to_numpy())
        mean = np.nanmean(output_values)
        means.append(mean)
        num_samples.append(len(output_values))
    return means, num_samples 

def get_test_set(base_run_dir):
    print('RETRIVING TEST SET FROM', base_run_dir)
    config = get_config(base_run_dir)
    test_dir = config.sampler['test_dir']
    print('USING TEST DIR:',test_dir)
    
    # listdir = os.listdir(base_run_dir)
    # test_dir = None
    # for di in listdir:
    #     if 'sobolseq' in di or 'testset' in di:
    #         test_dir = os.path.join(base_run_dir,di)
    #         break
    # if test_dir == None:
    #     raise FileNotFoundError('NO FOLDERS WITH sosbol_seq OR testset IN NAME, THAT COULD CONTAIN A TEST SET WERE FOUND')
    # print('FOUND TEST DIR', test_dir)
    
    file_path = os.path.join(test_dir,'merged_runner_return.csv')
    file_path_txt = os.path.join(test_dir, 'merged_runner_return.txt')
    if os.path.exists(file_path):        
        df_test = pd.read_csv(file_path)
        print('got merged_runner_return.csv', file_path)
    elif os.path.exists(file_path_txt):
        df_test = pd.read_csv(file_path_txt)
        print('got merged_runner_return.txt', file_path_txt)
    # elif os.path.exists(os.path.join(test_dir, 'runner_return.txt')):
    #     df_test = pd.read_csv(os.path.join(test_dir, 'runner_return.txt'))
    #     print('got runner_return.txt')    
    elif os.path.exists(os.path.join(test_dir, 'active_cycle_0', 'runner_return.txt')):
        print('MERGING RUNNER RETURNS')
        test_cycle_dirs = get_cycle_dirs(test_dir)
        dfs = []
        for cycle_dir in test_cycle_dirs:
           df = pd.read_csv(os.path.join(cycle_dir,'runner_return.txt'))
           dfs.append(df)
        df_test = pd.concat(dfs)
        df_test.to_csv(os.path.join(test_dir, 'merged_runner_return.txt')) 
    else:
        print('NO RUNNER RETURN FOUND, BEGINNIGN PARSING')
        finished_result = find_files(test_dir, 'GENE.finished')
        stopped_result = find_files(test_dir, 'stopped_by_monitor')
        result = finished_result + stopped_result
        run_dirs = [os.path.dirname(path) for path in result]
        if len(result) == 0:        
            raise FileNotFoundError(f'NO RUNNER RETURN PATH WAS FOUND IN:{test_dir},\n ALSO THERE SEEM TO BE NO FINNISHED OR EARLY STOPPED GENE RUNS IN: {test_dir}\n{file_path}\n{file_path_txt}')
        else:
            outputs = []
            for i, run_dir in enumerate(run_dirs):
                if i % 10 == 0:
                    print('NUMBER OF RUN_DIR PARSED:',i)
                outputs.append(parse_run_dir(run_dir, parameters))
            with open(os.path.join(test_dir, 'merged_runner_return.txt'), 'w') as file:
                lines = [runner_return_headder] + outputs
                lines = [line+'\n' for line in lines]
                file.writelines(lines)
            df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
    
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    test_x = np.array(df_test.iloc[:,0:-1].astype('float'))
    test_y = np.array(df_test.iloc[:,-1].astype('float'))
    
    # sasg = get_sasg(os.path.join(base_run_dir,'active_cycle_0'))
    # unit_test_x = sasg.points_transform_box2unit(test_x)
    # parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    # closest_boarder_distance = []
    # for tx in unit_test_x:
    #     lower = np.abs(tx-bounds.T[0])
    #     upper = np.abs(tx-bounds.T[1])
    #     closest_boarder_distance.append(np.min([lower,upper]))
    # closest_boarder_distance = np.array(closest_boarder_distance)
    # test_x = test_x#[closest_boarder_distance>0.1]
    # test_y = test_y#[closest_boarder_distance>0.1]
    test_x = test_x[~np.isnan(test_y)]
    test_y = test_y[~np.isnan(test_y)]
    if test_x.shape[1] <= 2:
        warnings.warn('test_x is 2D or less, this is probably not the correct dimensionality and can be caused by no runner_return_headder being at the top of the runner_return.txt file or in the configs file')
    return test_x, test_y

def get_rmse_boundary(base_run_dir, cycle_dir_index=None):
    print('PERFORMING POLY DEGREE HYPER SCAN FROM', base_run_dir)
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    test_x, test_y = get_test_set(base_run_dir)
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
          u'#bcbd22', u'#1f77b4', u'#8c564b', u'#e377c2']
    
    RMSE = []
    if cycle_dir_index!=None:
        cycle_dirs = [cycle_dirs[cycle_dir_index]]
        
    for cycle_dir in cycle_dirs:
        print('get_rmse_boundary: CYCLE DIR', cycle_dir)
        eval_many = get_predict_with_bounds(cycle_dir)
        pred_y = eval_many(test_x)
        # max_residuals.append(np.max(np.abs(test_y - pred_y)))
        # mean_residuals.append(np.mean(np.abs(test_y - pred_y)))
        rmse = np.sqrt(np.mean((test_y-pred_y)**2))
        RMSE.append(rmse)
        with open(os.path.join(cycle_dir,'boundary_rmse.txt'), 'w') as file:
            file.write(str(rmse))
    
    # plt.plot(df_cycle_info['num_samples'],max_residuals, '--', color=colors[i])
    plt.plot(df_cycle_info['num_samples'],RMSE, '-o')

    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Root Mean Square Error')
    #plt.legend()

    fig.tight_layout()
    print('SAVING PLOT:',os.path.join(base_run_dir,'rms_BOUNDARY.png'))
    fig.savefig(os.path.join(base_run_dir,'rms_BOUNDARY.png'),dpi=200)
    plt.close(fig)

def hyper_scan_poly_degree(base_run_dir, poly_degrees:list):
    print('PERFORMING POLY DEGREE HYPER SCAN FROM', base_run_dir)
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    test_x, test_y = get_test_set(base_run_dir)
    df_cycle_info = pd.read_csv(os.path.join(base_run_dir, 'all_cycle_info.csv'))
    fig = plt.figure()
    colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
          u'#bcbd22', u'#1f77b4', u'#8c564b', u'#e377c2']
    for i, poly_degree in enumerate(poly_degrees):
        print('POLY DEGREE:',poly_degree)
        max_residuals = []
        mean_residuals = []
        for cycle_dir in cycle_dirs:
            # print('CYCLE DIR', cycle_dir)
            eval_many = get_predict_new_poly_degree(cycle_dir, poly_degree)
            pred_y = eval_many(test_x)
            max_residuals.append(np.max(np.abs(test_y - pred_y)))
            mean_residuals.append(np.mean(np.abs(test_y - pred_y)))
        # plt.plot(df_cycle_info['num_samples'],max_residuals, '--', color=colors[i])
        plt.plot(df_cycle_info['num_samples'],mean_residuals, color=colors[i], label=f'Poly Degree: {poly_degree}')

    plt.xlabel(f'Number of {parent_model} Evaluations')
    plt.ylabel(f'Max Residual')
    #plt.legend()

    fig.tight_layout()
    print('SAVING PLOT:',os.path.join(base_run_dir,'hyper_scan_poly_degree.png'))
    fig.savefig(os.path.join(base_run_dir,'hyper_scan_poly_degree.png'),dpi=200)
    plt.close(fig)

def plot_outlier_slices_MMMG(base_run_dir, key_outliers, x_outliers, cycle_dirs=None, base_save_dir=None, sasg_type='get_sasg'):
    print('PLOTTING OUTLIER SLICES')
    if base_save_dir == None: 
        save_dir = os.path.join(base_run_dir,'outlier_slices')
    else:
        save_dir = os.path.join(base_save_dir,'outlier_slices')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if type(cycle_dirs) == type(None):    
        cycle_dirs = get_cycle_dirs(base_run_dir)
        
    config = get_config(base_run_dir)
    value_of_interest = config.general.get('value_of_interest', 'function value')
    parent_model = config.general.get('simulation_name', 'Parent Model')
    parameters, bounds = get_parameters_bounds(base_run_dir)
    from runners.MMMGrunner import MMMGrunner
    runner = MMMGrunner(**config.executor['static_executor']['runner'])
    
    cycle_dirs = list(set(cycle_dirs))
    for cycle_dir in cycle_dirs:
        print('plot_outlier_slices_MMMG, CYCLE DIR:', cycle_dir)
        sasg = eval(f"{sasg_type}(cycle_dir)")
        eval_many = lambda points: sasg.surrogate_predict(points)
        # eval_single = lambda point: sasg.surrogate_predict([tuple(point)])[0]
        fig_contours = plot_matrix_contour(function=eval_many, bounds=bounds, dimension_labels=parameters, points=x_outliers, indicies_to_do=None)
        fig_contours.suptitle(f"N_Evals_{len(sasg.train)}")
        fig_contours.tight_layout()
        fig_contours.savefig(os.path.join(save_dir,'contours_outliers.png'), dpi=300)
        plt.close(fig_contours)
        for i, outlier in enumerate(key_outliers):
            slices = runner.mmg.get_slices(nominals=outlier)
            fig_slices = plot_slices(function=eval_many, bounds=bounds, dimension_labels=parameters, ylabel=value_of_interest, parent_model=parent_model, slices=slices, nominals=outlier)
            
            title = f"N_Evals_{len(sasg.train)}_outlier_{i}"
            fig_slices.suptitle(title)
            fig_slices.tight_layout()
            fig_slices.savefig(os.path.join(save_dir, title))
            plt.close(fig_slices)

def inspect_outliers(base_run_dir):
    save_dir = os.path.join(base_run_dir, 'inspect_outliers')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
    cycle_dirs = get_cycle_dirs(base_run_dir)
    x_test, y_test = get_test_set(base_run_dir)
    sasg = get_sasg(cycle_dirs[-1])
    trueist_mean = np.mean([np.mean(list(sasg.train.values())), np.mean(y_test)])
    print('trueist mean:',trueist_mean)
    for cycle_dir in cycle_dirs:
        print('CYCLE DIR',cycle_dir)
        eval_many=get_predict_from_cycle_dir(cycle_dir)
        x_test, y_test = get_test_set(base_run_dir)
        y_pred = eval_many(x_test)
        residuals = np.abs(y_pred-y_test)
        indexes = np.flip(np.argsort(residuals))
        print('PRINTING WORST RESIDUALS')
        for i in indexes[0:3]:
            print('='*100)
            print('='*100)
            print(i, 'Residual:', residuals[i], 'y test:', y_test[i], 'y pred:', y_pred[i])
            for j, p in enumerate(parameters):
                print(p, x_test[i][j])
    
def test_boundary_hypothesis(base_run_dir, sasg_type='get_sasg'):
    cycle_dirs = get_cycle_dirs(base_run_dir)
    # cycle_dir = cycle_dirs[-1]
    
    for cycle_dir in [cycle_dirs[-1]]:
        sasg = eval(f"{sasg_type}(cycle_dir)")
        
        test_x, test_y = get_test_set(base_run_dir)
        unit_test_x = sasg.points_transform_box2unit(test_x)
        
        pred_y = sasg.surrogate_predict(test_x)
        residuals = np.abs(test_y - pred_y)
        parameters, bounds, parent_model, value_of_interest = get_config_info(base_run_dir)
        
        closest_boarder_distance = []
        average_boarder_distance = []
        for tx in unit_test_x:
            lower = np.abs(tx-bounds.T[0])
            upper = np.abs(tx-bounds.T[1])
            closest_boarder_distance.append(np.min([lower,upper]))
            average_boarder_distance.append(np.mean([np.mean(lower), np.mean(upper)]))
        
        fig = plt.figure()
        plt.hexbin(closest_boarder_distance, residuals, gridsize=50, cmap='plasma', bins=None, mincnt=1)
        plt.xlabel('Closest Boarder Distance')
        plt.ylabel('|Residual|')
        fig.tight_layout()
        fig.savefig(os.path.join(cycle_dir,sasg_type+'_boundary_hypothesis_hexbin.png'),dpi=200)
        plt.close(fig)

        # fig = plt.figure()
        # plt.hexbin(average_boarder_distance, residuals, gridsize=50, cmap='plasma', bins=None, mincnt=1)
        # plt.xlabel('Average Boarder Distance')
        # plt.ylabel('|Residual|')
        # fig.tight_layout()
        # fig.savefig(os.path.join(cycle_dir,sasg_type+'_avbd_boundary_hypothesis_hexbin.png'),dpi=200)
        # plt.close(fig)
        
        # fig = plt.figure()
        # plt.scatter(closest_boarder_distance, residuals)
        # plt.xlabel('Closest Boarder Distance')
        # plt.ylabel('|Residual|')
        # fig.tight_layout()
        # fig.savefig(os.path.join(cycle_dir,'boundary_hypothesis_scatter.png'),dpi=200)
        # plt.close(fig)

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    # outlier_investigation(base_run_dir)
    # weighted_mean_test(base_run_dir)
    
    # sasg_test(base_run_dir, cycle_num='every-30', isMMMG=True, name='boundary_tree_')
    # sasg_test(base_run_dir, cycle_num='every-30', isMMMG=True, name='_0_boundary')
    # sasg_test(base_run_dir, cycle_num='every-20', isMMMG=True, name='boundary_anchors_')

    sasg_test(base_run_dir, cycle_num='every-1', isMMMG=False, name='larger_set')

    # montecarlo_UQ_test(base_run_dir, sasg_type='get_sasg_zero_bounds')
    # trunc1d(base_run_dir)
    
    # MCvsQUAD(base_run_dir, sasg_type='get_sasg')
    # hyper_scan_poly_degree(base_run_dir,poly_degrees=[3,4,5,6,7,8,9,10,11,20])
    # inspect_outliers(base_run_dir)
    
    # test_boundary_hypothesis(base_run_dir)
    
    # get_rmse_boundary(base_run_dir)
    
    
    
    
    