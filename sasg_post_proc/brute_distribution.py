import sys
import os
import time
import argparse

sys.path.append('/users/danieljordan/enchanted-surrogates2/src')
sys.path.append('/users/danieljordan/DEEPlasma')
from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import Sobol
import yaml
import pysgpp
import numpy as np
from scipy.stats import entropy

plt.rcParams.update({'font.size': 18})  # Adjust number as needed



from joblib import Parallel, delayed
import time

# def process(i):
#     time.sleep(1)
#     return i * i

# results = Parallel(n_jobs=4)(delayed(process)(i) for i in range(10))
# print(results)


def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    print('LOADING CONFIGURATION FILE')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.executor["config_filepath"] = config_path
    return config

def brute_distribution(cycle_dir, parameters, bounds, n_jobs=1, prec=0.00001, fix_params=None, title=''):
    print('FINDING OUTPUT DISTRIBUTION BY BRUTE FORCE')
    sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
    
    grid_file_path = os.path.join(cycle_dir, 'pysgpp_grid.txt')
    surpluses_file_path = os.path.join(cycle_dir, 'surpluses.mat')
    # train_points_file = os.path.join(cycle_dir, 'train_points.pkl')
    with open(grid_file_path, 'r') as file:
        serialized_grid = file.read()
        sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
        surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
        sasg.alpha = surpluses
    eval_many = lambda points: sasg.surrogate_predict(points)
    # with open(train_points_file, 'rb') as file:
    #     train_points_dict = pickle.load(file)
    # train_points = np.array(list(train_points_dict.keys()))
    
    engine = Sobol(d=len(parameters), scramble=True)  # d = number of dimensions
    
    # Define the bounds for each dimension
    lower_bounds = np.array(bounds).T[0]
    upper_bounds = np.array(bounds).T[1]
    
    def more_samples(m):
        # start = time.perf_counter()
        samples = engine.random_base2(m)
        # Scale the points to the desired bounds
        scaled_points = lower_bounds + samples * (upper_bounds - lower_bounds)
        # end = time.perf_counter()
        # print(f'm={m},GET MORE SAMPLES TIME: {end-start} sec')
        if type(fix_params) != type(None):
            for p,v in fix_params.items():
                arg = parameters.index(p)
                scaled_points.T[arg] = v
                print('debug',arg,v,scaled_points[0:3])
        return scaled_points
    
    engine2 = Sobol(d=len(parameters), scramble=True)  # d = number of dimensions    
    many_samples = engine2.random_base2(20)
    ms_index = 0
    num_samples_per_call = 100
    def more_samples2(m):
        # m should start at 1 and rise by step 1
        return many_samples[m*100:m*100+100]


    def eval_para(samples):
        from samplers.SpatiallyAdaptiveSparseGrids import SpatiallyAdaptiveSparseGrids
        import pysgpp
        sasg = SpatiallyAdaptiveSparseGrids(bounds, parameters)
        with open(grid_file_path, 'r') as file:
            serialized_grid = file.read()
            sasg.grid = pysgpp.Grid.unserialize(serialized_grid)
            surpluses = pysgpp.DataVector.fromFile(surpluses_file_path)
            sasg.alpha = surpluses
            eval_many = lambda points: sasg.surrogate_predict(points)
        return eval_many(samples)

    first_m=10
    samples = more_samples(first_m)
    
    start = time.perf_counter()
    indices = np.linspace(0, len(samples), n_jobs + 1, dtype=int)
    results = Parallel(n_jobs=n_jobs)(delayed(eval_para)(samples[indices[i]:indices[i+1]]) for i in range(n_jobs))
    p1 = np.concatenate(results)
    # p1 = eval_many(samples)
    end = time.perf_counter()
    print(f'm={first_m}, EVALUATION TIME FOR', len(samples), 'POINTS:', end-start, 'sec')
    
    converged=False
    num_bins = 100
    entropy_difs = []
    num_samples = [len(p1)]
    last_m=200
    means=[np.mean(p1)]
    stds=[np.sqrt(np.var(p1))]
    first_m, last_m = 1, 20
    for m in range(first_m,last_m):
        # samples = np.concatenate((samples, more_samples(m)),axis=0)
        samples = np.concatenate((samples, more_samples2(m)),axis=0)
        start = time.perf_counter()
        # p2 = eval_many(samples)
        indices = np.linspace(0, len(samples), n_jobs + 1, dtype=int)
        results = Parallel(n_jobs=n_jobs)(delayed(eval_para)(samples[indices[i]:indices[i+1]]) for i in range(n_jobs))
        p2 = np.concatenate(results)
        end = time.perf_counter()
        print(f'm={m}, n_jobs={n_jobs}, {len(parameters)}D pysgpp EVALUATION TIME FOR', len(samples), 'POINTS:', end-start, 'sec')
        
        entropy_difs.append(relative_entropy(p2,p1, num_bins=num_bins))
        means.append(np.mean(p2))
        stds.append(np.sqrt(np.var(p2)))
        
        num_samples.append(len(samples))
        if len(means)>2:
            print(f'IS CONVERGED? MEAN DIFFS: {np.abs(np.diff(means[-2:]))}, STD DIFFS: {np.abs(np.diff(stds[-2:]))}, PREC: {prec}')
            if all(np.abs(np.diff(means[-2:]))<prec) and all(np.abs(np.diff(stds[-2:]))<prec):
                print('CONVERGED')
                converged = True
            # if m == last_m-1:
            #     converged = True
            if converged:
                # fig = plt.figure(dpi=200)
                # # hist_x = np.linspace(np.min(self.f),np.max(self.f), 1000)
                # n, bins, _ = plt.hist(p1, bins=num_bins, density=True)
                # plt.title('p1')
                # # plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate')
                # fig.savefig(os.path.join(cycle_dir, 'p1.png'))
                # plt.close(fig)
                
                fig = plt.figure(dpi=200)
                # hist_x = np.linspace(np.min(self.f),np.max(self.f), 1000)
                n, bins, _ = plt.hist(p2, bins=num_bins, density=True, label=f'p2, mean={np.round(means[-1], 4)} 2sigma={np.round(2*stds[-1],4)}')
                n2, bins2, _ = plt.hist(p1, bins=num_bins, density=True, histtype='step', color='black', label=f'p1, mean={np.round(means[-2],4)} 2sigma={np.round(2*stds[-2],4)}')
                plt.ylabel('Growthrate Probability Density')
                plt.xlabel('Growthrate')
                plt.legend()
                #
                plt.title(title)
                # plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate'
                fig.savefig(os.path.join(cycle_dir, title+'uncertainty_dist.png'))
                plt.close(fig)
                break
        #---- setup for next cycle
        p1=p2
    
    fig = plt.figure()
    # plt.plot(num_samples,means)
    plt.plot(num_samples,np.array(stds))
    fig.savefig(os.path.join(cycle_dir, title+'uncertainty_dist_stds.png'))
    plt.xlabel('Number of Samples')
    plt.ylabel('Standard Deviation of Function')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(num_samples,means)
    # plt.plot(num_samples,np.array(means))
    fig.savefig(os.path.join(cycle_dir, title+'uncertainty_dist_means.png'))
    plt.xlabel('Number of Samples')
    plt.ylabel('Expectation of Function')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(num_samples[1:], entropy_difs)
    plt.ylabel('Entropy Difference')
    plt.xlabel('Number of Samples')
    fig.savefig(os.path.join(cycle_dir,title+'entropy_dif.png'))
    plt.close(fig)
    return p2
def relative_entropy(samples_p, samples_q, num_bins):
    """
    Calculate the relative entropy (Kullback-Leibler divergence) between two distributions.

    Parameters:
    samples_p (list or np.array): Samples from the first distribution.
    samples_q (list or np.array): Samples from the second distribution.

    Returns:
    float: The relative entropy between the two distributions.
    """
    # Convert samples to numpy arrays
    samples_p = np.array(samples_p)
    samples_q = np.array(samples_q)

    # Calculate the probability density functions
    p_values, _ = np.histogram(samples_p, bins=num_bins, density=True)
    q_values, _ = np.histogram(samples_q, bins=num_bins, density=True)

    # Add a small value to avoid division by zero and log of zero
    p_values += 1e-10
    q_values += 1e-10

    # Calculate the relative entropy
    rel_entropy = entropy(p_values, q_values)

    return rel_entropy

def exclude_comparson(cycle_dir, parameters, bounds, exclude_parameters, n_jobs=1):
    nominals = [np.mean(b) for b in bounds]
    nominals_ = {p:n for p,n in zip(parameters, nominals)}
    fix_exclude = {p:e for p,e in nominals_.items() if p in exclude_parameters}
    
    p_ex=brute_distribution(cycle_dir, parameters, bounds, fix_params=fix_exclude, title='excluded', n_jobs=n_jobs)
    p2=brute_distribution(cycle_dir, parameters, bounds, n_jobs=128)
    fig = plt.figure(dpi=200)
    num_bins=100
    n, bins, _ = plt.hist(p2, bins=num_bins, density=True, label=f'mean={np.round(np.mean(p2), 4)} | 2sigma={np.round(2*np.sqrt(np.var(p2)),4)}')
    n2, bins2, _ = plt.hist(p_ex, bins=num_bins, density=True, histtype='step', color='black', label=f'fixed: mean={np.round(np.mean(p_ex),4)} | 2sigma={np.round(2*np.sqrt(np.var(p_ex)),4)}')
    plt.ylabel('Growthrate Probability Density')
    plt.xlabel('Growthrate')
    plt.legend(fontsize=10)
    plt.xlim(-1,1)
    #
    plt.title(f'fixed: {exclude_parameters}')
    # plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate'
    fig.tight_layout()
    fig.savefig(os.path.join(cycle_dir, f"excluded_{'_'.join(exclude_parameters)}.png"))
    plt.close(fig)

    
def equilibriumVgradients(cycle_dir, parameters, bounds, n_jobs=1):
    nominals = [np.mean(b) for b in bounds]
    nominals_ = {p:n for p,n in zip(parameters, nominals)}
    # bounds_ = {p:b for p,b in zip(parameters,bounds)}
    equilibrium_parameters = ['beta', 'q0', 'shat', 's_kappa', 'kappa', 'delta']
    other_parameters = ['coll','omn', 'temp1', 'omt2', 'zeff', 'omt1']
    
    fix_equilibrium = {p:n for p,n in nominals_.items() if p in equilibrium_parameters}
    fix_others = {p:o for p,o in nominals_.items() if p in other_parameters}
    
    p_eq=brute_distribution(cycle_dir, parameters, bounds, fix_params=fix_equilibrium, title='EQUILIBRIUM FIXED', n_jobs=n_jobs)
    brute_distribution(cycle_dir, parameters, bounds, fix_params=fix_others, title='OTHERS FIXED', n_jobs=n_jobs)
    
    p2=brute_distribution(cycle_dir, parameters, bounds, n_jobs=128)

    fig = plt.figure(dpi=200)
    num_bins=100
    n, bins, _ = plt.hist(p2, bins=num_bins, density=True, label=f'mean={np.round(np.mean(p2), 4)} | 2sigma={np.round(2*np.sqrt(np.var(p2)),4)}')
    n2, bins2, _ = plt.hist(p_eq, bins=num_bins, density=True, histtype='step', color='black', label=f'eq_fixed: mean={np.round(np.mean(p_eq),4)} | 2sigma={np.round(2*np.sqrt(np.var(p_eq)),4)}')
    plt.ylabel('Growthrate Probability Density')
    plt.xlabel('Growthrate')
    plt.legend(fontsize=10)
    plt.xlim(-1,1)
    #
    plt.title('Is EQUILIBRIUM uncertainty important?')
    # plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate'
    fig.tight_layout()
    fig.savefig(os.path.join(cycle_dir, 'is_equilibrium_important.png'))
    plt.close(fig)
    
if __name__ == '__main__':
    _, cycle_dir = sys.argv
    
    base_run_dir = os.path.dirname(cycle_dir)
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    bounds=np.array(config.sampler['bounds'])
    parameters = config.sampler['parameters']
    
    brute_distribution(cycle_dir, parameters, bounds, n_jobs=128, prec=0.001)
    # # equilibriumVgradients(cycle_dir, parameters,bounds, n_jobs=128)
    # exclude_comparson(cycle_dir,parameters, bounds, exclude_parameters=['beta','coll', 'zeff'], n_jobs=128)
    # exclude_comparson(cycle_dir,parameters, bounds, exclude_parameters=['omn','omt2','omt1'], n_jobs=128)