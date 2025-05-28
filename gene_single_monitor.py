import sys
import os

import re
import f90nml
# from parsers.GENEparser import GENEparser
from time import time, sleep

import numpy as np
from scipy.signal import find_peaks

sys.path.append('/users/danieljordan/DEEPlasma/GENE_ML/')
sys.path.append('/users/danieljordan/DEEPlasma/GENE_ML/IFS_scripts')

from IFS_scripts.fieldlib import fieldfile
from IFS_scripts.ParIO import Parameters
# from IFS_scripts.nrgWrapper import read_from_nrg_files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from parsers.GENEparser_timeseries import GENEparserTimeseries

class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

def begin_early_convergence_monitor(problem_dir, diagdir, early_convergence_2sigma_threshold=None, time_window_multiplyer=30, stability_threshold_time_to_flux=100, do_plots=False, comparison_mode=False, gene_timeout=2000, report_path='./early_stopping_report.csv'):
    print('EARLY CONVERGENCE MONITOR')
    
    stability_threshold_time_to_flux = float(stability_threshold_time_to_flux)
    gene_timeout = float(gene_timeout)

    # Do early convergence monitoring
    print('STARTING EARLY CONVERGENCE MONITORING')
    print('WAITING FOR NEW SCANFILES DIR')
    
    sys.stdout = Tee(os.path.join(diagdir,'early_convergence_monitor.out'))
    sys.stderr = sys.stdout

    print('MONITORING GENE DIAGDIR DIRECTORY FOR EARLY CONVERGENCE:', diagdir)
    
    df = pd.DataFrame({'early_growthrate':np.repeat(np.nan,1),
                       'gene_growthrate':np.repeat(np.nan,1),
                       'converged_early?':np.repeat(False,1),
                       'timeout_reached?':np.repeat(False,1),
                       'time_to_GENE_finnish':np.repeat(np.nan,1), 
                       'time_to_EARLY_convergence':np.repeat(np.nan,1),
                       'simtime_early_converge': np.repeat(np.nan,1),
                       'sim_time_window':np.repeat(np.nan,1),
                       'stable':np.repeat(np.nan,1)})
    
    # simulation_time = np.zeros(len(suffix_s))
    print('MAKING GR DICT')
    print('getting start time')
    start_time = None#np.repeat(time(),len(suffix_s)) #NOTE!! only accurate if num_paralell_sims is 1 in gene parameters file. 
    field_file_exists = False
    check_stability_counter = 0
    
    
    print('SLEEPING 2sec TO GIVE GENE A CHANCE TO RUN')
    for i in range(2):
        print(i+1)
        sleep(1)
    finished = False
    while not finished:
        print( 'SLEEPING 2sec TO GIVE GENE A CHANCE TO RUN')
        for i in range(2):
            print( i+1)
            sleep(1)
        print( 'CHECKING IF STARTED')
        started = has_started(diagdir)
        if started and start_time == None:
            print( 'HAS STARTED, SETTING START TIME')
            start_time = time()
        elif started:
            print( 'HAS STARTED')
        elif not started:
            print( 'HAS NOT STARTED YET')
            continue
        if finished:
            print( 'IS ALREADY FINNISHED:')
            continue
        if has_GENE_finished(diagdir):
            print( 'GENE HAS FINNISHED')
            finished=True
            df.loc[ 'time_to_GENE_finnish'] = time()-start_time
            print( 'SLEEPING 0.5s TO ALLOW GENE TO MAKE OMEGA FILE')
            sleep(0.5)
            ky, growthrate, frequency = read_omega(diagdir, '.dat')
            df.loc[ 'gene_growthrate'] = growthrate
            print( 'WRITTING REPORT TO:', report_path)
            df.loc[ 'stable'] = False
            df.to_csv(report_path)
            continue
        # check stability every (stability_threshold_time_to_flux)sec 
        if (time()-start_time) // stability_threshold_time_to_flux > check_stability_counter:
            check_stability_counter+=1
            print( 'CHECKING IF STABLE')
            Q_e, Q_time = get_logQ_e_corrected(diagdir, '.dat')
            if Q_e[-1] < 10e-2:
                df.loc[ 'stable'] = True
                print( f'DID NOT PRODUCE HEAT FLUX AFTER {stability_threshold_time_to_flux}s AND SO IS DECLARED STABLE')    
                if not comparison_mode:
                    stop_file_path = os.path.join(diagdir, f'GENE.stop') 
                    print( 'MAKING STOP FILE AT:',stop_file_path)
                    os.system(f'touch {stop_file_path}')
                    os.system(f'touch stopped_by_monitor')
                    finished = True
                    continue
            else:
                df.loc[ 'stable'] = False
                print( 'DECLARED AS UNSTABLE')
            print( 'WRITTING REPORT TO:', report_path)
            df.to_csv(report_path)

        #check if the gene_timeout is reached
        if time() - start_time > gene_timeout:
            print( 'REACHED THE TIMEOUT OF:', gene_timeout, 'sec')
            df.loc[ 'timeout_reached?'] = True
            finished = True
            print( 'WRITTING REPORT TO:', report_path)
            df.to_csv(report_path)
            # df.to_csv(os.path.join(diagdir,'early_convergence_report.csv'))
        
        if not df.loc[ 'converged_early?']:
            #check if it has early converged
            field_path = os.path.join(diagdir, f'field.dat')
            nrg_path = os.path.join(diagdir, f'nrg.dat')
            if not (os.path.exists(field_path) and os.path.exists(nrg_path)):
                print( 'FIELD FILE AND/OR NRG FILE DOES NOT EXIST YET')
                continue
            print( 'FIELD FILE AND NRG FILE EXISTS')
            # just to ensure this sleep only happens once when the field file is just created.
            if not field_file_exists:
                print( 'SLEEPING 2SEC TO GIVE GENE A CHANCE TO FILL THEM')
                for i in range(2):
                    print( i+1)
                    sleep(1)
            field_file_exists = True
            _, gr_time = calculate_growthrate_raw(diagdir)
            min_number_of_file_updates = 44
            if len(gr_time) > min_number_of_file_updates and np.isnan(df.loc[ 'sim_time_window']):
                # check for early convergence
                # Get time window as 3*time_period of mode beating oscillations
                print( 'SEARCHING FOR SUITABLE TIME WINDOW')
                time_window = find_time_window(diagdir, time_window_multiplyer)
                if time_window==None:
                    print( 'NOT ENOUGH PEAKS OR NAN OR PEAKS ARE IRREGULARLY SPACED')
                    continue
                df.loc[ 'sim_time_window'] = time_window
                print( 'WRITTING REPORT TO:', report_path)
                df.to_csv(report_path)
                print( 'TIME WINDOW:', time_window)
            elif not np.isnan(df.loc[ 'sim_time_window']):
                time_window=df.loc[ 'sim_time_window']
            else:
                print( f'AT {len(gr_time)} FILE UPATES, WAITING FOR {min_number_of_file_updates}')
                continue
            if 2*time_window > gr_time[-1]:
                print( f'TIME WINDOW {time_window} IS LARGER THAN SIM TIME PASSED {gr_time[-1]} + TIME WINDOW = {gr_time[-1] + time_window}.\n',
                            'AT LEAST 2 TIME WINDOWS OF DATA IS NEEDED FOR THE STATS')
                continue
            print( 'CALCULATING WINDOW STATS')
            growthrate_window_mean, growthrate_window_sigma, growthrate_window_mean_time, growthrate, growthrate_time = get_growthrate_mean_window(diagdir,  window_time_span=time_window)
            if type(growthrate_window_mean) == type(None):
                print( 'AT LEAST 2 TIME WINDOWS OF DATA IS NEEDED FOR THE STATS, MUST WAIT LONGER')
                continue
            # Check if early converged
            # simulation_time[i] = growthrate_time[-1]
            print( f'HAS GENE CONVERGED EARLY?', '| 2 SIGMA:', 2*growthrate_window_sigma[-1], '| THRESHOLD:',early_convergence_2sigma_threshold, '| TIMER s:',time()-start_time)

            if 2*growthrate_window_sigma[-1] < early_convergence_2sigma_threshold:
                #we have converged early, end the gene simulation.
                print( 'EARLY CONVERGENCE TIMER:',time()-start_time,'s')
                print( 'EARLY CONVERGENCE DETECTED FOR SUFFIX:',  'MEAN GROWTHRATE:',growthrate_window_mean[-1], '2 SIGMA OF MEAN WINDOW:', 2*growthrate_window_sigma[-1])
                
                stop_file_path = os.path.join(diagdir, f'GENE.stop') 
                if not comparison_mode:
                    print( 'MAKING STOP FILE AT:',stop_file_path)
                    os.system(f'touch {stop_file_path}')
                    os.system(f'touch stopped_by_monitor')
                    finished = True
                else:
                    print( '''IN COMPARISON MODE, NO STOP FILE BEING MADE
                            GENE WILL CONTINUE RUNNING
                            THE EARLY CONVERGENCE TIME CAN THEN BE COMPARED WITH GENE CONVEREGENCE TIME''')
                
                print( 'WRITING EARLY CONVERGED OMEGA FILE')
                gamma, omega = calculate_latest_omega(diagdir)
                with open(os.path.join(diagdir,f'early_converged_omega'), 'w') as file:
                    file.write(f'{gamma},{omega}')
                
                df.loc[ 'converged_early?'] = True
                df.loc[ 'time_to_EARLY_convergence'] = time()-start_time
                df.loc[ 'simtime_early_converge'] = growthrate_window_mean_time[-1]
                df.loc[ 'early_growthrate'] = growthrate_window_mean[-1]
                print( 'WRITTING REPORT TO:', report_path)
                df.to_csv(report_path)

        else:
            print( 'IN COMPARISON MODE')
            
                
    print('EARLY CONVERGENCE FINNISHED')
    print('WRITTING REPORT TO:', report_path)
    df.to_csv(report_path)

    # if do_plots:
    #     def plot_suffix(suffix):
    #         print( 'MAKING PLOTS')
    #         gr, gr_time, logQ_e, eQ_time = get_growthrate_corrected(diagdir, suffix)
    #         points_path = os.path.join(diagdir,f't_logQ_e_for_window_calc_{suffix}.npy')
            
    #         fig = plt.figure()
    #         plt.plot(eQ_time, logQ_e)
    #         if os.path.exists(points_path):
    #             points = np.load(points_path)
    #             tp, num_peaks, peaks, dist_between_peaks = calculate_time_period(points)
    #             plt.plot(points.T[0], points.T[1], '.', color='red',label='time period calc points')
    #             plt.plot(points.T[0][peaks],points.T[1][peaks], '.', color='orange')
    #         plt.xlabel('simulation time, cref/Lref')
    #         plt.ylabel('Linear Electron Heat Flux')
    #         plt.legend()
    #         fig.savefig(os.path.join(diagdir,f'linear_heat_flux_{suffix}.png'))
            
    #         if os.path.exists(points_path):
    #             fig = plt.figure()
    #             plt.plot(points.T[0], points.T[1], '.', color='red',label='time period calc points')
    #             plt.plot(points.T[0][peaks],points.T[1][peaks], '.', color='orange')
    #             plt.xlabel('simulation time, cref/Lref')
    #             plt.ylabel('Linear Electron Heat Flux')
    #             plt.legend()
    #             fig.savefig(os.path.join(diagdir,f'linear_heat_flux_time_period_calc_{suffix}.png'))

    #         fig = plt.figure()
    #         time_window = df.loc[ 'sim_time_window']
    #         if not np.isnan(time_window):
    #             tw_arg1 = np.argmin(np.abs(gr_time - (df.loc[ 'simtime_early_converge']-time_window)))
    #             tw_arg2 = np.argmin(np.abs(gr_time - df.loc[ 'simtime_early_converge']))
    #             plt.plot(gr_time[tw_arg1:tw_arg2], gr[tw_arg1:tw_arg2], label=f'time window: {time_window}', color='magenta')
    #         plt.plot(gr_time, gr)
    #         plt.legend()
    #         plt.xlabel('simulation time, cref/Lref')
    #         plt.ylabel('Growthrate, phi')
    #         fig.savefig(os.path.join(diagdir,f'growthrate_{suffix}.png'))
            
    #         if not np.isnan(time_window) and df.loc[ 'converged_early?']:
    #             fig = plt.figure()
    #             growthrate_window_mean, growthrate_window_sigma, growthrate_window_mean_time, growthrate, growthrate_time = get_growthrate_mean_window(diagdir,  window_time_span=time_window)
    #             tw_arg1 = np.argmin(np.abs(gr_time - (df.loc[ 'simtime_early_converge']-time_window)))
    #             tw_arg2 = np.argmin(np.abs(gr_time - df.loc[ 'simtime_early_converge']))
    #             gr_m_arg1 = np.argmin(np.abs(growthrate_window_mean_time - (df.loc[ 'simtime_early_converge']-time_window)))
    #             gr_m_arg2 = np.argmin(np.abs(growthrate_window_mean_time - df.loc[ 'simtime_early_converge']))
    #             growthrate_window_sigma_time = growthrate_window_mean_time[-len(growthrate_window_sigma):]
    #             gr_s_arg2 = np.argmin(np.abs(growthrate_window_sigma_time - df.loc[ 'simtime_early_converge']))
                
    #             gr_s_arg2 = np.argmin(np.abs(growthrate_window_mean_time - df.loc[ 'simtime_early_converge']))
    #             plt.plot(gr_time[tw_arg1:tw_arg2], 
    #                     gr[tw_arg1:tw_arg2], 
    #                     label=f'time window: {time_window}', 
    #                     color='magenta')
    #             plt.plot(growthrate_window_mean_time[gr_m_arg1:gr_m_arg2], 
    #                     growthrate_window_mean[gr_m_arg1:gr_m_arg2], 
    #                     label=f'sliding_window_mean, 2*std:{np.round(2*np.sqrt(np.var(growthrate_window_mean[gr_m_arg1:gr_m_arg2])),7)}::{2*growthrate_window_sigma[gr_s_arg2]}', 
    #                     color='green')
    #             plt.legend()
    #             plt.xlabel('simulation time, cref/Lref')
    #             plt.ylabel('Growthrate, phi')
    #             fig.savefig(os.path.join(diagdir,f'growthrate_window_{suffix}.png'))
    #     # do plots in paralell
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=len(suffix_s)) as executor:
    #         results = list(executor.map(plot_ suffix_s))    
            
    print('EARLY CONVERGENCE MONITORING ALL DONE, see:\n',report_path)

                        
def has_GENE_finished(diagdir):
    if os.path.exists(os.path.join(diagdir,'GENE.finished')):
        # gene has converged
        print('GENE FINISHED BY ITS OWN STANDARDS FOR SUFFIX',  f'\nMAYBE IT REACHED A LIMIT, MAYBE IT CONVERGED, CHECK THE {diagdir}/geneerr.log and {diagdir}/scan.log')
        return True
    else:
        return False

############################
def get_all_suffixes(diagdir):
    files = os.listdir(diagdir)
    pattern = r'\d{4}'
    matched_strings = []
    for filename in files:
        matches = re.findall(pattern, filename)
        matched_strings.extend(matches)
    suffix_s = np.unique(matched_strings)
    return suffix_s

def read_parameters_dict(parameters_path):
    # for some reason f90nml fails to parse with 'FCVERSION' line in the parameters file, so I comment it
    with open(parameters_path, 'r') as parameters_file:
        lines = parameters_file.readlines()
        for i, line in enumerate(lines):
            if 'FCVERSION' in line:
                lines[i] = '!'+line

    with open(parameters_path, 'w') as parameters_file:
        parameters_file.writelines(lines)

    with open(parameters_path, 'r') as parameters_file:
        nml = f90nml.read(parameters_file)
        parameters_dict= nml.todict()
    return parameters_dict

def read_status(diagdir, timeout=100):
    start = time()
    global status_path 
    status_path = os.path.join(diagdir, 'in_par','gene_status')
    global status
    status = None
    # nescessary complication as GENE sometimes removes the status file and replaces it a moment later.
    def open_status():
        global status_path
        global status
        try:
            with open(status_path, 'r') as status_file:
                status = status_file.read()#.decode('utf8')
            return True
        except FileNotFoundError:
            return False
    
    while not open_status() and time()-start < timeout:
        sleep(0.1)
    if time()-start > timeout:
        return None
    return status

# def read_status(diagdir):
#     status_dir = os.path.join(diagdir, 'in_par','gene_status' )
#     try:
#         with open(status_dir, 'r') as status_file:
#             status = status_file.readline().strip()
#     except FileNotFoundError:
#         return None
#     return status

def has_started(diagdir, suffix='.dat'):
    if os.path.exists(os.path.join(diagdir, f'parameters{suffix}')):
        return True
    else:
        return False
#############################
            
        
from IFS_scripts.fieldlib import fieldfile
from IFS_scripts.ParIO import Parameters
import os
import numpy as np
import matplotlib.pyplot as plt
# from .IFS_scripts.fieldlib import 
# from IFS_scripts.get_nrg import get_nrg0
# from .IFS_scripts.ParIO import 
# from .IFS_scripts.finite_differences import 

# from parsers.GENEparser_timeseries import GENEparserTimeseries

# parser_ts = GENEparserTimeseries()

def calculate_growthrate_raw(diagdir, suffix='.dat', calc_from_apar=False):
    calc_from_apar=calc_from_apar
    parameters_path = os.path.join(diagdir, f'parameters{suffix}')
    
    par = Parameters()
    par.Read_Pars(parameters_path)
    pars = par.pardict

    field = fieldfile(os.path.join(diagdir,'field'+suffix),pars)
    field.set_time(field.tfld[-1])
    imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
    phi = np.empty(0,dtype='complex128')
    if pars['n_fields'] > 1:
        imaxa = np.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(field.nz,field.nx))
        apar = np.empty(0,dtype='complex128')

    time = np.empty(0)
        
    # Using list comprehension to speed up the process
    phi = np.array([field.set_time(field.tfld[i]) or field.phi()[imax[0], 0, imax[1]] for i in range(len(field.tfld))])
    time = np.array([field.tfld[i] for i in range(len(field.tfld))])
    if pars['n_fields'] > 1:
        apar = np.array([field.set_time(field.tfld[i]) or field.apar()[imaxa[0], 0, imaxa[1]] for i in range(len(field.tfld))])
        
    if len(phi) < 2.0:
        output_zeros = True
        omega = 0.0+0.0J
    else:
        output_zeros = False
        if calc_from_apar:
            print( "Calculating omega from apar")
            if pars['n_fields'] < 2:
                NotImplemented
                #stop
            omega = np.log(apar/np.roll(apar,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)
        else:
            omega = np.log(phi/np.roll(phi,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)

    gamma = np.real(omega)
    omega = np.imag(omega)
    
    return gamma, time

def calculate_latest_omega(diagdir, suffix='.dat', calc_from_apar=False):
    calc_from_apar=calc_from_apar
    parameters_path = os.path.join(diagdir, f'parameters{suffix}')
    
    par = Parameters()
    par.Read_Pars(parameters_path)
    pars = par.pardict

    field = fieldfile(os.path.join(diagdir,'field'+suffix),pars)
    field.set_time(field.tfld[-1])
    imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
    phi = np.empty(0,dtype='complex128')
    if pars['n_fields'] > 1:
        imaxa = np.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(field.nz,field.nx))
        apar = np.empty(0,dtype='complex128')

    time = np.empty(0)
        
    # Using list comprehension to speed up the process
    phi = np.array([field.set_time(field.tfld[i]) or field.phi()[imax[0], 0, imax[1]] for i in range(len(field.tfld))])
    time = np.array([field.tfld[i] for i in range(len(field.tfld))])
    if pars['n_fields'] > 1:
        apar = np.array([field.set_time(field.tfld[i]) or field.apar()[imaxa[0], 0, imaxa[1]] for i in range(len(field.tfld))])
    
    if len(phi) < 2.0:
        output_zeros = True
        omega = 0.0+0.0J
    else:
        output_zeros = False
        if calc_from_apar:
            print( "Calculating omega from apar")
            if pars['n_fields'] < 2:
                NotImplemented
                #stop
            omega = np.log(apar/np.roll(apar,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)
        else:
            omega = np.log(phi/np.roll(phi,1))
            dt = time - np.roll(time,1)
            omega /= dt
            omega = np.delete(omega,0)
            time = np.delete(time,0)

    gamma = np.real(omega)
    omega = np.imag(omega)
    
    return gamma[-1], omega[-1]

def correct_growthrate(gamma, gamma_time, Q_e, Q_e_time):
    deltaQ = Q_e - np.roll(Q_e, 1)
    ignore_args = np.argwhere(deltaQ < -1e10).flatten()
    ignore_args = [arg for arg in ignore_args if Q_e[arg-1] > 1e25]
    # print('NUMBER OF PERCISION BLIPS',len(ignore_args))
    ignore_times = Q_e_time[ignore_args]
    logQ_e = np.log(Q_e)
    for index in ignore_args:
        logQ_e[index:] = logQ_e[index:] + (logQ_e[index-1] - logQ_e[index])
    # Check the gamma near the ignore times and removes ones that have dropped.
    for t in ignore_times:
        tdif = np.abs(gamma_time-t)
        arg_t_closest = np.argmin(tdif)
        # sometimes the closes to the ignore time is not the blip, so we take the min within 5 steps away
        if arg_t_closest < 5:
            arg_delete = arg_t_closest + np.argmin(gamma[0: arg_t_closest+5])
        else:
            arg_delete = arg_t_closest - 5 + np.argmin(gamma[arg_t_closest-5: arg_t_closest+5])
        gamma = np.delete(gamma, arg_delete)
        gamma_time = np.delete(gamma_time, arg_delete)
    # print('GAMMA TIME AFTER CORRECTION',gamma_time)
    return gamma, gamma_time, logQ_e, Q_e_time

def get_growthrate_corrected(diagdir, suffix):
    Q_e, Q_time = eQ_history(diagdir, suffix)#species provides order in parameters file
    growthrate, g_time = calculate_growthrate_raw(diagdir, suffix)
    growthrate_corrected, time_corrected, logQ_e, Q_e_time = correct_growthrate(growthrate, g_time, Q_e, Q_time)
    return growthrate_corrected, time_corrected, logQ_e, Q_e_time

def get_logQ_e_corrected(diagdir, suffix):
    Q_e, Q_e_time = eQ_history(diagdir, suffix)#species provides order in parameters file
    deltaQ = Q_e - np.roll(Q_e, 1)
    ignore_args = np.argwhere(deltaQ < -1e10).flatten()
    import matplotlib.pyplot as plt
    plt.plot(Q_e_time, Q_e)
    plt.show()
    print('debug', Q_e[ignore_args-1])
    print('NUMBER OF PERCISION BLIPS',len(ignore_args), 'AT', Q_e_time[ignore_args])
    ignore_args = [arg for arg in ignore_args if Q_e[arg-1] > 1e25]
    print('NUMBER OF PERCISION BLIPS',len(ignore_args), 'AT', Q_e_time[ignore_args])
    logQ_e = np.log(Q_e)
    for index in ignore_args:
        logQ_e[index:] = logQ_e[index:] + (logQ_e[index-1] - logQ_e[index])
    return logQ_e, Q_e_time

def get_growthrate_mean_window(diagdir,  window_time_span=10):
    growthrate, growthrate_time, _, _ = get_growthrate_corrected(diagdir, suffix)
    # print('debug, gr, gr time', growthrate, growthrate_time)
    # print('grothrate time -1', growthrate_time[-1], 'len(growthrate_time)', len(growthrate_time))
    time_in_1_index_shift = (growthrate_time[-1]-growthrate_time[0]) / len(growthrate_time)
    # print('ti11', time_in_1_index_shift)
    # print('time in 1 shift:',time_in_1_index_shift)
    index_shift = int(window_time_span/time_in_1_index_shift)
    # print('is1', index_shift)
    # print('indext shift:',index_shift)
    growthrate_window_mean = np.empty(0)
    growthrate_window_mean_time = np.empty(0)
    if index_shift > len(growthrate_time):
        print('debug returning none')
        return [None]*5
    
    growthrate_window_mean = np.array([np.mean(growthrate[i-index_shift:i]) for i in range(index_shift, len(growthrate_time))])
    # print('debug, grwm', growthrate_window_mean)
    growthrate_window_mean_time = np.array([np.mean(growthrate_time[i]) for i in range(index_shift, len(growthrate_time))])
    # print('debug, grwmt', growthrate_window_mean_time)
    # for i in range(index_shift, len(growthrate_time)):
    #     gr_m = np.mean(growthrate[i-index_shift:i])
    #     growthrate_window_mean = np.append(growthrate_window_mean, gr_m)
    #     growthrate_window_mean_time = np.append(growthrate_window_mean_time, growthrate_time[i])
    
    # print('gwm, gwmt',growthrate_window_mean, growthrate_window_mean_time)

    # finding the sigma with the mean not the actual growthrate
    time_in_1_index_shift = (growthrate_window_mean_time[-1]-growthrate_window_mean_time[0]) / len(growthrate_window_mean_time)
    # print('ti12', time_in_1_index_shift)
    index_shift = int(window_time_span/time_in_1_index_shift)
    growthrate_window_sigma = np.empty(0)
    if index_shift > len(growthrate_window_mean_time):
        print('debug returning none')
        return [None]*5
    growthrate_window_sigma = np.array([np.sqrt(np.var(growthrate_window_mean[i-index_shift:i])) for i in range(index_shift, len(growthrate_window_mean_time))])
    # for i in range(index_shift, len(growthrate_window_mean_time)):
    #     gr_sigma = np.sqrt(np.var(growthrate_window_mean[i-index_shift:i]))
    #     growthrate_window_sigma = np.append(growthrate_window_sigma, gr_sigma)
    return growthrate_window_mean, growthrate_window_sigma, growthrate_window_mean_time, growthrate, growthrate_time 

def find_time_window(diagdir,  time_period_multiplyer=30):
    # we look from the end of time back to the start. We look for oscillations, once we have 4 oscillations we take the mean time_period
    # if we don't have 6 oscillations we continue the run. This allows for some buffer incase the start of the simulation is unstable
    # nrg_path = os.path.join(diagdir, f'nrg_{suffix}')
    # Q, Q_time = parser_ts.Q_history(nrg_path)
    print('GETTING CORRECTED GROWTHRATE AND logQ_e')
    logQ_e, Q_time = get_logQ_e_corrected(diagdir, suffix)
    points = [(t, lq) for t,lq in zip(Q_time, logQ_e)]
    # Go from late to early 20 points at a time
    for i in range(20,len(points),20):
        points_set = points[-i:]
        if np.isnan(points_set).any():
            print('THERE ARE NAN VALUES')
            return None
        time_period, num_peaks, peaks, dist_between_peaks = calculate_time_period(points[-i:])
        print('n points', len(points[-i:]))
        print('peak distances', dist_between_peaks)
    
        if num_peaks < 6:            
            continue
        if np.array(np.abs(dist_between_peaks-np.mean(dist_between_peaks)) > 2*np.sqrt(np.var(dist_between_peaks))).any():
            print('SOME PEAK DISTANCES ARE FURTHER THAN 2 SIGMA FROM THE MEAN. THIS INDICATES THE PEAKS ARE NOT PERIODIC.')
            return None
        else:
            print( 'TIME WINDOW FOUND:', time_period_multiplyer*time_period)
            print('saving points to file')
            np.save(os.path.join(diagdir, f't_logQ_e_for_window_calc_{suffix}.npy'), np.array(points_set))
            return time_period_multiplyer*time_period
    return None
            
    
def calculate_time_period(points):
    """
    Calculate the time_period of oscillations around a linear line given an array of points.
    
    Parameters:
    points (list of tuples): List of (x, y) points
    
    Returns:
    float: The time_period of the oscillations
    """
    # Convert points to numpy array for easier manipulation
    points = np.array(points)
    # Fit a linear line to the points
    x = points[:, 0]
    y = points[:, 1]
    coefficients = np.polyfit(x, y, 1)
    linear_fit = np.polyval(coefficients, x)
    
    # Calculate the deviations from the linear fit
    deviations = y - linear_fit
    # print('linear_fit', linear_fit)
    # print('deviations',deviations)
    # Find peaks in the deviations
    peaks, _ = find_peaks(deviations)
    num_peaks = len(peaks)
    # Calculate the distances between consecutive peaks
    peak_distances = np.diff(x[peaks])
    # Calculate the average time_period
    if len(peak_distances) > 0:
        time_period = np.mean(peak_distances)
    else:
        time_period = None
    
    return time_period, num_peaks, peaks, peak_distances

def read_omega(run_dir, suffix='.dat'):
    omega_path = os.path.join(run_dir,'omega'+suffix)
    with open(omega_path, 'r') as file:
        line = file.read()
        vars = line.split(' ')
        vars = [v for v in vars if ' ' not in v]
        vars = [float(v) for v in vars if v != '']
        ky, growthrate, frequency = vars
    return ky, growthrate, frequency

def eQ_history(diagdir, suffix):
    par = Parameters()
    par.Read_Pars(os.path.join(diagdir,'parameters'))
    pars = par.pardict
    n_spec = pars['n_spec']
    
    nrg_path = os.path.join(diagdir,'nrg'+suffix)
    df = pd.read_csv(nrg_path, header=None)
    df_time = df.iloc[::n_spec+1].astype(float)
    df_species = []
    for i in range(n_spec):
        i+=1
        s_index = df_time.index+i
        df_s = df.iloc[s_index]
        df_s = df_s[0].str.split(expand=True).astype(float)
        df_species.append(df_s)
    
    species_order = get_species_order(diagdir)
    e_index = species_order.index('e')
    df_e = df_species[e_index]
    df_eQ = df_e[6] + df_e[7]
    return np.array(df_eQ), np.array(df_time[0])

def get_species_order(diagdir):
    paramaters_path = os.path.join(diagdir,'parameters')
    par = Parameters()
    par.Read_Pars(paramaters_path)
    pars = par.pardict
    n_spec = pars['n_spec']
    species_order = []
    for i in range(n_spec):
        i+=1
        if pars[f'charge{i}']==-1:
            species_order.append('e') # electron
        if pars[f'charge{i}']==1:
            species_order.append('i') # ion 
        if pars[f'charge{i}']>1:
            species_order.append('z') # impurity
    return species_order

if __name__ == '__main__':
    import ast
    print('NAME == MAIN')
    this_file_path, diagdir, diagdir, early_convergence_2sigma_threshold, time_period_multiplyer, stability_threshold_time_to_flux, do_plots, comparison_mode, gene_timeout, report_path = sys.argv
    print('SYS.ARGV',sys.argv)
    begin_early_convergence_monitor(diagdir, diagdir, float(early_convergence_2sigma_threshold), float(time_period_multiplyer), float(stability_threshold_time_to_flux), do_plots, comparison_mode, gene_timeout, report_path)



'''
example sbatch

#!/bin/bash -l
## LUMI-C (CPU partition) submit script template
## Submit via: sbatch submit.cmd (parameters below can be overwritten by command line options)
#SBATCH -t 12:00:00                # wallclock limit
#SBATCH -N 2                       # total number of nodes, 2 CPUs with 64 rank each
#SBATCH --ntasks-per-node=128      # 64 per CPU (i.e. 128 per node). Additional 2 hyperthreads disabled
#SBATCH --mem=0                    # Allocate all the memory on each node
#SBATCH -p standard                # all options see: scontrol show partition
#SBATCH -J GENE                    # Job name
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
##uncomment to set specific account to charge or define SBATCH_ACCOUNT/SALLOC_ACCOUNT globally in ~/.bashrc
#SBATCH -A project_462000451

export MEMORY_PER_CORE=1800

## set openmp threads
export OMP_NUM_THREADS=1

#do not use file locking for hdf5
export HDF5_USE_FILE_LOCKING=FALSE

### START EARLY CONVERGENCE MONITOR

### monitor inputs: diagdir, diagdir, early_convergence_2sigma_threshold=None, time_window_multiplyer=30, stability_threshold_time_to_flux=100, do_plots=False, comparison_mode=False
export PATH=/project/project_462000451/enchanted_container_lumi3/bin:$PATH
python3 -u /users/danieljordan/DEEPlasma/gene_monitor.py ./ '/scratch/project_462000451/daniel/sprint_out/gene/early_convergence_test/' 0.001 30 100 True True > ./early_convergence_monitor.out &

PID=$!

## I need this sleep to give the python a chance to get the the point where it has the current latest scanfiles dir and waits for GENE to make the next one
echo "sleeping 20 to ensure new scanfiles isn't made before python has a chance to grab old one"
sleep 10s

set -x
# run GENE
#srun -l -K -n $SLURM_NTASKS ./gene_lumi_csc

# run scanscript
./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall="srun -l -K -n $SLURM_NTASKS ./gene_lumi_csc"

set +x

wait $PID
echo "gene is finished and python monitor is finnished"

'''


'''
these parameters must be set as such
istep_nrg = 20
istep_field = 20

# should match number of nodes specified
n_parallel_sims = 2

'''