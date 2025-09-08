import sys,os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/users/danieljordan/enchanted-surrogates/src')
from runners.gene_single_monitor import eQ_history

def merge_early_stopping_reports(base_run_dir):
    dfs = []
    for run_dir in os.listdir(base_run_dir):
        run_dir = os.path.join(base_run_dir, run_dir)
        report_path = os.path.join(run_dir, 'early_stopping_report.csv')
        if os.path.exists(report_path):
            dfi = pd.read_csv(os.path.join(run_dir, 'early_stopping_report.csv'))
            dfs.append(dfi)
        
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(base_run_dir, 'merged_early_stopping_report.csv'), index=False)

def comparison_plots(base_run_dir):
    save_dir = os.path.join('./early_convergence_post_proc/plots', os.path.basename(os.path.dirname(base_run_dir))+'_plots')
    os.makedirs(save_dir, exist_ok=True)
    merge_early_stopping_reports(base_run_dir)
    dfs = []
    for run_dir in os.listdir(base_run_dir):
        run_dir = os.path.join(base_run_dir, run_dir)
        report_path = os.path.join(run_dir, 'early_stopping_report.csv')
        if os.path.exists(report_path):
            dfi = pd.read_csv(os.path.join(run_dir, 'early_stopping_report.csv'))
            dfs.append(dfi)
    
    df = pd.concat(dfs, ignore_index=True)
    print(df)

    zero_gr_mask = df['gene_growthrate'].to_numpy() == 0
    df = df[~zero_gr_mask]
    print(df)
    bin_edges = np.histogram_bin_edges(df['time_to_GENE_finnish'], bins=20)
    # time hist plots
    nan_mask = df['time_to_EARLY_convergence'].isna().to_numpy()
    fig, ax = plt.subplots()
    early_time = np.round(np.sum(df['time_to_EARLY_convergence'].to_numpy()[~nan_mask]) + np.sum(df['time_to_GENE_finnish'].to_numpy()[nan_mask]))
    gene_time = np.round(np.sum(df['time_to_GENE_finnish'].to_numpy()))
    ax.hist([df['time_to_EARLY_convergence'],df['time_to_GENE_finnish'].to_numpy()[nan_mask]], bins=bin_edges, stacked=True, color=['salmon','skyblue'], label=[f"Converged Early, total: {np.round(np.nansum(df['time_to_EARLY_convergence'].to_numpy()),1)}s",f"Did not Converge Early, total: {np.round(np.nansum(df['time_to_GENE_finnish'].to_numpy()[nan_mask]),1)}s"])    
    ax.hist(df['time_to_GENE_finnish'], color='grey', alpha=0.5, bins=bin_edges, label=f'Early Convergence OFF: {gene_time}s')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Num GENE runs')
    ax.set_title(f'{np.round((1-(early_time/gene_time))*100,1)}% faster\nN runs: {len(df)} Saved: {np.round(gene_time - early_time, 1)}s | {np.round((gene_time - early_time)*128/(60*60),1)} cpuh\nSaved: {np.round((gene_time - early_time)/len(df),1)}s/run | {np.round(((gene_time - early_time)*128/(60*60))/len(df),1)} cpuh/run')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,'time_hist_early_stopping.png'))
    plt.close(fig)
    print(len(df))
    print(len(df['time_to_GENE_finnish'].to_numpy()) + len(df['time_to_EARLY_convergence'].to_numpy()))

    # early convergence vs gene convergence
    fig, ax, = plt.subplots()
    nan_mask = df['early_growthrate'].isna().to_numpy()
    ax.scatter(df['early_growthrate'].to_numpy()[~nan_mask], df['gene_growthrate'].to_numpy()[~nan_mask], label='Early Growth Rate', marker='o', linestyle='None', color='salmon')
    ax.plot(np.linspace(np.min(df['early_growthrate'].to_numpy()[~nan_mask]), np.max(df['early_growthrate'].to_numpy()[~nan_mask]), 100),
            np.linspace(np.min(df['early_growthrate'].to_numpy()[~nan_mask]), np.max(df['early_growthrate'].to_numpy()[~nan_mask]), 100), 
            color='black', linestyle='--')
    ax.set_xlabel('Early Growth Rate')
    ax.set_ylabel('Gene Growth Rate')
    fig.tight_layout()
    # fig.savefig(f'early_convergence_post_proc/plots/early_growthrate_vs_gene_growthrate_{os.path.basename(base_run_dir)}.png')
    fig.savefig(os.path.join(save_dir,'early_growthrate_vs_gene_growthrate.png'))
    plt.close(fig)
    # early convergence vs gene convergence residuals
    fig, ax, = plt.subplots()
    nan_mask = df['early_growthrate'].isna().to_numpy()
    early_growthrate = df['early_growthrate'].to_numpy()[~nan_mask]
    gene_growthrate = df['gene_growthrate'].to_numpy()[~nan_mask]
    
    residuals = np.abs(early_growthrate - gene_growthrate)
    ax.scatter(early_growthrate, residuals, label='Early Growth Rate Residuals', marker='o', linestyle='None', color='salmon')
    ax.set_xlabel('Early Growth Rate')
    ax.set_ylabel('Residuals (Early - Gene)')
    fig.tight_layout()
    # fig.savefig(f'early_convergence_post_proc/plots/early_growthrate_residuals_vs_gene_growthrate_{os.path.basename(base_run_dir)}.png')
    fig.savefig(os.path.join(save_dir,'early_growthrate_residuals_vs_gene_growthrate.png'))
    plt.close(fig)
def debug_plots(base_run_dir):
    save_dir = os.path.join('./early_convergence_post_proc/plots', os.path.basename(os.path.dirname(base_run_dir))+'_plots')
    df = pd.read_csv(os.path.join(base_run_dir,'merged_early_stopping_report.csv'))
    fig, ax, = plt.subplots()
    nan_mask = df['sim_time_window'].isna().to_numpy()
    early_growthrate = df['sim_time_window'].to_numpy()[~nan_mask]
    gene_growthrate = df['gene_growthrate'].to_numpy()[~nan_mask]
    
    ax.hist([df['sim_time_window'][~nan_mask],df['sim_time_window'][df['time_to_GENE_finnish']>999].to_numpy()], bins=20, stacked=True, color=['salmon','skyblue'], label=['sim_time_window','hit_time_limit'])    
    
    fig.tight_layout()
    # fig.savefig(f'early_convergence_post_proc/plots/early_growthrate_residuals_vs_gene_growthrate_{os.path.basename(base_run_dir)}.png')
    fig.savefig(os.path.join(save_dir,'sim_time_window_hist.png'))
    plt.close(fig)
def plot_eq_history(run_dir):
    save_dir = os.path.join('./early_convergence_post_proc/plots', os.path.basename(run_dir)+'_plots')
    os.makedirs(save_dir, exist_ok=True)
    print('debug savedir', save_dir)
    eQ, time = eQ_history(run_dir)
    
    fig, ax = plt.subplots()
    ax.plot(time, np.log(eQ))
    ax.set_xlim(0,100)
    ax.set_ylim(-10,10)
    fig.savefig(os.path.join(save_dir, 'eQ_history.png'))
    plt.close(fig)


if __name__ == '__main__':
    _, base_run_dir = sys.argv
    
    # comparison_plots(base_run_dir)
    # debug_plots(base_run_dir)
    
    plot_eq_history('/scratch/project_462000954/daniel/enchanted_test/gene_beta_scan_test_early_stopping_nrg5-field100/0fc93421-27a4-487a-bb7d-0ab66c58915f')