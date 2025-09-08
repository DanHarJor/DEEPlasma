import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
def plot_beta_scan(base_run_dir):
    df = pd.read_csv(os.path.join(base_run_dir,'merged_runner_return.txt'))
    print(df)
    df = df.dropna()
    beta_nominal = 0.58999087e-3
    beta_fraction = df['beta'].to_numpy()/beta_nominal #df['beta'].to_numpy()[0]
    df['beta_fraction'] = beta_fraction
    
    df.columns = [a.strip() for a in df.columns]
    df = df.dropna()
    print(df.sort_values(by='beta'))
    dfs = [group for _, group in df.groupby('beta')]
    # take the max of each growthrate, make KBM true if only one growthrate is TRUE for each beta
    idx = df.groupby('beta')['growthrate'].idxmax()
    KBM = df.groupby('beta')['KBM'].any() #apply(lambda g: any(g.to_numpy()))
    
    print('KBM\n', KBM)
    df = df.loc[idx]
    df['KBM'] = KBM.to_numpy()
    print(df)
    
    color = np.where(df['KBM'].to_numpy(), 'blue', 'black')
    fig, ax = plt.subplots()
    axt = ax.twinx()
    axt.set_ylabel('frequency')
    axt.scatter(df['beta_fraction'],df['frequency'], color='grey', alpha=0.4, marker='.')
    ax.scatter(df['beta_fraction'].to_numpy(), df['growthrate'].to_numpy(), color=list(color), marker='.')
    ax.hlines([0.5], np.min(df['beta_fraction'].to_numpy()), np.max(df['beta_fraction'].to_numpy()))
    ax.set_xlabel(r'$\beta / \beta_{nominal}$')
    ax.set_ylabel('Growthrate, (cref / Lref)')
    ax.hlines(0,0,1, label='ITG', color = 'red')
    ax.hlines(0,1,2.8, label='TEM', color = 'black')
    ax.hlines(0,2.8,5, label='KBM', color = 'blue')
    ax.scatter(df['beta_fraction'].to_numpy()[-1], df['growthrate'].to_numpy()[-1], color='blue', marker='.', label='KBM classifier')
    ax.set_title('ky=0.09 | x0=0.95 | JET #97781')
    fig.legend()
    fig.savefig('./beta_scan.png')
    plt.close(fig)
    to_plot = ['growthrate','frequency', 'chi_ratio', 'D_ratio_i', 'D_ratio_e']
    figa, AX = plt.subplots(len(to_plot),1, figsize=(5,5*len(to_plot)))
    for ind, ax in zip(to_plot, AX):
        ax.scatter(df['beta_fraction'],df[ind])
        ax.set_xlabel(r'$\beta / \beta_{nominal}$')
        ax.set_ylabel(ind)
        
    figa.savefig('./fingerprints.png')
    plt.close(figa)
    
    # figure, AX = plt.subplots(1,len(dfs), figsize=(5*len(dfs),5))
    # for ax, dfi in zip(AX, dfs):
    #     color = np.where(dfi['KBM'].to_numpy(), 'blue', 'black')    
    #     ax.scatter(dfi['kymin'], dfi['growthrate'], color=color)
    #     ax.set_title(str(dfi['beta_fraction'].iloc[0]))
    # figure.tight_layout()
    # figure.savefig('./kymin.png')
    
    
if __name__ == '__main__':
    _, base_run_dir = sys.argv
    
    plot_beta_scan(base_run_dir)