import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({'font.size': 24}) 

def slice1d_post_proc(base_run_dir):
    # Load the CSV
    df = pd.read_csv(os.path.join(base_run_dir,"merged_runner_return.txt"))
    # print(df)
    columns = [df.columns[i] for i in range(len(df.columns)-1)]
    output_col = df.columns[-1]
    print('COL',columns, output_col)
    h=5
    w=5
    nr=1
    nc=len(columns)#int(len(columns)/2)
    fig, AX = plt.subplots(nr,nc, figsize = (w*nc, h*nr), sharey=True)
    slices = []
    AX = AX.flatten()
    for i, col, ax in zip(range(len(columns)), columns, AX):
        df_filtered = df[df[col].map(df[col].value_counts()) == 1]
        df_filtered = df_filtered.sort_values(by=col)
        x = df_filtered[col].astype('float')
        y = df_filtered[output_col].astype('float')
        x = x[y!=0]
        y = y[y!=0]
        # x = x[np.argsort(x)]
        # y = y[np.argsort(x)]
        ax.scatter(x, y)
        ax.set_xlabel(col)
        if i == 0: ax.set_ylabel(output_col)
        # else: ax.set_ylabel('')
        slices.append((x,y))
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(base_run_dir, 'slice1d.png'))
    plt.close(fig)
    return slices

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    slice1d_post_proc(base_run_dir)