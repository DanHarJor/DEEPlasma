import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
def slice1d_post_proc(base_run_dir):
    # Load the CSV
    df = pd.read_csv(os.path.join(base_run_dir,"runner_return.csv"))
    # print(df)
    columns = [df.columns[i] for i in range(len(df.columns)-1)]
    output_col = df.columns[-1]
    print('COL',columns, output_col)
    h=5/2
    w=6/2
    nr=2
    nc=int(len(columns)/2)
    fig, AX = plt.subplots(nr,nc, figsize = (w*nc, h*nr), sharey=True)
    slices = []
    AX = AX.flatten()
    for i, col, ax in zip(range(len(columns)), columns, AX):
        df_filtered = df[df[col].map(df[col].value_counts()) == 1]
        df_filtered = df_filtered.sort_values(by=col)
        x = df_filtered[col].astype('float')
        y = df_filtered[output_col].astype('float')
        # x = x[np.argsort(x)]
        # y = y[np.argsort(x)]
        ax.plot(x, y)
        ax.set_xlabel(col)
        if i == 0: ax.set_ylabel('function_output')
        # else: ax.set_ylabel('')
        slices.append((x,y))
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(base_run_dir, 'slice1d.png'))
    return slices

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    slice1d_post_proc(base_run_dir)