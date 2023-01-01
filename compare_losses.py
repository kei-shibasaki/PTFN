import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def smoothing(array, a):
    new_array = np.zeros_like(array)
    new_array[0] = array[0]
    for i in range(1, len(new_array)):
        new_array[i] = (1-a)*array[i] + a*new_array[i-1]
    return new_array

def compare_losses():
    names = ['ptfn_v4_small_b16', 'ptfn_inter000', 'ptfn_inter001', 'ptfn_inter100']
    #names = ['ptfn_b16', 'ptfn_v4_b16']
    train_logs = {}
    val_logs = {}
    for name in names:
        train_logs[name] = pd.read_csv(f'experiments/{name}/logs/train_losses_{name}.csv')
        val_logs[name] = pd.read_csv(f'experiments/{name}/logs/test_losses_{name}_50.csv')
    metric = 'psnr'
    figsize = (16,9)

    plt.figure(figsize=figsize)
    for name in names:
        tlog = train_logs[name]
        vlog = val_logs[name]
        plt.plot(tlog['step'], smoothing(tlog['loss_G'], 0.99), alpha=0.5, label=f'{name} train')
        plt.plot(vlog['step'], smoothing(vlog['loss_G'+'_inter'], 0.0), alpha=0.5, label=f'{name} val inter')
        plt.plot(vlog['step'], smoothing(vlog['loss_G'+'_final'], 0.0), alpha=0.5, label=f'{name} val final')
    plt.ylim(-42, -30)
    plt.legend()
    plt.grid()
    plt.savefig(f'temp/compare_train_{"_".join(names)}.png')

    plt.figure(figsize=figsize)
    for name in names:
        vlog = val_logs[name]
        plt.plot(vlog['step'], smoothing(vlog[metric], 0.0), alpha=0.5, label=f'{name}')
        plt.plot(vlog['step'], smoothing(vlog[metric+'_inter'], 0.0), alpha=0.5, label=f'{name} inter')

    plt.ylim(30, 35)
    plt.legend()
    plt.grid()
    plt.savefig(f'temp/compare_val_{"_".join(names)}_{metric}.png')


if __name__=='__main__':
    compare_losses()