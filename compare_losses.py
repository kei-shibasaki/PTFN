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
    name1 = 'ptfn_b8'
    name2 = 'ptfn_progressive2'
    train_log1 = pd.read_csv(f'experiments/{name1}/logs/train_losses_{name1}.csv')
    train_log2 = pd.read_csv(f'experiments/{name2}/logs/train_losses_{name2}.csv')
    val_log1 = pd.read_csv(f'experiments/{name1}/logs/test_losses_{name1}_50.csv')
    val_log2 = pd.read_csv(f'experiments/{name2}/logs/test_losses_{name2}_50.csv')
    metric = 'psnr'

    plt.figure(figsize=(8,4.5))
    plt.plot(train_log1['step'], smoothing(train_log1['loss_G'], 0.99), alpha=0.5, label=f'{name1} train')
    plt.plot(train_log2['step'], smoothing(train_log2['loss_G'], 0.99), alpha=0.5, label=f'{name2} train')
    plt.plot(val_log1['step'], smoothing(val_log1['loss_G'], 0.0), alpha=0.5, label=f'{name1} val')
    plt.plot(val_log2['step'], smoothing(val_log2['loss_G'], 0.0), alpha=0.5, label=f'{name2} val')
    #plt.ylim(0.0002, 0.0008) 
    plt.ylim(-42, -30)
    plt.legend()
    plt.grid()
    plt.savefig(f'temp/compare_train_{name1}_{name2}.png')

    plt.figure(figsize=(8,4.5))
    plt.plot(val_log1['step'], val_log1[metric], alpha=0.5, label=f'{name1}')
    plt.plot(val_log2['step'], val_log2[metric], alpha=0.5, label=f'{name2}')
    #plt.ylim(0.0005, 0.001)
    plt.ylim(30, 35)
    plt.legend()
    plt.grid()
    plt.savefig(f'temp/compare_val_{name1}_{name2}_{metric}.png')


if __name__=='__main__':
    compare_losses()