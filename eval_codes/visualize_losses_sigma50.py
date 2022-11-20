import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from easydict import EasyDict
from utils.utils import load_option

def smoothing(array, a):
    new_array = np.zeros_like(array)
    new_array[0] = array[0]
    for i in range(1, len(new_array)):
        new_array[i] = (1-a)*array[i] + a*new_array[i-1]
    return new_array

def plot_losses(opt_paths, a):
    for opt_path in opt_paths:
        opt = EasyDict(load_option(opt_path))
        
        model_name = opt.name
        log_dir = os.path.join('experiments', model_name, 'logs')
        train_log = pd.read_csv(os.path.join(log_dir, f'train_losses_{model_name}.csv'))
        
        plt.figure()
        plt.plot(train_log['step'], smoothing(train_log['loss_G'], a), label=f'train_loss_G', alpha=0.75)
        plt.ylim(top=-30, bottom=-40)
        plt.xlabel('Steps')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.title(f'Loss {model_name}')
        plt.grid(axis='y')
        plt.savefig(os.path.join(log_dir, f'losses_{model_name}.png'))
        
        fig = plt.figure(figsize=(19.2, 4.8))
        
        ax1 = fig.add_subplot(131)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss_G')
        ax1.grid(axis='y')
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('PSNR')
        ax2.grid(axis='y')
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('SSIM')
        ax3.grid(axis='y')
        for sigma in [50]:
            test_log = pd.read_csv(os.path.join(log_dir, f'test_losses_{model_name}_{sigma}.csv'))
            min_idx_loss_G = test_log['loss_G'].idxmin()
            point_loss_G = test_log['step'][min_idx_loss_G], test_log['loss_G'][min_idx_loss_G]
            max_idx_psnr = test_log['psnr'].idxmax()
            point_psnr = test_log['step'][max_idx_psnr], test_log['psnr'][max_idx_psnr], test_log['ssim'][max_idx_psnr]
            max_idx_ssim = test_log['ssim'].idxmax()
            point_ssim = test_log['step'][max_idx_ssim], test_log['ssim'][max_idx_ssim]
            if sigma==50: points = [sigma, point_loss_G, point_psnr, point_ssim]

            ax1.plot(test_log['step'], test_log['loss_G'], label=f'{sigma}', alpha=1.0)
            ax1.scatter(*point_loss_G, color='red', s=10)
            ax2.plot(test_log['step'], test_log['psnr'], label=f'{sigma}', alpha=1.0)
            ax2.scatter(*point_psnr[:2], color='red', s=10)
            ax3.plot(test_log['step'], test_log['ssim'], label=f'{sigma}', alpha=1.0)
            ax3.scatter(*point_ssim, color='red', s=10)
        ax1.set_title(f'Loss_G {model_name}, sigma{points[0]}: {points[1]}')
        ax2.set_title(f'PSNR {model_name}, sigma{points[0]}: {points[2]}')
        ax3.set_title(f'SSIM {model_name}, sigma{points[0]}: {points[3]}')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        
        plt.savefig(os.path.join(log_dir, f'metrics_{model_name}.png'))
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of plot losses.')
    parser.add_argument('-c', '--config_files', nargs='+', required=True)
    parser.add_argument('-a', '--smoothing_ratio', default=0.9, type=float)
    
    args = parser.parse_args()
    
    plot_losses(args.config_files, args.smoothing_ratio)