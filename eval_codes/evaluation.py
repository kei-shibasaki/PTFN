import glob
import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
import pandas as pd

from scripts.metrics import calculate_psnr, calculate_ssim
from scripts.utils import load_option


def eval_from_image(out_path, model_name, noise_levels):
    # create result files
    for sigma in noise_levels:
        with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'w', encoding='utf-8') as fp:
            fp.write(f'video_name,psnr_inter,ssim_inter,psnr,ssim\n')
    
    # caluculating psnr/ssim in each video
    dir_paths = sorted([d for d in glob.glob(f'results/{model_name}/*') if os.path.isdir(d)])
    for idx_dir, dir_path in enumerate(dir_paths):
        video_name = os.path.basename(dir_path)
        print(f'{idx_dir}/{len(dir_paths)}: Processing {video_name} ...')
        for sigma in tqdm(noise_levels):
            images_gen_inter = sorted(glob.glob(os.path.join(dir_path, 'generated_inter', str(sigma), '*.png')))
            images_gen = sorted(glob.glob(os.path.join(dir_path, 'generated', str(sigma), '*.png')))
            images_gt = sorted(glob.glob(os.path.join(dir_path, 'GT', '*.png')))
            assert len(images_gen)==len(images_gt), f'{len(images_gen)} vs {len(images_gt)}'
            n_images = len(images_gen)

            psnr_inter, ssim_inter = 0.0, 0.0
            psnr, ssim = 0.0, 0.0
            with torch.no_grad():
                for imgpath_gen_inter, imgpath_gen, imgpath_gt in zip(images_gen_inter, images_gen, images_gt):
                    with Image.open(imgpath_gen_inter) as img_gen_inter, Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                        img_gen_inter = np.array(img_gen_inter)
                        img_gen = np.array(img_gen)
                        img_gt = np.array(img_gt)

                        psnr_inter += calculate_psnr(img_gen_inter, img_gt, crop_border=0, test_y_channel=False) / n_images
                        ssim_inter += calculate_ssim(img_gen_inter, img_gt, crop_border=0, test_y_channel=False) / n_images
                        psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                        ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
            
            with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
                fp.write(f'{video_name},{psnr_inter:f},{ssim_inter:f},{psnr:f},{ssim:f}\n')
    
    # add psnr/ssim average
    for sigma in noise_levels:
        df = pd.read_csv(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'))
        psnr_mean_inter = df['psnr_inter'].mean()
        ssim_mean_inter = df['ssim_inter'].mean()
        psnr_mean = df['psnr'].mean()
        ssim_mean = df['ssim'].mean()
        with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
            fp.write(f'All,{psnr_mean_inter:f},{ssim_mean_inter:f},{psnr_mean:f},{ssim_mean:f}\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    parser.add_argument('-nl', '--noise_levels', nargs='*', type=int, default=[10,20,30,40,50], help='List of level of gaussian noise [0, 255]')
    parser.add_argument('-set8', '--set8', action='store_true', help='Wether you want to evaluate Set8 dataset')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))

    if args.set8:
        model_name = opt.name + '_set8'
    else:
        model_name = opt.name
    
    out_path = f'results/{model_name}'

    noise_levels = args.noise_levels

    eval_from_image(out_path, model_name, noise_levels)