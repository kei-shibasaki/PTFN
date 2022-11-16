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

from metrics import calculate_psnr, calculate_ssim
from utils.utils import load_option



def eval_from_image():
    gen_path = '/home/shibasaki/MyTask/BSVD/results/bsvd_c64/visualization/davis_50_fp32'
    log_name = 'temp/eval_log_bsvd_fp32.csv'
    with open(log_name, 'w', encoding='utf-8') as fp:
        fp.write(f'video_name,psnr,ssim\n')
    
    # caluculating psnr/ssim in each video
    dir_paths = sorted([d for d in glob.glob(os.path.join(gen_path, '*')) if os.path.isdir(d)])
    for idx_dir, dir_path in enumerate(dir_paths):
        video_name = os.path.basename(dir_path)
        print(f'{idx_dir}/{len(dir_paths)}: Processing {video_name} ...')
        gt_dir = f'results/naf_small/{video_name}/GT'

        images_gen = sorted(glob.glob(os.path.join(dir_path, '*.png')))
        images_gt = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        assert len(images_gen)==len(images_gt), f'{len(images_gen)} vs {len(images_gt)}'
        n_images = len(images_gen)

        psnr, ssim = 0.0, 0.0
        with torch.no_grad():
            for imgpath_gen, imgpath_gt in zip(images_gen, images_gt):
                with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                    img_gen = np.array(img_gen)
                    img_gt = np.array(img_gt)

                    psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                    ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
        
        with open(log_name, 'a', encoding='utf-8') as fp:
            fp.write(f'{video_name},{psnr:f},{ssim:f}\n')
    
    # add psnr/ssim average
    df = pd.read_csv(log_name)
    psnr_mean = df['psnr'].mean()
    ssim_mean = df['ssim'].mean()
    with open(log_name, 'a', encoding='utf-8') as fp:
        fp.write(f'All,{psnr_mean:f},{ssim_mean:f}\n')

if __name__=='__main__':
    eval_from_image()