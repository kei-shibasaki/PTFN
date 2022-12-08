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



def eval_from_image(out_path, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create result files
    for sigma in [50]:
        with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'w', encoding='utf-8') as fp:
            fp.write(f'video_name,psnr_inter,ssim_inter,psnr,ssim\n')
    
    # caluculating psnr/ssim in each video
    dir_paths = sorted([d for d in glob.glob(f'results/{model_name}_set8/*') if os.path.isdir(d)])
    for idx_dir, dir_path in enumerate(dir_paths):
        video_name = os.path.basename(dir_path)
        print(f'{idx_dir}/{len(dir_paths)}: Processing {video_name} ...')
        for sigma in [50]:
            images_gen1 = sorted(glob.glob(os.path.join(dir_path, 'generated_inter', str(sigma), '*.png')))
            images_gen2 = sorted(glob.glob(os.path.join(dir_path, 'generated', str(sigma), '*.png')))
            images_gt = sorted(glob.glob(os.path.join(dir_path, 'GT', '*.png')))
            assert len(images_gen1)==len(images_gt), f'{len(images_gen1)} vs {len(images_gt)}'
            n_images = len(images_gen1)

            psnr1, ssim1 = 0.0, 0.0
            psnr2, ssim2 = 0.0, 0.0
            with torch.no_grad():
                for imgpath_gen1, imgpath_gen2, imgpath_gt in zip(tqdm(images_gen1), images_gen2, images_gt):
                    with Image.open(imgpath_gen1) as img_gen1, Image.open(imgpath_gen2) as img_gen2, Image.open(imgpath_gt) as img_gt:
                        img_gen1 = np.array(img_gen1)
                        img_gen2 = np.array(img_gen2)
                        img_gt = np.array(img_gt)

                        psnr1 += calculate_psnr(img_gen1, img_gt, crop_border=0, test_y_channel=False) / n_images
                        ssim1 += calculate_ssim(img_gen1, img_gt, crop_border=0, test_y_channel=False) / n_images
                        psnr2 += calculate_psnr(img_gen2, img_gt, crop_border=0, test_y_channel=False) / n_images
                        ssim2 += calculate_ssim(img_gen2, img_gt, crop_border=0, test_y_channel=False) / n_images
            
            with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
                fp.write(f'{video_name},{psnr1:f},{ssim1:f},{psnr2:f},{ssim2:f}\n')
    
    # add psnr/ssim average
    for sigma in [50]:
        df = pd.read_csv(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'))
        psnr1_mean = df['psnr_inter'].mean()
        ssim1_mean = df['ssim_inter'].mean()
        psnr2_mean = df['psnr'].mean()
        ssim2_mean = df['ssim'].mean()
        with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
            fp.write(f'All,{psnr1_mean:f},{ssim1_mean:f},{psnr2_mean:f},{ssim2_mean:f}\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    model_name = opt.name
    
    out_path = f'results/{model_name}_set8'

    eval_from_image(out_path, model_name)