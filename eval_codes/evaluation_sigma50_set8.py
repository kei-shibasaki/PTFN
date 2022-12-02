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
            fp.write(f'video_name,psnr,ssim\n')
    
    # caluculating psnr/ssim in each video
    dir_paths = sorted([d for d in glob.glob(f'results/{model_name}_set8/*') if os.path.isdir(d)])
    for idx_dir, dir_path in enumerate(dir_paths):
        video_name = os.path.basename(dir_path)
        print(f'{idx_dir}/{len(dir_paths)}: Processing {video_name} ...')
        for sigma in [50]:
            images_gen = sorted(glob.glob(os.path.join(dir_path, 'generated', str(sigma), '*.png')))
            images_gt = sorted(glob.glob(os.path.join(dir_path, 'GT', '*.png')))
            assert len(images_gen)==len(images_gt), f'{len(images_gen)} vs {len(images_gt)}'
            n_images = len(images_gen)

            psnr, ssim = 0.0, 0.0
            with torch.no_grad():
                for imgpath_gen, imgpath_gt in zip(tqdm(images_gen), images_gt):
                    with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                        img_gen = np.array(img_gen)
                        img_gt = np.array(img_gt)

                        psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                        ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
            
            with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
                fp.write(f'{video_name},{psnr:f},{ssim:f}\n')
    
    # add psnr/ssim average
    for sigma in [50]:
        df = pd.read_csv(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'))
        psnr_mean = df['psnr'].mean()
        ssim_mean = df['ssim'].mean()
        with open(os.path.join(out_path, f'{model_name}_{sigma:02}_results.csv'), 'a', encoding='utf-8') as fp:
            fp.write(f'All,{psnr_mean:f},{ssim_mean:f}\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    model_name = opt.name
    
    out_path = f'results/{model_name}_set8'

    eval_from_image(out_path, model_name)