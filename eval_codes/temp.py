import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import argparse
from collections import OrderedDict
import glob
import cv2
import numpy as np

from utils.utils import tensor2ndarray, load_option, pad_tensor
from dataloader import SingleVideoDenoisingTestDataset
from easydict import EasyDict
from models.network import FastDVDNet
from models.networkM import FastDVDNetM

def generate_images(opt, checkpoint_path, out_dir):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset_paths = sorted([d for d in glob.glob('datasets/DAVIS-test/JPEGImages/480p/*') if os.path.isdir(d)])
    # dataset_paths = ['datasets/DAVIS-test/JPEGImages/480p/rollercoaster']
    for idx_dataset, dataset_path in enumerate(dataset_paths):
        dataset_name = os.path.basename(dataset_path)
        print(f'{idx_dataset}/{len(dataset_paths)}: Processing {dataset_name}')
        out_dataset_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(out_dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dataset_dir, 'GT'), exist_ok=True)
        opt.val_dataset_path = dataset_path

        val_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        for i, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(val_loader):
            _, _, H, W = x0.shape
            x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
            x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: pad_tensor(x, divisible_by=8), [x0, x1, x2, x3, x4, noise_map, gt])

            img, gen, gt = map(lambda x: tensor2ndarray(x), [x2, out, gt])
            img, gen, gt = map(lambda x: x[:,:H,:W,:], [img, gen, gt])

            #img = Image.fromarray(img[0,:,:,:])
            #gen = Image.fromarray(gen[0,:,:,:])
            gt = Image.fromarray(gt[0,:,:,:])

            fname = f'{i:03}.png'
            gt.save(os.path.join(out_dataset_dir, 'GT', fname), 'PNG')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of generate images.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, help='Path to the chenckpoint')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
        
    model_name = opt.name
    if args.checkpoint_path==None:
        checkpoint_path = os.path.join('experiments', model_name, 'ckpt', f'{model_name}_{opt.steps}.ckpt')
    else:
        checkpoint_path = args.checkpoint_path
    
    out_dir = f'results/{model_name}'
    
    generate_images(opt, checkpoint_path, out_dir)