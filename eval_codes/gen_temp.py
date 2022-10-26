import os
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import argparse
import glob
import cv2
import numpy as np

from utils.utils import tensor2ndarray, load_option, pad_tensor
from dataloader import SingleVideoDenoisingTestDataset
from easydict import EasyDict
from models.network import FastDVDNet2

def generate_images(opt, checkpoint_path, out_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #seeds = random.sample([x for x in range(9999)], 50)
    #print(seeds)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)
    
    for seed in tqdm(range(50)):
        dataset_paths = ['datasets/DAVIS-test/JPEGImages/480p/rollercoaster']
        for dataset_path in dataset_paths:
            dataset_name = os.path.basename(dataset_path)
            out_dataset_dir = os.path.join(out_dir, dataset_name+f'_{seed:04}')
            os.makedirs(out_dataset_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dataset_dir, 'GT'), exist_ok=True)

            net = FastDVDNet2().to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['netG_state_dict'])
            net.eval()

            opt.val_dataset_path = dataset_path
            for sigma in [50]:
                os.makedirs(os.path.join(out_dataset_dir, 'input', str(sigma)), exist_ok=True)
                os.makedirs(os.path.join(out_dataset_dir, 'generated', str(sigma)), exist_ok=True)

                val_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
                for i, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(val_loader):
                    _, _, H, W = x0.shape
                    x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
                    x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: pad_tensor(x, divisible_by=8), [x0, x1, x2, x3, x4, noise_map, gt])
                    with torch.no_grad():
                        out = net(x0, x1, x2, x3, x4, noise_map)

                    img, gen, gt = map(lambda x: tensor2ndarray(x), [x2, out, gt])
                    img, gen, gt = map(lambda x: x[:,:H,:W,:], [img, gen, gt])

                    img = Image.fromarray(img[0,:,:,:])
                    gen = Image.fromarray(gen[0,:,:,:])
                    gt = Image.fromarray(gt[0,:,:,:])

                    fname = f'{i:03}.png'
                    if sigma==50: gt.save(os.path.join(out_dataset_dir, 'GT', fname), 'PNG')
                    img.save(os.path.join(out_dataset_dir, 'input', str(sigma), fname), 'PNG')
                    gen.save(os.path.join(out_dataset_dir, 'generated', str(sigma), fname), 'PNG')


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