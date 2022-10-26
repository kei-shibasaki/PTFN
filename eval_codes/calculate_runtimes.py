import os
import torch
from torch.nn import functional as F
import torch.utils.data
import argparse
import numpy as np
import time

from utils.utils import load_option, pad_tensor
from dataloader import SingleVideoDenoisingTestDataset
from easydict import EasyDict
from models.network import FastDVDNet
from models.networkM import FastDVDNetM

def calc_speed(opt, checkpoint_path, resolution):
    device = torch.device('cuda:0')
    
    print('Setting Dataloader...')
    dataset_path = 'datasets/DAVIS-test/JPEGImages/480p/rollercoaster'
    opt.val_dataset_path = dataset_path
    val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=50)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Creating Network...')
    net = FastDVDNetM().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['netG_state_dict'])
    net.eval()

    # Blank Shot
    print('Processing Blank Shot...')
    for _ in range(10):
        with torch.no_grad():
            x0, x1, x2, x3, x4, gt = [torch.rand([1, 3, *resolution]).to(device) for _ in range(6)]
            noise_map = torch.rand([1, 1, *resolution]).to(device)
            torch.cuda.synchronize()
            out = net(x0, x1, x2, x3, x4, noise_map)
            torch.cuda.synchronize()

    runtimes = []
    for i, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(val_loader):
        _, _, H, W = x0.shape
        x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
        # x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: pad_tensor(x, divisible_by=8), [x0, x1, x2, x3, x4, noise_map, gt])
        x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: F.interpolate(x, size=resolution), [x0, x1, x2, x3, x4, noise_map, gt])
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            out = net(x0, x1, x2, x3, x4, noise_map)
            torch.cuda.synchronize()
            elapsed = time.time()-start
            runtimes.append(elapsed)
            print(f'{i} -> {elapsed*1e3:f} ms')
    
    return np.array(runtimes)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of generate images.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    model_name = opt.name
    checkpoint_path = os.path.join('experiments', model_name, 'ckpt', f'{model_name}_{opt.steps}.ckpt')
    
    runtimes = calc_speed(opt, checkpoint_path, resolution=(256, 256))

    print(f'{np.mean(runtimes)*1e3:f} ± {np.std(runtimes)*1e3:f} ms')