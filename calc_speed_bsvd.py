import torch
import numpy as np
import time
from easydict import EasyDict
import importlib
from models.wnet_bsvd import BSVD

from utils.utils import load_option

def calc_speed(resolution):
    #opt_path = 'experiments/wnet/config_wnet.json'
    opt_path = 'config/config_test3.json'
    opt = EasyDict(load_option(opt_path))
    divisible_by = 8
    resolution[0] = divisible_by*(resolution[0]//divisible_by) if resolution[0]%divisible_by==0 else divisible_by*(resolution[0]//divisible_by+1)
    resolution[1] = divisible_by*(resolution[1]//divisible_by) if resolution[1]%divisible_by==0 else divisible_by*(resolution[1]//divisible_by+1)

    print('Creating Network...')
    device = torch.device('cuda:0')
    network_module = importlib.import_module('models.network_mimo3')
    #net = BSVD(pretrain_ckpt='/home/shibasaki/MyTask/BSVD/experiments/pretrained_ckpt/bsvd-64.pth').to(device)
    net = getattr(network_module, 'NAFBBB')(opt).to(device)
    net.eval()

    # Blank Shot
    print('Processing Denoising...')
    runtimes = []
    with torch.no_grad():
        n_frames = 50
        input_seq = torch.rand([1, n_frames, 3, *resolution]).to(device)
        noise_map = torch.rand([1, 1, 1, *resolution]).to(device)

        torch.cuda.synchronize()
        start = time.time()
        out = net(input_seq, noise_map)
        torch.cuda.synchronize()

        elapsed = time.time()-start
        #runtimes.append(elapsed)
        print(f'{elapsed:f} s')

        for i in range(10):
            input_seq = torch.rand([1, n_frames, 3, *resolution]).to(device)
            noise_map = torch.rand([1, 1, 1, *resolution]).to(device)

            torch.cuda.synchronize()
            start = time.time()
            out = net(input_seq, noise_map)
            torch.cuda.synchronize()

            elapsed = time.time()-start
            runtimes.append(elapsed)
            print(f'{i} -> {elapsed:f} s')

    
    return np.array(runtimes)


if __name__=='__main__':
    #H, W = 256, 256
    #H, W = 480, 720
    #H, W = 480, 852
    H, W = 1080, 1920
    #H, W = 540, 960
    runtimes = calc_speed(resolution=[H, W])
    print(f'{np.mean(runtimes)/50:f} ± {np.std(runtimes)/50:f} s')