import torch
import numpy as np
import time
from easydict import EasyDict
import importlib

from scripts.utils import load_option

def calc_speed(resolution):
    opt_path = 'config/config_test.json'
    opt = EasyDict(load_option(opt_path))
    divisible_by = 16
    resolution[0] = divisible_by*(resolution[0]//divisible_by) if resolution[0]%divisible_by==0 else divisible_by*(resolution[0]//divisible_by+1)
    resolution[1] = divisible_by*(resolution[1]//divisible_by) if resolution[1]%divisible_by==0 else divisible_by*(resolution[1]//divisible_by+1)

    print('Creating Network...')
    device = torch.device('cuda:0')
    network_module = importlib.import_module('models.network')
    net = getattr(network_module, opt['model_type_test'])(opt).to(device)
    net.eval()

    # Blank Shot
    print('Processing Denoising...')
    runtimes = []
    with torch.no_grad():
        for i in range(51):
            input_seq = torch.rand([1, opt.n_frames, 3, *resolution]).to(device)
            noise_map = torch.rand([1, 1, 1, *resolution]).to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = net(input_seq, noise_map)
            torch.cuda.synchronize()
            elapsed = time.time()-start
            if i!=0: runtimes.append(elapsed)
            print(f'{i} -> {elapsed:f} s')
    
    return np.array(runtimes)


if __name__=='__main__':
    #H, W = 256, 256
    #H, W = 480, 854
    #H, W = 720, 1280
    #H, W = 1080, 1920
    runtimes = calc_speed(resolution=[H, W])
    print(f'{np.mean(runtimes):f} ± {np.std(runtimes):f} s')