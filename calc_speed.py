import torch
import numpy as np
import time

from models.network import FastDVDNet
from models.networkM import FastDVDNetM, ExtremeStageDenoisingNetwork, ExtremeStageDenoisingNetwork2

def calc_speed(resolution):
    device = torch.device('cuda:0')

    print('Creating Network...')
    net = ExtremeStageDenoisingNetwork2().to(device)
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # net.load_state_dict(checkpoint['netG_state_dict'])
    net.eval()

    # Blank Shot
    print('Processing Denoising...')
    runtimes = []
    with torch.no_grad():
        for i in range(51):
            x0, x1, x2, x3, x4 = [torch.rand([1, 3, *resolution]).to(device) for _ in range(5)]
            noise_map = torch.rand([1, 1, *resolution]).to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = net(x0, x1, x2, x3, x4, noise_map)
            torch.cuda.synchronize()
            elapsed = time.time()-start
            if i!=0: runtimes.append(elapsed)
            print(f'{i} -> {elapsed:f} s')
    
    return np.array(runtimes)


if __name__=='__main__':
    H, W = 256, 256
    #H, W = 480, 720
    #H, W = 1080, 1920
    runtimes = calc_speed(resolution=(H, W))
    print(f'{np.mean(runtimes):f} ± {np.std(runtimes):f} s')