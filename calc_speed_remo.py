import torch
import numpy as np
import time
from easydict import EasyDict
import importlib
import thop

from UNUSED.models_ReMoNet import ReMoNet

def calc_speed(resolution):
    divisible_by = 16
    resolution[0] = divisible_by*(resolution[0]//divisible_by) if resolution[0]%divisible_by==0 else divisible_by*(resolution[0]//divisible_by+1)
    resolution[1] = divisible_by*(resolution[1]//divisible_by) if resolution[1]%divisible_by==0 else divisible_by*(resolution[1]//divisible_by+1)

    print('Creating Network...')
    device = torch.device('cuda:0')
    net = ReMoNet().to(device)
    net.eval()

    # Blank Shot
    print('Processing Denoising...')
    runtimes = []
    with torch.no_grad():
        for i in range(11):
            input_seq = torch.rand([1, 5, 3, *resolution]).to(device)
            noise_map = torch.rand([1, 1, 1, *resolution]).to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = net(input_seq, noise_map)
            torch.cuda.synchronize()
            #print(out.shape)
            elapsed = time.time()-start
            if i!=0: runtimes.append(elapsed)
            print(f'{i} -> {elapsed:f} s')
    
    return np.array(runtimes)

def calc_gmacs(resolution):
    divisible_by = 16
    resolution[0] = divisible_by*(resolution[0]//divisible_by) if resolution[0]%divisible_by==0 else divisible_by*(resolution[0]//divisible_by+1)
    resolution[1] = divisible_by*(resolution[1]//divisible_by) if resolution[1]%divisible_by==0 else divisible_by*(resolution[1]//divisible_by+1)

    print('Creating Network...')
    device = torch.device('cuda:0')
    net = ReMoNet().to(device)
    net.eval()

    input_seq = torch.rand([1, 5, 3, *resolution]).to(device)
    noise_map = torch.rand([1, 1, 1, *resolution]).to(device)
    #out = net(input_seq, noise_map)
    macs, params = thop.profile(net, inputs=[input_seq, noise_map])
    print(f'#Params: {params/1e6:f}, GMACs: {macs/1e9:f}')




if __name__=='__main__':
    #H, W = 256, 256
    #H, W = 480, 720
    #H, W = 480, 856
    H, W = 720, 1280
    #H, W = 1080, 1920
    #H, W = 2560, 1440
    #H, W = 540, 960
    #runtimes = calc_speed(resolution=[H, W])
    #print(f'{np.mean(runtimes)/5:f} ± {np.std(runtimes)/5:f} s')
    calc_gmacs(resolution=[H,W])