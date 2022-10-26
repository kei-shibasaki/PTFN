import argparse
import datetime
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.utils
from easydict import EasyDict
from PIL import Image
from torch.nn import functional as F

from dataloader import DAVISVideoDenoisingTrainDataset, SingleVideoDenoisingTestDataset
from metrics import calculate_psnr, calculate_ssim
from models.networkM import ExtremeStageDenoisingNetwork2
from utils.utils import load_option, pad_tensor, tensor2ndarray, send_line_notify

def train(opt_path):
    opt = EasyDict(load_option(opt_path))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_name = opt.name
    batch_size = opt.batch_size
    print_freq = opt.print_freq
    eval_freq = opt.eval_freq
    
    model_ckpt_dir = f'./experiments/{model_name}/ckpt'
    image_out_dir = f'./experiments/{model_name}/generated'
    log_dir = f'./experiments/{model_name}/logs'
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(f'{log_dir}/log_{model_name}.log', mode='w', encoding='utf-8') as fp:
        fp.write('')
    with open(f'{log_dir}/train_losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,lr,loss_G\n')
    for sigma in [10,20,30,40,50]:
        with open(f'{log_dir}/test_losses_{model_name}_{sigma}.csv', mode='w', encoding='utf-8') as fp:
            fp.write('step,loss_G,psnr,ssim\n')
    
    shutil.copy(opt_path, f'./experiments/{model_name}/{os.path.basename(opt_path)}')
    
    netG = ExtremeStageDenoisingNetwork2().to(device)
    if opt.pretrained_path:
        netG_state_dict = torch.load(opt.pretrained_path, map_location=device)
        netG_state_dict = netG_state_dict['netG_state_dict']
        netG.load_state_dict(netG_state_dict, strict=False)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=opt.milestones, gamma=0.1)
    
    train_dataset = DAVISVideoDenoisingTrainDataset(opt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loaders = {}
    for sigma in [10,20,30,40,50]:
        val_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        val_loaders[sigma] = val_loader

    print('Start Training')
    start_time = time.time()
    total_step = 0
    netG.train()
    for e in range(1, 42*opt.steps):
        for i, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(train_loader):
            x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
            
            # Training G
            netG.zero_grad()
            gen = netG(x0, x1, x2, x3, x4, noise_map)
            # gen = torch.sigmoid(gen)
            loss_G = F.mse_loss(gen, gt)

            loss_G.backward()
            optimG.step()
            
            total_step += 1
            
            schedulerG.step()
            
            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                with open(f'{log_dir}/train_losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    txt = f'{total_step},{lr_G[0]},{loss_G:f}\n'
                    fp.write(txt)
            
            if total_step%print_freq==0 or total_step==1:
                rest_step = opt.steps-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.steps}, Epoch:{str(e).zfill(len(str(opt.steps)))}, elepsed: {elapsed}, eta: {eta}, loss_G: {loss_G:f}'
                print(lg)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if total_step%eval_freq==0:
                # Validation
                netG.eval()
                for sigma in [10,20,30,40,50]:
                    val_loader = val_loaders[sigma]
                    psnr = 0.0
                    ssim = 0.0
                    loss_G = 0.0
                    for j, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(val_loader):
                        x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
                        x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: pad_tensor(x, divisible_by=8), [x0, x1, x2, x3, x4, noise_map, gt])
                        with torch.no_grad():
                            gen = netG(x0, x1, x2, x3, x4, noise_map)
                            #gen = torch.sigmoid(gen)
                            loss_G += F.mse_loss(gen, gt)
                        
                        img = x2
                        img = tensor2ndarray(img)
                        gen = tensor2ndarray(gen)
                        gt = tensor2ndarray(gt)

                        os.makedirs(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}'), exist_ok=True)
                        psnr += calculate_psnr(gen[0,:,:,:], gt[0,:,:,:], crop_border=0, test_y_channel=False)
                        ssim += calculate_ssim(gen[0,:,:,:], gt[0,:,:,:], crop_border=0, test_y_channel=False)
                            
                        # Visualization
                        if opt.color_channels==1:
                            img = Image.fromarray(img[0,:,:,0])
                            gen = Image.fromarray(gen[0,:,:,0])
                            gt = Image.fromarray(gt[0,:,:,0])
                        else:
                            img = Image.fromarray(img[0,:,:,:])
                            gen = Image.fromarray(gen[0,:,:,:])
                            gt = Image.fromarray(gt[0,:,:,:])
                        compare_img = Image.new('RGB', size=(3*img.width, img.height), color=0)
                        compare_img.paste(img, box=(0, 0))
                        compare_img.paste(gen, box=(img.width, 0))
                        compare_img.paste(gt, box=(2*img.width, 0))
                        compare_img.save(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}', f'{j:03}.png'), 'PNG')
                    
                    loss_G = loss_G / len(val_loader)
                    psnr = psnr / len(val_loader)
                    ssim = ssim / len(val_loader)
                
                    txt = f'sigma: {sigma}, loss_G: {loss_G:f}, PSNR: {psnr:f}, SSIM: {ssim:f}'
                    print(txt)
                    with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                        fp.write(txt+'\n')
                    with open(f'{log_dir}/test_losses_{model_name}_{sigma}.csv', mode='a', encoding='utf-8') as fp:
                        fp.write(f'{total_step},{loss_G:f},{psnr:f},{ssim:f}\n')
                
                if total_step%opt.save_freq==0 and opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'{opt.name} Step: {total_step}\n{lg}\n{txt}')

            if total_step%opt.save_freq==0:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{model_name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))
                    
            if total_step==opt.steps:
                if opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'Complete training {opt.name}.')
                
                print('Completed.')
                exit()
                

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of training network with adversarial loss.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    
    train(args.config)
