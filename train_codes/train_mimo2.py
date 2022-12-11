import argparse
import datetime
import json
import os
import random
import shutil
import time
import importlib

import numpy as np
import torch
import torch.utils
from easydict import EasyDict
from PIL import Image
from torch.nn import functional as F

from dataset import DAVISVideoDenoisingTrainDatasetMIMO, SingleVideoDenoisingTestDataset
from metrics import calculate_psnr, calculate_ssim
from utils.utils import load_option, pad_tensor, tensor2ndarray, send_line_notify, convert_state_dict
from losses import PSNRLoss

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
    #for sigma in [10,20,30,40,50]:
    for sigma in [50]:
        with open(f'{log_dir}/test_losses_{model_name}_{sigma}.csv', mode='w', encoding='utf-8') as fp:
            fp.write('step,loss_G,psnr,ssim\n')
    
    shutil.copy(opt_path, f'./experiments/{model_name}/{os.path.basename(opt_path)}')
    
    loss_fn = PSNRLoss().to(device)
    network_module = importlib.import_module('models.network_mimo2')
    netG = getattr(network_module, opt['model_type_train'])(opt).to(device)
    netG_val = getattr(network_module, opt['model_type_test'])(opt).to(device)
    if opt.pretrained_path:
        netG_state_dict = torch.load(opt.pretrained_path, map_location=device)
        netG_state_dict = netG_state_dict['netG_state_dict']
        netG.load_state_dict(netG_state_dict, strict=False)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, T_max=opt.T_max, eta_min=opt.eta_min)
    
    train_dataset = DAVISVideoDenoisingTrainDatasetMIMO(opt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loaders = {}
    #for sigma in [10,20,30,40,50]:
    for sigma in [50]:
        val_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        val_loaders[sigma] = val_loader

    print('Start Training')
    start_time = time.time()
    total_step = 0
    best_psnr = -float('inf')
    netG.train()
    for e in range(1, 42*opt.steps):
        for i, data in enumerate(train_loader):
            # [(B,C,H,W)]*F -> (B,F,C,H,W)
            input_seq = torch.stack(data['input_seq'], dim=1).to(device)
            gt_seq = torch.stack(data['gt_seq'], dim=1).to(device)
            noise_map = data['noise_map'].unsqueeze(1).to(device)

            b,f,c,h,w = input_seq.shape
            
            # Training G
            netG.zero_grad()
            gen = netG(input_seq, noise_map)
            loss_G = loss_fn(gen.reshape(b*f,c,h,w), gt_seq.reshape(b*f,c,h,w))

            loss_G.backward()
            if opt.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(netG.parameters(), 0.01)
            optimG.step()
            
            total_step += 1
            
            schedulerG.step()

            if total_step%eval_freq==0:
                # (B,F,C,H,W) -> [(B,C,H,W)]*F
                imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, gen, gt_seq])
                imgs = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], imgs))
                gens = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gens))
                gts = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gts))
                os.makedirs(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}'), exist_ok=True)
                for j, (img, gen, gt) in enumerate(zip(imgs, gens, gts)):
                    # Visualization
                    img, gen, gt = map(lambda x: Image.fromarray(x), [img, gen, gt])
                    compare_img = Image.new('RGB', size=(3*img.width, img.height), color=0)
                    compare_img.paste(img, box=(0, 0))
                    compare_img.paste(gen, box=(img.width, 0))
                    compare_img.paste(gt, box=(2*img.width, 0))
                    compare_img.save(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}', f'train_{i:03}_{j:03}.png'), 'PNG')

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
                netG_val.load_state_dict(convert_state_dict(netG.state_dict()), strict=True)
                netG_val.eval()
                #for sigma in [10,20,30,40,50]:
                for sigma in [50]:
                    val_loader = val_loaders[sigma]
                    psnr, ssim, loss_G = 0.0, 0.0, 0.0
                    for i, data in enumerate(val_loader):
                        # [(B,C,H,W)]*F -> (B,F,C,H,W)
                        input_seq = torch.cat(data['input_seq'], dim=0).unsqueeze(0).to(device)
                        gt_seq = torch.cat(data['gt_seq'], dim=0).unsqueeze(0).to(device)
                        noise_map = data['noise_map'].unsqueeze(1).to(device)
                        input_seq, gt_seq, noise_map = map(lambda x: pad_tensor(x, divisible_by=8), [input_seq, gt_seq, noise_map])
                        b,f,c,h,w = input_seq.shape
                        with torch.no_grad():
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                gen = netG_val(input_seq, noise_map)
                                loss_G = loss_fn(gen.reshape(b*f,c,h,w), gt_seq.reshape(b*f,c,h,w))
                        
                        # (B,F,C,H,W) -> [(B,C,H,W)]*F
                        imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, gen, gt_seq])
                        
                        imgs = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], imgs))
                        gens = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gens))
                        gts = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gts))

                        os.makedirs(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}'), exist_ok=True)
                        for j, (img, gen, gt) in enumerate(zip(imgs, gens, gts)):
                            psnr += calculate_psnr(gen, gt, crop_border=0, test_y_channel=False)
                            ssim += calculate_ssim(gen, gt, crop_border=0, test_y_channel=False)
                            
                            # Visualization
                            img, gen, gt = map(lambda x: Image.fromarray(x), [img, gen, gt])
                            compare_img = Image.new('RGB', size=(3*img.width, img.height), color=0)
                            compare_img.paste(img, box=(0, 0))
                            compare_img.paste(gen, box=(img.width, 0))
                            compare_img.paste(gt, box=(2*img.width, 0))
                            compare_img.save(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}', f'{i:03}_{j:03}.png'), 'PNG')
                    
                    #loss_G = loss_G
                    psnr = psnr / f
                    ssim = ssim / f
                
                    txt = f'sigma: {sigma}, loss_G: {loss_G:f}, PSNR: {psnr:f}, SSIM: {ssim:f}'
                    print(txt)
                    with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                        fp.write(txt+'\n')
                    with open(f'{log_dir}/test_losses_{model_name}_{sigma}.csv', mode='a', encoding='utf-8') as fp:
                        fp.write(f'{total_step},{loss_G:f},{psnr:f},{ssim:f}\n')
                    
                    if psnr >= best_psnr:
                        best_psnr = psnr
                        torch.save({
                            'total_step': total_step,
                            'netG_state_dict': netG.state_dict(),
                            'optimG_state_dict': optimG.state_dict(),
                        }, os.path.join(model_ckpt_dir, f'{model_name}_best.ckpt'))

                
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
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{model_name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))

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
