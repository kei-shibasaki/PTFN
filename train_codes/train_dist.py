import argparse
import datetime
import json
import os
import random
import shutil
import time

import lpips
import numpy as np
import torch
import torch.utils
from easydict import EasyDict
from PIL import Image
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from dataloader import Market1501TrainDataset, Market1501ValDataset
from losses import VGGLoss, gradient_penalty
from metrics import calculate_psnr, calculate_ssim
from model.discriminator import Discriminator
from model.pose_transformer import PoseTransformer
from utils.pose_utils import draw_pose_from_map
from utils.utils import load_option, tensor2ndarray, send_line_notify


def train(rank, opt_path):
    opt = EasyDict(load_option(opt_path))
    dist.init_process_group('nccl', rank=rank, world_size=opt.n_gpu)
    torch.backends.cudnn.benchmark = True
        
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
        fp.write('step,lr_D,loss_D,loss_D_real,loss_D_fake,gp,lr_G,loss_G,advloss,l1loss,ploss,sloss\n')
    with open(f'{log_dir}/test_losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,psnr,ssim,lpips\n')
    
    shutil.copy(opt_path, f'./experiments/{model_name}/{os.path.basename(opt_path)}')
    
    netG = PoseTransformer(opt).to(rank)
    if opt.pretrained_path:
        dist.barrier()
        map_location = {f'cuda:0': f'cuda:{rank}'}
        netG_state_dict = torch.load(opt.pretrained_path, map_location=map_location)
        netG_state_dict = netG_state_dict['netG_state_dict']
        netG.load_state_dict(netG_state_dict, strict=False)
    netG = DDP(netG, device_ids=[rank], find_unused_parameters=True)
    netD = Discriminator().to(rank)
    netD = DDP(netD, device_ids=[rank])
    perceptual_loss = VGGLoss().to(rank)
    loss_fn_alex = lpips.LPIPS(net='alex').to(torch.device('cuda:0'))
    loss_fn_alex.eval()
    
    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    optimD = torch.optim.Adam(netD.parameters(), lr=opt.learning_rate_D, betas=opt.betas)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=opt.milestones, gamma=0.5)
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimD, milestones=opt.milestones, gamma=0.5)
    
    if opt.dataset_type=='fashion':
        train_dataset = DeepFashionTrainDataset(opt)
        val_dataset = DeepFashionValDataset(opt)
    elif opt.dataset_type=='market':
        train_dataset = Market1501TrainDataset(opt)
        val_dataset = Market1501ValDataset(opt)

    train_sampler = DistributedSampler(train_dataset, num_replicas=opt.n_gpu, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=opt.n_gpu, rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
    
    print('Start Training')
    start_time = time.time()
    total_step = 0
    netG.train()
    for e in range(1, 9999):
        for i, data_dict in enumerate(train_loader):
            P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
            P1, P2, map1, map2 = P1.to(rank), P2.to(rank), map1.to(rank), map2.to(rank)
            b_size = P1.size(0)
            
            # Training D
            netD.zero_grad()
            logits_real = netD(P2)
            loss_D_real = -logits_real.mean()
            
            fake = netG(P1,map1,map2)
            fake_img = fake.sigmoid()
            logits_fake = netD(fake_img.detach())
            loss_D_fake = logits_fake.mean()
            
            gp = opt.coef_gp*gradient_penalty(netD, P2, fake_img)
            
            loss_D = loss_D_real + loss_D_fake + gp
            
            loss_D.backward(retain_graph=True)
            optimD.step()
            schedulerD.step()
            
            # Training G
            if total_step%1==0:
                netG.zero_grad()
                logits_fake = netD(fake_img)
                advloss = -logits_fake.mean()
                l1loss = opt.coef_l1*F.l1_loss(fake_img, P2)
                ploss, sloss = perceptual_loss(fake_img, P2)
                ploss, sloss = opt.coef_perc*ploss, opt.coef_style*sloss
                loss_G = advloss + l1loss + ploss + sloss
                
                loss_G.backward()
                optimG.step()
            
            total_step += 1
            
            schedulerG.step()
            
            if rank==0 and total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                lr_D = [group['lr'] for group in optimD.param_groups]
                with open(f'{log_dir}/train_losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    txt = f'{total_step},{lr_D[0]},{loss_D:f},{loss_D_real:f},{loss_D_fake:f},{gp:f},'
                    txt = txt + f'{lr_G[0]},{loss_G:f},{advloss:f},{l1loss:f},{ploss:f},{sloss:f}\n'
                    fp.write(txt)
            
            if rank==0 and (total_step%print_freq==0 or total_step==1):
                rest_step = opt.steps-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.steps}, Epoch:{e:03}, elepsed: {elapsed}, eta: {eta}, '
                lg = lg + f'loss_D: {loss_D:f}, '
                lg = lg + f'loss_D_real: {loss_D_real:f}, loss_D_fake: {loss_D_fake:f}, gp: {gp:f}, '
                lg = lg + f'loss_G: {loss_G:f}, advloss: {advloss:f}, '
                lg = lg + f'l1loss: {l1loss:f}, ploss: {ploss:f}, sloss: {sloss:f}'
                print(lg)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if rank==0:
                if total_step%eval_freq==0:
                    # Validation
                    netG.eval()
                    val_step = 1
                    psnr_fake = 0.0
                    ssim_fake = 0.0
                    lpips_val = 0.0
                    for j, val_data_dict in enumerate(val_loader):
                        P1, P2, map1, map2, P1_path, P2_path = val_data_dict.values()
                        P1, P2, map1, map2 = P1.to(rank), P2.to(rank), map1.to(rank), map2.to(rank)
                        with torch.no_grad():
                            fake_val_logits = netG(P1,map1,map2)
                            fake_vals = fake_val_logits.sigmoid()
                    
                        lpips_val += loss_fn_alex(fake_vals, P2, normalize=True).sum()
                        
                        input_vals = tensor2ndarray(P1)
                        fake_vals = tensor2ndarray(fake_vals)
                        real_vals = tensor2ndarray(P2)
                        
                        for b in range(fake_vals.shape[0]):
                            if total_step%eval_freq==0:
                                os.makedirs(os.path.join(image_out_dir, f'{total_step:06}'), exist_ok=True)
                                psnr_fake += calculate_psnr(
                                    fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=True)
                                ssim_fake += calculate_ssim(
                                    fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=True)
                                
                                # Visualization
                                mp1 = map1[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                                mp1, _ = draw_pose_from_map(mp1)
                                mp2 = map2[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                                mp2, _ = draw_pose_from_map(mp2)

                                input_val = Image.fromarray(input_vals[b,:,:,:])
                                mp1 = Image.fromarray(mp1)
                                fake_val = Image.fromarray(fake_vals[b,:,:,:])
                                mp2 = Image.fromarray(mp2)
                                real_val = Image.fromarray(real_vals[b,:,:,:])
                                img = Image.new('RGB', size=(5*input_val.width, input_val.height), color=0)
                                img.paste(input_val, box=(0, 0))
                                img.paste(mp1, box=(input_val.width, 0))
                                img.paste(fake_val, box=(2*input_val.width, 0))
                                img.paste(mp2, box=(3*input_val.width, 0))
                                img.paste(real_val, box=(4*input_val.width, 0))
                                img.save(os.path.join(image_out_dir, f'{total_step:06}', f'{j:03}_{b:02}.jpg'), 'JPEG')

                            val_step += 1
                            if val_step==opt.val_step: break
                        if val_step==opt.val_step: break

                    psnr_fake = psnr_fake / val_step
                    ssim_fake = ssim_fake / val_step
                    lpips_val = lpips_val / val_step
                    
                    txt = f'PSNR: {psnr_fake:f}, SSIM: {ssim_fake:f}, LPIPS: {lpips_val:f}'
                    print(txt)
                    with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                        fp.write(txt+'\n')
                    with open(f'{log_dir}/test_losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                        fp.write(f'{total_step},{psnr_fake:f},{ssim_fake:f},{lpips_val:f}\n')
                    
                    if total_step%(50*eval_freq)==0 and opt.enable_line_nortify:
                        with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                            token = json.load(fp)['token']
                        send_line_notify(token, f'{opt.name} Step: {total_step}\n{lg}\n{txt}')

                    torch.save({
                        'total_step': total_step,
                        'netG_state_dict': netG.module.state_dict(),
                        'optimG_state_dict': optimG.state_dict(),
                        'PSNR': psnr_fake, 
                        'SSIM': ssim_fake,
                        'LPIPS': lpips_val,
                    }, os.path.join(model_ckpt_dir, f'{model_name}_{total_step:06}.ckpt'))
                        
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
    
    opt = EasyDict(load_option(args.config))
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train, args=(args.config, ), nprocs=opt.n_gpu, join=True)
