import argparse
import datetime
import importlib
import json
import os
import shutil
import time

import torch
import torch.utils
from easydict import EasyDict
from PIL import Image

from datasets.dataset import VideoDenoisingDatasetTrain, SingleVideoDenoisingDatasetTest
from scripts.losses import PSNRLoss
from scripts.metrics import calculate_psnr, calculate_ssim
from scripts.utils import convert_state_dict, load_option, pad_tensor, send_line_notify, tensor2ndarray, arrange_images, convert_state_dict_to_half


def train(opt_path):
    opt = EasyDict(load_option(opt_path))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    model_ckpt_dir = f'./experiments/{opt.name}/ckpt'
    image_out_dir = f'./experiments/{opt.name}/generated'
    log_dir = f'./experiments/{opt.name}/logs'
    log_path = f'{log_dir}/log_{opt.name}.log'
    log_train_losses_path = f'{log_dir}/train_losses_{opt.name}.csv'
    log_test_losses_paths = {sigma: f'{log_dir}/test_losses_{opt.name}_{sigma}.csv' for sigma in opt.sigmas_for_eval}

    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(log_path, mode='w', encoding='utf-8') as fp: fp.write('')
    with open(log_train_losses_path, mode='w', encoding='utf-8') as fp: fp.write('step,lr,loss_G\n')
    for sigma in opt.sigmas_for_eval:
        with open(log_test_losses_paths[sigma], mode='w', encoding='utf-8') as fp:
            fp.write('step,loss_G,psnr,ssim\n')
    
    shutil.copy(opt_path, f'./experiments/{opt.name}/{os.path.basename(opt_path)}')
    
    loss_fn = PSNRLoss().to(device)
    network_module = importlib.import_module('models.network')
    netG = getattr(network_module, opt['model_type_train'])(opt).to(device)
    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, T_max=opt.T_max, eta_min=opt.eta_min)
    if opt.resume_step is not None:
        for _ in range(opt.resume_step): schedulerG.step()

    if opt.pretrained_path:
        state_dict = torch.load(opt.pretrained_path, map_location=device)
        netG_state_dict = convert_state_dict_to_half(state_dict['netG_state_dict'])
        netG.load_state_dict(netG_state_dict, strict=True)

    netG_val = getattr(network_module, opt['model_type_test'])(opt).to(device)
    
    train_dataset = VideoDenoisingDatasetTrain(opt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loaders = {}
    for sigma in opt.sigmas_for_eval:
        val_dataset = SingleVideoDenoisingDatasetTest(opt, sigma)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        val_loaders[sigma] = val_loader

    print('Start Training')
    start_time = time.time()
    total_step = 0 if opt.resume_step is None else opt.resume_step
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
            gens = netG(input_seq, noise_map)
            loss_G = loss_fn(gens.reshape(b*f,c,h,w), gt_seq.reshape(b*f,c,h,w))

            loss_G.backward()
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.grad_clip_val)
            optimG.step()
            schedulerG.step()
            
            total_step += 1

            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                with open(log_train_losses_path, mode='a', encoding='utf-8') as fp:
                    fp.write(f'{total_step},{lr_G[0]},{loss_G:f}\n')
            
            if total_step%opt.print_freq==0 or total_step==1:
                rest_step = opt.steps-total_step 
                time_per_step = int(time.time()-start_time) / total_step if opt.resume_step is None else int(time.time()-start_time) / (total_step-opt.resume_step)
                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.steps}, Epoch:{str(e).zfill(len(str(opt.steps)))}, elepsed: {elapsed}, eta: {eta}, '
                lg = lg + f'loss_G: {loss_G:f}'
                print(lg)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')

            if total_step%opt.eval_freq==0:
                # Save Train images
                # (B,F,C,H,W) -> [(B,C,H,W)]*F
                imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, gens, gt_seq])
                imgs, gens, gts = map(lambda arr: list(map(lambda x: tensor2ndarray(x)[0,:,:,:], arr)), [imgs, gens, gts])
                os.makedirs(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}'), exist_ok=True)
                for j, (img, gen, gt) in enumerate(zip(imgs, gens, gts)):
                    # Visualization
                    img, gen, gt = map(lambda x: Image.fromarray(x), [img, gen, gt])
                    compare_img = arrange_images([img, gen, gt])
                    compare_img.save(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}', f'train_{i:03}_{j:03}.png'), 'PNG')
                
                # Validation
                netG_val.load_state_dict(convert_state_dict(netG.state_dict()), strict=True)
                netG_val.eval()
                for sigma in opt.sigmas_for_eval:
                    val_loader = val_loaders[sigma]
                    psnr1, psnr2, ssim1, ssim2 = 0.0, 0.0, 0.0, 0.0
                    for i, data in enumerate(val_loader):
                        # [(B,C,H,W)]*F -> (B,F,C,H,W)
                        input_seq = torch.cat(data['input_seq'], dim=0).unsqueeze(0).to(device)
                        gt_seq = torch.cat(data['gt_seq'], dim=0).unsqueeze(0).to(device)
                        noise_map = data['noise_map'].unsqueeze(1).to(device)
                        input_seq, gt_seq, noise_map = map(lambda x: pad_tensor(x, divisible_by=8), [input_seq, gt_seq, noise_map])
                        b,f,c,h,w = input_seq.shape
                        with torch.no_grad():
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                gens = netG_val(input_seq, noise_map)
                                loss_G = loss_fn(gens.reshape(b*f,c,h,w), gt_seq.reshape(b*f,c,h,w))
                        
                        # (B,F,C,H,W) -> [(B,C,H,W)]*F
                        imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, gens, gt_seq])
                        imgs, gens, gts = map(lambda arr: list(map(lambda x: tensor2ndarray(x)[0,:,:,:], arr)), [imgs, gens, gts])

                        os.makedirs(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}'), exist_ok=True)
                        for j, (img, gen, gt) in enumerate(zip(imgs, gens, gts)):
                            psnr2 += calculate_psnr(gen, gt, crop_border=0, test_y_channel=False)
                            ssim2 += calculate_ssim(gen, gt, crop_border=0, test_y_channel=False)
                            
                            # Visualization
                            img, gen, gt = map(lambda x: Image.fromarray(x), [img, gen, gt])
                            compare_img = arrange_images([img, gen, gt])
                            compare_img.save(os.path.join(image_out_dir, str(sigma), f'{str(total_step).zfill(len(str(opt.steps)))}', f'{i:03}_{j:03}.png'), 'PNG')
                    
                    psnr2, ssim2 = psnr2/f, ssim2/f
                
                    txt = f'sigma: {sigma}, loss_G: {loss_G:f}, '
                    txt = txt + f'PSNR: {psnr2:f}, SSIM: {ssim2:f}'
                    print(txt)
                    with open(log_path, mode='a', encoding='utf-8') as fp:
                        fp.write(txt+'\n')
                    with open(log_test_losses_paths[sigma], mode='a', encoding='utf-8') as fp:
                        fp.write(f'{total_step},{loss_G:f},{psnr2:f},{ssim2:f}\n')
                    
                    if psnr2 >= best_psnr:
                        best_psnr = psnr2
                        torch.save({
                            'total_step': total_step,
                            'netG_state_dict': netG.state_dict(),
                            'optimG_state_dict': optimG.state_dict(),
                        }, os.path.join(model_ckpt_dir, f'{opt.name}_best.ckpt'))
                
                if total_step%opt.save_freq==0 and opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'{opt.name} Step: {total_step}\n{lg}\n{txt}')
            
            if total_step%opt.save_freq==0:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))
                    
            if total_step==opt.steps:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))

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
