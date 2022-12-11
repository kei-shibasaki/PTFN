import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import argparse
from collections import OrderedDict
import glob
import cv2
import numpy as np
import importlib

from utils.utils import tensor2ndarray, load_option, pad_tensor, convert_state_dict
from dataset import SingleVideoDenoisingTestDataset
from easydict import EasyDict


def generate_images(opt, checkpoint_path, out_dir):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset_paths = sorted([d for d in glob.glob('datasets/DAVIS-test/JPEGImages/480p/*') if os.path.isdir(d)])
    opt.data_extention = 'jpg'
    #dataset_paths = ['datasets/DAVIS-test/JPEGImages/480p/rollercoaster']
    network_module = importlib.import_module('models.network_mimo_multi')
    net = getattr(network_module, opt['model_type_test'])(opt).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(convert_state_dict(checkpoint['netG_state_dict']), strict=True)
    net.eval()

    for idx_dataset, dataset_path in enumerate(dataset_paths):
        dataset_name = os.path.basename(dataset_path)
        print(f'{idx_dataset}/{len(dataset_paths)}: Processing {dataset_name}')
        out_dataset_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(out_dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dataset_dir, 'GT'), exist_ok=True)

        opt.val_dataset_path = dataset_path
        for sigma in [50]:
            os.makedirs(os.path.join(out_dataset_dir, 'input', str(sigma)), exist_ok=True)
            os.makedirs(os.path.join(out_dataset_dir, 'generated', str(sigma)), exist_ok=True)
            os.makedirs(os.path.join(out_dataset_dir, 'generated_inter', str(sigma)), exist_ok=True)
            os.makedirs(os.path.join(out_dataset_dir, 'comparison', str(sigma)), exist_ok=True)

            val_dataset = SingleVideoDenoisingTestDataset(opt, sigma, max_frames=130, margin_frames=10, return_idx=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

            cnt = 0
            for i, data in enumerate(tqdm(val_loader)):
                input_seq = torch.stack(data['input_seq'], dim=1).to(device)
                _,_,_,h_raw,w_raw = input_seq.shape
                gt_seq = torch.stack(data['gt_seq'], dim=1).to(device)
                noise_map = data['noise_map'].unsqueeze(1).to(device)
                input_seq, gt_seq, noise_map = map(lambda x: pad_tensor(x, divisible_by=8), [input_seq, gt_seq, noise_map])
                #b,f,c,h,w = input_seq.shape
                #noise_map = noise_map.repeat(1,f,1,1,1)
                with torch.no_grad():
                    gens, inter_imgs = net(input_seq, noise_map)
                
                offset = data['idxs'][1].numpy()[0]
                frame1 = (data['idxs'][1] if i==0 else data['idxs'][0]).numpy()[0]
                frame2 = (data['idxs'][3] if i==len(val_loader)-1 else data['idxs'][2]).numpy()[0]
                idx1, idx2 = frame1 - offset, frame2 - offset

                imgs, inter_imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, inter_imgs, gens, gt_seq])
                imgs = list(map(lambda x: tensor2ndarray(x)[0,:,:w_raw,:h_raw], imgs[idx1:idx2]))
                inter_imgs = list(map(lambda x: tensor2ndarray(x)[0,:,:w_raw,:h_raw], inter_imgs[idx1:idx2]))
                gens = list(map(lambda x: tensor2ndarray(x)[0,:,:w_raw,:h_raw], gens[idx1:idx2]))
                gts = list(map(lambda x: tensor2ndarray(x)[0,:,:w_raw,:h_raw], gts[idx1:idx2]))

                for _, (img, inter_img, gen, gt) in enumerate(zip(imgs, inter_imgs, gens, gts)):
                    fname = f'{cnt:03}.png'
                    # Visualization
                    img, inter_img, gen, gt = map(lambda x: Image.fromarray(x), [img, inter_img, gen, gt])
                    img.save(os.path.join(out_dataset_dir, 'input', str(sigma), fname), 'PNG')
                    inter_img.save(os.path.join(out_dataset_dir, 'generated_inter', str(sigma), fname), 'PNG')
                    gen.save(os.path.join(out_dataset_dir, 'generated', str(sigma), fname), 'PNG')
                    if sigma==50: gt.save(os.path.join(out_dataset_dir, 'GT', fname), 'PNG')
                    compare_img = Image.new('RGB', size=(4*img.width, img.height), color=0)
                    compare_img.paste(img, box=(0, 0))
                    compare_img.paste(inter_img, box=(img.width, 0))
                    compare_img.paste(gen, box=(2*img.width, 0))
                    compare_img.paste(gt, box=(3*img.width, 0))
                    compare_img.save(os.path.join(out_dataset_dir, 'comparison', str(sigma), fname), 'PNG')

                    cnt += 1

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