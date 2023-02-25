import glob
import os
import random

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import functional as TF
from tqdm import tqdm

from scripts.utils import read_img


class VideoDenoisingDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.n_frames = opt.n_frames
        self.surrounding_frames = opt.n_frames//2
        self.crop_h, self.crop_w = opt.input_resolution
        self.sigma_range = opt.sigma_range
        self.random_flip = opt.random_flip
        self.random_rotate_range = opt.random_rotate_range
        self.rot_interp_mode = torchvision.transforms.InterpolationMode.BILINEAR

        video_dirs = sorted([d for d in glob.glob(os.path.join(opt.dataset_path, f'*'))])
        
        self.imgs = {}
        for video_dir in tqdm(video_dirs):
            name = os.path.basename(video_dir)
            images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
            self.imgs[name] = {}
            for j, img_path in enumerate(images):
                self.imgs[name][j] = read_img(img_path)
        self.video_dirs = sorted(list(self.imgs.keys()))
    
    def change_configs(self, n_frames, input_resolution):
        self.n_frames = n_frames
        self.surrounding_frames = n_frames//2
        self.crop_h, self.crop_w = input_resolution

    def set_crop_position(self, h, w):
        top = random.randint(0, h-self.crop_h-1) if h-self.crop_h-1>=0 else 0
        left = random.randint(0, w-self.crop_w-1) if w-self.crop_w-1>=0 else 0
        return top, left
    
    def __getitem__(self, idx):
        name = self.video_dirs[idx]
        frame_idx = random.randint(self.surrounding_frames, len(self.imgs[name])-self.surrounding_frames)

        temp = self.imgs[name][frame_idx]
        top, left = self.set_crop_position(temp.shape[1], temp.shape[2])
        a_min, a_max = self.random_rotate_range
        angle = random.random() * (a_max-a_min) + a_min
        use_flip = random.random()>0.5 if self.random_flip else False

        sigma = ((random.random()*(self.sigma_range[1]-self.sigma_range[0])) + self.sigma_range[0]) / 255.0
        noise_level = torch.ones((1,1,1)) * sigma
        noise_map = noise_level.expand(1, self.crop_h, self.crop_w)
        
        imgs = []
        gts = []
        for i in range(self.n_frames):
            temp_idx = i + frame_idx-self.surrounding_frames
            temp_idx = min(max(temp_idx, 0), len(self.imgs[name])-1)
            img = self.imgs[name][temp_idx]
            img = TF.crop(img, top, left, self.crop_h, self.crop_w)
            noise = torch.normal(mean=0, std=noise_level.expand_as(img))
            img_n = img + noise
            if use_flip:
                img = TF.hflip(img)
                img_n = TF.hflip(img_n)
            img = TF.rotate(img, angle, self.rot_interp_mode)
            img_n = TF.rotate(img_n, angle, self.rot_interp_mode)
            gts.append(img)
            imgs.append(img_n)

        return {'input_seq': imgs, 'gt_seq': gts, 'noise_map': noise_map}

    def __len__(self):
        return len(self.video_dirs)

class SingleVideoDenoisingDatasetTest(torch.utils.data.Dataset):
    def __init__(self, opt, sigma, max_frames=130, margin_frames=10, return_idx=False):
        super().__init__()
        self.max_frames = max_frames
        self.margin_frames = margin_frames
        self.return_idx = return_idx
        self.sigma = sigma / 255.0
        self.noise_level = torch.ones((1,1,1)) * self.sigma
        
        video_dir = opt.val_dataset_path
        
        self.imgs = []
        self.noise_imgs = []
        images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))

        for i, img_path in enumerate(tqdm(images)):
            img = read_img(img_path)
            noise = torch.normal(mean=0, std=self.noise_level.expand_as(img))
            self.imgs.append(img)
            self.noise_imgs.append(img + noise)
        
        self.noise_map = self.noise_level.expand(1, self.imgs[0].shape[1], self.imgs[0].shape[2])
    
    def __getitem__(self, idx):
        # input_seq: (max_frames+2*margin_frames,C,H,W)
        if len(self.imgs)<=self.max_frames+2*self.margin_frames:
            if self.return_idx:
                forward_idx, start_idx, backward_idx, last_idx = 0, 0, len(self.imgs), len(self.imgs)
                return {'input_seq': self.noise_imgs, 'gt_seq': self.imgs, 'noise_map': self.noise_map, 'idxs': [forward_idx, start_idx, backward_idx, last_idx]}
            else:
                return {'input_seq': self.noise_imgs, 'gt_seq': self.imgs, 'noise_map': self.noise_map}
        else:
            forward_idx = self.max_frames*idx
            backward_idx = min(self.max_frames*(idx+1), len(self.imgs))
            start_idx = max(forward_idx-self.margin_frames, 0)
            last_idx = min(backward_idx+self.margin_frames, len(self.imgs))
            input_seq = self.noise_imgs[start_idx:last_idx]
            gt_seq = self.imgs[start_idx:last_idx]
            if self.return_idx:
                return {'input_seq': input_seq, 'gt_seq': gt_seq, 'noise_map': self.noise_map, 'idxs': [forward_idx, start_idx, backward_idx, last_idx]}
            else:
                return {'input_seq': input_seq, 'gt_seq': gt_seq, 'noise_map': self.noise_map}

    def __len__(self):
        if len(self.noise_imgs)<=self.max_frames+2*self.margin_frames: 
            out = 1
        else:
            out = len(self.imgs)//self.max_frames if len(self.imgs)%self.max_frames==0 else len(self.imgs)//self.max_frames+1
        return out

# Not maintained. It may have bugs.
class SingleVideoDenoisingTestDatasetMIMO(torch.utils.data.Dataset):
    def __init__(self, opt, sigma, stride=None):
        super().__init__()
        # assert opt.n_frames%2==1
        self.n_frames = opt.n_frames
        self.surrounding_frames = opt.n_frames//2
        self.stride = self.n_frames if stride is None else stride
        self.sigma = sigma / 255.0
        self.noise_level = torch.ones((1,1,1)) * self.sigma

        video_dir = opt.val_dataset_path
        
        self.imgs = {}
        images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
        for j, img_path in enumerate(images):
            self.imgs[j] = read_img(img_path)

    def __getitem__(self, idx):
        idx = idx*self.stride + self.surrounding_frames
        idx = min(idx, len(self.imgs)-self.surrounding_frames)
        temp = self.imgs[idx]
        noise_map = self.noise_level.expand(1, temp.shape[1], temp.shape[2])

        gts = []
        imgs = []
        for i in range(self.n_frames):
            temp_idx = i + idx-self.surrounding_frames
            # print(temp_idx)
            temp_idx = min(max(temp_idx, 0), len(self.imgs)-1)
            img = self.imgs[temp_idx]
            noise = torch.normal(mean=0, std=self.noise_level.expand_as(img))
            gts.append(img)
            imgs.append(img + noise)

        return {'input_seq': imgs, 'gt_seq': gts, 'noise_map': noise_map}

    def __len__(self):
        return len(self.imgs) // self.stride + 1