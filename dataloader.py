import os

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision
import numpy as np
import json
import glob
import os
from PIL import Image
import random
import io
import numpy as np
import glob
from tqdm import tqdm
from utils.utils import read_img

class DAVISVideoDenoisingTrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        assert opt.n_frames==5
        self.n_frames = opt.n_frames
        self.surrounding_frames = opt.n_frames//2
        self.crop_h, self.crop_w = opt.input_resolution
        self.sigma_range = opt.sigma_range

        video_dirs = sorted([d for d in glob.glob(os.path.join(opt.dataset_path, f'*'))])
        self.cache_data = opt.cache_data
        
        self.imgs = {}
        for video_dir in tqdm(video_dirs):
            name = os.path.basename(video_dir)
            images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
            self.imgs[name] = {}
            for j, img_path in enumerate(images):
                if self.cache_data:
                    self.imgs[name][j] = read_img(img_path)
                else:
                    self.imgs[name][j] = img_path
        self.video_dirs = sorted(list(self.imgs.keys()))
    
    def set_crop_position(self, h, w):
        top = random.randint(0, h-self.crop_h-1)
        left = random.randint(0, w-self.crop_w-1)
        return top, left
    
    def __getitem__(self, idx):
        name = self.video_dirs[idx]
        frame_idx = random.randint(0, len(self.imgs[name])-1)

        gt = self.imgs[name][frame_idx]
        top, left = self.set_crop_position(gt.shape[1], gt.shape[2])
        gt = TF.crop(gt, top, left, self.crop_h, self.crop_w)

        sigma = ((random.random()*(self.sigma_range[1]-self.sigma_range[0])) + self.sigma_range[0]) / 255.0
        noise_level = torch.ones((1,1,1)) * sigma
        noise_map = noise_level.expand(1, self.crop_h, self.crop_w)
        
        imgs = []
        for i in range(self.n_frames):
            temp_idx = i + frame_idx-self.surrounding_frames
            temp_idx = min(max(temp_idx, 0), len(self.imgs[name])-1)
            if self.cache_data:
                img = self.imgs[name][temp_idx]
            else:
                img = read_img(self.imgs[name][temp_idx])
            img = TF.crop(img, top, left, self.crop_h, self.crop_w)
            noise = torch.normal(mean=0, std=noise_level.expand_as(img))
            imgs.append(img+noise)

        return imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], noise_map, gt, noise_level.flatten()

    def __len__(self):
        return len(self.video_dirs)

class DAVISVideoDenoisingTrainDatasetMIMO(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.n_frames = opt.n_frames
        self.surrounding_frames = opt.n_frames//2
        self.crop_h, self.crop_w = opt.input_resolution
        self.sigma_range = opt.sigma_range

        video_dirs = sorted([d for d in glob.glob(os.path.join(opt.dataset_path, f'*'))])
        self.cache_data = opt.cache_data
        
        self.imgs = {}
        for video_dir in tqdm(video_dirs):
            name = os.path.basename(video_dir)
            images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
            self.imgs[name] = {}
            for j, img_path in enumerate(images):
                if self.cache_data:
                    self.imgs[name][j] = read_img(img_path)
                else:
                    self.imgs[name][j] = img_path
        self.video_dirs = sorted(list(self.imgs.keys()))
    
    def set_crop_position(self, h, w):
        top = random.randint(0, h-self.crop_h-1)
        left = random.randint(0, w-self.crop_w-1)
        return top, left
    
    def __getitem__(self, idx):
        name = self.video_dirs[idx]
        frame_idx = random.randint(self.surrounding_frames, len(self.imgs[name])-self.surrounding_frames)

        temp = self.imgs[name][frame_idx]
        top, left = self.set_crop_position(temp.shape[1], temp.shape[2])

        sigma = ((random.random()*(self.sigma_range[1]-self.sigma_range[0])) + self.sigma_range[0]) / 255.0
        noise_level = torch.ones((1,1,1)) * sigma
        noise_map = noise_level.expand(1, self.crop_h, self.crop_w)
        
        imgs = []
        gts = []
        for i in range(self.n_frames):
            temp_idx = i + frame_idx-self.surrounding_frames
            temp_idx = min(max(temp_idx, 0), len(self.imgs[name])-1)
            # print(name, frame_idx, temp_idx)
            if self.cache_data:
                img = self.imgs[name][temp_idx]
            else:
                img = read_img(self.imgs[name][temp_idx])
            img = TF.crop(img, top, left, self.crop_h, self.crop_w)
            noise = torch.normal(mean=0, std=noise_level.expand_as(img))
            gts.append(img)
            imgs.append(img+noise)

        #return imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], noise_map
        return {'input_seq': imgs, 'gt_seq': gts, 'noise_map': noise_map}

    def __len__(self):
        return len(self.video_dirs)


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
        self.cache_data = opt.cache_data
        
        self.imgs = {}
        images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
        for j, img_path in enumerate(images):
            if self.cache_data:
                self.imgs[j] = read_img(img_path)
            else:
                self.imgs[j] = img_path
    
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
            if self.cache_data:
                img = self.imgs[temp_idx]
            else:
                img = read_img(self.imgs[temp_idx])
            noise = torch.normal(mean=0, std=self.noise_level.expand_as(img))
            gts.append(img)
            imgs.append(img+noise)

        return {'input_seq': imgs, 'gt_seq': gts, 'noise_map': noise_map}

    def __len__(self):
        return len(self.imgs) // self.stride + 1

class SingleVideoDenoisingTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, sigma):
        super().__init__()
        assert opt.n_frames%2==1
        self.n_frames = opt.n_frames
        self.surrounding_frames = opt.n_frames//2
        self.sigma = sigma / 255.0
        self.noise_level = torch.ones((1,1,1)) * self.sigma

        video_dir = opt.val_dataset_path
        self.cache_data = opt.cache_data
        
        self.imgs = {}
        images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
        for j, img_path in enumerate(images):
            if self.cache_data:
                self.imgs[j] = read_img(img_path)
            else:
                self.imgs[j] = img_path
    
    def __getitem__(self, idx):
        gt = self.imgs[idx]
        noise_map = self.noise_level.expand(1, gt.shape[1], gt.shape[2])

        imgs = []
        for i in range(self.n_frames):
            temp_idx = i + idx-self.surrounding_frames
            temp_idx = min(max(temp_idx, 0), len(self.imgs)-1)
            if self.cache_data:
                img = self.imgs[temp_idx]
            else:
                img = read_img(self.imgs[temp_idx])
            noise = torch.normal(mean=0, std=self.noise_level.expand_as(img))
            imgs.append(img+noise)

        return imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], noise_map, gt, self.noise_level.flatten()

    def __len__(self):
        return len(self.imgs)


class SingleVideoDenoisingTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, sigma):
        super().__init__()
        self.n_frames = opt.n_frames
        self.sigma = sigma / 255.0
        self.noise_level = torch.ones((1,1,1)) * self.sigma
        
        video_dir = opt.val_dataset_path
        
        self.imgs = []
        self.noise_imgs = []
        images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))

        for i, img_path in enumerate(images):
            img = read_img(img_path)
            noise = torch.normal(mean=0, std=self.noise_level.expand_as(img))
            self.imgs.append(img)
            self.noise_imgs.append(img+noise)
        
        self.noise_map = self.noise_level.expand(1, self.imgs[0].shape[1], self.imgs[0].shape[2])
    
    def __getitem__(self, idx):
        return {'input_seq': self.noise_imgs, 'gt_seq': self.imgs, 'noise_map': self.noise_map}

    def __len__(self):
        return 1