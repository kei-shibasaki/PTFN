import json
import os
import numpy as np
import torch
from torch.nn import functional as F
import requests
import cv2
import glob
import math

def load_option(opt_path):
    with open(opt_path, 'r') as json_file:
        json_obj = json.load(json_file)
        return json_obj

def tensor2ndarray3d(tensor):
    # Pytorch Tensor (B, C, D, H, W), [0, 1] -> ndarray (B, D, H, W, C) [0, 255]
    img = tensor.detach()
    img = img.cpu().permute(0,2,3,4,1).numpy()
    img = np.clip(img, a_min=0, a_max=1.0)
    img = (img*255).astype(np.uint8)
    return img

def pad_tensor(x, divisible_by=8, mode='reflect'):
    if len(x.shape)==5:
        b,f,c,h,w = x.shape
        x = x.reshape(b*f,c,h,w)
    else:
        f = None
        _,_,h,w = x.shape
    
    nh = h//divisible_by+1 if h%divisible_by!=0 else h//divisible_by
    nw = w//divisible_by+1 if w%divisible_by!=0 else w//divisible_by
    nh, nw = int(nh*divisible_by), int(nw*divisible_by)
    pad_h, pad_w = nh-h, nw-w

    x = F.pad(x, [0,pad_w,0,pad_h], mode)

    if f is not None:
        x = x.reshape(b,f,c,nh,nw)

    return x


def read_img(path):
    """Read a sequence of images from a given folder path.
    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.
    Returns:
        Tensor: size (c, h, w), RGB, [0, 1].
    """
    img = cv2.imread(path).astype(np.float32) / 255.
    img = img2tensor(img, bgr2rgb=True, float32=True)

    return img

def read_img_seq(path):
    """Read a sequence of images from a given folder path.
    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.
    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
    """
    img_paths = path
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    return imgs

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2ndarray(tensor):
    # Pytorch Tensor (B, C, H, W), [0, 1] -> ndarray (B, H, W, C) [0, 255]
    img = tensor.detach()
    img = img.cpu().permute(0,2,3,1).numpy()
    img = np.clip(img, a_min=0, a_max=1.0)
    img = (img*255).astype(np.uint8)
    return img

def send_line_notify(line_notify_token, nortification_message):
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'{nortification_message}'}
    requests.post(line_notify_api, headers=headers, data=data)