import glob
import json
from easydict import EasyDict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchinfo
import time
from PIL import Image
import os
from tqdm import tqdm
from scripts.metrics import calculate_psnr, calculate_ssim
import cv2
import time

from scripts.utils import pad_tensor, tensor2ndarray, read_img


def check_block():
    from models.network import PseudoTemporalFusionNetwork, PseudoTemporalFusionNetworkEval
    import thop

    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    device = torch.device('cuda')
    b,f,c,h,w = 2,2,3,480,480
    opt.n_frames = f
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)
    net = PseudoTemporalFusionNetwork(opt).to(device)
    out1, out2 = net(x, noise_map)
    print(out1.shape, out2.shape)
    torchinfo.summary(net, input_data=[x, noise_map])
    #macs, params = thop.profile(net, inputs=[x, noise_map])
    #print(f'GMACs: {macs/1e9/f:f}, #Params: {params/1e6:f}')

def check_block2():
    from models.network_blind import PseudoTemporalFusionNetwork, PseudoTemporalFusionNetworkEval, PseudoTemporalFusionNetworkEvalHalf
    from models.network_blind import PseudoTemporalFusionNetworkL, PseudoTemporalFusionNetworkEvalL, PseudoTemporalFusionNetworkEvalHalfL
    #from UNUSED.wnet_bsvd import BSVD
    import thop

    with open('config/config_ptfn.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    device = torch.device('cuda')
    #b,f,c,h,w = 8,opt.n_frames,3,96,96
    b,f,c,h,w = 1,50,3,480,856
    #b,f,c,h,w = 1,50,3,720,1280

    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)
    net = PseudoTemporalFusionNetworkEvalL(opt).to(device)
    #net = PseudoTemporalFusionNetworkEvalHalf(opt).to(device)
    #net = PseudoTemporalFusionNetwork(opt).to(device)
    #net = BSVD(pretrain_ckpt=None).to(device)
    out1, out2 = net(x)
    print(out1.shape, out2.shape)
    #torchinfo.summary(net, input_data=[x, noise_map])
    #macs, params = thop.profile(net, inputs=(x))
    #print(f'GMACs: {macs/1e9/50:f}')
    #print(f'#Params (M): {params/1e6:f}')


def yoyaku():
    device = torch.device('cuda')
    x = torch.rand((100000,60000), device=device)
    input('Press Enter...')

def check_vis():
    device = torch.device('cuda')
    with open('config/config_unet.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    sigma = 10
    pyramid = LaplacianPyramid(opt.level)
    train_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for i, (imgs, gt, noise_level) in enumerate(train_loader):
        img = torch.chunk(imgs[:,:-1,:,:], chunks=opt.n_frames, dim=1)[2]
        img, gt = img.to(device), gt.to(device)
        temp_img = tensor2ndarray(img)[0,:,:,:]
        temp_gt = tensor2ndarray(gt)[0,:,:,:]
        print(calculate_psnr(temp_img, temp_gt, crop_border=0, test_y_channel=False))

        pyr_img = pyramid.pyramid_decom(img)
        pyr_gt = pyramid.pyramid_decom(gt)
        target_size = [img.shape[2], img.shape[3]]
        for i, (p_img, p_gt) in enumerate(zip(pyr_img, pyr_gt)):
            mse = F.mse_loss(p_img, p_gt)
            print(mse.cpu().numpy())
            """
            p_img = F.interpolate(p_img, size=target_size, mode='nearest')
            p_gt = F.interpolate(p_gt, size=target_size, mode='nearest')
            p_img = tensor2ndarray(p_img)
            p_gt = tensor2ndarray(p_gt)
            
            img = Image.fromarray(p_img[0,:,:,:])
            gt = Image.fromarray(p_gt[0,:,:,:])
            compare_img = Image.new('RGB', size=(2*img.width, img.height), color=0)
            compare_img.paste(img, box=(0, 0))
            compare_img.paste(gt, box=(img.width, 0))
            compare_img.save(os.path.join('temp', f'{sigma:02}_{i:02}.png'), 'PNG')
            """

        exit()

def check_vis2():
    device = torch.device('cuda')
    with open('config/config_test2.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    sigma = 50
    pixel_unshufle = nn.PixelUnshuffle(2)
    train_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for i, (imgs, gt, noise_level) in enumerate(train_loader):
        img = torch.chunk(imgs[:,:-1,:,:], chunks=opt.n_frames, dim=1)[2]
        
        img, gt = img.to(device), gt.to(device)
        # target_size = [img.shape[2], img.shape[3]]
        order_idx = [0,4,8,1,5,9,2,6,10,3,7,11]
        img = pixel_unshufle(img)
        img = img[:, order_idx, :, :]
        gt = pixel_unshufle(gt)
        gt = gt[:, order_idx, :, :]
        imgs = torch.chunk(img, chunks=4, dim=1)
        gts = torch.chunk(gt, chunks=4, dim=1)
        for i, (img, gt) in enumerate(zip(imgs, gts)):
            img = tensor2ndarray(img)[0,:,:,:]
            gt = tensor2ndarray(gt)[0,:,:,:]

            # img = F.interpolate(img, size=target_size, mode='nearest')
            # gt = F.interpolate(gt, size=target_size, mode='nearest')

            img = Image.fromarray(img)
            gt = Image.fromarray(gt)
            img.save(os.path.join('temp', f'input_{sigma:02}_{i:02}.png'), 'PNG')
            gt.save(os.path.join('temp', f'gt_{sigma:02}_{i:02}.png'), 'PNG')
            compare_img = Image.new('RGB', size=(2*img.width, img.height), color=0)
            compare_img.paste(img, box=(0, 0))
            compare_img.paste(gt, box=(img.width, 0))
            compare_img.save(os.path.join('temp', f'compare_{sigma:02}_{i:02}.png'), 'PNG')

        exit()

def check_vis3():
    device = torch.device('cuda')
    with open('config/config_test2.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    sigma = 50
    pixel_unshufle = nn.PixelUnshuffle(2)
    train_dataset = SingleVideoDenoisingTestDataset(opt, sigma)
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for i, (imgs, gt, noise_level) in enumerate(train_loader):
        img = torch.chunk(imgs[:,:-1,:,:], chunks=opt.n_frames, dim=1)[2]
        img, _ = pad_tensor(img, divisible_by=32)
        img = tensor2ndarray(img)[0,:,:,:]
        img = Image.fromarray(img)
        img.save(os.path.join('temp', f'input_{sigma:02}.png'), 'PNG')
        exit()

def pixel_unshuffle_from_img():
    img_path = 'temp/input_50_00_00_00_00.png'
    name_base = os.path.splitext(os.path.basename(img_path))[0]
    img = read_img(img_path).unsqueeze(0)

    pixel_unshufle = nn.PixelUnshuffle(2)
    order_idx = [0,4,8,1,5,9,2,6,10,3,7,11]
    img = pixel_unshufle(img)
    img = img[:, order_idx, :, :]
    imgs = torch.chunk(img, chunks=4, dim=1)
    for i, img in enumerate(imgs):
        img = tensor2ndarray(img)[0,:,:,:]

        # img = F.interpolate(img, size=target_size, mode='nearest')
        # gt = F.interpolate(gt, size=target_size, mode='nearest')

        img = Image.fromarray(img)
        img.save(os.path.join('temp', f'{name_base}_{i:02}.png'), 'PNG')

def check_wiener():
    import scipy
    from scipy.signal import wiener
    import cv2
    import numpy as np

    img_path = f'temp/input_50.png'
    img = cv2.imread(img_path) / 255.0
    print(img.min(), img.max())
    out = wiener(img, (5,5,1))
    print(out.min(), out.max())
    out = np.clip(out*255, 0, 255).astype(np.uint8)
    cv2.imwrite('temp.png', out)

def check_img():
    import numpy as np

    with open('temp.csv', 'w', encoding='utf-8') as fp:
        fp.write('psnr,ssim\n')

    img_dirs = sorted(glob.glob('./results/fastdvd_level5/rollercoaster_*/generated/50'))
    gt_dirs = sorted(glob.glob('./results/fastdvd_level5/rollercoaster_*/GT'))
    for img_dir, gt_dir in zip(tqdm(img_dirs), gt_dirs):
        images_gen = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        images_gt = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

        assert len(images_gen)==len(images_gt)
        n_images = len(images_gen)

        psnr, ssim = 0.0, 0.0
        for imgpath_gen, imgpath_gt in zip(images_gen, images_gt):
            with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                img_gen = np.array(img_gen)
                img_gt = np.array(img_gt)

                psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images

        with open('temp.csv', 'a', encoding='utf-8') as fp:
            fp.write(f'{psnr:f},{ssim:f}\n')

def check_img2():
    import numpy as np

    with open('ttemp.csv', 'w', encoding='utf-8') as fp:
        fp.write('psnr,ssim\n')

    img_dirs = [
        'results/_fastdvd_level5/rollercoaster/generated/50',
        'results/fastdvd_level5_1/rollercoaster/generated/50',
        'results/fastdvd_level5_2/rollercoaster/generated/50',
        'results/fastdvd_level5_3/rollercoaster/generated/50',
    ]
    gt_dirs = [
        'results/_fastdvd_level5/rollercoaster/GT',
        'results/fastdvd_level5_1/rollercoaster/GT',
        'results/fastdvd_level5_2/rollercoaster/GT',
        'results/fastdvd_level5_3/rollercoaster/GT',
    ]
    gt_dirs = [
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
    ]
    for img_dir, gt_dir in zip(tqdm(img_dirs), gt_dirs):
        images_gen = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        images_gt = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))

        assert len(images_gen)==len(images_gt), f'{len(images_gen)}, {len(images_gt)}'
        n_images = len(images_gen)

        psnr, ssim = 0.0, 0.0
        for imgpath_gen, imgpath_gt in zip(images_gen, images_gt):
            with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                img_gen = np.array(img_gen)
                img_gt = np.array(img_gt)

                psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images

        with open('ttemp.csv', 'a', encoding='utf-8') as fp:
            fp.write(f'{psnr:f},{ssim:f}\n')


def check_gif():
    from PIL import Image
    import glob
    img_paths1 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/input/50/*.png'))
    img_paths2 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/generated/50/*.png'))
    print(len(img_paths1), len(img_paths2))

    images = []
    for i, (img_path1, img_path2) in enumerate(zip(img_paths1, img_paths2)):
        with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            w, h = img1.size
            img1, img2 = img1.crop([0,0,w//2,h]), img2.crop([w//2,0,w,h])
            img = Image.new('RGB', size=[w,h], color=0)
            img.paste(img1, [0,0])
            img.paste(img2, [w//2,0])
            if i==0: 
                first_img = img
                continue
            images.append(img)
    first_img.save('temp.gif', save_all=True, append_images=images)
    
def check_mp4():
    import cv2
    import glob
    img_paths = sorted(glob.glob('results/fastdvd_level5/rollercoaster/generated/50/*.png'))

    images = [cv2.imread(img_path) for img_path in img_paths]
    video_size = [images[0].shape[1], images[0].shape[0]]
    frame_rate = 24
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('temp.mp4', fmt, frame_rate, video_size)
    
    for img in images:
        writer.write(img)
    
    writer.release()
    cv2.destroyAllWindows()

def check_compare_video_mp4():
    import cv2
    import glob
    import numpy as np

    img_paths1 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/input/50/*.png'))
    images1 = [cv2.imread(img_path) for img_path in img_paths1]
    img_paths2 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/GT/*.png'))
    images2 = [cv2.imread(img_path) for img_path in img_paths2]
    img_paths3 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/generated/50/*.png'))
    images3 = [cv2.imread(img_path) for img_path in img_paths3]
    
    video_size = [3*images1[0].shape[1], images1[0].shape[0]]
    frame_rate = 24
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('temp/temp.mp4', fmt, frame_rate, video_size)
    
    for img1, img2, img3 in zip(images1, images2, images3):
        writer.write(np.hstack([img1, img2, img3]))
    
    writer.release()
    cv2.destroyAllWindows()

def check_compare_video_gif():
    import cv2
    import glob
    import numpy as np

    img_paths1 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/input/50/*.png'))
    images1 = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths1]
    img_paths2 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/generated/50/*.png'))
    images2 = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths2]
    img_paths3 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/GT/*.png'))
    images3 = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths3]
    
    frame_rate = 24
    images = [np.hstack([img1, img2, img3]) for img1, img2, img3 in zip(images1, images2, images3)]
    images = [Image.fromarray(img) for img in images]
    first_img, images = images[0], images[1:]

    first_img.save('temp.gif', save_all=True, append_images=images)

def check_boxplot():
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('temp.csv')
    vmin, vmax = df['psnr'].min(), df['psnr'].max()
    vavg = df['psnr'].mean()
    plt.figure(figsize=None)
    plt.boxplot(df['psnr'])
    plt.grid()
    plt.title(f'min: {vmin:f}, max: {vmax:f}, avg: {vavg:f}')
    plt.savefig('temp.png')

def check_gt():
    import numpy as np

    with open('tttemp.csv', 'w', encoding='utf-8') as fp:
        fp.write('psnr,ssim\n')

    img_dirs = [
        'results/_fastdvd_level5/rollercoaster/GT',
        'results/fastdvd_level5_1/rollercoaster/GT',
        'results/fastdvd_level5_2/rollercoaster/GT',
        'results/fastdvd_level5_3/rollercoaster/GT',
    ]
    gt_dirs = [
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
        'datasets/DAVIS-test/JPEGImages/480p/rollercoaster',
    ]
    for img_dir, gt_dir in zip(tqdm(img_dirs), gt_dirs):
        images_gen = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        images_gt = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))

        assert len(images_gen)==len(images_gt), f'{len(images_gen)}, {len(images_gt)}'
        n_images = len(images_gen)

        psnr, ssim = 0.0, 0.0
        for imgpath_gen, imgpath_gt in zip(images_gen, images_gt):
            with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                img_gen = np.array(img_gen)
                img_gt = np.array(img_gt)

                psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images
                ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=False) / n_images

        with open('tttemp.csv', 'a', encoding='utf-8') as fp:
            fp.write(f'{psnr:f},{ssim:f}\n')

def check_wiener():
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    device = torch.device('cuda')
    layer = WienerFilter(kernel_size=9, device=device)

    dataset = SingleVideoDenoisingTestDataset(opt, sigma=50)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for step, (x0, x1, x2, x3, x4, noise_map, gt, noise_level) in enumerate(dataloader):
        x0, x1, x2, x3, x4, noise_map, gt = map(lambda x: x.to(device), [x0, x1, x2, x3, x4, noise_map, gt])
        normed_sigma = noise_map[:,0,0,0].reshape(-1,1,1,1)
        x0, x1, x2, x3, x4 = map(lambda x: layer(x, noise_power=normed_sigma**2), [x0, x1, x2, x3, x4])

        # Visualization
        x0, x1, x2, x3, x4, gt = map(lambda x: tensor2ndarray(x), [x0,x1,x2,x3,x4,gt])
        x0, x1, x2, x3, x4, gt = map(lambda x: Image.fromarray(x[0,:,:,:]), [x0,x1,x2,x3,x4,gt])

        x0.save('temp/x0.png')
        x1.save('temp/x1.png')
        x2.save('temp/x2.png')
        x3.save('temp/x3.png')
        x4.save('temp/x4.png')
        gt.save('temp/gt.png')

        exit()

def compare_images():
    import cv2
    import glob
    import numpy as np

    temp = cv2.imread('experiments/fastdvd_b8/generated/50/400000/000.png')
    H, W, C = temp.shape
    W = W//3

    img_paths1 = sorted(glob.glob('results/wiener/rollercoaster/generated/50/*.png'))
    images1 = [cv2.imread(img_path) for img_path in img_paths1]
    images1 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images1]

    img_paths2 = sorted(glob.glob('results/fastdvd_b8/rollercoaster/generated/50/*.png'))
    images2 = [cv2.imread(img_path) for img_path in img_paths2]
    images2 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images2]

    img_paths3 = sorted(glob.glob('results/fastdvd_b8/rollercoaster/GT/*.png'))
    images3 = [cv2.imread(img_path) for img_path in img_paths3]
    images3 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images3]

    images1 = [np.abs(img3-img1) for img1, img3 in zip(images1, images3)]
    images2 = [np.abs(img3-img2) for img2, img3 in zip(images2, images3)]

    temp1, temp2 = [], []
    for img1, img2 in zip(images1, images2):
        temp1.append(img1.mean())
        temp2.append(img2.mean())
    print(sum(temp1)/len(temp1), sum(temp2)/len(temp2))
    
    #images = [np.hstack([img1, img2, img3]) for img1, img2, img3 in zip(images1, images2, images3)]
    images = [np.hstack([img1, img2]) for img1, img2 in zip(images1, images2)]
    
    for i, img in enumerate(tqdm(images)):
        cv2.imwrite(os.path.join('temp', f'{i:03}.png'), img)

def check_disk():
    from utils.utils import read_img
    import random

    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))

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
            for i, video_dir in enumerate(video_dirs):
                start_total = time.time()
                name = os.path.basename(video_dir)
                images = sorted(glob.glob(os.path.join(video_dir, f'*.{opt.data_extention}')))
                self.imgs[name] = {}
                for j, img_path in enumerate(images):
                    start = time.time()
                    if self.cache_data:
                        self.imgs[name][j] = read_img(img_path)
                    else:
                        self.imgs[name][j] = img_path
                    elapsed = time.time() - start 
                    print(f'{i},{j},{elapsed}')
                    with open('temp/temp11.csv', mode='a', encoding='utf-8') as fp:
                        fp.write(f'{i},{j},{elapsed}\n')
                elapsed_total = time.time() - start_total
                #with open('temp/temp.csv', mode='a', encoding='utf-8') as fp:
                #    fp.write(f'{i},total,{elapsed_total}\n')
                #print(f'{i},total,{elapsed_total}')
            self.video_dirs = sorted(list(self.imgs.keys()))
        
        def set_crop_position(self, h, w):
            top = random.randint(0, h-self.crop_h-1)
            left = random.randint(0, w-self.crop_w-1)
            return top, left
        
        def __getitem__(self, idx):
            name = self.video_dirs[idx]
            frame_idx = random.randint(0, len(self.video_dirs[idx])-1)

            gt = self.imgs[name][frame_idx]
            top, left = self.set_crop_position(gt.shape[1], gt.shape[2])
            gt = TF.crop(gt, top, left, self.crop_h, self.crop_w)

            sigma = ((random.random()*(self.sigma_range[1]-self.sigma_range[0])) + self.sigma_range[0]) / 255.0
            noise_level = torch.ones((1,1,1)) * sigma
            noise_map = noise_level.expand(1, self.crop_h, self.crop_w)
            
            imgs = []
            for i in range(self.n_frames):
                temp_idx = i + frame_idx-self.surrounding_frames
                temp_idx = min(max(temp_idx, 0), len(self.video_dirs[idx])-1)
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
    
    dataset = DAVISVideoDenoisingTrainDataset(opt)
    
def check_cpu():
    images = []
    for i in range(6000):
        start = time.time()
        x = (torch.rand((3,5,480,720))*255).to(torch.uint8)
        images.append(x)
        elapsed = time.time() - start
        print(f'{i},{elapsed}')
        with open('temp/temp.csv', mode='a', encoding='utf-8') as fp:
            fp.write(f'{i},{elapsed}\n')

def check_importlib():
    import importlib

    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    module = importlib.import_module('models.network_mimo')
    print(module)

    net = getattr(module, opt['model_type'])(opt)

    print(net)

def check_tsm():
    from models.network_mimo3 import NAFTSM
    import torchinfo
    device = torch.device('cuda')
    with open('config/config_test3.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    opt.batch_size = 2
    opt.n_frames = 11
    res = 96
    netG = NAFTSM(opt).to(device)

    b,f,c,h,w = opt.batch_size,opt.n_frames,3,res,res
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)

    out = netG(x, noise_map)

    print(out.shape)
    #print(netG)

    #torchinfo.summary(netG, input_data=[x, noise_map])


def check_bbb():
    from models.network_mimo2 import NAFBBB
    import torchinfo
    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    layer = NAFBBB(opt).to(device)

    b,f,c,h,w = 1,11,3,96,96
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)

    out = layer(x, noise_map)

    print(out.shape)
    print(layer)

    #torchinfo.summary(layer, input_data=[x, noise_map])

def check_state_dict():
    import torch 
    device = torch.device('cuda')
    checkpoint = torch.load('experiments/naf_tsm/ckpt/naf_tsm2_400000.ckpt', map_location=device)
    
    for key, value in checkpoint['netG_state_dict'].items():
        print(key, value.shape)

def check_load_model():
    from models.network_mimo2 import NAFTSM, NAFBBB
    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    opt.n_frames = 5
    layer = NAFTSM(opt).to(device)
    state_dict = torch.load('experiments/naf_tsm2/ckpt/naf_tsm2_400000.ckpt', map_location=device)
    layer.load_state_dict(state_dict['netG_state_dict'], strict=True)
    
    res = 32
    b,f,c,h,w = opt.batch_size,opt.n_frames,3,res,res
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)

    out = layer(x, noise_map)

    print(out.shape)

def check_flow2():
    from models.network_mimo2 import NAFTSM, NAFBBB
    from utils.utils import convert_state_dict
    from collections import OrderedDict
    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    dataset_paths = []
    opt.val_dataset_path = 'datasets/Set8/sunflower'
    opt.data_extention = 'png'
    #val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=50)
    val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=50, max_frames=100, margin_frames=10, return_idx=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(len(val_loader))
    net = NAFBBB(opt).to(device)
    ckpt_path = 'experiments/naf_tsm2/ckpt/naf_tsm2_400000.ckpt'
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = state_dict['netG_state_dict']
    new_state_dict = convert_state_dict(state_dict)

    net.load_state_dict(new_state_dict, strict=True)

    start = time.time()
    cnt = 0
    for i, data in enumerate(val_loader):
        input_seq = torch.cat(data['input_seq'], dim=0).unsqueeze(0)#.to(device)
        gt_seq = torch.cat(data['gt_seq'], dim=0).unsqueeze(0)#.to(device)
        noise_map = data['noise_map'].unsqueeze(1)#.to(device)
        print(input_seq.shape, gt_seq.shape, noise_map.shape)
        print(data['idxs'])
        offset = data['idxs'][1].numpy()[0]
        frame1 = (data['idxs'][1] if i==0 else data['idxs'][0]).numpy()[0]
        frame2 = (data['idxs'][3] if i==len(val_loader)-1 else data['idxs'][2]).numpy()[0]
        idx1 = frame1 - offset
        idx2 = frame2 - offset
        print(offset, frame1, frame2, idx1, idx2)
        temp = gt_seq.unbind(dim=1)[idx1:idx2]
        print(len(temp))
        for _ in temp:
            cnt += 1

        continue
    print(cnt)

    '''
        # input_seq = (torch.arange(0,13685760, dtype=torch.float32, device=device) / 13685760).reshape(1,11,3,480,864)

        input_seq, gt_seq, noise_map = map(lambda x: pad_tensor(x, divisible_by=16), [input_seq, gt_seq, noise_map])

        # print(input_seq.shape, gt_seq.shape, noise_map.shape)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                gen = net(input_seq, noise_map)
        
        mse = torch.mean((gen-gt_seq)**2)
        print(20*torch.log10(1/mse**0.5))
        # 33.1216 vs 33.1262        

        # (B,F,C,H,W) -> [(B,C,H,W)]*F
        imgs, gens, gts = map(lambda x: x.unbind(dim=1), [input_seq, gen, gt_seq])
        
        imgs = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], imgs))
        gens = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gens))
        gts = list(map(lambda x: tensor2ndarray(x)[0,:,:,:], gts))

        for j, (img, gen, gt) in enumerate(zip(imgs, gens, gts)):
            # Visualization
            img, gen, gt = map(lambda x: Image.fromarray(x), [img, gen, gt])
            compare_img = Image.new('RGB', size=(3*img.width, img.height), color=0)
            compare_img.paste(img, box=(0, 0))
            compare_img.paste(gen, box=(img.width, 0))
            compare_img.paste(gt, box=(2*img.width, 0))
            compare_img.save(os.path.join('temp', f'{j:03}.png'), 'PNG')
        exit()
    '''
        

    #torchinfo.summary(net, input_data=[input_seq, noise_map])

def check_state_dict2():
    from models.network_mimo_multi import NAFTSM, NAFBBB
    from utils.utils import convert_state_dict
    from collections import OrderedDict
    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))

    netG = NAFTSM(opt).to(device)
    netG = nn.DataParallel(netG, device_ids=[0,1])

    state_dict = netG.module.state_dict()
    for key, value in state_dict.items():
        print(key, value.shape)

def check_flow3():
    from models.network_mimo4 import NAFTSM, NAFBBB
    from utils.utils import convert_state_dict
    from collections import OrderedDict
    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    #val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=50)
    #val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=50)

    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    netG = NAFTSM(opt).to(device)

    #netG_val = BSVD(pretrain_ckpt='temp.pth').to(device)
    netG_val = NAFBBB(opt).to(device)
    netG_val.load_state_dict(convert_state_dict(netG.state_dict()), strict=True)
    
    exit()

    start = time.time()
    for i, data in enumerate(val_loader):
        input_seq = torch.cat(data['input_seq'], dim=0).unsqueeze(0).to(device)
        gt_seq = torch.cat(data['gt_seq'], dim=0).unsqueeze(0).to(device)
        noise_map = data['noise_map'].unsqueeze(1).to(device)

        # input_seq = (torch.arange(0,13685760, dtype=torch.float32, device=device) / 13685760).reshape(1,11,3,480,864)

        input_seq, gt_seq, noise_map = map(lambda x: pad_tensor(x, divisible_by=16), [input_seq, gt_seq, noise_map])

        # print(input_seq.shape, gt_seq.shape, noise_map.shape)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                gen = net(input_seq, noise_map)
        
        mse = torch.mean((gen-gt_seq)**2)
        print(20*torch.log10(1/mse**0.5))
        # 33.1216 vs 33.1262
        exit()

def check_remo():
    import thop
    device = torch.device('cuda')
    from UNUSED.wnet_bsvd import BSVD

    net = BSVD(pretrain_ckpt=None).to(device)
    b,f,c,h,w = 1,50,3,480,856
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,f,1,h,w)).to(device)

    #out = net(x, noise_map)
    #print(out.shape)
    #torchinfo.summary(net, input_data=[x, noise_map])

    macs, params = thop.profile(net, inputs=(x, noise_map))
    print(f'GMACs: {macs/1e9/50:f}')
    print(f'#Params (M): {params/1e6:f}')
    
def check_halfmodel():
    from models.network2 import PseudoTemporalFusionNetworkEval
    from scripts.utils import convert_state_dict
    import thop

    device = torch.device('cuda')
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    net = PseudoTemporalFusionNetworkEval(opt).to(device)

    checkpoint = torch.load('experiments/ptfn_b16/ckpt/ptfn_b16_400000.ckpt', map_location=device)
    net.load_state_dict(convert_state_dict(checkpoint['netG_state_dict']), strict=False)

    b,f,c,h,w = 1,50,3,480,856
    x = torch.rand((b,f,c,h,w)).to(device)
    noise_map = torch.rand((b,1,1,h,w)).to(device)

    macs, params = thop.profile(net, inputs=(x, noise_map))
    print(f'GMACs: {macs/1e9/50:f}')
    print(f'#Params (M): {params/1e6:f}')

    #with torch.no_grad():
    #    out = net(x, noise_map)

    #print(out.shape)

def check_convert_state_dict():
    from models.network import PseudoTemporalFusionNetwork, PseudoTemporalFusionNetworkEval
    #from models.network8 import PseudoTemporalFusionNetwork, PseudoTemporalFusionNetworkEval
    from scripts.utils import convert_state_dict

    device = torch.device('cuda')
    with open('config/config_ptfn.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    net = PseudoTemporalFusionNetworkEval(opt).to(device)

    checkpoint = torch.load('experiments/ptfn_v4_small_b16/ckpt/ptfn_v4_small_b16_400000.ckpt', map_location=device)
    state_dict = convert_state_dict(checkpoint['netG_state_dict'])
    net.load_state_dict(state_dict, strict=True)

    for key, value in state_dict.items():
        print(key, value.shape)

def check_schedular():
    import pandas as pd
    import matplotlib.pyplot as plt 
    from models.network import PseudoTemporalFusionNetworkL

    device = torch.device('cuda')
    with open('config/config_ptfn4.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    
    tlog = pd.read_csv('experiments/ptfn_v4_small_b16/logs/train_losses_ptfn_v4_small_b16.csv')
    
    netG = PseudoTemporalFusionNetworkL(opt).to(device)
    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, T_max=opt.T_max, eta_min=opt.eta_min)
    state_dict = torch.load(opt.pretrained_path, map_location=device)

    nums1 = []
    lrs1 = []
    for i in range(opt.resume_step): 
        lr_G = [group['lr'] for group in optimG.param_groups]
        nums1.append(i)
        lrs1.append(lr_G[0])
        schedulerG.step()

    netG.load_state_dict(state_dict['netG_state_dict'], strict=True)
    optimG.load_state_dict(state_dict['optimG_state_dict'])

    nums2 = []
    lrs2 = []
    for i in range(opt.resume_step, opt.steps): 
        lr_G = [group['lr'] for group in optimG.param_groups]
        nums2.append(i)
        lrs2.append(lr_G[0])
        schedulerG.step()
    
    plt.figure(figsize=(8,4.5))
    plt.plot(tlog['step'], tlog['lr'], label='correct', alpha=0.5)
    plt.plot(nums1, lrs1, label='resume 1', alpha=0.5)
    plt.plot(nums2, lrs2, label='resume 2', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.savefig('temp/temp.png')

def check_lr():
    import pandas as pd
    import matplotlib.pyplot as plt 

    tlog1 = pd.read_csv('experiments/ptfn_v4_small_b16/logs/train_losses_ptfn_v4_small_b16.csv')
    tlog2 = pd.read_csv('experiments/ptfn_progressive3/logs/train_losses_ptfn_progressive3.csv')
    
    plt.figure(figsize=(8,4.5))
    plt.plot(tlog1['step'], tlog1['lr'], label='correct', alpha=0.5)
    plt.plot(tlog2['step'], tlog2['lr'], label='a', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.savefig('temp/temp.png')

def check_diff():
    import pandas as pd 
    log1 = pd.read_csv('results/bsvd_pretrained/bsvd_pretrained_50_results.csv')
    log2 = pd.read_csv('results/ptfn_inter010/ptfn_inter010_50_results.csv')

    diff = log2['psnr']-log1['psnr']
    for name, val in zip(log1['video_name'], diff):
        print(f'{name}: {val:f}')
    
    #print(log2['psnr'].idxmin())

def concat_images():
    import glob 
    from PIL import Image
    from tqdm import tqdm
    import numpy as np

    images_list = [
        sorted(glob.glob(f'results/ptfn_inter010/skate-jump/input/50/*.png')),
        sorted(glob.glob(f'results/ptfn_inter010/skate-jump/GT/*.png')),
        sorted(glob.glob(f'results/bsvd_pretrained/skate-jump/generated/50/*.png')),
        sorted(glob.glob(f'results/ptfn_inter010/skate-jump/generated/50/*.png')),
    ]
    w, h = 854, 480
    n_images = 25
    n_columns = len(images_list)
    for i in range(n_images):
        print(f'{i:03}: ', end='')
        img_gt = np.array(Image.open(images_list[1][i]))
        compare_img = Image.new('RGB', size=(w*n_columns, h))
        for j in range(n_columns):
            img = Image.open(images_list[j][i])
            compare_img.paste(img, box=(w*j,0))

            img = np.array(img)
            mse = np.mean((img_gt-img)**2)
            psnr = 20*np.log10(255.0/mse**0.5)
            print(f'{psnr:f} dB, ', end='')
        print()

        compare_img.save(f'temp/cmp_{i:03}.png')
        

def check_speed_unpix():
    import torch 
    from torch import nn 
    import numpy as np 
    from tqdm import tqdm 
    from models.layers import PseudoTemporalFusionBlock, TemporalShift, LayerNorm2d

    device = torch.device('cuda')

    class Toy1(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.tsm = TemporalShift(n_segment=11, shift_type='TSM')
            self.layer = PseudoTemporalFusionBlock(dim)

        def forward(self, x):
            x = self.tsm(x)
            x = self.layer(x)
            return x
    
    class Toy2(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = LayerNorm2d(dim)
            self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
            self.gelu = nn.GELU()

        def forward(self, x):
            f,c,h,w = x.shape
            x = self.norm(x)
            x = x.reshape(-1,c,f,h,w)
            x = self.conv1(x)
            x = x.reshape(f,c,h,w)
            x = self.gelu(x)
            return x
    
    f, h, w = 11, 128, 128
    dim = 64
    net1 = Toy1(dim).to(device)
    net2 = Toy2(dim).to(device)

    with torch.no_grad():
        runtimes1 = []
        for i in range(51):
            x = torch.rand((f,dim,h,w)).to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = net1(x)
            torch.cuda.synchronize()
            elapsed = time.time()-start
            if i!=0: runtimes1.append(elapsed)
            print(f'{i:02} -> {elapsed:f} s')
    
    with torch.no_grad():
        runtimes2 = []
        for i in range(51):
            x = torch.rand((f,dim,h,w)).to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = net2(x)
            torch.cuda.synchronize()
            elapsed = time.time()-start
            if i!=0: runtimes2.append(elapsed)
            print(f'{i:02} -> {elapsed:f} s')
    
    runtimes1 = np.array(runtimes1)
    runtimes2 = np.array(runtimes2)
    print(f'{runtimes1.mean():f} ± {runtimes1.std():f}')
    print(f'{runtimes2.mean():f} ± {runtimes2.std():f}')


if __name__=='__main__':
    #yoyaku()
    #check_block2()
    #check_diff()
    check_speed_unpix()
    #check_gif()
    #check_compare_video_mp4()


# 250001,0.0003087237902913379,-36.982525
# 250001,9.535928945656333e-05,-36.157455
# 250001,0.0003087237902913379,-36.872372

# 300001,0.00014652918823252866,-35.504791
# 300001,0.00014652918823252866,-35.962147
