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
from metrics import calculate_psnr, calculate_ssim

from models.layers import LaplacianPyramid, MBConvBlock, LaplacianPyramid, UNetBlock, WienerFilter
from models.network import FastDVDNet, U2NetDenoisingBlock, VideoDenoisingNetwork, FastDVDNetA, LapDenoisingBlock, VideoDenoisingNetwork2, FastDVDNetWiener, NAFDenoisingBlock, NAFDenoisingNet, MultiStageNAF, MultiStageNAF2
from models.networkM import FastDVDNetM, TinyDenoisingBlock, TinyDenoisingBlockSingle, ExtremeStageDenoisingNetwork, ExtremeStageDenoisingNetwork2
from utils.utils import pad_tensor, tensor2ndarray, read_img
from dataloader import DAVISVideoDenoisingTrainDataset, SingleVideoDenoisingTestDataset

def check_lap3d():
    device = torch.device('cuda')
    lap3d = LaplacianPyramid(3)

    x = torch.rand((1,3,6,64,64)).to(device)
    out = lap3d.pyramid_decom(x)
    for o in out:
        print(o.shape)

def check_layer():
    device = torch.device('cuda')
    H, W = 256, 256
    x0 = torch.rand((1,3,H,W)).to(device)
    x1 = torch.rand((1,3,H,W)).to(device)
    x2 = torch.rand((1,3,H,W)).to(device)
    noise_map = torch.rand((1,1,H,W)).to(device)
    layer = TinyDenoisingBlockSingle().to(device)
    # out = layer(x0, x1, x2, noise_map)
    out = layer(x0, noise_map)
    print(out.shape)

    # torchinfo.summary(layer, input_data=[x0, x1, x2, noise_map])
    torchinfo.summary(layer, input_data=[x0, noise_map])

def check_lap():
    device = torch.device('cuda')
    x = torch.rand((1,3,256,256)).to(device)
    lap_pyr = LaplacianPyramid(level=5)
    pyr = lap_pyr.pyramid_decom(x)
    for i, p in enumerate(pyr):
        print(i, p.shape)


def check_net():
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    device = torch.device('cuda')
    # x = torch.rand((1,16,128,128)).to(device)
    b = 8
    h, w = 128, 128
    x0 = torch.rand((b,3,h,w)).to(device)
    x1 = torch.rand((b,3,h,w)).to(device)
    x2 = torch.rand((b,3,h,w)).to(device)
    x3 = torch.rand((b,3,h,w)).to(device)
    x4 = torch.rand((b,3,h,w)).to(device)
    noise_map = torch.rand((b,1,h,w)).to(device)
    net = MultiStageNAF2(opt).to(device)
    out = net(x0, x1, x2, x3, x4, noise_map)
    print(out.shape)
    torchinfo.summary(net, input_data=[x0, x1, x2, x3, x4, noise_map])
    
def check_block():
    with open('config/config_test.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    device = torch.device('cuda')
    # x = torch.rand((1,16,128,128)).to(device)
    b = 8
    h, w = 256, 256
    x0 = torch.rand((b,3,h,w)).to(device)
    x1 = torch.rand((b,3,h,w)).to(device)
    x2 = torch.rand((b,3,h,w)).to(device)
    #x3 = torch.rand((b,3,h,w)).to(device)
    #x4 = torch.rand((b,3,h,w)).to(device)
    noise_map = torch.rand((b,1,h,w)).to(device)
    net = NAFDenoisingBlock(opt).to(device)
    out = net(x0, x1, x2, noise_map)
    print(out.shape)
    torchinfo.summary(net, input_data=[x0, x1, x2, noise_map])


def check_dataloader():
    with open('config/config_unet.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    train_dataset = DAVISDenoisingDataset(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    start = time.time()
    for i, (imgs, gt, noise_level) in enumerate(train_loader):
        print(imgs.shape)
        print(gt.shape)
        print(noise_level)
        exit()

def check_dataloader2():
    with open('config/config_unet.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    val_dataset = SingleVideoDenoisingTestDataset(opt, sigma=10)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    start = time.time()
    for i, (imgs, gt, noise_level) in enumerate(val_loader):
        print(imgs.shape)
        print(gt.shape)
        print(noise_level)
        exit()

def check_flow():
    device = torch.device('cuda')
    with open('config/config_unet.json', 'r', encoding='utf-8') as fp:
        opt = EasyDict(json.load(fp))
    train_dataset = DAVISVideoDenoisingTrainDataset(opt)
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    start = time.time()
    for i, (imgs, gt, noise_level) in enumerate(train_loader):
        net = VideoDenoisingNetwork(opt).to(device)
        imgs, gt = imgs.to(device), gt.to(device)
        out = net(imgs)
        print(out.shape, gt.shape)
        break
    torchinfo.summary(net, input_data=[imgs])

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

def check_pad():
    from utils.utils import pad_tensor
    h, w = 480, 720
    x = torch.arange(0,h*w).reshape(1,1,h,w).float()
    y, original_size = pad_tensor(x, divisible_by=32)
    print(x.shape, y.shape)

def check_pd():
    import pandas as pd
    df = pd.read_csv('results/fastdvdnet/fastdvdnet_50_results.csv')
    print(f'All,{df["psnr"].mean():f},{df["ssim"].mean():f}')
    print(abs(df['psnr'] - df["psnr"].mean()))

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
    img_paths = sorted(glob.glob('results/fastdvd_level5/rollercoaster/generated/50/*.png'))

    images = []
    for i, img_path in enumerate(img_paths):
        with Image.open(img_path) as img:
            img = img.convert('RGB')
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

    img_paths1 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/input/50/*.png'))
    images1 = [cv2.imread(img_path) for img_path in img_paths1]
    img_paths2 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/generated/50/*.png'))
    images2 = [cv2.imread(img_path) for img_path in img_paths2]
    img_paths3 = sorted(glob.glob('results/fastdvd_level5/rollercoaster/GT/*.png'))
    images3 = [cv2.imread(img_path) for img_path in img_paths3]
    
    video_size = [3*images1[0].shape[1], images1[0].shape[0]]
    frame_rate = 24
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('temp.mp4', fmt, frame_rate, video_size)
    
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

if __name__=='__main__':
    check_net()