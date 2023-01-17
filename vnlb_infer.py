import vnlb 
import numpy as np 
import glob 
import os
from PIL import Image

def denoise_vnlb():
    n_images = 25
    noisy1 = sorted(glob.glob('results/ptfn_inter010_set8/snowboard/input/50/*.png'))[n_images-2:n_images+3]
    noisy2 = sorted(glob.glob('results/ptfn_inter010_set8/tractor/input/50/*.png'))[n_images-2:n_images+3]

    images1 = []
    images2 = []
    for img_path in noisy1:
        img = np.array(Image.open(img_path)).transpose(2,0,1).astype(np.float32) #/ 255.
        images1.append(img)
    images1 = np.stack(images1, axis=0)

    for img_path in noisy2:
        img = np.array(Image.open(img_path)).transpose(2,0,1).astype(np.float32) #/ 255.
        images2.append(img)
    images2 = np.stack(images2, axis=0)
    

    denoised1, basic, dtime = vnlb.denoise(images1, 50)
    denoised1 = denoised1.cpu().numpy()
    denoised1 = np.clip(denoised1, 0.0, 255.0).transpose(0,2,3,1)
    print(denoised1.shape, denoised1.dtype, denoised1.min(), denoised1.max())

    for i, img in enumerate(np.split(denoised1, indices_or_sections=5, axis=0)):
        img = Image.fromarray(img[0,:,:,:].astype(np.uint8))
        img.save(f'temp/snow_{i}.png')
    
    denoised2, basic, dtime = vnlb.denoise(images2, 50)
    denoised2 = denoised2.cpu().numpy()
    denoised2 = np.clip(denoised2, 0.0, 255.0).transpose(0,2,3,1)
    print(denoised2.shape, denoised2.dtype, denoised2.min(), denoised2.max())

    for i, img in enumerate(np.split(denoised2, indices_or_sections=5, axis=0)):
        img = Image.fromarray(img[0,:,:,:].astype(np.uint8))
        img.save(f'temp/tractor_{i}.png')



if __name__=='__main__':
    denoise_vnlb()


