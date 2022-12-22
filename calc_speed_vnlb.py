import vnlb 
import numpy as np 

def denoise_vnlb(n,c,h,w, total_step=1):
    dtimes = []
    for i in range(total_step):
        noisy = np.random.random([n,c,h,w])
        std = 50

        _,_,dtime = vnlb.denoise(noisy, std)
        if i!=0: dtimes.append(dtime)
        print(f'{i}, dtime: {dtime:f}s')

if __name__=='__main__':
    n,c = 5, 3
    #h,w = 480, 856
    #h,w = 720, 1280
    h,w = 1080, 1920
    denoise_vnlb(n,c,h,w)


