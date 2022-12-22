import numpy as np
from bm4d import bm4d
import time 

def main(n,c,h,w, total_step=5):
    dtimes = []
    for i in range(total_step):
        noisy = np.random.random([h,w,c])
        std = 50
        start = time.time()
        bm4d(noisy, std)
        dtime = time.time() - start
        if i!=0: dtimes.append(dtime)
        print(f'{i}, dtime: {dtime:f}s')
    return dtimes 

if __name__=='__main__':
    n,c = 5, 3
    #h,w = 480, 856
    #h,w = 720, 1280
    #h,w = 1080, 1920
    for (h, w) in [[720,1280], [1080,1920]]:
        print(h, w)
        dtimes = main(n,c,h,w)
        print(f'({h},{w}): {sum(dtimes)/len(dtimes):f}')
