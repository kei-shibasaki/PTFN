import torch
from torch import nn
from torch.nn import functional as F
from models.layers import NAFBlock2, TemporalShift, NAFBlockBBB2, MemSkip

class NAFDenoisingBlockTSM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        self.intro = nn.Conv2d(in_channels=opt.color_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_tsm_{i}_{j}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
                setattr(self, f'enc_block_{i}_{j}', NAFBlock2(chan))
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_tsm_{i}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
            setattr(self, f'middle_block_{i}', NAFBlock2(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_tsm_{i}_{j}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
                setattr(self, f'dec_block_{i}_{j}', NAFBlock2(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    def forward(self, seq, noise_map):
        # seq: (B, F, C, H, W), noise_map: (B, 1, 1, H, W)
        b, f, c, h, w = seq.shape
        x = torch.cat([seq, noise_map.repeat(1,f,1,1,1)], dim=2).reshape(b*f,c+1,h,w)
        x = self.intro(x)

        encs = []
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                x = getattr(self, f'enc_tsm_{i}_{j}')(x)
                x = getattr(self, f'enc_block_{i}_{j}')(x)
            encs.append(x)
            x = getattr(self, f'enc_down_{i}')(x)

        for i in range(self.middle_blk_num):
            x = getattr(self, f'middle_tsm_{i}')(x)
            x = getattr(self, f'middle_block_{i}')(x)

        for i, (num, enc) in enumerate(zip(self.dec_blk_nums, encs[::-1])):
            x = getattr(self, f'dec_upconv_{i}')(x)
            x = getattr(self, f'dec_up_{i}')(x)
            x = x + enc
            for j in range(num):
                x = getattr(self, f'dec_tsm_{i}_{j}')(x)
                x = getattr(self, f'dec_block_{i}_{j}')(x)

        x = self.ending(x).reshape(b,f,c,h,w)
        x = seq + x

        return x

class NAFDenoisingBlockBBB(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        self.skip_intro = MemSkip()
        for i in range(len(self.enc_blk_nums)):
            setattr(self, f'skip_{i}', MemSkip())

        self.intro = nn.Conv2d(in_channels=opt.color_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_block_{i}_{j}', NAFBlockBBB2(chan))
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_block_{i}', NAFBlockBBB2(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_block_{i}_{j}', NAFBlockBBB2(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
    
    def none_add(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            return x1+x2

    def forward(self, seq, noise_map):
        # seq: (B, F, C, H, W), noise_map: (B, 1, 1, H, W)
        self.skip_intro.push(seq)
        if seq is not None:
            self.b, self.f, self.c, self.h, self.w = seq.shape
            x = torch.cat([seq, noise_map.repeat(1,self.f,1,1,1)], dim=2).reshape(self.b*self.f,self.c+1,self.h,self.w)
            x = self.intro(x)
        else:
            x = None

        cnt = 0
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                x = getattr(self, f'enc_block_{i}_{j}')(x)
            getattr(self, f'skip_{i}').push(x)
            if x is not None:
                x = getattr(self, f'enc_down_{i}')(x)

        for i in range(self.middle_blk_num):
            x = getattr(self, f'middle_block_{i}')(x)

        for i, num in enumerate(self.dec_blk_nums):
            if x is not None:
                x = getattr(self, f'dec_upconv_{i}')(x)
                x = getattr(self, f'dec_up_{i}')(x)
                enc_feature = getattr(self, f'skip_{len(self.enc_blk_nums)-1-i}').pop(x)
                x = self.none_add(x, enc_feature)
            for j in range(num):
                x = getattr(self, f'dec_block_{i}_{j}')(x)
        
        if x is not None:
            x = self.ending(x)
            x = self.none_add(self.skip_intro.pop(x), x.reshape(self.b,self.f,self.c,self.h,self.w))

        return x

class NAFTSM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = NAFDenoisingBlockTSM(opt)
        self.temp2 = NAFDenoisingBlockTSM(opt)
    
    def forward(self, seq, noise_map):
        # seq: (B, F, C, H, W)
        seq = self.temp1(seq, noise_map)
        seq = self.temp2(seq, noise_map)
        return seq

class NAFBBB(nn.Module):
    def __init__(self, opt, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.temp1 = NAFDenoisingBlockBBB(opt)
        self.temp2 = NAFDenoisingBlockBBB(opt)
        self.shift_num = self.count_shift()
    
    def feed_in_one_element(self, seq, noise_map):
        #seq: (B,F,C,H,W)
        seq = self.temp1(seq, noise_map)
        seq = self.temp2(seq, noise_map)
        return seq
    
    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            if 'NAFBlockBBB' in str(type(module)):
                count += 1
        return count
    
    def reset(self):
        for name, module in self.named_modules():
            if 'NAFBlockBBB' in str(type(module)):
                module.reset()

    def forward(self, seq, noise_map):
        # seq: (1,F,C,H,W), noise_map: (1,1,1,H,W)
        b,f,c,h,w = seq.shape
        # (1,F,C,H,W) -> [(1,C,H,W)]*F
        seq = torch.unbind(seq, dim=1)
        noise_map = noise_map.to(self.device)

        out_seq = []
        with torch.no_grad():
            cnt = 0
            for i, x in enumerate(seq):
                # (1,C,H,W) -> (1,1,C,H,W)
                x = x.unsqueeze(0).to(self.device)
                x = self.feed_in_one_element(x, noise_map)
                out_seq.append(x)
            end_out = self.feed_in_one_element(None, noise_map)
            out_seq.append(end_out)

            while True:
                end_out = self.feed_in_one_element(None, noise_map)
                if len(out_seq)==(self.shift_num+len(seq)): break
                out_seq.append(end_out)

            out_seq_clip = out_seq[self.shift_num:]
            self.reset()

            return torch.cat(out_seq_clip, dim=1)