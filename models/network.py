import torch
from torch import nn
from models.layers import PseudoTemporalFusionBlock, TemporalShift, PseudoTemporalFusionBlockBBB, MemSkip

class DenoisingBlock(nn.Module):
    def __init__(self, opt, in_channels=None, out_channels=None):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        in_channels = in_channels if in_channels is not None else opt.color_channels
        out_channels = out_channels if out_channels is not None else opt.color_channels

        self.intro = nn.Conv2d(in_channels=in_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_tsm_{i}_{j}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
                setattr(self, f'enc_block_{i}_{j}', PseudoTemporalFusionBlock(chan))
                
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_tsm_{i}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
            setattr(self, f'middle_block_{i}', PseudoTemporalFusionBlock(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_tsm_{i}_{j}', TemporalShift(opt.n_frames, 'TSM', fold_div=opt.tsm_fold, stride=1))
                setattr(self, f'dec_block_{i}_{j}', PseudoTemporalFusionBlock(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

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

        x = self.ending(x) + self.expand_dims(seq.reshape(b*f,c,h,w))
        x = x.reshape(b,f,-1,h,w)

        return x

class DenoisingBlockBBB(nn.Module):
    def __init__(self, opt, in_channels=None, out_channels=None):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        in_channels = in_channels if in_channels is not None else opt.color_channels
        out_channels = out_channels if out_channels is not None else opt.color_channels

        self.skip_intro = MemSkip()
        for i in range(len(self.enc_blk_nums)):
            setattr(self, f'skip_{i}', MemSkip())

        self.intro = nn.Conv2d(in_channels=in_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_block_{i}_{j}', PseudoTemporalFusionBlockBBB(chan))
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_block_{i}', PseudoTemporalFusionBlockBBB(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_block_{i}_{j}', PseudoTemporalFusionBlockBBB(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
    
    def none_add(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            return x1+x2
    
    def none_reshape(self, x, order):
        if x is not None:
            return x.reshape(*order)
        else:
            return None
    
    def none_expand_dims(self, seq):
        # seq: (B,F,C,H,W) or None
        if seq is not None:
            seq = self.expand_dims(seq.reshape(self.b*self.f,-1,self.h,self.w)).reshape(self.b,self.f,-1,self.h,self.w)
            return seq
        else:
            return None      

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
            x = self.none_add(self.ending(x), self.none_expand_dims(self.skip_intro.pop(x)))
            x = self.none_reshape(x, [self.b,self.f,-1,self.h,self.w])

        return x

class PseudoTemporalFusionNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = DenoisingBlock(opt, out_channels=opt.width)
        self.temp2 = DenoisingBlock(opt, in_channels=opt.width)
        self.to_rgb = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=1)
    
    def forward(self, seq, noise_map):
        # seq: (B, F, C, H, W)
        b,f,c,h,w = seq.shape
        seq = self.temp1(seq, noise_map)
        inter_img = self.to_rgb(seq.reshape(b*f,-1,h,w)).reshape(b,f,-1,h,w)
        seq = self.temp2(seq, noise_map)
        return seq, inter_img

class PseudoTemporalFusionNetworkEval(nn.Module):
    def __init__(self, opt, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.temp1 = DenoisingBlockBBB(opt, out_channels=opt.width)
        self.temp2 = DenoisingBlockBBB(opt, in_channels=opt.width)
        self.to_rgb = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=1)
        self.shift_num = self.count_shift()
    
    def none_reshape(self, x, order):
        if x is not None:
            return x.reshape(*order)
        else:
            return None
    
    def feed_in_one_element(self, seq, noise_map):
        #seq: (B,F,C,H,W)
        seq = self.temp1(seq, noise_map)
        if seq is not None:
            b,f,c,h,w = seq.shape
            inter_img = self.to_rgb(seq.reshape(b*f,-1,h,w)).reshape(b,f,-1,h,w)
        else:
            inter_img = None
        seq = self.temp2(seq, noise_map)
        return seq, inter_img
    
    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            if 'PseudoTemporalFusionBlockBBB' in str(type(module)):
                count += 1
        return count
    
    def reset(self):
        for name, module in self.named_modules():
            if 'PseudoTemporalFusionBlockBBB' in str(type(module)):
                module.reset()

    def forward(self, seq, noise_map):
        # (1,F,C,H,W) -> [(1,C,H,W)]*F
        seq = torch.unbind(seq, dim=1)
        noise_map = noise_map.to(self.device)

        out_seq1 = []
        out_seq2 = []
        with torch.no_grad():
            cnt = 0
            for i, x in enumerate(seq):
                # (1,C,H,W) -> (1,1,C,H,W)
                x = x.unsqueeze(0).to(self.device)
                x, inter_img = self.feed_in_one_element(x, noise_map)
                out_seq1.append(x)
                out_seq2.append(inter_img)
            end_out, inter_img = self.feed_in_one_element(None, noise_map)
            out_seq1.append(end_out)
            out_seq2.append(inter_img)

            while True:
                end_out, inter_img = self.feed_in_one_element(None, noise_map)
                if len(out_seq1)==(self.shift_num+len(seq)): break
                out_seq1.append(end_out)
                out_seq2.append(inter_img)

            out_seq_clip1 = out_seq1[self.shift_num:]
            out_seq_clip2 = []
            for inter_img in out_seq2:
                if inter_img is not None:
                    out_seq_clip2.append(inter_img)

            self.reset()

            return torch.cat(out_seq_clip1, dim=1), torch.cat(out_seq_clip2, dim=1)