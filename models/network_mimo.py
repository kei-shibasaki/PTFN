import torch
from torch import nn
from torch.nn import functional as F
from models.layers import NAFBlock
from models.temporal_shift import TemporalShift

class NAFDenoisingBlockMIMO(nn.Module):
    def __init__(self, opt, n_frames=None):
        super().__init__()
        n_frames = opt.n_frames if n_frames is None else n_frames

        self.intro = nn.Conv2d(in_channels=n_frames*opt.color_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=n_frames*opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = opt.width
        for num in opt.enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(opt.middle_blk_num)]
            )

        for num in opt.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, seq, noise_map):
        # seq: (B, C*F, H, W)
        x = torch.cat([seq, noise_map], dim=1)
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = seq + x

        return x

class MultiStageNAFMIMO(nn.Module):
    def __init__(self, opt):
        super().__init__()
        assert opt.n_frames==9
        self.n_blocks = opt.n_frames//3
        self.temp1 = NAFDenoisingBlockMIMO(opt, n_frames=3)
        self.temp2 = NAFDenoisingBlockMIMO(opt, n_frames=9)
        self.temp3 = NAFDenoisingBlockMIMO(opt, n_frames=3)

    def forward(self, seq, noise_map):
        # Stage1
        seq = torch.chunk(seq, self.n_blocks, dim=1)
        seq = list(map(lambda x: self.temp1(x, noise_map), seq))

        # Stage2
        seq = torch.cat(seq, dim=1)
        seq = self.temp2(seq, noise_map)

        # Stage3
        seq = torch.chunk(seq, self.n_blocks, dim=1)
        seq = list(map(lambda x: self.temp3(x, noise_map), seq))
        seq = torch.cat(seq, dim=1)
        return seq

class MultiStageNAFMIMO2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        assert opt.n_frames==12
        self.n_blocks1 = 4
        self.n_blocks2 = 2
        self.temp1 = NAFDenoisingBlockMIMO(opt, n_frames=3)
        self.temp2 = NAFDenoisingBlockMIMO(opt, n_frames=6)
        self.temp3 = NAFDenoisingBlockMIMO(opt, n_frames=12)
        self.temp4 = NAFDenoisingBlockMIMO(opt, n_frames=6)
        self.temp5 = NAFDenoisingBlockMIMO(opt, n_frames=3)

    def forward(self, seq, noise_map):
        # Stage1
        seq = torch.chunk(seq, self.n_blocks1, dim=1)
        seq = list(map(lambda x: self.temp1(x, noise_map), seq))
        seq = torch.cat(seq, dim=1)

        # Stage2
        seq = torch.chunk(seq, self.n_blocks2, dim=1)
        seq = list(map(lambda x: self.temp2(x, noise_map), seq))
        seq = torch.cat(seq, dim=1)

        # Stage3
        seq = self.temp3(seq, noise_map)
        
        # Stage4
        seq = torch.chunk(seq, self.n_blocks2, dim=1)
        seq = list(map(lambda x: self.temp4(x, noise_map), seq))
        seq = torch.cat(seq, dim=1)

        # Stage5
        seq = torch.chunk(seq, self.n_blocks1, dim=1)
        seq = list(map(lambda x: self.temp5(x, noise_map), seq))
        seq = torch.cat(seq, dim=1)

        return seq

class NAFDenoisingBlockTSM(nn.Module):
    def __init__(self, opt, n_frames=None):
        super().__init__()
        n_frames = opt.n_frames if n_frames is None else n_frames

        self.intro = nn.Conv2d(in_channels=opt.color_channels+opt.n_noise_channel, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = TemporalShift(self.ending, opt.n_frames, opt.tsm_fold, 'TSM', inplace=False, enable_past_buffer=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = opt.width
        for num in opt.enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[TemporalShift(NAFBlock(chan), opt.n_frames, opt.tsm_fold, 'TSM', inplace=False, enable_past_buffer=True) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[TemporalShift(NAFBlock(chan), opt.n_frames, opt.tsm_fold, 'TSM', inplace=False, enable_past_buffer=True) for _ in range(opt.middle_blk_num)]
            )

        for num in opt.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    TemporalShift(nn.Conv2d(chan, chan * 2, 1, bias=False), opt.n_frames, opt.tsm_fold, 'TSM', inplace=False, enable_past_buffer=True),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[TemporalShift(NAFBlock(chan), opt.n_frames, opt.tsm_fold, 'TSM', inplace=False, enable_past_buffer=True) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, seq, noise_map):
        # seq: (B, F, C, H, W), noise_map: (B, 1, 1, H, W)
        b, f, c, h, w = seq.shape
        x = torch.cat([seq, noise_map.repeat(1,f,1,1,1)], dim=2).reshape(b*f,c+1,h,w)
        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x).reshape(b,f,c,h,w)
        x = seq + x

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