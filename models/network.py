import torch
from torch import nn
from torch.nn import functional as F
from models.layers import ConvBNReLU, LaplacianPyramid, UNetBlock, RConvBNReLU, RDConvBNReLU, WienerFilter
from models.layers import InputConvBlock, ConvBlock, OutputConvBlock, NAFBlock, MotionCompensationAttention, MotionCompensationAttention2

class U2NetDenoisingBlock(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.level = opt.level

        for l in range(self.level):
            in_channels = opt.channel_info[l-1][1] if l!=0 else 3*(3+1)
            hidden_channels = opt.channel_info[l][0]
            out_channels = opt.channel_info[l][1]
            setattr(self, f'enc_{l}', UNetBlock(opt.sub_levels[l], in_channels, hidden_channels, out_channels))
            setattr(self, f'downsample_{l}', nn.MaxPool2d(2, ceil_mode=True))
        
        hidden_channels = opt.channel_info[-1][1]
        bottom_layers = []
        for d in range(opt.bottom_depth):
            bottom_layers.append(ConvBNReLU(hidden_channels, hidden_channels, dilation=opt.bottom_dilations[d]))
        self.bottom = nn.Sequential(*bottom_layers)
    
        for l in reversed(range(self.level)):
            in_channels = opt.channel_info[l+1][0] if l!=self.level-1 else opt.channel_info[-1][1]
            hidden_channels = opt.channel_info[l][1]
            out_channels = opt.channel_info[l][0]
            # setattr(self, f'upsample_{l}', nn.PixelShuffle(2))
            setattr(self, f'dec_{l}', UNetBlock(opt.sub_levels[l], in_channels, hidden_channels, out_channels))
        
        self.to_out = nn.Conv2d(opt.channel_info[0][0], 3, kernel_size=3, padding=1)

    def forward(self, x0, x1, x2, noise_map):
        x = torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1)
        encoded = []
        for l in range(self.level):
            x = getattr(self, f'enc_{l}')(x)
            encoded.append(x)
            x = getattr(self, f'downsample_{l}')(x)
        x = self.bottom(x)
        for l in reversed(range(self.level)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = getattr(self, f'dec_{l}')(x+encoded[l])
        x = self.to_out(x)
        return x1 - x

class VideoDenoisingNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = U2NetDenoisingBlock(opt)
        self.temp2 = U2NetDenoisingBlock(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class DenoisingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputConvBlock(num_in_frames=3, out_ch=32)
        self.down = ConvBlock(32, 64, 64, depth=2, downsample=True)
        self.bottom = ConvBlock(64, 128, 64, depth=4, downsample=True, upsample=True)
        self.up = ConvBlock(64, 64, 32, depth=2, upsample=True)
        self.out = OutputConvBlock(32, 3)
    
    def forward(self, x0, x1, x2, noise_map):
        enc0 = self.input(torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1))
        enc1 = self.down(enc0)
        enc2 = self.bottom(enc1)
        enc1 = self.up(enc1+enc2)
        out = self.out(enc0+enc1)
        out = x1 - out
        return out

class DenoisingBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputConvBlock(num_in_frames=3, out_ch=32)
        self.down1 = ConvBlock(32, 64, 64, depth=2, downsample=True)
        self.down2 = ConvBlock(64, 64, 64, depth=2, downsample=True)
        self.bottom = ConvBlock(64, 128, 64, depth=4, downsample=True, upsample=True)
        self.up1 = ConvBlock(64, 64, 64, depth=2, upsample=True)
        self.up2 = ConvBlock(64, 64, 32, depth=2, upsample=True)
        self.out = OutputConvBlock(32, 3)
    
    def forward(self, x0, x1, x2, noise_map):
        enc0 = self.input(torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1))
        enc1 = self.down1(enc0)
        enc2 = self.down2(enc1)
        enc3 = self.bottom(enc2)
        enc2 = self.up1(enc2+enc3)
        enc1 = self.up2(enc1+enc2)
        out = self.out(enc0+enc1)
        out = x1 - out
        return out

class FastDVDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = DenoisingBlock()
        self.temp2 = DenoisingBlock()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class FastDVDNetWiener(nn.Module):
    def __init__(self):
        super().__init__()
        self.wiener = WienerFilter(kernel_size=9)
        self.temp1 = DenoisingBlock()
        self.temp2 = DenoisingBlock()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        normed_sigma = noise_map[:,0,0,0].reshape(-1,1,1,1)
        x0, x1, x2, x3, x4 = map(lambda x: self.wiener(x, noise_power=normed_sigma**2), [x0, x1, x2, x3, x4])
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class FastDVDNetWiener2(nn.Module):
    def __init__(self):
        super().__init__()
        self.wiener = WienerFilter(kernel_size=3)
        self.temp1 = DenoisingBlock()
        self.temp2 = DenoisingBlock()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        normed_sigma = noise_map[:,0,0,0].reshape(-1,1,1,1)
        x0, x1, x2, x3, x4 = map(lambda x: self.wiener(x, noise_power=normed_sigma**2), [x0, x1, x2, x3, x4])
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class FastDVDNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = DenoisingBlock2()
        self.temp2 = DenoisingBlock2()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x


class LapDenoisingBlock(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.level = opt.level
        self.depths = opt.depths
        self.lap_pyr = LaplacianPyramid(self.level)
        for l in range(self.level+1):
            for d in range(self.depths[l]):
                if d==0:
                    if l==self.level-1:
                        in_channels = 3*(3+1)+3*(3+1)+3
                    else:
                        in_channels = 3*(3+1)
                else:
                    in_channels = opt.dims[l]
                
                out_channels = 3 if d==self.depths[l]-1 else opt.dims[l]

                if (d==0) and (l<self.level-1):
                    setattr(self, f'expand_dim_{l}', nn.Conv2d(3, in_channels, kernel_size=1))

                setattr(self, f'conv_{l}_{d}', RConvBNReLU(in_channels, out_channels))
            
    def forward(self, x0, x1, x2, noise_map):
        pyr_x0 = self.lap_pyr.pyramid_decom(x0)
        pyr_x1 = self.lap_pyr.pyramid_decom(x1)
        pyr_x2 = self.lap_pyr.pyramid_decom(x2)
        pyr_out = []
        for l in reversed(range(self.level+1)):
            x0, x1, x2 = pyr_x0[l], pyr_x1[l], pyr_x2[l]
            n_map = F.interpolate(noise_map, size=[x0.shape[2], x0.shape[3]], mode='nearest')

            if l==self.level:
                temp = torch.cat([x0, n_map, x1, n_map, x2, n_map], dim=1)
                x = temp
            elif l==self.level-1:
                temp = F.interpolate(temp, scale_factor=2, mode='bilinear', align_corners=False)
                x = torch.cat([x, temp, x0, n_map, x1, n_map, x2, n_map], dim=1)
            else:
                temp = torch.cat([x0, n_map, x1, n_map, x2, n_map], dim=1)
                x = getattr(self, f'expand_dim_{l}')(x)
                x = temp*x + temp

            for d in range(self.depths[l]):
                x = getattr(self, f'conv_{l}_{d}')(x)
            
            pyr_out.append(x)
            if l!=0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        out = self.lap_pyr.pyramid_recons(list(reversed(pyr_out)))

        return out

class VideoDenoisingNetwork2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = LapDenoisingBlock(opt)
        self.temp2 = LapDenoisingBlock(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class DenoisingBlockA(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputConvBlock(num_in_frames=3, out_ch=32)
        self.down = ConvBlock(32, 64, 64, depth=2, downsample=True)

        self.bottom = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            AxialTransformerBlock(128, heads=8, mlp_ratio=1, sum_axial_out=True), 
            AxialTransformerBlock(128, heads=8, mlp_ratio=1, sum_axial_out=True), 
            nn.Conv2d(128, 4*64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.PixelShuffle(2),
        )

        #self.bottom = ConvBlock(64, 128, 64, depth=4, downsample=True, upsample=True)

        self.up = ConvBlock(64, 64, 32, depth=2, upsample=True)
        self.out = OutputConvBlock(32, 3)
    
    def forward(self, x0, x1, x2, noise_map):
        enc0 = self.input(torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1))
        enc1 = self.down(enc0)
        enc2 = self.bottom(enc1)
        enc1 = self.up(enc1+enc2)
        out = self.out(enc0+enc1)
        out = x1 - out
        return out

class FastDVDNetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = DenoisingBlockA()
        self.temp2 = DenoisingBlockA()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x


class NAFDenoisingBlock(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=3*(opt.color_channels+1), out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
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

    def forward(self, x0, x1, x2, noise_map):
        x = torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1)

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
        x = x + x1

        return x

class NAFDenoisingBlockWithMC(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.mc = MotionCompensationAttention(3*(opt.color_channels+1), kernel_size=3)
        self.intro = nn.Conv2d(in_channels=3*(opt.color_channels+1), out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
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

    def forward(self, x0, x1, x2, noise_map):
        x = torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1)

        x = self.intro(self.mc(x))

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
        x = x + x1

        return x

class NAFDenoisingBlockWithMC2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.mc = MotionCompensationAttention2(3*(opt.color_channels+1), kernel_size=3)
        self.intro = nn.Conv2d(in_channels=3*(opt.color_channels+1), out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
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

    def forward(self, x0, x1, x2, noise_map):
        x = torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1)

        x = self.intro(self.mc(x))

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
        x = x + x1

        return x

class NAFDenoisingBlockSingle(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=opt.color_channels+1, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=opt.color_channels, kernel_size=3, padding=1, stride=1, groups=1,
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

    def forward(self, x0, noise_map):
        x = torch.cat([x0, noise_map], dim=1)

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
        x = x + x0

        return x

class MultiStageNAF(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = NAFDenoisingBlock(opt)
        self.temp2 = NAFDenoisingBlock(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class MultiStageNAFWithMC(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = NAFDenoisingBlockWithMC(opt)
        self.temp2 = NAFDenoisingBlockWithMC(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class MultiStageNAFWithMC2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = NAFDenoisingBlockWithMC2(opt)
        self.temp2 = NAFDenoisingBlockWithMC2(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)

        return x

class MultiStageNAF2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.temp1 = NAFDenoisingBlockSingle(opt)
        self.temp2 = NAFDenoisingBlock(opt)
        self.temp3 = NAFDenoisingBlock(opt)
        self.temp4 = NAFDenoisingBlockSingle(opt)
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x0, x1, x2, x3, x4 = map(lambda x: self.temp1(x, noise_map), [x0, x1, x2, x3, x4])
        x20 = self.temp2(x0, x1, x2, noise_map)
        x21 = self.temp2(x1, x2, x3, noise_map)
        x22 = self.temp2(x2, x3, x4, noise_map)
        x = self.temp3(x20, x21, x22, noise_map)
        x = self.temp4(x, noise_map)

        return x

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