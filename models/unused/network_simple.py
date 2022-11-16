from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from models.layers import MBConvBlock, FusedMBConvBlock

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.level = opt.level
        self.depths = opt.depths

        for l in range(self.level):
            if l==0:
                in_channel = opt.color_channels*opt.n_frames+opt.n_noise_channel
                layer = nn.Conv2d(in_channel, opt.dims[l], kernel_size=1)
            else:
                layer = nn.Conv2d(opt.dims[l-1], opt.dims[l], kernel_size=1)
            setattr(self, f'expand_dim_{l}', layer)
        
            for d in range(self.depths[l]):
                layer = FusedMBConvBlock(opt.dims[l], opt.dims[l], opt.expand_ratios[l])
                setattr(self, f'fused_mbconv_{l}_{d}', layer)
            
            layer = nn.AvgPool2d(2)
            setattr(self, f'avg_pool_{l}', layer)

    def forward(self, x):
        encoded = []
        for l in range(self.level):
            layer = getattr(self, f'expand_dim_{l}')
            x = layer(x)
            for d in range(self.depths[l]):
                layer = getattr(self, f'fused_mbconv_{l}_{d}')
                x = layer(x)
            encoded.append(x)
            layer = getattr(self, f'avg_pool_{l}')
            x = layer(x)
        return x, encoded

class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.level = opt.level
        self.depths = opt.depths

        for l in reversed(range(self.level)):
            if l==self.level-1:
                layer = nn.Conv2d(2*opt.dims[l], opt.dims[l], kernel_size=1)
            else:
                layer = nn.Conv2d(opt.dims[l]+opt.dims[l+1], opt.dims[l], kernel_size=1)
            setattr(self, f'expand_dim_{l}', layer)
        
            for d in range(self.depths[l]):
                layer = MBConvBlock(opt.dims[l], opt.dims[l], opt.expand_ratios[l])
                setattr(self, f'mbconv_{l}_{d}', layer)
        self.to_out = nn.Conv2d(opt.dims[0], opt.color_channels, kernel_size=1)
    
    def forward(self, x, encoded):
        for l in reversed(range(self.level)):
            B,C,H,W = x.shape
            target_h, target_w = encoded[l].shape[-2], encoded[l].shape[-1]
            x = F.interpolate(
                x, size=[target_h, target_w], 
                mode='bilinear', align_corners=False
                ).reshape(B,C,target_h,target_w)
            x = torch.cat([x, encoded[l]], dim=1)
            layer = getattr(self, f'expand_dim_{l}')
            x = layer(x)
            for d in range(self.depths[l]):
                layer = getattr(self, f'mbconv_{l}_{d}')
                x = layer(x)
        x = self.to_out(x)
        return x

class VideoDenoisingNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
    
    def forward(self, x):
        x, encoded = self.encoder(x)
        x = self.decoder(x, encoded)
        return x