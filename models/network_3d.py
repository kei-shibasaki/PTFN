import torch
from torch import nn
from torch.nn import functional as F
from models.layers import MBConvBlock3d, FusedMBConvBlock3d

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.level = opt.level
        self.depths = opt.depths

        for l in range(self.level):
            if l==0:
                layer = nn.Conv3d(opt.color_channels, opt.dims[l], kernel_size=1)
            else:
                layer = nn.Conv3d(opt.dims[l-1], opt.dims[l], kernel_size=1)
            setattr(self, f'expand_dim_{l}', layer)
        
            for d in range(self.depths[l]):
                layer = FusedMBConvBlock3d(opt.dims[l], opt.dims[l], opt.expand_ratios[l])
                setattr(self, f'fused_mbconv_{l}_{d}', layer)
            
            layer = nn.AvgPool3d((1,2,2))
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
                layer = nn.Conv3d(2*opt.dims[l], opt.dims[l], kernel_size=1)
            else:
                layer = nn.Conv3d(opt.dims[l]+opt.dims[l+1], opt.dims[l], kernel_size=1)
            setattr(self, f'expand_dim_{l}', layer)
        
            for d in range(self.depths[l]):
                layer = MBConvBlock3d(opt.dims[l], opt.dims[l], opt.expand_ratios[l])
                setattr(self, f'mbconv_{l}_{d}', layer)
    
    def forward(self, x, encoded):
        for l in reversed(range(self.level)):
            B,C,D,H,W = x.shape
            target_h, target_w = encoded[l].shape[-2], encoded[l].shape[-1]
            x = F.interpolate(
                x.reshape(B*D,C,H,W), size=[target_h, target_w], 
                mode='bilinear', align_corners=False
                ).reshape(B,C,D,target_h,target_w)
            x = torch.cat([x, encoded[l]], dim=1)
            layer = getattr(self, f'expand_dim_{l}')
            x = layer(x)
            for d in range(self.depths[l]):
                layer = getattr(self, f'mbconv_{l}_{d}')
                x = layer(x)
        return x

class VideoDenoisingNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        