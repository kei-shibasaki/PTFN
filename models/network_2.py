import torch
from torch import nn
from torch.nn import functional as F
from models.layers import MBConvBlock, FusedMBConvBlock, TransformerBlock, InverseSigmoid

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
        layer = nn.Conv2d(opt.dims[-2], opt.dims[-1], kernel_size=1)
        setattr(self, f'expand_dim_{self.level}', layer)

    def forward(self, x):
        encoded = []
        for l in range(self.level):
            x = getattr(self, f'expand_dim_{l}')(x)
            for d in range(self.depths[l]):
                x = getattr(self, f'fused_mbconv_{l}_{d}')(x)
            encoded.append(x)
            x = getattr(self, f'avg_pool_{l}')(x)
        x = getattr(self, f'expand_dim_{self.level}')(x)
        return x, encoded


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        assert opt.level+1==opt.n_high+opt.n_low
        self.level = opt.level
        self.depths = opt.depths
        self.n_high = opt.n_high
        self.n_low = opt.n_low

        for l in reversed(range(self.level+1)):
            is_high = l<self.n_high
            for d in range(self.depths[l]):
                if is_high:
                    layer = MBConvBlock(opt.dims[l], opt.dims[l], opt.expand_ratios[l])
                else:
                    layer = TransformerBlock(opt.dims[l], opt.num_heads[l], opt.shrink_ratios[l], opt.mlp_ratios[l])
                setattr(self, f'block_{l}_{d}', layer)
            out_channel = opt.dims[l-1] if l!=0 else opt.color_channels
            layer = nn.Conv2d(opt.dims[l], out_channel, kernel_size=1)
            setattr(self, f'out_conv_{l}', layer)
    
    def forward(self, x, encoded):
        for l in reversed(range(self.level+1)):
            B,C,H,W = x.shape
            a = x

            if l!=self.level: x = x + encoded[l]
            
            for d in range(self.depths[l]):
                x = getattr(self, f'block_{l}_{d}')(x)
            
            x = x + a if l==self.level else x + encoded[l]
            x = getattr(self, f'out_conv_{l}')(x)
            
            if l!=0:
                target_h, target_w = encoded[l-1].shape[-2], encoded[l-1].shape[-1]
                x = F.interpolate(x, size=[target_h, target_w], mode='bilinear', align_corners=False)

        return x




class VideoDenoisingNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.inv_sigmoid = InverseSigmoid()
    
    def forward(self, x):
        #a = self.inv_sigmoid(torch.chunk(x[:,:-1,:,:], chunks=5, dim=1)[2])
        x, encoded = self.encoder(x)
        x = self.decoder(x, encoded)
        #x = a + x
        return x

# FastDVDを再現してみるよ
# Pixel Unshuffleでノイズ画像は保存されるか