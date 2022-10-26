from operator import mod
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from models.layers import ConvBNReLU, LaplacianPyramid, UNetBlock, RConvBNReLU, RDConvBNReLU
from models.layers import InputConvBlock, ConvBlock, OutputConvBlock
from models.axial_transformer import AxialTransformerBlock

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

class DenoisingBlockSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputConvBlock(num_in_frames=1, out_ch=32)
        self.down = ConvBlock(32, 64, 64, depth=2, downsample=True)
        self.bottom = ConvBlock(64, 128, 64, depth=4, downsample=True, upsample=True)
        self.up = ConvBlock(64, 64, 32, depth=2, upsample=True)
        self.out = OutputConvBlock(32, 3)
    
    def forward(self, x, noise_map):
        enc0 = self.input(torch.cat([x, noise_map], dim=1))
        enc1 = self.down(enc0)
        enc2 = self.bottom(enc1)
        enc1 = self.up(enc1+enc2)
        out = self.out(enc0+enc1)
        out = x - out
        return out

class FastDVDNetM(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = DenoisingBlockSingle()
        self.temp2 = DenoisingBlock()
        self.temp3 = DenoisingBlock()
        self.temp4 = DenoisingBlockSingle()
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        x0, x1, x2, x3, x4 = map(lambda x: self.temp1(x, noise_map), [x0, x1, x2, x3, x4])
        x20 = self.temp2(x0, x1, x2, noise_map)
        x21 = self.temp2(x1, x2, x3, noise_map)
        x22 = self.temp2(x2, x3, x4, noise_map)
        x = self.temp3(x20, x21, x22, noise_map)
        x = self.temp4(x, noise_map)

        return x


class TinyDenoisingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = ConvBNReLU(3*(3+1), 32)
        self.layer1 = ConvBNReLU(32, 64, stride=2)
        self.layer2 = ConvBNReLU(64, 128, stride=2)
        self.expand_dim1 = nn.Conv2d(128, 64*4, kernel_size=1)
        self.upsample1 = nn.PixelShuffle(2)
        self.layer3 = ConvBNReLU(64, 32)
        self.expand_dim2 = nn.Conv2d(32, 32*4, kernel_size=1)
        self.upsample2 = nn.PixelShuffle(2)
        self.layer4 = ConvBNReLU(32, 32)
        self.to_out = nn.Conv2d(32, 3, kernel_size=1)
    
    def forward(self, x0, x1, x2, noise_map):
        enc0 = self.layer0(torch.cat([x0, noise_map, x1, noise_map, x2, noise_map], dim=1))
        enc1 = self.layer1(enc0)
        x = self.layer2(enc1)
        x = self.expand_dim1(x)
        x = self.upsample1(x)
        x = self.layer3(x+enc1)
        x = self.expand_dim2(x)
        x = self.upsample2(x)
        x = self.layer4(x+enc0)
        x = self.to_out(x)

        return x1 - x

class TinyDenoisingBlockSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = ConvBNReLU(3+1, 32)
        self.layer1 = ConvBNReLU(32, 64, stride=2)
        self.layer2 = ConvBNReLU(64, 128, stride=2)
        self.expand_dim1 = nn.Conv2d(128, 64*4, kernel_size=1)
        self.upsample1 = nn.PixelShuffle(2)
        self.layer3 = ConvBNReLU(64, 32)
        self.expand_dim2 = nn.Conv2d(32, 32*4, kernel_size=1)
        self.upsample2 = nn.PixelShuffle(2)
        self.layer4 = ConvBNReLU(32, 32)
        self.to_out = nn.Conv2d(32, 3, kernel_size=1)
    
    def forward(self, x0, noise_map):
        enc0 = self.layer0(torch.cat([x0, noise_map], dim=1))
        enc1 = self.layer1(enc0)
        x = self.layer2(enc1)
        x = self.expand_dim1(x)
        x = self.upsample1(x)
        x = self.layer3(x+enc1)
        x = self.expand_dim2(x)
        x = self.upsample2(x)
        x = self.layer4(x+enc0)
        x = self.to_out(x)

        return x0 - x

class ExtremeStageDenoisingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth1 = 5
        self.depth2 = 5
        self.depth3 = 5
        for d in range(self.depth1):
            setattr(self, f'block1_{d}', TinyDenoisingBlockSingle())
        self.fusion1 = TinyDenoisingBlock()
        for d in range(self.depth2):
            setattr(self, f'block2_{d}', TinyDenoisingBlock())
        self.fusion2 = TinyDenoisingBlock()
        for d in range(self.depth3):
            setattr(self, f'block3_{d}', TinyDenoisingBlockSingle())
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        for d in range(self.depth1):
            layer = getattr(self, f'block1_{d}')
            x0, x1, x2, x3, x4 = map(lambda x: layer(x, noise_map), [x0, x1, x2, x3, x4])
        
        x20 = self.fusion1(x0, x1, x2, noise_map)
        x21 = self.fusion1(x1, x2, x3, noise_map)
        x22 = self.fusion1(x2, x3, x4, noise_map)

        for d in range(self.depth2):
            layer = getattr(self, f'block2_{d}')
            x20 = layer(x20, x20, x21, noise_map)
            x21 = layer(x20, x20, x22, noise_map)
            x22 = layer(x21, x22, x22, noise_map)
        
        x = self.fusion2(x20, x21, x22, noise_map)

        for d in range(self.depth3):
            layer = getattr(self, f'block3_{d}')
            x = layer(x, noise_map)

        return x

class ExtremeStageDenoisingNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth1 = 3
        self.depth2 = 3
        self.depth3 = 3
        for d in range(self.depth1):
            setattr(self, f'block1_{d}', DenoisingBlockSingle())
        self.fusion1 = DenoisingBlock()
        for d in range(self.depth2):
            setattr(self, f'block2_{d}', DenoisingBlock())
        self.fusion2 = DenoisingBlock()
        for d in range(self.depth3):
            setattr(self, f'block3_{d}', DenoisingBlockSingle())
    
    def forward(self, x0, x1, x2, x3, x4, noise_map):
        for d in range(self.depth1):
            layer = getattr(self, f'block1_{d}')
            x0, x1, x2, x3, x4 = map(lambda x: layer(x, noise_map), [x0, x1, x2, x3, x4])
        
        x20 = self.fusion1(x0, x1, x2, noise_map)
        x21 = self.fusion1(x1, x2, x3, noise_map)
        x22 = self.fusion1(x2, x3, x4, noise_map)

        for d in range(self.depth2):
            layer = getattr(self, f'block2_{d}')
            x20 = layer(x20, x20, x21, noise_map)
            x21 = layer(x20, x20, x22, noise_map)
            x22 = layer(x21, x22, x22, noise_map)
        
        x = self.fusion2(x20, x21, x22, noise_map)

        for d in range(self.depth3):
            layer = getattr(self, f'block3_{d}')
            x = layer(x, noise_map)

        return x