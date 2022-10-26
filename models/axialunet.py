import torch
from torch import nn
from torch.nn import functional as F
from models.axial_transformer import AxialTransformerBlock
from models.positional_encoding import ConditionalPositionalEncoding

class ResBlock(nn.Module):
    def __init__(self, channels, ksize, mode='BRC'):
        super().__init__()
        self.mode = mode
        layer_dict = {
            'C': nn.Conv2d(channels, channels, ksize, padding=ksize//2), 
            'R': nn.ReLU(inplace=True),
            'B': nn.BatchNorm2d(channels),
        }
        layers = []
        for m in mode:
            layers.append(layer_dict[m])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.layers(x)

class ResBlocks(nn.Module):
    def __init__(self, depth, in_channels, out_channels, ksize):
        super().__init__()
        self.depth = depth

        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        for i in range(self.depth):
            block = ResBlock(out_channels, ksize, mode='CBR')
            setattr(self, f'block_{i}', block)
    
    def forward(self, x):
        x = self.expand_dims(x)
        for i in range(self.depth):
            block = getattr(self, f'block_{i}')
            x = block(x)
        return x

class AxialTransformerEncoders(nn.Module):
    def __init__(self, depth, in_channels, out_channels, cpek, heads, mlp_ratio):
        super().__init__()
        self.depth = depth
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cpe = ConditionalPositionalEncoding(out_channels, k=cpek)
        for i in range(depth):
            block = AxialTransformerBlock(out_channels, heads=heads, mlp_ratio=mlp_ratio)
            setattr(self, f'block_{i}', block)
    
    def forward(self, x):
        x = self.expand_dims(x)
        x = self.cpe(x)
        for i in range(self.depth):
            block = getattr(self, f'block_{i}')
            x = block(x)
        return x

class AxialUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.expand_dims = nn.Conv2d(opt.color_channels, opt.dims_cnn[0], kernel_size=1)

        # Encoder
        for i in range(opt.l_cnn):
            block = ResBlocks(opt.depth, opt.dims_cnn[i] if i==0 else opt.dims_cnn[i-1], opt.dims_cnn[i], opt.ksizes_cnn[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_cnn_block_{i}', block)
            setattr(self, f'enc_cnn_pool_{i}', pool)
        
        for i in range(opt.l_trans):
            block = AxialTransformerEncoders(opt.depth, opt.dims_cnn[-1] if i==0 else opt.dims_trans[i-1], opt.dims_trans[i], opt.ksizes_cpe[i], opt.heads_trans[i], opt.mlp_ratios_trans[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_trans_block_{i}', block)
            setattr(self, f'enc_trans_pool_{i}', pool)

        # Bottom
        block = AxialTransformerEncoders(opt.depth, opt.dims_trans[-1], opt.dims_trans[-1], opt.ksizes_cpe[-1], opt.heads_trans[-1], opt.mlp_ratios_trans[-1])
        setattr(self, f'bottom_block_{i}', block)

        # Decoder
        for i in range(opt.l_trans):
            idx = opt.l_trans-1-i
            in_channels = 2*opt.dims_trans[idx]
            out_channels = opt.dims_trans[idx-1] if i!=opt.l_trans-1 else opt.dims_cnn[-1]
            block = AxialTransformerEncoders(opt.depth, in_channels, out_channels, opt.ksizes_cpe[idx], opt.heads_trans[idx], opt.mlp_ratios_trans[idx])
            setattr(self, f'dec_trans_block_{i}', block)
        
        for i in range(opt.l_cnn):
            idx = idx = opt.l_cnn-1-i
            in_channels = 2*opt.dims_cnn[idx]
            out_channels = opt.dims_cnn[idx-1] if i!=opt.l_cnn-1 else opt.dims_cnn[0]
            block = ResBlocks(opt.depth, in_channels, out_channels, opt.ksizes_cnn[idx])
            setattr(self, f'dec_cnn_block_{i}', block)
        
        self.to_out = nn.Conv2d(opt.dims_cnn[0], opt.color_channels, opt.ksizes_cnn[0], stride=1, padding=opt.ksizes_cnn[0]//2)
    
    def forward(self, x):
        x = self.expand_dims(x)

        encoded_cnn = []
        encoded_trans = []
        for i in range(self.opt.l_cnn):
            block = getattr(self, f'enc_cnn_block_{i}')
            pool = getattr(self, f'enc_cnn_pool_{i}')
            x = block(x)
            encoded_cnn.append(x)
            x = pool(x)
        for i in range(self.opt.l_trans):
            block = getattr(self, f'enc_trans_block_{i}')
            pool = getattr(self, f'enc_trans_pool_{i}')
            x = block(x)
            encoded_trans.append(x)
            x = pool(x)
        
        block = getattr(self, f'bottom_block_{i}')
        x = block(x)
    
        for i in range(self.opt.l_trans):
            idx = self.opt.l_trans-1-i
            block = getattr(self, f'dec_trans_block_{i}')
            x = F.interpolate(x, size=(encoded_trans[idx].size(2), encoded_trans[idx].size(3)), mode='bilinear', align_corners=False)
            x = torch.cat([x, encoded_trans[idx]], dim=1)
            x = block(x)
        for i in range(self.opt.l_cnn):
            idx = self.opt.l_cnn-1-i
            block = getattr(self, f'dec_cnn_block_{i}')
            x = F.interpolate(x, size=(encoded_cnn[idx].size(2), encoded_cnn[idx].size(3)), mode='bilinear', align_corners=False)
            x = torch.cat([x, encoded_cnn[idx]], dim=1)
            x = block(x)

        x = self.to_out(x)

        return x