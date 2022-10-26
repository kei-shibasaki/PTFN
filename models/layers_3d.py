from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pconv3d = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.dconv3d = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
    
    def forward(self, x):
        x = self.pconv3d(x)
        x = self.dconv3d(x)
        return x

class ShrinkedMultiDeconvHeadAttention3d(nn.Module):
    def __init__(self, dim, num_heads, r):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.to_qkv1 = nn.Conv3d(dim, 3*(dim//r), kernel_size=1)
        self.to_qkv2 = nn.Conv3d(3*(dim//r), 3*(dim//r), kernel_size=3, padding=1, groups=3*(dim//r))
        self.to_out = nn.Conv3d(dim//r, dim, kernel_size=1)
        self.merge_heads = lambda x: rearrange(x, 'b (head c) d h w -> b head c (d h w)', head=num_heads)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.to_qkv2(self.to_qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q,k,v = map(self.merge_heads, [q,k,v])
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, h=H, w=W)
        out = self.to_out(out)
        return out

class SELayer3d(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim//reduction), 
            nn.SiLU(inplace=True),
            nn.Linear(dim//reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B,C,*_ = x.shape
        a = self.avg_pool(x).view(B,C)
        a = self.fc(a).view(B,C,1,1,1)
        return a*x

class MBConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio):
        super().__init__()
        self.identity = in_channels==out_channels
        hidden_dim = round(in_channels*expand_ratio)
        self.mbconv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False), 
            nn.BatchNorm3d(hidden_dim), 
            nn.SiLU(inplace=True), 
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, groups=hidden_dim), 
            nn.BatchNorm3d(hidden_dim), 
            nn.SiLU(inplace=True), 
            SELayer3d(hidden_dim), 
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
    
    def forward(self, x):
        return x + self.mbconv(x) if self.identity else self.mbconv(x)

class FusedMBConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio):
        super().__init__()
        self.identity = in_channels==out_channels
        hidden_dim = round(in_channels*expand_ratio)
        self.mbconv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False), 
            nn.BatchNorm3d(hidden_dim), 
            nn.SiLU(inplace=True), 
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm3d(hidden_dim), 
            nn.SiLU(inplace=True), 
            SELayer3d(hidden_dim), 
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
    
    def forward(self, x):
        return x + self.mbconv(x) if self.identity else self.mbconv(x)

# Unused
# https://github.com/csjliang/LPTN/blob/main/codes/models/archs/LPTN_arch.py
class LaplacianPyramid3d(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.kernel = self.gauss_kernel()
    
    def gauss_kernel(self, channels=3, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel
    
    def downsample(self, x):
        return x[:,:,::2,::2]
    
    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)
    
    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def pyramid_decom(self, img):
        B, C, D, H, W = img.shape
        img = img.reshape(B*D,C,H,W)
        current = img
        pyr = []
        for i in range(self.level):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = F.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            _,_,h,w = diff.shape
            pyr.append(diff.reshape(B,C,D,h,w))
            current = down
        _,_,h,w = current.shape
        pyr.append(current.reshape(B,C,D,h,w))
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        b,c,d,h,w = image.shape
        image = image.reshape(b*d,c,h,w)
        for level in reversed(pyr[:-1]):
            b,c,d,h,w = image.shape
            level = level.reshape(b*d,c,h,w)
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = F.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        _,c,h,w = image.shape
        return image.reshape(b,c,d,h,w)
    




# Unused
class SortAndMask(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sort_tensor(self, x):
        B,C,D,H,W = x.shape
        b_idx = torch.arange(B).unsqueeze(1).repeat((1,C)).flatten()
        val_mean = torch.mean(x.abs(), dim=(2,3,4))
        c_idx = torch.argsort(val_mean, dim=1, descending=True).flatten()
        return x[b_idx, c_idx, ...].reshape(*x.shape)
    
    def create_mask(self, x, exist_ratio=0.75):
        B,C,D,H,W = x.shape
        n_exist = int(C*exist_ratio)
        n_cut = C - n_exist
        mask1 = torch.ones(size=(1,n_exist,1,1,1), device=x.device)
        mask2 = torch.zeros(size=(1,n_cut,1,1,1), device=x.device)
        return torch.cat([mask1, mask2], dim=1).to(x.dtype)
    
    def forward(self, x, exist_ratio):
        x = self.sort_tensor(x)
        mask = self.create_mask(x, exist_ratio)
        return x*mask
