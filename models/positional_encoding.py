from torch import nn

class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        assert k%2==1, 'k must be odd.'
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k//2, groups=dim)
    
    def forward(self, x):
        assert len(x.shape)==4, 'Input shape must be (B, C, H, W)'
        return x + self.proj(x)