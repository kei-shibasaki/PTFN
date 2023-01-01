import torch
from torch import nn

### https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class PseudoTemporalGate(nn.Module):
    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=1)
        return 0.5*(x1*x2 + x2*x3)

class TemporalShift(nn.Module):
    def __init__(self, n_segment, shift_type, fold_div=8, stride=1):
        super().__init__()
        self.n_segment = n_segment
        self.shift_type = shift_type
        self.fold_div = fold_div
        self.stride = stride 

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div # 32/8 = 4
        
        out = torch.zeros_like(x)
        if not 'toFutureOnly' in self.shift_type:
            out[:, :-self.stride, :fold] = x[:, self.stride:, :fold]  # backward (left shift)
            out[:, self.stride:, fold: 2 * fold] = x[:, :-self.stride, fold: 2 * fold]  # forward (right shift)
        else:
            out[:, self.stride:, : 2 * fold] = x[:, :-self.stride, : 2 * fold] # right shift only
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class MemSkip(nn.Module):
    def __init__(self):
        super(MemSkip, self).__init__()
        self.mem_list = []
    def push(self, x):
        if x is not None:
            self.mem_list.insert(0,x)
            return 1
        else:
            return 0
    def pop(self, x):
        if x is not None:
            return self.mem_list.pop()
        else:
            return None

class PseudoTemporalFusionSpatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.alpha = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(2*dim, 2*dim, kernel_size=3, padding=1, groups=2*dim)
        self.sg = nn.GELU()
        self.conv3 = nn.Conv2d(2*dim, dim, kernel_size=1, bias=True)
    
    def forward(self, x):
        a = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        return a + x

class PseudoTemporalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.alpha = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.sg = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
    
    def forward(self, x):
        a = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return a + x

class PseudoTemporalFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ptfs = PseudoTemporalFusionSpatial(dim)
        self.ptf1 = PseudoTemporalFusion(dim)
        self.ptf2 = PseudoTemporalFusion(dim)

    def forward(self, x):
        x = self.ptfs(x)
        x = self.ptf1(x)
        x = self.ptf2(x)
        return x

class ShiftPseudoTemporalFusionBlock(nn.Module):
    def __init__(self, dim, fold_div=8):
        super().__init__()
        self.ptfs = PseudoTemporalFusionSpatial(dim)
        self.ptf1 = PseudoTemporalFusion(dim)
        self.ptf2 = PseudoTemporalFusion(dim)
        self.fold_div = fold_div

    def forward(self,  left_fold_2fold, center, right):
        _, c, _, _ = center.size()
        fold = c//self.fold_div
        x = torch.cat([right[:, :fold, :, :], left_fold_2fold, center[:, 2*fold:, :, :]], dim=1)
        x = self.ptfs(x)
        x = self.ptf1(x)
        x = self.ptf2(x)
        return x

class PseudoTemporalFusionBlockBBB(nn.Module):
    def __init__(self, dim, fold_div=8):
        super().__init__()
        self.op = ShiftPseudoTemporalFusionBlock(dim, fold_div)
        self.left_fold_2fold = None
        self.center = None
        
    def reset(self):
        self.left_fold_2fold = None
        self.center = None
        
    def forward(self, input_right):
        fold_div = 8
        if input_right is not None:
            self.n, self.c, self.h, self.w = input_right.size()
            self.fold = self.c//fold_div
        # Case1: In the start or end stage, the memory is empty
        if self.center is None:
            self.center = input_right
            if input_right is not None:
                if self.left_fold_2fold is None:
                    # In the start stage, the memory and left tensor is empty
                    self.left_fold_2fold = torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda'))
            else:
                # in the end stage, both feed in and memory are empty
                pass
                # print("self.center is None")
            return None
        # Case2: Center is not None, but input_right is None
        elif input_right is None:
            # In the last procesing stage, center is 0
            output =  self.op(self.left_fold_2fold, self.center, torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda')))
        else:
            output =  self.op(self.left_fold_2fold, self.center, input_right)
        self.left_fold_2fold = self.center[:, self.fold:2*self.fold, :, :]
        self.center = input_right
        return output