import torch.nn.functional as F
import torch
from torch import nn
import torchvision

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y):
        # x,y: [B,C,H,W]
        loss = ((y-x).pow(2)+self.eps**2).sqrt()
        return torch.mean(loss)

def gradient_penalty(netD, real, fake):
    b_size = real.size(0)
    alpha = torch.rand(b_size,1,1,1).to(real.device)
    alpha = alpha.expand_as(real)
    
    interpolation = alpha*real + (1-alpha)*fake
    interpolation = torch.autograd.Variable(interpolation, requires_grad=True)
    logits = netD(interpolation)
    
    grad_outputs = torch.ones_like(logits).to(real.device)
    grads = torch.autograd.grad(
        outputs=logits, inputs=interpolation, 
        grad_outputs=grad_outputs, 
        create_graph=True, retain_graph=True, 
        only_inputs=True)[0]
    grads = grads.view(b_size, -1)
    grad_norm = grads.norm(2,1)
    return torch.mean((grad_norm-1)**2)