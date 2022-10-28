import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import torchvision

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y):
        # x,y: [B,C,H,W]
        loss = ((y-x).pow(2)+self.eps**2).sqrt()
        return torch.mean(loss)

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

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