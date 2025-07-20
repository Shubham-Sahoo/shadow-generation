import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_vgg = self.vgg_layers(pred)
        target_vgg = self.vgg_layers(target)
        return F.l1_loss(pred_vgg, target_vgg)

def rectified_flow_loss(pred: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Rectified Flow loss: ||(x1 - x0) - fψ(xt, t)||^2
    """
    target = x1 - x0
    return F.mse_loss(pred, target)