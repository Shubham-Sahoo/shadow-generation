import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ModifiedSDXLUNet(nn.Module):
    def __init__(self, pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Load pre-trained SDXL U-Net and remove cross-attention blocks.
        Input: Concatenated object image (3), mask (1), light param map (3), time (1).
        Output: RGB shadow map (3 channels, first channel used at inference).
        """
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet", torch_dtype=torch.float16
        )
        
        for block in self.unet.down_blocks + self.unet.mid_block.attentions + self.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    if hasattr(attn, "transformer_blocks"):
                        attn.transformer_blocks = nn.ModuleList([])
        
        self.unet.conv_in = nn.Conv2d(8, self.unet.conv_in.out_channels, kernel_size=3, padding=1)
        self.unet.conv_out = nn.Conv2d(
            self.unet.conv_out.in_channels, 3, kernel_size=3, padding=1
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, dummy_cond: torch.Tensor = None) -> torch.Tensor:
        t = t.view(-1)
        return self.unet(x, t, dummy_cond).sample