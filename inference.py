import torch
from .model import ModifiedSDXLUNet

def inference(model, object_image: torch.Tensor, object_mask: torch.Tensor, light_params: torch.Tensor, background: torch.Tensor, device: torch.device, image_size: int = 256) -> torch.Tensor:
    """
    Inference pipeline following section 3.2.3 of the paper.
    """
    model.eval()
    with torch.no_grad():
        light_map = light_params.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, image_size, image_size)
        light_map = light_map[:, :3, :, :]
        t = torch.ones(object_image.size(0), device=device)
        input_x = torch.cat([object_image, object_mask, light_map, t[:, None, None, None]], dim=1).to(device)
        
        dummy_cond = torch.zeros(object_image.size(0), 2, 768, device=device)
        pred = model(input_x, t, dummy_cond)
        
        shadow_map = pred[:, 0:1, :, :]
        shadow_map = 1 - shadow_map
        
        output = object_image * object_mask + background * (1 - object_mask) * shadow_map
        return output