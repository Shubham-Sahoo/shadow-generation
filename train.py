import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .dataset import ShadowSynthDataset
from .model import ModifiedSDXLUNet
from .losses import rectified_flow_loss, PerceptualLoss

# Configuration
BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
TOTAL_STEPS = 100000
PERCEPTUAL_LOSS_WEIGHT = 0.1

def train_step(model, data, optimizer, perceptual_loss, device):
    model.train()
    object_image, object_mask, light_params, shadow_map = [x.to(device) for x in data]
    
    shadow_map_rgb = shadow_map.repeat(1, 3, 1, 1)
    light_map = light_params.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    light_map = light_map[:, :3, :, :]
    
    x0 = torch.randn_like(shadow_map_rgb)
    x1 = shadow_map_rgb
    t = torch.rand(x0.size(0), device=device)
    xt = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
    
    input_x = torch.cat([object_image, object_mask, light_map, t[:, None, None, None]], dim=1)
    
    dummy_cond = torch.zeros(x0.size(0), 2, 768, device=device)
    pred = model(input_x, t, dummy_cond)
    
    rf_loss = rectified_flow_loss(pred, x0, x1, t)
    perc_loss = perceptual_loss(pred, x1)
    loss = rf_loss + PERCEPTUAL_LOSS_WEIGHT * perc_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(rank: int, world_size: int, data_dir: str):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    
    model = ModifiedSDXLUNet().to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    perceptual_loss = PerceptualLoss().to(device)
    
    dataset = ShadowSynthDataset(data_dir=data_dir, image_size=IMAGE_SIZE)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / WARMUP_STEPS, 1.0))
    
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        for step, data in enumerate(dataloader):
            loss = train_step(model, data, optimizer, perceptual_loss, device)
            scheduler.step()
            
            if rank == 0 and step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss:.4f}")
            
            if step >= TOTAL_STEPS:
                break
    
    dist.destroy_process_group()