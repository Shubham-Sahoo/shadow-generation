import os
import torch
import torch.multiprocessing as mp
from train import train
from inference import inference
from model import ModifiedSDXLUNet

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    world_size = 8
    data_dir = "path/to/shadowsynth1m"
    
    # Run training
    mp.spawn(train, args=(world_size, data_dir), nprocs=world_size, join=True)
    
    # Example inference
    """
    device = torch.device("cuda:0")
    model = ModifiedSDXLUNet().to(device)
    object_image = torch.rand(1, 3, 256, 256).to(device)
    object_mask = torch.rand(1, 1, 256, 256).to(device)
    light_params = torch.tensor([[0.5, 0.5, 1.0]]).to(device)  # Normalized
    background = torch.rand(1, 3, 256, 256).to(device)
    output = inference(model, object_image, object_mask, light_params, background, device)
    print("Inference output shape:", output.shape)
    """

if __name__ == "__main__":
    main()