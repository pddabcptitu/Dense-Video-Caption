import torch
import os

def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    return model

def save_checkpoint(model, out, epoch):
    os.makedirs(out, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch},
               f"{out}/checkpoint_{epoch}.pth")