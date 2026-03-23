import torch
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval", leave=False):
            video    = batch["video"].to(device)
            out_ids  = batch["output_input_ids"].to(device)
            out_mask = batch["output_attention_mask"].to(device)
            loss = model.forward(
                video=video,
                output_tokenized={"input_ids": out_ids, "attention_mask": out_mask},
            )["loss"]
            total_loss += loss.mean().item()
    return total_loss / len(loader)