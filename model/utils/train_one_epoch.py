import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, grad_clip=1.0):
    model.train()
    total_loss = total_cap = total_con = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        video    = batch["video"].to(device)
        out_ids  = batch["output_input_ids"].to(device)
        out_mask = batch["output_attention_mask"].to(device)

        out  = model.forward(
            video=video,
            output_tokenized={"input_ids": out_ids, "attention_mask": out_mask},
        )
        loss = out["loss"].mean()
        cap  = out["caption_loss"].mean().item()
        con  = out["contrastive_loss"].mean().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_cap  += cap
        total_con  += con
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cap=f"{cap:.4f}",
            con=f"{con:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    n = len(loader)
    return total_loss / n, total_cap / n, total_con / n
