from tqdm import tqdm
from torch.utils.data import DataLoader
from model.datasets.video_dataset import VideoCaptionDataset
import torch

def collate_fn(batchs):
    batchs = [b for b in batchs if b[0] is not None]

    if len(batchs) == 0:
        return None

    features = [b[0] for b in batchs]
    attention_mask = [b[1] for b in batchs]
    captions = [b[-1] for b in batchs]

    features = torch.stack(features)
    attention_mask = torch.stack(attention_mask)

    return features, attention_mask, captions

def train_one_epoch(
        model,
        tokenizer,
        optimizer,
        train_loader,
):
    total_loss = 0
    num_batches = 0
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for data in tqdm(train_loader):
        if data is None:
            continue

        optimizer.zero_grad()
        features, attention_mask, captions = data
        features.to(device)
        attention_mask.to(device)
        video = {
            'video_features': features,
            'attention_mask': attention_mask
        }
        out = model(video, captions)
        loss = out['loss']
        loss.backward()
        total_loss += loss.item()
        num_batches += 1
        optimizer.step()
    avg_loss = loss/num_batches if num_batches > 0 else 0
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, tokenizer, val_loader):
    model.eval()  # quan trọng!
    total_loss = 0
    n_batches = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():  # không tính gradient
        for data in tqdm(val_loader):
            if data is None:
                continue

            features, attention_mask, captions = data
            features = features.to(device)
            attention_mask.to(device)
            video = {
                'video_features': features,
                'attention_mask': attention_mask
            }
            outputs = model(video, captions)
            
            # với ViT + T5 model, loss được trả về là .loss
            loss = outputs.loss  
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss