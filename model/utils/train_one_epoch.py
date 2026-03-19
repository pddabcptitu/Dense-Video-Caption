from tqdm import tqdm
from torch.utils.data import DataLoader
from model.datasets.video_dataset import VideoCaptionDataset
import torch

def collate_fn(batchs):
    if batchs is None:
      return None
    batchs = [b for b in batchs if b is not None]

    if len(batchs) == 0:
        return None

    features = [b[0] for b in batchs]
    feature_masks = [b[1] for b in batchs]
    label_input_ids = [b[2] for b in batchs]
    label_attention_masks = [b[3] for b in batchs]

    features = torch.stack(features)
    feature_masks = torch.stack(feature_masks)
    
    # Pad label sequences to same length
    label_input_ids = torch.nn.utils.rnn.pad_sequence(
        label_input_ids, batch_first=True, padding_value=-100
    )
    label_attention_masks = torch.nn.utils.rnn.pad_sequence(
        label_attention_masks, batch_first=True, padding_value=0
    )

    return features, feature_masks, label_input_ids, label_attention_masks

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
        features, feature_masks, label_input_ids, label_attention_masks = data
        features = features.to(device)
        feature_masks = feature_masks.to(device)
        label_input_ids = label_input_ids.to(device)
        label_attention_masks = label_attention_masks.to(device)
        video = {
            'video_features': features,
            'attention_mask': feature_masks
        }
        out = model(video, label_input_ids, label_attention_masks)
        loss = out['loss']
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss += loss.item()
        num_batches += 1
        optimizer.step()

    avg_loss = total_loss/num_batches if num_batches > 0 else 0
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

            features, feature_masks, label_input_ids, label_attention_masks = data
            features = features.to(device)
            feature_masks = feature_masks.to(device)
            label_input_ids = label_input_ids.to(device)
            label_attention_masks = label_attention_masks.to(device)
            video = {
                'video_features': features,
                'attention_mask': feature_masks
            }
            outputs = model(video, label_input_ids, label_attention_masks)
            
            # với ViT + T5 model, loss được trả về là .loss
            loss = outputs['loss']
            loss = loss.mean()
            
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss