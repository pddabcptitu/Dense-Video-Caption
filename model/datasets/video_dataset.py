import torch
from torch.utils.data import Dataset
import os
from model.utils.convert_to_dvc import convert_activitynet


class VideoCaptionDataset(Dataset):
    def __init__(self, data, feature_dir, max_len=300, tokenizer=None):
        self.data = data
        self.feature_dir = feature_dir
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def pad_or_truncate(self, feat):
        T, D = feat.shape

        if T > self.max_len:
            feat = feat[:self.max_len]
            mask = torch.ones(self.max_len, dtype=torch.long)
        else:
            pad_len = self.max_len - T
            pad = torch.zeros(pad_len, D, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)

            # attention mask: 1 cho frame thật, 0 cho padding
            mask = torch.cat([torch.ones(T, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        return feat, mask

    def __getitem__(self, idx):
        item = self.data[idx]

        video_id = item["video_id"]
        caption = convert_activitynet(item)
        
        # FIX: Check tokenizer is initialized
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided. Pass tokenizer to VideoCaptionDataset.__init__")
        
        # FIX: Tokenize once and extract only tensor components for better multiprocessing
        tokenized = self.tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        try:
            path = os.path.join(self.feature_dir, f"{video_id}.pt")
            feat = torch.load(path)  # Tensor (T, D)

            feat, attention_mask = self.pad_or_truncate(feat)

        except Exception as e:
            print(f"Error {video_id}: {e}")
            return None

        # Return tensors instead of tokenized object for proper multiprocessing
        label_input_ids = tokenized.input_ids.squeeze(0)  # Remove batch dim
        label_attention_mask = tokenized.attention_mask.squeeze(0)  # Remove batch dim
        
        return feat, attention_mask, label_input_ids, label_attention_mask