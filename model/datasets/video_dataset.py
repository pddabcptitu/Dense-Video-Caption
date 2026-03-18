import torch
from torch.utils.data import Dataset
import os

class VideoCaptionDataset(Dataset):
    def __init__(self, data, feature_dir, max_len=300):
        self.data = data
        self.feature_dir = feature_dir
        self.max_len = max_len

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
        caption = item["caption"]

        try:
            path = os.path.join(self.feature_dir, f"{video_id}.pt")
            feat = torch.load(path)  # Tensor (T, D)

            feat, attention_mask = self.pad_or_truncate(feat)

        except Exception as e:
            print(f"Error {video_id}: {e}")
            return None

        return feat, attention_mask, caption