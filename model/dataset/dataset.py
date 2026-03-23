import os
import json
import random
import torch
from torch.utils.data import Dataset
from augment import temporal_speed_jitter, boundary_emphasis, temporal_feature_dropout, gaussian_feature_noise, temporal_crop
# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════
class Vid2SeqDataset(Dataset):
    """
    JSON format:
    [
      {
        "video_id"   : "v_xxx",
        "duration"   : 82.73,
        "sentences"  : ["...", "..."],
        "timestamps" : [[s, e], [s, e]]
      }, ...
    ]
    Feature files: {feature_dir}/{video_id}.pt  → Tensor (T, 768) or (1, T, 768)
    """

    def __init__(self, data_path, tokenizer, feature_dir,
                 max_output_tokens=256, num_bins=100, max_feats=100,
                 augment=False,
                 use_speed_jitter=True,
                 use_temporal_crop=True,
                 use_gaussian_noise=True,
                 use_feature_dropout=True,
                 use_boundary_emphasis=True):
        with open(data_path) as f:
            raw = json.load(f)
        self.data = [
            item for item in raw
            if os.path.exists(os.path.join(feature_dir, self._feat_name(item)))
        ]
        skipped = len(raw) - len(self.data)
        if skipped:
            print(f"  [Dataset] Skipped {skipped} items (missing .pt files)")

        self.tokenizer         = tokenizer
        self.feature_dir       = feature_dir
        self.max_output_tokens = max_output_tokens
        self.max_feats         = max_feats
        self.num_bins          = num_bins
        self.augment               = augment
        self.use_speed_jitter      = use_speed_jitter
        self.use_temporal_crop     = use_temporal_crop
        self.use_gaussian_noise    = use_gaussian_noise
        self.use_feature_dropout   = use_feature_dropout
        self.use_boundary_emphasis = use_boundary_emphasis

    @staticmethod
    def _feat_name(item):
        if "feature"  in item: return item["feature"]
        if "video_id" in item: return item["video_id"] + ".pt"
        return os.path.splitext(item["video"])[0] + ".pt"

    def __len__(self):
        return len(self.data)

    def _load_video(self, item):
        feat_path = os.path.join(self.feature_dir, self._feat_name(item))
        video = torch.load(feat_path, map_location="cpu")
        if video.dim() == 3:
            video = video.squeeze(0)
        T, D      = video.shape
        max_feats = self.max_feats
        if T >= max_feats:
            indices = [(j * T) // max_feats for j in range(max_feats)]
            video   = video[indices]
        else:
            video = torch.cat([video, torch.zeros(max_feats - T, D)], dim=0)
        return video

    def _time_tok(self, x, duration):
        tok = int((self.num_bins - 1) * x / duration)
        return f"<time={min(tok, self.num_bins - 1)}>"

    def _build_target(self, sentences, timestamps, duration):
        """
        FIX: no space between the two time tokens.
        Format: "<time=X><time=Y> sentence <time=A><time=B> sentence ..."
        Must match decode_prediction pattern: r"<time=(\d+)><time=(\d+)>\s*([^<]+)"
        """
        parts = []
        for s, (t0, t1) in zip(sentences, timestamps):
            s_tok = self._time_tok(t0, duration)
            e_tok = self._time_tok(t1, duration)
            parts.append(f"{s_tok}{e_tok} {s}")   # ← FIX: no space between tokens
        return " ".join(parts)

    def __getitem__(self, idx):
        item      = self.data[idx]
        duration  = float(item["duration"])
        video     = self._load_video(item)
        sentences  = list(item["sentences"])
        timestamps = [list(t) for t in item["timestamps"]]

        if self.augment:
            if self.use_speed_jitter and random.random() < 0.5:
                video, timestamps, duration = temporal_speed_jitter(
                    video, timestamps, duration, speed_range=(0.8, 1.2))
            if self.use_temporal_crop and random.random() < 0.4:
                video, timestamps, duration = temporal_crop(
                    video, timestamps, duration, crop_ratio_range=(0.7, 1.0))
                sentences = sentences[:len(timestamps)]
            if self.use_gaussian_noise and random.random() < 0.5:
                video = gaussian_feature_noise(video, std=0.02)
            if self.use_feature_dropout and random.random() < 0.3:
                video = temporal_feature_dropout(video, p=0.1)
            if self.use_boundary_emphasis and random.random() < 0.5:
                video = boundary_emphasis(video, timestamps, duration, self.max_feats, window=2)

        target_str = self._build_target(sentences, timestamps, duration)
        target = self.tokenizer(
            target_str,
            max_length=self.max_output_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "video":                 video,
            "output_input_ids":      target["input_ids"].squeeze(0),
            "output_attention_mask": target["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    return {
        "video":                 torch.stack([b["video"] for b in batch]),
        "output_input_ids":      torch.stack([b["output_input_ids"] for b in batch]),
        "output_attention_mask": torch.stack([b["output_attention_mask"] for b in batch]),
    }
