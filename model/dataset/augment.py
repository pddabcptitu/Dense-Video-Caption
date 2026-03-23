import torch
import random

# ═══════════════════════════════════════════════════════════════
# Augmentation utilities
# ═══════════════════════════════════════════════════════════════
def temporal_speed_jitter(video, timestamps, duration, speed_range=(0.8, 1.2)):
    T, D   = video.shape
    speed  = random.uniform(*speed_range)
    new_T  = max(1, int(T / speed))
    indices = torch.linspace(0, T - 1, new_T).long().clamp(0, T - 1)
    video  = video[indices]
    if new_T < T:
        video = torch.cat([video, torch.zeros(T - new_T, D)], dim=0)
    elif new_T > T:
        video = video[:T]
    new_dur = duration / speed
    ts_aug  = [[min(s / speed, new_dur), min(e / speed, new_dur)] for s, e in timestamps]
    return video, ts_aug, new_dur


def temporal_crop(video, timestamps, duration, crop_ratio_range=(0.7, 1.0)):
    T, D     = video.shape
    ratio    = random.uniform(*crop_ratio_range)
    crop_len = max(1, int(T * ratio))
    start_f  = random.randint(0, T - crop_len)
    end_f    = start_f + crop_len
    video_crop = video[start_f:end_f]
    if crop_len < T:
        video_crop = torch.cat([video_crop, torch.zeros(T - crop_len, D)], dim=0)
    t_start  = start_f / T * duration
    t_end    = end_f   / T * duration
    dur_crop = t_end - t_start
    ts_crop  = []
    for s, e in timestamps:
        s_new = max(s - t_start, 0.0)
        e_new = min(e - t_start, dur_crop)
        if e_new > s_new + 0.1:
            ts_crop.append([s_new, e_new])
    if not ts_crop:
        return video, timestamps, duration
    return video_crop, ts_crop, dur_crop


def gaussian_feature_noise(video, std=0.02):
    return video + torch.randn_like(video) * std


def temporal_feature_dropout(video, p=0.1):
    mask = (torch.rand(video.shape[0]) > p).float().unsqueeze(-1)
    return video * mask


def boundary_emphasis(video, timestamps, duration, num_feats, window=2):
    video = video.clone()
    for seg in timestamps:
        for t_sec in seg:
            idx = int(t_sec / duration * (num_feats - 1))
            for delta in range(-window, window + 1):
                i        = max(0, min(num_feats - 1, idx + delta))
                neighbour = max(0, min(num_feats - 1, idx))
                video[i] = (video[i] + video[neighbour]) * 0.5
    return video


