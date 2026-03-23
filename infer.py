# infer.py

import os
import json
import argparse

import torch

from tqdm import tqdm

# ===== Import từ project =====
from model.vid2seq import Vid2Seq
from model.utils.tokenizer import get_tokenizer
from model.utils.checkpoint import load_checkpoint
from model.utils.format import decode_prediction


# ═══════════════════════════════════════════════════════════════
# Load & preprocess video feature
# ═══════════════════════════════════════════════════════════════
def load_video_feature(path, max_feats):
    video = torch.load(path, map_location="cpu")

    if video.dim() == 3:
        video = video.squeeze(0)

    T, D = video.shape

    if T >= max_feats:
        indices = [(j * T) // max_feats for j in range(max_feats)]
        video = video[indices]
    else:
        video = torch.cat([video, torch.zeros(max_feats - T, D)], dim=0)

    return video


# ═══════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════
def infer(model, tokenizer, video, device, max_length=256):
    model.eval()
    video = video.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            video=video,
            max_length=max_length,
            num_beams=4,
            repetition_penalty=1.9,
        )

    # 🔥 FIX QUAN TRỌNG
    if isinstance(output, list):
        pred = output[0]   # đã là string rồi
    else:
        pred = tokenizer.decode(output[0], skip_special_tokens=True)

    return pred


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--t5_path", default="t5-base")
    parser.add_argument("--max_feats", type=int, default=225)
    parser.add_argument("--num_bins", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ──
    print("\n[1] Loading model...")
    tokenizer = get_tokenizer(args.t5_path, num_bins=args.num_bins)

    model = Vid2Seq(
        t5_path=args.t5_path,
        num_features=args.max_feats,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
    )

    model = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    # ── Load feature ──
    print("\n[2] Loading video feature...")
    video = load_video_feature(args.feature_path, args.max_feats)

    # ── Infer ──
    print("\n[3] Generating caption...")
    pred_text = infer(model, tokenizer, video, device)

    print("\nRaw prediction:")
    print(pred_text)

    # ── Decode ──
    print("\n[4] Decoding timestamps...")
    results = decode_prediction(
        pred_text,
        duration=args.duration,
        num_bins=args.num_bins,
    )

    print("\nFinal results:")
    for r in results:
        print(r)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()