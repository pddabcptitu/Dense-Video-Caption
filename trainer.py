# train.py

import os
import json
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

from model.dataset.dataset import Vid2SeqDataset, collate_fn
from model.vid2seq import Vid2Seq
from model.utils.checkpoint import load_checkpoint, save_checkpoint
from model.utils.tokenizer import get_tokenizer
from model.utils.train_one_epoch import train_one_epoch
from model.utils.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data",        required=True)
    parser.add_argument("--test_data",         required=True)
    parser.add_argument("--feature_dir",       required=True)
    parser.add_argument("--checkpoint",        required=True)

    parser.add_argument("--t5_path",           default="t5-base")
    parser.add_argument("--output_dir",        default="checkpoints")

    parser.add_argument("--epochs",            type=int,   default=30)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=3e-5)
    parser.add_argument("--warmup_ratio",      type=float, default=0.1)
    parser.add_argument("--patience",          type=int,   default=5)

    parser.add_argument("--contrastive_weight",type=float, default=0.1)
    parser.add_argument("--max_feats",         type=int,   default=100)
    parser.add_argument("--num_bins",          type=int,   default=100)
    parser.add_argument("--max_output_tokens", type=int,   default=256)

    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_workers",       type=int,   default=2)

    parser.add_argument("--no_augment",            action="store_true")
    parser.add_argument("--no_speed_jitter",       action="store_true")
    parser.add_argument("--no_temporal_crop",      action="store_true")
    parser.add_argument("--no_gaussian_noise",     action="store_true")
    parser.add_argument("--no_feature_dropout",    action="store_true")
    parser.add_argument("--no_boundary_emphasis",  action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    
    print("\n[1] Building model...")
    tokenizer = get_tokenizer(args.t5_path, num_bins=args.num_bins)

    model = Vid2Seq(
        t5_path=args.t5_path,
        num_features=args.max_feats,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        vis_drop=0.1,
        contrastive_weight=args.contrastive_weight,
    )

    model = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    augment = not args.no_augment
    print(f"\n[2] Loading datasets... (augmentation={'ON' if augment else 'OFF'})")

    train_ds = Vid2SeqDataset(
        args.train_data, tokenizer, args.feature_dir,
        args.max_output_tokens, args.num_bins, args.max_feats,
        augment=augment,
        use_speed_jitter      = not args.no_speed_jitter,
        use_temporal_crop     = not args.no_temporal_crop,
        use_gaussian_noise    = not args.no_gaussian_noise,
        use_feature_dropout   = not args.no_feature_dropout,
        use_boundary_emphasis = not args.no_boundary_emphasis,
    )

    test_ds = Vid2SeqDataset(
        args.test_data, tokenizer, args.feature_dir,
        args.max_output_tokens, args.num_bins, args.max_feats,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\n[3] Training...")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        train_loss, train_cap, train_con = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        test_loss = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.4f} (cap={train_cap:.4f}, con={train_con:.4f}) | "
            f"test={test_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        save_checkpoint(model, args.output_dir, epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            save_checkpoint(model, args.output_dir, epoch, tag="best")
            print(f"*** New best: {best_loss:.4f} ***")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()