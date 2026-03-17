import torch
import os
from extract.video_loader import VideoLoader
from torch.utils.data import DataLoader
from extract.model_video_extract import VideoExtract
from tqdm import tqdm

def extract_and_save(
    video_paths,
    model_name='ViT-L-14',
    output_dim=768,
    batch_size=128,
    size=(224, 224),
    save_dir='features_output',
    target_fps=2
):
    os.makedirs(save_dir, exist_ok=True)

    loader = VideoLoader(video_paths, fps=target_fps)
    dataloader = DataLoader(loader, batch_size=1, shuffle=False)

    extractor = VideoExtract(
        model_name=model_name,
        output_dim=output_dim,
        batch_size=batch_size,
        size=size
    )

    for i, v in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            features = extractor(v[0])

        video_name = os.path.basename(video_paths[i]).split('.')[0]
        save_path = os.path.join(save_dir, f"{video_name}.pt")

        torch.save(features.cpu(), save_path)

    print(f"--- Hoàn thành! Toàn bộ feature nằm trong: {save_dir} ---")