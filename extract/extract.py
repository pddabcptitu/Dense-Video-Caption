import torch
import os
from extract.video_loader import VideoLoader
from extract.model_video_extract import VideoExtract

def extract_and_save(video_paths, model_name='ViT-L-14', output_dim=768, batch_size=128, size=(224, 224), save_dir='features_output', target_fps=2):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loader = VideoLoader(video_paths, fps=target_fps, save_dir=save_dir)
    extractor = VideoExtract(model_name=model_name, output_dim=output_dim, batch_size=batch_size, size=size)

    for i, v in enumerate(loader):
        features = extractor(v) 

        video_name = os.path.basename(video_paths[i]).split('.')[0]
        save_path = os.path.join(save_dir, f"{video_name}.pt")

        torch.save(features, save_path)
        
        # print(f"Đã lưu feature của {video_name} - Shape: {features.shape}")

    print(f"--- Hoàn thành! Toàn bộ feature nằm trong: {save_dir} ---")