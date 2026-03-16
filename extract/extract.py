import torch
import os
from extract.video_loader import VideoLoader
from extract.model_video_extract import VideoExtract

def extract_and_save(video_paths, save_dir='features_output'):
    # Tạo thư mục lưu trữ nếu chưa có
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Khởi tạo loader và extractor
    # batch_size=1 để đảm bảo mỗi lần loop là trọn vẹn 1 video
    loader = VideoLoader(video_paths)
    extractor = VideoExtract()

    for i, v in enumerate(loader):
        # Vì DataLoader batch_size=1 trả về shape (1, N, 3, 224, 224)
        # Ta lấy v[0] để đưa về (N, 3, 224, 224) cho extractor
        features = extractor(v[0]) 

        # Lấy tên file gốc để đặt tên cho file feature (ví dụ: video1.pt)
        video_name = os.path.basename(video_paths[i]).split('.')[0]
        save_path = os.path.join(save_dir, f"{video_name}.pt")

        # Lưu lại dưới dạng file PyTorch (.pt) cực nhẹ
        torch.save(features, save_path)
        
        print(f"Đã lưu feature của {video_name} - Shape: {features.shape}")

    print(f"--- Hoàn thành! Toàn bộ feature nằm trong: {save_dir} ---")