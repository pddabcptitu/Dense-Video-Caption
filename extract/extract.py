import torch
import os
from extract.video_loader import VideoLoader
from torch.utils.data import DataLoader
from extract.model_video_extract import VideoExtract
from tqdm import tqdm
from extract.utils import CURD_driver

def extract_and_save(
    video_paths=None,
    model_name='ViT-L-14',
    output_dim=768,
    batch_size=128,
    size=(224, 224),
    save_dir='features_output',
    target_fps=2,
    start=0,
    end=-1,
    is_local=True
):
    os.makedirs(save_dir, exist_ok=True)
    if is_local:
        exists_paths = set(os.listdir(save_dir))
    else:
        exists_paths = set(CURD_driver.list_all_files_with_id('1lF_AiDorN7UpDE-W5_DHDUg4EX9sT4SO').keys())

    if is_local:
        end = min(end, len(video_paths))
        video_paths = video_paths[start:end]
        video_paths = [
            path for path in video_paths
            if os.path.splitext(os.path.basename(path))[0] + '.pt' not in exists_paths and not os.path.splitext(video_path)[-1] == '.webm'
        ]
    else:
        video_paths = CURD_driver.list_all_files_with_id('14yuk3BTCVgqsWJPSpaxMDu2Lmv7LpjjS')
        tmp_paths = list(video_paths.keys())
        end = min(end, len(tmp_paths))
        tmp_paths = tmp_paths[start:end]
        video_paths = {
            video_path: video_paths[video_path]
            for video_path in tmp_paths
            if os.path.splitext(video_path)[0] + '.pt' not in exists_paths and not os.path.splitext(video_path)[-1] == '.webm'
        }

    loader = VideoLoader(video_paths, fps=target_fps, is_local=is_local)
    def collate_fn(batch):
        batch = [x for x in batch if x[0] is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    dataloader = DataLoader(loader, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    extractor = VideoExtract(
        model_name=model_name,
        output_dim=output_dim,
        batch_size=batch_size,
        size=size
    )
    if not is_local:
        video_paths = list(video_paths.keys())
    for v in tqdm(dataloader):
        try:
            if v is None:
                continue

            frames, names = v
            frames = frames[0]
            names = names[0]

            with torch.no_grad():
                features = extractor(frames)

            video_name = os.path.basename(names).split('.')[0]
            save_path = os.path.join(save_dir, f"{video_name}.pt")

            torch.save(features.cpu(), save_path)

            if not is_local:
                CURD_driver.upload_file(save_path, '1lF_AiDorN7UpDE-W5_DHDUg4EX9sT4SO')
                os.remove(save_path)

        except Exception as e:
            print(f"❌ Skip video: {names}")
            print(f"Error: {e}")
            continue
    print(f"--- Hoàn thành! Toàn bộ feature nằm trong: {save_dir} ---")

def extract(
    video_paths,
    model_name='ViT-L-14',
    output_dim=768,
    batch_size=128,
    size=(224, 224),
    target_fps=2
):

    loader = VideoLoader(video_paths, fps=target_fps)
    dataloader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=4, pin_memory=True  )

    extractor = VideoExtract(
        model_name=model_name,
        output_dim=output_dim,
        batch_size=batch_size,
        size=size
    )

    features_video = []
    for i, v in enumerate(tqdm(dataloader)):
        try:
            with torch.no_grad():
                features = extractor(v[0])
            features_video.append(features)
        except Exception as e:
            print(f"❌ Skip video: {video_paths[i]}")
            print(f"Error: {e}")
            continue

    print(f"--- Hoàn thành! ---")
    return features_video