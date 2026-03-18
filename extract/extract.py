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
    is_local=True
):
    os.makedirs(save_dir, exist_ok=True)
    if is_local:
        exists_paths = set(os.listdir(save_dir))
    else:
        exists_paths = set(CURD_driver.list_all_files_with_id('1lF_AiDorN7UpDE-W5_DHDUg4EX9sT4SO').keys())

    if is_local:
        video_paths = [
            path for path in video_paths
            if os.path.basename(path).split('.')[0] + '.pt' not in exists_paths
        ]
    else:
        video_paths = CURD_driver.list_all_files_with_id('14yuk3BTCVgqsWJPSpaxMDu2Lmv7LpjjS')
        video_paths = {video_path:video_paths[video_path] for video_path in video_paths if (video_path.split('.')[0] + 'pt') not in exists_paths}

    loader = VideoLoader(video_paths, fps=target_fps, is_local=is_local)
    def collate_fn(batch):
        batch = [x for x in batch if x[0] is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    dataloader = DataLoader(loader, batch_size=1, shuffle=False, collate_fn=collate_fn)

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

            for feat, name in zip(features, names):
                video_name = os.path.basename(name).split('.')[0]
                save_path = os.path.join(save_dir, f"{video_name}.pt")

                torch.save(feat.cpu(), save_path)

                if not is_local:
                    CURD_driver.upload_file(save_path, '1lF_AiDorN7UpDE-W5_DHDUg4EX9sT4SO')
                    os.remove(save_path)

        except Exception as e:
            print(f"❌ Skip video: {name}")
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
    dataloader = DataLoader(loader, batch_size=1, shuffle=False)

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