from torch.utils.data import Dataset
from extract.utils.get_frames import get_frames
from extract.utils import CURD_driver
import os

class VideoLoader(Dataset):
    def __init__(self, video_paths:list, fps=1, size=(224, 224), is_local=True):
        self.video_paths = video_paths
        self.fps = fps
        self.size = size
        self.is_local = is_local

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        if not self.is_local:
            name = self.video_paths.keys()[idx]
            id_ = self.video_paths[name]
            CURD_driver.download_file(id_, name)
            frames = get_frames(name, self.fps, self.size)
            os.remove(name)
        else:
            video_path = self.video_paths[idx]
            frames = get_frames(video_path, self.fps, self.size)

        return frames