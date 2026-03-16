from torch.utils.data import Dataset
from extract.utils.get_frames import get_frames

class VideoLoader(Dataset):
    def __init__(self, video_paths:list, fps=1, size=(224, 224)):
        self.video_paths = video_paths
        self.fps = fps
        self.size = size

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = get_frames(video_path, self.fps, self.size)

        return frames