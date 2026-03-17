from decord import VideoReader, cpu
import numpy as np
import torch

def get_frames(video_path, target_fps=2, size=(224, 224)):
    vr = VideoReader(video_path, ctx=cpu(0), width=size[0], height=size[1])

    video_fps = vr.get_avg_fps()
    step = max(1, int(video_fps / target_fps)) 

    idx = np.arange(0, len(vr), step)
    frames = vr.get_batch(idx).asnumpy()   

    frames = torch.from_numpy(frames).float() / 255.0 
    frames = frames.permute(0, 3, 1, 2) 

    return frames