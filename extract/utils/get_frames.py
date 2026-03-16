from decord import VideoReader, cpu
import numpy as np

def get_frames(video_path, target_fps=2, size=(224, 224)):

    vr = VideoReader(video_path, ctx=cpu(0), width=size[0], height=size[1])

    video_fps = vr.get_avg_fps()

    step = int(video_fps / target_fps)

    idx = np.arange(0, len(vr), step)

    frames = vr.get_batch(idx).asnumpy()

    return frames