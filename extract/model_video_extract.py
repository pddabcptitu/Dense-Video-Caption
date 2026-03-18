import open_clip
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import math

class VideoExtract:
    def __init__(self, model_name='ViT-L-14', output_dim=768, batch_size=128, size=(224, 224)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, frames):
        if frames is None or len(frames) == 0:
            return None

        if not isinstance(frames, torch.Tensor):
            frames = torch.stack([
                self.preprocess(Image.fromarray(frame)) for frame in frames
            ])
        else:
            normalize = transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
            frames = torch.stack([normalize(f) for f in frames])

        num_frame = len(frames)
        features = []

        for i in range(0, num_frame, self.batch_size):
            batch = frames[i:i+self.batch_size].to(self.device).float()

            batch_features = self.model.encode_image(batch)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            features.append(batch_features.cpu())

        return torch.cat(features, dim=0)