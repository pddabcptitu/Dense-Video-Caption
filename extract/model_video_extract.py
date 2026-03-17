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
            model_name=model_name, pretrained='openai', device=self.device
        )
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.size = size

    @torch.no_grad()
    def __call__(self, frames):
        if not isinstance(frames, torch.Tensor):
            if isinstance(frames[0], np.ndarray):
                frames = [Image.fromarray(frame) for frame in frames]
            
            transform_small = transforms.Compose([
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            frames = torch.stack([transform_small(frame) for frame in frames])

        frames = frames.to(self.device)
        
        num_frame = len(frames)
        n_iter = math.ceil(num_frame / self.batch_size)
        features = []

        for i in range(n_iter):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, num_frame)
            
            batch_features = self.model.encode_image(frames[start:end])
            
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            features.append(batch_features)

        return torch.cat(features, dim=0)