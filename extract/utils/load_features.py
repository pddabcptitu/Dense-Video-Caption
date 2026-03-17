import torch

def load_features(feature_path):
    ft = torch.load(feature_path, map_location='cpu')

    return ft