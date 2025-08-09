# torchreid_embedding.py
from torchreid.utils import FeatureExtractor
import torch

class TorchReID:
    def __init__(self, model_name='osnet_x1_0', device='cuda'):
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_root='models',
            device=device
        )

    def __call__(self, image):
        features = self.extractor(image)  # Returns a list of numpy arrays
        return torch.tensor(features[0])
