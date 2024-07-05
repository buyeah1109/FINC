'''
    This code is borrowed from https://github.com/marcojira/fld
    Thanks for their great work
'''

import torch
import torchvision.transforms as transforms
from src.features.ImageFeatureExtractor import ImageFeatureExtractor


class DINOv2FeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = f"dinov2"
        super().__init__(save_path, logger)

        self.features_size = 768
        # From https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L44
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        self.model.to("cuda")
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        return self.model(img_batch)
