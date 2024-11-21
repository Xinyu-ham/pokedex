from dataclasses import dataclass
from PIL import Image
import torch 
from torch import nn 
from transformers import ViTImageProcessor, ViTModel

@dataclass
class LabeledImage:
    image: Image.Image
    label: str
    @classmethod
    def load_image(cls, path: str, label: str='queryImage'):
        return cls(image=Image.open(path).convert("RGB"), label=label)

class ImageSimilarity:
    def __init__(self, model_name: str):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit = ViTModel.from_pretrained(model_name)
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def compare(self, image_set: list[LabeledImage], query: LabeledImage) -> dict[str, float]:
        image_features = torch.stack([self.processor(images=image.image, return_tensors='pt')['pixel_values'].squeeze() for image in image_set])
        query_features = self.processor(images=query.image, return_tensors='pt')['pixel_values']
        image_features = self.vit(image_features).last_hidden_state[:, 0].squeeze()
        query_features = self.vit(query_features).last_hidden_state[:, 0].squeeze()
        similarity = self.similarity(image_features, query_features)
        return {image.label: sim.item() for image, sim in zip(image_set, similarity)}